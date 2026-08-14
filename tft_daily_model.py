"""TFT model for daily targets (2pm snapshot) — multi-day forecast.

One model per target, all sharing this same blueprint: the daily: section's
real_known/real_unknown/hyperparameters from train_config.yaml (DegC columns
are real_known because EC's 6-day daily forecast — pull_forecast_daily() in
ec_scrape.py — actually covers them; KPa/Hum/squamishDegC are real_unknown
since EC doesn't forecast pressure or humidity, and squamishDegC is the local
sensor, never forecastable). Reuses tft_with_ignore/_build_dataset/
_apply_mask_intervals from tft_model.py rather than duplicating them.

Available targets are train_config.yaml's daily: targets list — currently
squamishSpeed (the raw 2pm speed) and speed_steadiness/direction_steadiness
(the 0-5 scores from build_dataset.py's add_speed_steadiness()/
add_direction_steadiness()). --target selects which one; defaults to the
first entry (squamishSpeed). The steadiness scores only exist in the
daily-built CSV, not the live DB, so use --source csv for those.

    python tft_daily_model.py --train [--target NAME]                      # train + save models/tft{target}DailyCheckpoint.ckpt
    python tft_daily_model.py --test [--target NAME] [--source db|csv]     # predict + plot
    python tft_daily_model.py [--target NAME]                              # train then test

CAVEAT: --test currently feeds the real_known window from OBSERVED station
temperatures (DB or CSV), not genuine EC forecast values — pull_forecast_daily()
isn't wired into either test path yet, so this is a "perfect foresight"
backtest, not a true live-inference test. Wiring in real forecast temps for
actual daily forecast generation is a follow-up (see forecast.py, which
doesn't have a daily path either right now).
"""

import argparse
import os
import sqlite3
from pathlib import Path

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.serialization
from dotenv import load_dotenv
from lightning.pytorch import Trainer
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting import QuantileLoss, RMSE, TemporalFusionTransformer, TimeSeriesDataSet

from config import DailyConfig
from ec_scrape import normalize_sky_series
from tft_model import _apply_mask_intervals, _build_dataset, tft_with_ignore

load_dotenv()
WORKING_DIR = Path(os.getenv('WORKING_DIRECTORY', '.'))
DB_PATH    = WORKING_DIR / 'web_data' / 'weather_data_hourly.db'
DAILY_HOUR = 14  # matches build_dataset.py's DAILY_HOUR

# Targets that only exist in the daily-built CSV (steadiness scores are
# computed by build_dataset.py, never written to the live SQLite DB).
CSV_ONLY_TARGETS = {'speed_steadiness', 'direction_steadiness'}


def _resolve_target(cfg: DailyConfig, target: str = None) -> str:
    if target is None:
        return cfg.targets[0]
    if target not in cfg.targets:
        raise SystemExit(f'--target {target!r} not in train_config.yaml daily: targets={cfg.targets}')
    return target


def train(epochs: int = None, target: str = None) -> None:
    cfg = DailyConfig.from_yaml()
    if epochs is not None:
        cfg.max_epochs = epochs
    target = _resolve_target(cfg, target)

    data = pd.read_csv(cfg.data_path)
    data.dropna(thresh=14, inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'], utc=True)
    data = data.sort_values('datetime')
    for col in (cfg.categorical or []):
        data[col] = data[col].astype(str)
    num_cols = data.select_dtypes(include='number').columns
    data[num_cols] = data[num_cols].ffill().bfill()

    data = _apply_mask_intervals(data)
    data['static'] = 'S'
    data['time_idx'] = np.arange(len(data))
    data = data.reset_index(drop=True)

    cutoff = cfg.training_cutoff(data['time_idx'].max())

    print(f'Training {target} (Daily)')
    print(f'real_known={cfg.real_known}  real_unknown={cfg.real_unknown}  categorical={cfg.categorical}')

    training = _build_dataset(data, target, cfg, cutoff)
    training.save(f'models/{target}_training_dataset_daily.pkl')

    val_data = data if cfg.val_full_data else data[data.time_idx > cutoff]
    validation = TimeSeriesDataSet.from_dataset(
        training, val_data, predict=cfg.val_predict_mode, stop_randomization=True,
    )

    # num_workers=0: ~600 rows already in memory, so spawning worker subprocesses is pure
    # multiprocessing overhead here, not a speedup (unlike hourly's much larger dataset).
    train_dl = training.to_dataloader(train=True, batch_size=cfg.batch_size, num_workers=0)
    val_dl = validation.to_dataloader(train=False, batch_size=cfg.batch_size, num_workers=0)

    pl.seed_everything(42)

    lr = cfg.learning_rate
    if cfg.find_lr:
        lr_probe = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=lr,
            hidden_size=cfg.hidden_size,
            attention_head_size=cfg.attention_head_size,
            dropout=cfg.dropout,
            hidden_continuous_size=cfg.hidden_continuous_size,
            output_size=1,
            loss=RMSE(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        lr_trainer = Trainer(
            accelerator=cfg.accelerator,
            gradient_clip_val=cfg.gradient_clip_val,
            max_epochs=1,
            enable_checkpointing=False,
            logger=False,
        )
        res = Tuner(lr_trainer).lr_find(
            lr_probe, train_dataloaders=train_dl, val_dataloaders=val_dl, max_lr=10.0, min_lr=1e-6,
        )
        lr = res.suggestion()
        print(f'  Optimal LR: {lr:.2e}')

    model_kwargs = dict(
        learning_rate=lr,
        hidden_size=cfg.hidden_size,
        attention_head_size=cfg.attention_head_size,
        dropout=cfg.dropout,
        hidden_continuous_size=cfg.hidden_continuous_size,
        output_size=cfg.output_size,
        loss=QuantileLoss(),
        logging_metrics=[],
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    if cfg.optimizer:
        model_kwargs['optimizer'] = cfg.optimizer

    tft = tft_with_ignore.from_dataset(training, **model_kwargs)

    trainer = Trainer(accelerator=cfg.accelerator, max_epochs=cfg.max_epochs, gradient_clip_val=cfg.gradient_clip_val)
    trainer.fit(tft, train_dl, val_dl)

    ckpt_path = f'models/tft{target}DailyCheckpoint.ckpt'
    trainer.save_checkpoint(ckpt_path)
    print(f'Saved {ckpt_path}')


def _load_model_and_dataset(target: str):
    ckpt = WORKING_DIR / 'models' / f'tft{target}DailyCheckpoint.ckpt'
    pkl  = WORKING_DIR / 'models' / f'{target}_training_dataset_daily.pkl'
    with torch.serialization.safe_globals([TimeSeriesDataSet]):
        training_dataset = torch.load(pkl, weights_only=False)
    model = tft_with_ignore.load_from_checkpoint(ckpt)
    model.eval()
    return model, training_dataset


def _load_test_df(cfg: DailyConfig, start_ts: pd.Timestamp, source: str,
                   end_ts: pd.Timestamp = None) -> pd.DataFrame:
    if source == 'csv':
        df = pd.read_csv(cfg.data_path)
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert('America/Vancouver')
        df = df[df['datetime'] > start_ts]
    else:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(
                'SELECT * FROM weather WHERE datetime > ?', conn, params=(start_ts.timestamp(),),
            )
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s', utc=True).dt.tz_convert('America/Vancouver')
        df = df[df['datetime'].dt.hour == DAILY_HOUR]

    if end_ts is not None:
        df = df[df['datetime'] <= end_ts]
    df = df.sort_values('datetime').reset_index(drop=True)

    for col in (cfg.categorical or []):
        if col in df.columns:
            df[col] = normalize_sky_series(df[col].ffill().bfill()).astype(str)
    if cfg.real_unknown:
        df[cfg.real_unknown] = df[cfg.real_unknown].ffill().bfill()
    if cfg.real_known:
        df[cfg.real_known] = df[cfg.real_known].ffill().bfill()
    return df


def _predict(cfg: DailyConfig, target: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp, source: str):
    """Run the sliding-window backtest for one target. Returns (df, merged, mae, rmse)
    or (df, None, None, None) if no predictions were produced."""
    model, training_dataset = _load_model_and_dataset(target)

    df = _load_test_df(cfg, start_ts, source, end_ts)
    df[target] = df[target].fillna(0)
    df = df.dropna(subset=(cfg.real_known or []) + (cfg.real_unknown or []) + [target]).reset_index(drop=True)
    df['static'] = 'S'
    df['time_idx'] = np.arange(len(df))

    window_size = cfg.encoder_length + cfg.prediction_length
    stride = cfg.prediction_length
    n_windows = max(0, (len(df) - window_size) // stride + 1)
    records = []

    for w, start_idx in enumerate(range(0, len(df) - window_size + 1, stride)):
        window = df.iloc[start_idx: start_idx + window_size].copy().reset_index(drop=True)
        window['time_idx'] = window.index
        try:
            inference_ds = TimeSeriesDataSet.from_dataset(
                training_dataset, window, predict=True, stop_randomization=True,
            )
            batch = inference_ds.to_dataloader(train=False, batch_size=len(inference_ds), shuffle=False, num_workers=0)
            raw = model.predict(batch, mode='raw', return_index=True, return_x=True)
            pred_start = int(raw.index['time_idx'].max())
            n_quantiles = raw.output.prediction.shape[-1]
            med_idx = n_quantiles // 2
            lo_idx, hi_idx = med_idx - 1, med_idx + 1
            for step in range(cfg.prediction_length):
                row_idx = pred_start + step
                if row_idx >= len(window):
                    continue
                records.append({
                    'datetime': window['datetime'].iloc[row_idx],
                    'pred': float(raw.output.prediction[0, step, med_idx]),
                    'pred_lo': float(raw.output.prediction[0, step, lo_idx]),
                    'pred_hi': float(raw.output.prediction[0, step, hi_idx]),
                })
        except Exception:
            pass
        if (w + 1) % 10 == 0 or (w + 1) == n_windows:
            print(f'  {target}: {w + 1}/{n_windows}', end='\r', flush=True)
    print()

    preds = pd.DataFrame(records).drop_duplicates(subset='datetime').sort_values('datetime')
    merged = df[['datetime', target]].merge(preds, on='datetime', how='inner')
    if merged.empty:
        return df, None, None, None

    actual = merged[target].values
    pred = merged['pred'].values
    mae = np.mean(np.abs(pred - actual))
    rmse = np.sqrt(np.mean((pred - actual) ** 2))
    return df, merged, mae, rmse


def _plot_target(ax, target: str, df: pd.DataFrame, merged: pd.DataFrame, source: str) -> None:
    q = QuantileLoss().quantiles
    med_idx = len(q) // 2
    lo_q, hi_q = q[med_idx - 1], q[med_idx + 1]

    ax.plot(df['datetime'], df[target], color='#333333', linewidth=0.8, alpha=0.5, marker='o', markersize=3, label='actual')
    ax.fill_between(merged['datetime'], merged['pred_lo'], merged['pred_hi'],
                     color='#1f77b4', alpha=0.15, label=f'Q{lo_q:.2f}–Q{hi_q:.2f}')
    ax.plot(merged['datetime'], merged['pred'], color='#1f77b4', linewidth=1.1, alpha=0.9, marker='o', markersize=3, label='predicted (median)')
    ax.set_ylabel(target)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.set_title(f'TFT — predicted vs actual {target} (daily, 2pm)  [source={source}]')


def test(start: str = None, end: str = None, save: bool = False, source: str = 'db', target: str = None) -> None:
    cfg = DailyConfig.from_yaml()

    targets = cfg.targets if target == 'all' else [_resolve_target(cfg, target)]
    for t in targets:
        if t in CSV_ONLY_TARGETS and source == 'db':
            raise SystemExit(f'{t} only exists in the daily-built CSV, not the live DB — use --source csv')

    start_ts = (
        pd.Timestamp(start, tz='America/Vancouver') if start
        else pd.Timestamp('2000-01-01', tz='America/Vancouver')
    )
    end_ts = pd.Timestamp(end, tz='America/Vancouver') if end else None

    fig, axes = plt.subplots(len(targets), 1, figsize=(16, 5 * len(targets)), squeeze=False)
    axes = axes[:, 0]
    any_plotted = False

    for ax, t in zip(axes, targets):
        df, merged, mae, rmse = _predict(cfg, t, start_ts, end_ts, source)
        if merged is None:
            print(f'{t}: no predictions produced — check that the model/dataset match the current feature config.')
            continue
        actual = merged[t].values
        print(f'{t}: source={source}  {len(merged):,} predicted rows from {merged["datetime"].min()} → {merged["datetime"].max()}')
        print(f'{t}: MAE={mae:.2f}  RMSE={rmse:.2f}  (actual mean={actual.mean():.2f}, max={actual.max():.2f})')
        _plot_target(ax, t, df, merged, source)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return

    plt.tight_layout()
    if save:
        suffix = 'all' if target == 'all' else targets[0]
        out = WORKING_DIR / 'forecasts' / f'tft_daily_test_{suffix}_{source}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved {out}')
    else:
        plt.show()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='TFT model for squamishSpeed (daily)')
    p.add_argument('--train', action='store_true', help='Train and save the model')
    p.add_argument('--test',  action='store_true', help='Predict against DB or CSV and plot')
    p.add_argument('--epochs', type=int, default=None, help='Override max_epochs from train_config.yaml')
    p.add_argument('--target', default=None,
                   help="Which daily: targets entry to train/test (default: first entry, squamishSpeed). "
                        "--test --target all plots every target in one stacked figure.")
    p.add_argument('--start', help='Test start date YYYY-MM-DD (default: all available data)')
    p.add_argument('--end', help='Test end date YYYY-MM-DD, inclusive (default: no upper bound)')
    p.add_argument('--source', choices=['db', 'csv'], default='db',
                   help="Where --test pulls data from: live SQLite DB filtered to each day's 2pm "
                        'row (default), or the full daily-built CSV — required for CSV-only targets')
    p.add_argument('--save', action='store_true', help='Save plot to forecasts/ instead of showing it')
    args = p.parse_args()

    run_train = args.train or not (args.train or args.test)
    run_test  = args.test or not (args.train or args.test)

    if run_train and args.target == 'all':
        raise SystemExit('--target all trains one model at a time — pass an explicit target, or omit --target')

    if run_train:
        train(epochs=args.epochs, target=args.target)
    if run_test:
        test(start=args.start, end=args.end, save=args.save, source=args.source, target=args.target)
