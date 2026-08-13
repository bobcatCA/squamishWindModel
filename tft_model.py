"""TFT model for squamishSpeed — hourly multi-horizon forecast.

A from-scratch rebuild of the TFT pipeline, mirroring simple_model.py's
shape: one file, train()/test() functions, the same --train/--test/
--source db|csv CLI. Reads the same real_known/real_unknown/categorical
feature configuration from train_config.yaml's hourly section.

    python tft_model.py --train                   # train + save models/tft{target}HourlyCheckpoint.ckpt
    python tft_model.py --test [--source db|csv]   # predict + plot; csv exposes Hum features the live DB lacks
    python tft_model.py                            # train then test
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
import yaml
from dotenv import load_dotenv
from lightning.pytorch import Trainer
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting import QuantileLoss, RMSE, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

from config import CONFIG_YAML, HourlyConfig
from ec_scrape import normalize_sky_series

load_dotenv()
WORKING_DIR = Path(os.getenv('WORKING_DIRECTORY', '.'))
DB_PATH    = WORKING_DIR / 'web_data' / 'weather_data_hourly.db'


class tft_with_ignore(TemporalFusionTransformer):
    """TFT subclass that skips serialising loss and logging_metrics into hparams.

    Without this, loading a checkpoint with a custom loss object fails.
    """

    def __init__(self, *args, loss=None, logging_metrics=None, **kwargs):
        self.save_hyperparameters(ignore=['loss', 'logging_metrics'])
        super().__init__(*args, loss=loss, logging_metrics=logging_metrics, **kwargs)


def _apply_mask_intervals(data: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose date falls within any mask_interval in train_config.yaml."""
    with open(CONFIG_YAML) as f:
        intervals = yaml.safe_load(f).get('data', {}).get('mask_intervals', [])
    for iv in intervals:
        start = pd.Timestamp(iv['start']).date()
        end = pd.Timestamp(iv['end']).date()
        mask = (data['datetime'].dt.date >= start) & (data['datetime'].dt.date <= end)
        n = int(mask.sum())
        data = data[~mask].reset_index(drop=True)
        print(f'  Masked {n:,} rows  ({iv["start"]} → {iv["end"]})')
    return data


def _build_dataset(data: pd.DataFrame, target: str, cfg: HourlyConfig, cutoff: int) -> TimeSeriesDataSet:
    required = set((cfg.real_known or []) + (cfg.real_unknown or []) + (cfg.categorical or []) + [target])
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f'Feature columns missing from data: {missing}')

    kwargs = dict(
        time_idx='time_idx',
        target=target,
        group_ids=['static'],
        static_categoricals=['static'],
        time_varying_unknown_reals=[target] + (cfg.real_unknown or []),
        min_encoder_length=cfg.min_encoder_length,
        max_encoder_length=cfg.encoder_length,
        min_prediction_length=cfg.min_prediction_length,
        max_prediction_length=cfg.prediction_length,
        target_normalizer=GroupNormalizer(groups=['static']),
        add_relative_time_idx=True,
        add_target_scales=True,
        randomize_length=None,
    )
    if cfg.categorical:
        kwargs['time_varying_known_categoricals'] = cfg.categorical
    if cfg.real_known:
        kwargs['time_varying_known_reals'] = cfg.real_known
    if cfg.allow_missing_timesteps:
        kwargs['allow_missing_timesteps'] = True
    return TimeSeriesDataSet(data[data.time_idx <= cutoff], **kwargs)


def train(epochs: int = None) -> None:
    cfg = HourlyConfig.from_yaml()
    if epochs is not None:
        cfg.max_epochs = epochs
    target = cfg.targets[0]

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

    print(f'Training {target} (Hourly)')
    print(f'real_known={cfg.real_known}  real_unknown={cfg.real_unknown}  categorical={cfg.categorical}')

    training = _build_dataset(data, target, cfg, cutoff)
    training.save(f'models/{target}_training_dataset_hourly.pkl')

    val_data = data if cfg.val_full_data else data[data.time_idx > cutoff]
    validation = TimeSeriesDataSet.from_dataset(
        training, val_data, predict=cfg.val_predict_mode, stop_randomization=True,
    )

    train_dl = training.to_dataloader(train=True, batch_size=cfg.batch_size, num_workers=4)
    val_dl = validation.to_dataloader(train=False, batch_size=cfg.batch_size, num_workers=4)

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

    ckpt_path = f'models/tft{target}HourlyCheckpoint.ckpt'
    trainer.save_checkpoint(ckpt_path)
    print(f'Saved {ckpt_path}')


def _load_model_and_dataset(target: str):
    ckpt = WORKING_DIR / 'models' / f'tft{target}HourlyCheckpoint.ckpt'
    pkl  = WORKING_DIR / 'models' / f'{target}_training_dataset_hourly.pkl'
    with torch.serialization.safe_globals([TimeSeriesDataSet]):
        training_dataset = torch.load(pkl, weights_only=False)
    model = tft_with_ignore.load_from_checkpoint(ckpt)
    model.eval()
    return model, training_dataset


def _load_test_df(cfg: HourlyConfig, start_ts: pd.Timestamp, source: str,
                   end_ts: pd.Timestamp = None) -> pd.DataFrame:
    if source == 'csv':
        df = pd.read_csv(cfg.data_path)
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert('America/Vancouver')
        df = df[df['datetime'] > start_ts]
        if end_ts is not None:
            df = df[df['datetime'] <= end_ts]
        df = df.sort_values('datetime').reset_index(drop=True)
    else:
        with sqlite3.connect(DB_PATH) as conn:
            if end_ts is not None:
                df = pd.read_sql_query(
                    'SELECT * FROM weather WHERE datetime > ? AND datetime <= ?',
                    conn, params=(start_ts.timestamp(), end_ts.timestamp()),
                )
            else:
                df = pd.read_sql_query(
                    'SELECT * FROM weather WHERE datetime > ?', conn, params=(start_ts.timestamp(),),
                )
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s', utc=True).dt.tz_convert('America/Vancouver')
        df = df.sort_values('datetime').reset_index(drop=True)

    for col in (cfg.categorical or []):
        if col in df.columns:
            df[col] = normalize_sky_series(df[col].ffill().bfill()).astype(str)
    if cfg.real_unknown:
        df[cfg.real_unknown] = df[cfg.real_unknown].ffill().bfill()
    if cfg.real_known:
        df[cfg.real_known] = df[cfg.real_known].ffill().bfill()
    return df


def test(start: str = None, end: str = None, save: bool = False, source: str = 'db') -> None:
    cfg = HourlyConfig.from_yaml()
    target = cfg.targets[0]
    model, training_dataset = _load_model_and_dataset(target)

    start_ts = (
        pd.Timestamp(start, tz='America/Vancouver') if start
        else pd.Timestamp('2000-01-01', tz='America/Vancouver')
    )
    end_ts = pd.Timestamp(end, tz='America/Vancouver') if end else None
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
            med_idx = raw.output.prediction.shape[-1] // 2
            for step in range(cfg.prediction_length):
                row_idx = pred_start + step
                if row_idx >= len(window):
                    continue
                records.append({
                    'datetime': window['datetime'].iloc[row_idx],
                    'pred': float(raw.output.prediction[0, step, med_idx]),
                })
        except Exception:
            pass
        if (w + 1) % 10 == 0 or (w + 1) == n_windows:
            print(f'  {w + 1}/{n_windows}', end='\r', flush=True)
    print()

    preds = pd.DataFrame(records).drop_duplicates(subset='datetime').sort_values('datetime')
    merged = df[['datetime', target]].merge(preds, on='datetime', how='inner')
    if merged.empty:
        print('No predictions produced — check that the model/dataset match the current feature config.')
        return

    actual = merged[target].values
    pred = merged['pred'].values
    mae = np.mean(np.abs(pred - actual))
    rmse = np.sqrt(np.mean((pred - actual) ** 2))
    print(f'source={source}  {len(merged):,} predicted rows from {merged["datetime"].min()} → {merged["datetime"].max()}')
    print(f'MAE={mae:.2f}  RMSE={rmse:.2f}  (actual mean={actual.mean():.2f}, max={actual.max():.2f})')

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(df['datetime'], df[target], color='#333333', linewidth=0.8, alpha=0.5, label='actual')
    ax.plot(merged['datetime'], merged['pred'], color='#1f77b4', linewidth=1.1, alpha=0.9, label='predicted (median)')
    ax.set_ylabel(target)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.set_title(f'TFT — predicted vs actual {target}  [source={source}]')
    plt.tight_layout()

    if save:
        out = WORKING_DIR / 'forecasts' / f'tft_test_{source}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved {out}')
    else:
        plt.show()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='TFT model for squamishSpeed (hourly)')
    p.add_argument('--train', action='store_true', help='Train and save the model')
    p.add_argument('--test',  action='store_true', help='Predict against DB or CSV and plot')
    p.add_argument('--epochs', type=int, default=None, help='Override max_epochs from train_config.yaml')
    p.add_argument('--start', help='Test start date YYYY-MM-DD (default: all available data)')
    p.add_argument('--end', help='Test end date YYYY-MM-DD, inclusive (default: no upper bound) — '
                                  'use with --start to bound a small window for fast iteration')
    p.add_argument('--source', choices=['db', 'csv'], default='db',
                   help='Where --test pulls data from: live SQLite DB (default), or the full '
                        'training CSV — use csv for features not in the live DB')
    p.add_argument('--save', action='store_true', help='Save plot to forecasts/ instead of showing it')
    args = p.parse_args()

    run_train = args.train or not (args.train or args.test)
    run_test  = args.test or not (args.train or args.test)

    if run_train:
        train(epochs=args.epochs)
    if run_test:
        test(start=args.start, end=args.end, save=args.save, source=args.source)
