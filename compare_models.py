"""Compare base and weighted hourly models across all targets.

Evaluates against 2023 rows in hourly_database.csv — a year that was masked
out of both training runs and is therefore a true holdout.

Usage:
    python compare_models.py [--windows N]
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch.serialization
from dotenv import load_dotenv
from pathlib import Path
from pytorch_forecasting import TimeSeriesDataSet

from config import HourlyConfig
from tft_common import tft_with_ignore

load_dotenv()
WORKING_DIR = Path(os.getenv('WORKING_DIRECTORY'))

_cfg = HourlyConfig.from_yaml()
ENCODER_LENGTH = _cfg.encoder_length
PREDICTION_LENGTH = _cfg.prediction_length
REAL_KNOWN = _cfg.real_known
REAL_UNKNOWN = _cfg.real_unknown
TARGETS = _cfg.targets
TARGET_START = _cfg.weight_target_start_hour
TARGET_END = _cfg.weight_target_end_hour

UNITS = {'speed': 'kts', 'gust': 'kts', 'lull': 'kts', 'direction': 'deg'}

MODELS = [
    ('base (unweighted)',       'tftBase'),
    ('weighted (3× 10AM–6PM)', 'tftWeighted'),
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_holdout_data() -> pd.DataFrame:
    """Load hourly_database.csv, return 2023 rows (masked from training)."""
    data = pd.read_csv(_cfg.data_path)
    data.dropna(thresh=14, inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'], utc=True)
    data = data.sort_values('datetime').reset_index(drop=True)
    num_cols = data.select_dtypes(include='number').columns
    data[num_cols] = data[num_cols].ffill().bfill()

    holdout = data[data['datetime'].dt.year == 2023].copy()
    holdout['static'] = 'S'
    holdout['time_idx'] = np.arange(len(holdout))
    return holdout.reset_index(drop=True)


# ── Inference ─────────────────────────────────────────────────────────────────

def load_model_and_dataset(prefix: str, target: str):
    pkl_prefix = '' if prefix == 'tft' else f'{prefix}_'
    ckpt_path = WORKING_DIR / f'{prefix}{target}HourlyCheckpoint.ckpt'
    pkl_path  = WORKING_DIR / f'{pkl_prefix}{target}_training_dataset_hourly.pkl'
    with torch.serialization.safe_globals([TimeSeriesDataSet]):
        training_dataset = torch.load(pkl_path, weights_only=False)
    model = tft_with_ignore.load_from_checkpoint(ckpt_path)
    model.eval()
    return model, training_dataset


def predict_window(window: pd.DataFrame, model, training_dataset,
                   target: str) -> pd.DataFrame:
    window = window.copy().reset_index(drop=True)
    window['time_idx'] = window.index

    if training_dataset.weight is not None and training_dataset.weight not in window.columns:
        window[training_dataset.weight] = 1.0

    inference_ds = TimeSeriesDataSet.from_dataset(
        training_dataset, window, predict=True, stop_randomization=True,
    )
    batch = inference_ds.to_dataloader(
        train=False, batch_size=len(inference_ds), shuffle=False, num_workers=0,
    )
    raw = model.predict(batch, mode='raw', return_index=True, return_x=True)
    pred_start = raw.index['time_idx'].max()
    y_pred = raw.output.prediction[:, :, 3].numpy().reshape(-1)  # median quantile
    dt_pred = window['datetime'].iloc[pred_start:len(window)].dt.tz_localize(None)
    return pd.DataFrame({'datetime': dt_pred.values, f'{target}_pred': y_pred})


def run_windows(df: pd.DataFrame, model, training_dataset,
                target: str, n_windows: int) -> pd.DataFrame:
    all_preds = []
    window_size = ENCODER_LENGTH + PREDICTION_LENGTH
    n_available = (len(df) - window_size) // PREDICTION_LENGTH

    for i in range(min(n_windows, n_available)):
        start_idx = i * PREDICTION_LENGTH
        window = df.iloc[start_idx: start_idx + window_size + 1]
        if len(window) >= window_size:
            try:
                all_preds.append(predict_window(window, model, training_dataset, target))
            except Exception as exc:
                if i == 0:
                    raise

    return pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()


# ── Metrics ───────────────────────────────────────────────────────────────────

def _circular_mae(pred: pd.Series, actual: pd.Series) -> float:
    """Mean absolute error in degrees, taking the shortest angular path."""
    diff = ((pred.values - actual.values + 180) % 360) - 180
    return float(np.abs(diff).mean())


def compute_metrics(preds: pd.DataFrame, holdout: pd.DataFrame,
                    target: str) -> dict:
    actuals = holdout[['datetime', target]].copy()
    actuals['datetime'] = actuals['datetime'].dt.tz_localize(None)
    actuals = actuals.rename(columns={target: 'actual'})

    df = preds.merge(actuals, on='datetime', how='inner').dropna()
    df['hour'] = pd.to_datetime(df['datetime']).dt.hour
    in_win = (df['hour'] >= TARGET_START) & (df['hour'] < TARGET_END)

    pred_col = f'{target}_pred'
    if target == 'direction':
        mae_fn = lambda sub: _circular_mae(sub[pred_col], sub['actual'])
    else:
        mae_fn = lambda sub: (sub[pred_col] - sub['actual']).abs().mean()

    return {
        'target_mae':  mae_fn(df[in_win]),
        'other_mae':   mae_fn(df[~in_win]),
        'overall_mae': mae_fn(df),
        'n_target': int(in_win.sum()),
        'n_other':  int((~in_win).sum()),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--windows', type=int, default=100,
                   help='Forecast windows to evaluate per target (default: 100)')
    args = p.parse_args()

    print('Loading 2023 holdout data ...')
    holdout = load_holdout_data()
    print(f'  {len(holdout):,} rows  '
          f'({holdout["datetime"].min().date()} → {holdout["datetime"].max().date()})')

    # results[model_label][target] = metrics dict
    all_results: dict[str, dict] = {}

    for label, prefix in MODELS:
        missing = [t for t in TARGETS
                   if not (WORKING_DIR / f'{prefix}{t}HourlyCheckpoint.ckpt').exists()]
        if missing:
            print(f'\nSkipping {label!r} — missing checkpoints for: {missing}')
            continue

        all_results[label] = {}
        print(f'\n── {label} ──')
        for target in TARGETS:
            print(f'  {target} ...', end='', flush=True)
            model, training_dataset = load_model_and_dataset(prefix, target)
            preds = run_windows(holdout, model, training_dataset, target, args.windows)
            if preds.empty:
                print(' no predictions')
                continue
            m = compute_metrics(preds, holdout, target)
            all_results[label][target] = m
            unit = UNITS.get(target, '')
            print(f'  10AM–6PM {m["target_mae"]:.2f} {unit}  '
                  f'other {m["other_mae"]:.2f} {unit}  '
                  f'overall {m["overall_mae"]:.2f} {unit}')

    if len(all_results) < 2:
        print('\nNeed both model checkpoints to print the comparison.')
        return

    # ── Comparison tables ─────────────────────────────────────────────────────
    labels = list(all_results.keys())
    bm_all = all_results[labels[0]]
    wm_all = all_results[labels[1]]

    w = 30
    sep = '=' * (w + 38)
    note = f'Holdout: 2023 (masked from training)  |  {args.windows} windows'

    for target in TARGETS:
        if target not in bm_all or target not in wm_all:
            continue
        unit = UNITS.get(target, '')
        circ = '  (circular distance)' if target == 'direction' else ''
        print(f'\n{sep}')
        print(f'TARGET: {target}  [{unit}{circ}]   —   {note}')
        print(sep)
        print(f'{"Model":<{w}}  {"10AM–6PM":>10}  {"Other hrs":>10}  {"Overall":>10}')
        print(f'{"-"*(w+38)}')
        for lbl in labels:
            m = all_results[lbl][target]
            print(f'{lbl:<{w}}  {m["target_mae"]:>10.2f}  '
                  f'{m["other_mae"]:>10.2f}  {m["overall_mae"]:>10.2f}')
        bm = bm_all[target]
        wm = wm_all[target]
        d_t = wm['target_mae']  - bm['target_mae']
        d_o = wm['other_mae']   - bm['other_mae']
        d_a = wm['overall_mae'] - bm['overall_mae']
        def fmt(v): return f'{v:+.2f}'
        print(f'{"-"*(w+38)}')
        print(f'{"Delta (weighted − base)":<{w}}  {fmt(d_t):>10}  '
              f'{fmt(d_o):>10}  {fmt(d_a):>10}')
        print(sep)


if __name__ == '__main__':
    main()
