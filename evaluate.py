"""Evaluate model accuracy against historical data.

Default mode — walk through DB data in forecast-sized windows and plot
measured vs predicted speed:

    python evaluate.py
    python evaluate.py --start 2025-06-20

Comparison mode — evaluate two named model variants on the 2023 holdout
(a year masked from training) across all 4 targets:

    python evaluate.py --compare
    python evaluate.py --compare --windows 100
"""

import argparse
import os
import sqlite3

import matplotlib.pyplot as plt
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
DB_PATH = WORKING_DIR / 'weather_data_hourly.db'

_cfg = HourlyConfig.from_yaml()
ENCODER_LENGTH  = _cfg.encoder_length
PREDICTION_LENGTH = _cfg.prediction_length
REAL_KNOWN    = _cfg.real_known
REAL_UNKNOWN  = _cfg.real_unknown
TARGETS       = _cfg.targets
TARGET_START  = _cfg.weight_target_start_hour
TARGET_END    = _cfg.weight_target_end_hour

COMPARE_MODELS = [
    ('base (unweighted)',       'tftBase'),
    ('weighted (3× 10AM–6PM)', 'tftWeighted'),
]
UNITS = {'speed': 'kts', 'gust': 'kts', 'lull': 'kts', 'direction': 'deg'}


# ── Shared helpers ────────────────────────────────────────────────────────────

def _load_model_and_dataset(prefix: str, target: str):
    pkl_prefix = '' if prefix == 'tft' else f'{prefix}_'
    ckpt = WORKING_DIR / f'{prefix}{target}HourlyCheckpoint.ckpt'
    pkl  = WORKING_DIR / f'{pkl_prefix}{target}_training_dataset_hourly.pkl'
    with torch.serialization.safe_globals([TimeSeriesDataSet]):
        training_dataset = torch.load(pkl, weights_only=False)
    model = tft_with_ignore.load_from_checkpoint(ckpt)
    model.eval()
    return model, training_dataset


def _predict_window(window: pd.DataFrame, model, training_dataset,
                    target: str, forecast_q: int = 3) -> pd.DataFrame:
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
    y_pred = raw.output.prediction[:, :, forecast_q].numpy().reshape(-1)
    dt_pred = window['datetime'].iloc[pred_start:len(window)]
    if any(name in target for name in ('direction', 'lull', 'gust')):
        return pd.DataFrame({
            'datetime': dt_pred,
            target: y_pred,
            f'{target}_Q1': raw.output.prediction[:, :, 0].numpy().reshape(-1),
            f'{target}_Q7': raw.output.prediction[:, :, 6].numpy().reshape(-1),
        })
    return pd.DataFrame({'datetime': dt_pred, target: y_pred})


# ── Default evaluate mode (DB data, speed plot) ───────────────────────────────

def _load_db_data(start: pd.Timestamp) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            'SELECT * FROM weather WHERE datetime > ?',
            conn, params=(start.timestamp(),),
        )
    df['datetime'] = (
        pd.to_datetime(df['datetime'], unit='s', utc=True)
        .dt.tz_convert('America/Vancouver')
    )
    df['sin_hour']     = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['year_fraction'] = (df['datetime'].dt.month * 30.416 + df['datetime'].dt.day) / 365
    df[REAL_UNKNOWN] = df[REAL_UNKNOWN].ffill().bfill()
    df[REAL_KNOWN]   = df[REAL_KNOWN].ffill().bfill()
    df[TARGETS]      = df[TARGETS].fillna(0)
    df.reset_index(drop=True, inplace=True)
    df['static']   = 'S'
    df['time_idx'] = np.arange(df.shape[0])
    return df


def run_evaluate(start: pd.Timestamp, n_windows: int = 100) -> None:
    df_measured = _load_db_data(start)
    df_all_preds = pd.DataFrame()
    window_size = ENCODER_LENGTH + PREDICTION_LENGTH
    model, training_dataset = _load_model_and_dataset('tft', 'speed')

    t = start
    for count in range(n_windows):
        end = t + pd.Timedelta(hours=window_size)
        window = df_measured[(df_measured['datetime'] >= t) & (df_measured['datetime'] <= end)]
        if len(window) > window_size - 2:
            df_preds = _predict_window(window, model, training_dataset, 'speed')
            df_all_preds = pd.concat([df_all_preds, df_preds], ignore_index=True)
        t += pd.Timedelta(hours=PREDICTION_LENGTH)
        print(f'Window {count+1}/{n_windows}')

    plt.plot(df_measured['datetime'], df_measured['speed'], label='measured')
    plt.plot(df_all_preds['datetime'], df_all_preds['speed'], label='predicted')
    plt.legend()
    plt.show()


# ── Comparison mode (CSV holdout, all targets) ────────────────────────────────

def _load_holdout_data() -> pd.DataFrame:
    data = pd.read_csv(_cfg.data_path)
    data.dropna(thresh=14, inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'], utc=True)
    data = data.sort_values('datetime').reset_index(drop=True)
    num_cols = data.select_dtypes(include='number').columns
    data[num_cols] = data[num_cols].ffill().bfill()
    holdout = data[data['datetime'].dt.year == 2023].copy()
    holdout['static']   = 'S'
    holdout['time_idx'] = np.arange(len(holdout))
    return holdout.reset_index(drop=True)


def _run_windows(df: pd.DataFrame, model, training_dataset,
                 target: str, n_windows: int) -> pd.DataFrame:
    all_preds = []
    window_size = ENCODER_LENGTH + PREDICTION_LENGTH
    n_available = (len(df) - window_size) // PREDICTION_LENGTH
    for i in range(min(n_windows, n_available)):
        start_idx = i * PREDICTION_LENGTH
        window = df.iloc[start_idx: start_idx + window_size + 1]
        if len(window) >= window_size:
            try:
                all_preds.append(_predict_window(window, model, training_dataset, target))
            except Exception:
                if i == 0:
                    raise
    return pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()


def _circular_mae(pred: pd.Series, actual: pd.Series) -> float:
    diff = ((pred.values - actual.values + 180) % 360) - 180
    return float(np.abs(diff).mean())


def _compute_metrics(preds: pd.DataFrame, holdout: pd.DataFrame, target: str) -> dict:
    pred_col = f'{target}' if target not in preds.columns else target
    # preds may have target column named directly
    if target not in preds.columns:
        return {}
    actuals = holdout[['datetime', target]].copy()
    actuals['datetime'] = actuals['datetime'].dt.tz_localize(None)
    actuals = actuals.rename(columns={target: 'actual'})
    preds_copy = preds[['datetime', target]].copy()
    preds_copy['datetime'] = pd.to_datetime(preds_copy['datetime']).dt.tz_localize(None) \
        if preds_copy['datetime'].dtype != 'datetime64[ns]' \
        else preds_copy['datetime']
    df = preds_copy.merge(actuals, on='datetime', how='inner').dropna()
    df['hour'] = pd.to_datetime(df['datetime']).dt.hour
    in_win = (df['hour'] >= TARGET_START) & (df['hour'] < TARGET_END)
    mae_fn = _circular_mae if target == 'direction' else \
             lambda sub: (sub[target] - sub['actual']).abs().mean()
    return {
        'target_mae':  mae_fn(df[in_win]),
        'other_mae':   mae_fn(df[~in_win]),
        'overall_mae': mae_fn(df),
        'n_target': int(in_win.sum()),
        'n_other':  int((~in_win).sum()),
    }


def run_compare(n_windows: int = 100) -> None:
    print('Loading 2023 holdout data ...')
    holdout = _load_holdout_data()
    print(f'  {len(holdout):,} rows  '
          f'({holdout["datetime"].min().date()} → {holdout["datetime"].max().date()})')

    all_results: dict[str, dict] = {}
    for label, prefix in COMPARE_MODELS:
        missing = [t for t in TARGETS
                   if not (WORKING_DIR / f'{prefix}{t}HourlyCheckpoint.ckpt').exists()]
        if missing:
            print(f'\nSkipping {label!r} — missing checkpoints: {missing}')
            continue
        all_results[label] = {}
        print(f'\n── {label} ──')
        for target in TARGETS:
            print(f'  {target} ...', end='', flush=True)
            model, ds = _load_model_and_dataset(prefix, target)
            preds = _run_windows(holdout, model, ds, target, n_windows)
            if preds.empty:
                print(' no predictions')
                continue
            m = _compute_metrics(preds, holdout, target)
            all_results[label][target] = m
            unit = UNITS.get(target, '')
            print(f'  10AM–6PM {m["target_mae"]:.2f} {unit}  '
                  f'other {m["other_mae"]:.2f} {unit}  '
                  f'overall {m["overall_mae"]:.2f} {unit}')

    if len(all_results) < 2:
        print('\nNeed both checkpoint sets to print the comparison table.')
        return

    labels = list(all_results.keys())
    w = 30
    for target in TARGETS:
        if target not in all_results[labels[0]] or target not in all_results[labels[1]]:
            continue
        unit = UNITS.get(target, '')
        circ = '  (circular)' if target == 'direction' else ''
        sep = '=' * (w + 38)
        print(f'\n{sep}')
        print(f'TARGET: {target}  [{unit}{circ}]')
        print(sep)
        print(f'{"Model":<{w}}  {"10AM–6PM":>10}  {"Other hrs":>10}  {"Overall":>10}')
        print(f'{"-"*(w+38)}')
        for lbl in labels:
            m = all_results[lbl][target]
            print(f'{lbl:<{w}}  {m["target_mae"]:>10.2f}  '
                  f'{m["other_mae"]:>10.2f}  {m["overall_mae"]:>10.2f}')
        bm = all_results[labels[0]][target]
        wm = all_results[labels[1]][target]
        d_t = wm['target_mae'] - bm['target_mae']
        d_o = wm['other_mae']  - bm['other_mae']
        d_a = wm['overall_mae'] - bm['overall_mae']
        def fmt(v): return f'{v:+.2f}'
        print(f'{"-"*(w+38)}')
        print(f'{"Delta (weighted − base)":<{w}}  {fmt(d_t):>10}  {fmt(d_o):>10}  {fmt(d_a):>10}')
        print(sep)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Evaluate hourly model accuracy')
    p.add_argument('--compare', action='store_true',
                   help='Compare tftBase vs tftWeighted on 2023 holdout')
    p.add_argument('--start', default='2025-06-20',
                   help='Start date for default evaluate mode (default: 2025-06-20)')
    p.add_argument('--windows', type=int, default=100,
                   help='Number of forecast windows (default: 100)')
    args = p.parse_args()

    if args.compare:
        run_compare(n_windows=args.windows)
    else:
        run_evaluate(
            start=pd.Timestamp(args.start, tz='America/Vancouver'),
            n_windows=args.windows,
        )
