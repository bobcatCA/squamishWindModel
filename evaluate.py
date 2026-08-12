"""Evaluate model accuracy against historical data.

Walk through DB data in forecast-sized windows and plot measured vs
predicted speed:

    python evaluate.py
    python evaluate.py --start 2025-06-20
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
DB_PATH = WORKING_DIR / 'web_data' / 'weather_data_hourly.db'

_cfg = HourlyConfig.from_yaml()
ENCODER_LENGTH  = _cfg.encoder_length
PREDICTION_LENGTH = _cfg.prediction_length
REAL_KNOWN    = _cfg.real_known
REAL_UNKNOWN  = _cfg.real_unknown
TARGETS       = _cfg.targets


# ── Shared helpers ────────────────────────────────────────────────────────────

def _load_model_and_dataset(prefix: str, target: str):
    pkl_prefix = '' if prefix == 'tft' else f'{prefix}_'
    ckpt = WORKING_DIR / 'models' / f'{prefix}{target}HourlyCheckpoint.ckpt'
    pkl  = WORKING_DIR / 'models' / f'{pkl_prefix}{target}_training_dataset_hourly.pkl'
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
    df['year_fraction'] = (df['datetime'].dt.month * 30.416 + df['datetime'].dt.day) / 365
    if REAL_UNKNOWN:
        df[REAL_UNKNOWN] = df[REAL_UNKNOWN].ffill().bfill()
    if REAL_KNOWN:
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


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Evaluate hourly model accuracy')
    p.add_argument('--start', default='2025-06-20',
                   help='Start date for evaluate mode (default: 2025-06-20)')
    p.add_argument('--windows', type=int, default=100,
                   help='Number of forecast windows (default: 100)')
    args = p.parse_args()

    run_evaluate(
        start=pd.Timestamp(args.start, tz='America/Vancouver'),
        n_windows=args.windows,
    )
