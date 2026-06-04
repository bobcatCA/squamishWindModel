"""Evaluate the hourly model against historical DB data.

Walks through the database from START_DATE in PREDICTION_LENGTH-hour steps,
runs inference for each window, and plots measured vs predicted speed.
"""

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
START_DATE = pd.Timestamp('2025-06-20', tz='America/Vancouver')

_cfg = HourlyConfig.from_yaml()
ENCODER_LENGTH = _cfg.encoder_length
PREDICTION_LENGTH = _cfg.prediction_length
REAL_KNOWN = _cfg.real_known
REAL_UNKNOWN = _cfg.real_unknown
TARGETS = _cfg.targets


def load_db_data() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            'SELECT * FROM weather WHERE datetime > ?',
            conn,
            params=(START_DATE.timestamp(),),
        )
    df['datetime'] = pd.to_datetime(df['datetime'], unit='s', utc=True).dt.tz_convert('America/Vancouver')
    df['sin_hour'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['year_fraction'] = (df['datetime'].dt.month * 30.416 + df['datetime'].dt.day) / 365
    df[REAL_UNKNOWN] = df[REAL_UNKNOWN].ffill().bfill()
    df[REAL_KNOWN] = df[REAL_KNOWN].ffill().bfill()
    df[TARGETS] = df[TARGETS].fillna(0)
    df.reset_index(drop=True, inplace=True)
    df['static'] = 'S'
    df['time_idx'] = np.arange(df.shape[0])
    return df


def predict_window(window: pd.DataFrame, target: str,
                   forecast_q: int = 3) -> pd.DataFrame:
    window = window.reset_index(drop=True)
    window['time_idx'] = window.index

    checkpoint_model = WORKING_DIR / f'tft{target}HourlyCheckpoint.ckpt'
    checkpoint_dataset = WORKING_DIR / f'{target}_training_dataset_hourly.pkl'

    with torch.serialization.safe_globals([TimeSeriesDataSet]):
        training_dataset = torch.load(checkpoint_dataset, weights_only=False)

    inference_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, window, predict=True, stop_randomization=True,
    )
    batch = inference_dataset.to_dataloader(
        train=False, batch_size=len(inference_dataset),
        shuffle=False, num_workers=4,
    )
    model = tft_with_ignore.load_from_checkpoint(checkpoint_model)

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


def main():
    df_measured = load_db_data()
    df_all_preds = pd.DataFrame()
    window_size = ENCODER_LENGTH + PREDICTION_LENGTH
    step = pd.Timedelta(hours=PREDICTION_LENGTH)

    t = START_DATE
    count = 0
    while count < 100:
        end = t + pd.Timedelta(hours=window_size)
        window = df_measured[(df_measured['datetime'] >= t) & (df_measured['datetime'] <= end)]

        if len(window) > window_size - 2:
            df_window_preds = pd.DataFrame()
            for target in TARGETS:
                df_t = predict_window(window, target, forecast_q=3)
                df_window_preds = df_t if df_window_preds.empty else df_window_preds.merge(df_t, on='datetime', how='outer')

            df_all_preds = pd.concat([df_all_preds, df_window_preds], ignore_index=True)

        t += step
        count += 1
        print(f'Window {count}/100')

    plt.plot(df_measured['datetime'], df_measured['speed'], label='measured')
    plt.plot(df_all_preds['datetime'], df_all_preds['speed'], label='predicted')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
