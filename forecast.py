"""Generate hourly and/or daily wind forecasts.

    python forecast.py                  # run both hourly and daily (one DB update)
    python forecast.py --mode hourly
    python forecast.py --mode daily
"""

import argparse
import os

import numpy as np
import pandas as pd
import pytz
import torch
import torch.serialization
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from pytorch_forecasting import TimeSeriesDataSet

from config import DailyConfig, HourlyConfig
from tft_common import tft_with_ignore
from update_data import get_conditions_table_daily, get_conditions_table_hourly, update_db

load_dotenv()
WORKING_DIR = Path(os.getenv('WORKING_DIRECTORY'))
TZ = pytz.timezone('America/Vancouver')

_hcfg = HourlyConfig.from_yaml()
_dcfg = DailyConfig.from_yaml()


def _warn_missing(data: pd.DataFrame, features: list, label: str) -> None:
    """Print a warning for any feature columns that contain NaN values."""
    for col in features:
        n_na = data[col].isna().sum()
        if n_na == 0:
            continue
        total = len(data)
        groups = data[col].isna().ne(data[col].isna().shift()).cumsum()
        max_gap = int(data[col].isna().groupby(groups).sum().max())
        if n_na == total:
            print(f'Warning [{label}] {col}: ALL {total} values missing — check scraper or DB')
        else:
            print(f'Warning [{label}] {col}: {n_na}/{total} NaN values will be filled '
                  f'(max consecutive gap: {max_gap})')


# ── Hourly forecast ───────────────────────────────────────────────────────────

def _prepare_hourly(update: bool = True) -> pd.DataFrame:
    data = get_conditions_table_hourly(
        encoder_length=_hcfg.encoder_length,
        prediction_length=_hcfg.prediction_length,
        update=update,
    )
    data['hour'] = data['datetime'].dt.hour
    data[_hcfg.targets] = data[_hcfg.targets].fillna(0).astype(float)
    if _hcfg.real_unknown:
        _warn_missing(data, _hcfg.real_unknown, 'hourly')
        data[_hcfg.real_unknown] = data[_hcfg.real_unknown].interpolate(method='linear', limit=2).ffill().bfill()
    if _hcfg.real_known:
        data[_hcfg.real_known]   = data[_hcfg.real_known].interpolate(method='linear')
    data.reset_index(drop=True, inplace=True)
    data['static']   = 'S'
    data['time_idx'] = np.arange(data.shape[0])
    return data


def _predict_hourly_target(data: pd.DataFrame, target: str,
                            forecast_q: int = 4) -> pd.DataFrame:
    ckpt_model   = WORKING_DIR / f'tft{target}HourlyCheckpoint.ckpt'
    ckpt_dataset = WORKING_DIR / f'{target}_training_dataset_hourly.pkl'
    with torch.serialization.safe_globals([TimeSeriesDataSet]):
        training_dataset = torch.load(ckpt_dataset, weights_only=False)
    inference_ds = TimeSeriesDataSet.from_dataset(
        training_dataset, data, predict=True, stop_randomization=True,
    )
    batch = inference_ds.to_dataloader(
        train=False, batch_size=len(inference_ds), shuffle=False, num_workers=4,
    )
    model = tft_with_ignore.load_from_checkpoint(ckpt_model)
    raw = model.predict(batch, mode='raw', return_index=True, return_x=True)
    pred_start = raw.index['time_idx'].max()
    y_mid = raw.output.prediction[:, :, forecast_q].numpy().reshape(-1)
    dt_pred = data['datetime'].iloc[pred_start:len(data)]
    if any(name in target for name in ('direction', 'lull', 'gust')):
        return pd.DataFrame({
            'datetime': dt_pred,
            target: y_mid,
            f'{target}_Q1': raw.output.prediction[:, :, 0].numpy().reshape(-1),
            f'{target}_Q7': raw.output.prediction[:, :, 6].numpy().reshape(-1),
        })
    return pd.DataFrame({'datetime': dt_pred, target: y_mid})


def _compute_hourly_quality(df: pd.DataFrame) -> pd.DataFrame:
    df['sailing_window'] = df['speed'] > 15
    df['speed_score'] = (df['gust'] - df['lull']) / df['speed']
    df['speed_score'] = np.clip(round(df['speed_score'] * -2.22 + 5.444), 1, 5)
    df['direction_score'] = np.clip(round((df['direction_Q7'] - df['direction_Q1']) / -22 + 12), 1, 5)
    df.loc[~df['sailing_window'], ['speed_score', 'direction_score']] = 0
    return df


def run_hourly(update: bool = True) -> None:
    print(f'Hourly forecast started at {datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")}')
    data = _prepare_hourly(update=update)
    df_out = pd.DataFrame()
    for target in _hcfg.targets:
        df_t = _predict_hourly_target(data, target)
        df_out = df_t if df_out.empty else df_out.merge(df_t, on='datetime', how='outer')
    df_out = _compute_hourly_quality(df_out)
    df_out = df_out[['datetime', 'speed', 'speed_score', 'direction_score']]
    df_out.to_csv(WORKING_DIR / 'hourly_speed_predictions.csv', index=False)
    df_out.to_json(WORKING_DIR / 'hourly_speed_predictions.json', orient='records', lines=True, date_format='iso')
    print(f'Hourly forecast complete at {datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")}')


# ── Daily forecast ────────────────────────────────────────────────────────────

def _prepare_daily(update: bool = True) -> pd.DataFrame:
    data = get_conditions_table_daily(
        encoder_length=_dcfg.encoder_length,
        prediction_length=_dcfg.prediction_length,
        update=update,
    )
    if _dcfg.real_unknown:
        _warn_missing(data, _dcfg.real_unknown, 'daily')
        data[_dcfg.real_unknown] = data[_dcfg.real_unknown].interpolate(method='linear', limit=2).ffill().bfill()
    if _dcfg.real_known:
        _warn_missing(data, _dcfg.real_known, 'daily')
        data[_dcfg.real_known]   = data[_dcfg.real_known].interpolate(method='linear', limit=2).ffill().bfill()
    data[_dcfg.targets]      = data[_dcfg.targets].fillna(0).astype(float)
    data.reset_index(drop=True, inplace=True)
    data['static']   = 'S'
    data['time_idx'] = np.arange(data.shape[0])
    return data


def _predict_daily_target(data: pd.DataFrame, target: str,
                           forecast_q: int = 3) -> pd.DataFrame:
    ckpt_dataset = WORKING_DIR / f'{target}_training_dataset_daily.pkl'
    with torch.serialization.safe_globals([TimeSeriesDataSet]):
        training_dataset = torch.load(ckpt_dataset, weights_only=False)
    inference_ds = TimeSeriesDataSet.from_dataset(
        training_dataset, data, predict=True, stop_randomization=True,
    )
    loader = inference_ds.to_dataloader(
        train=False, batch_size=len(inference_ds), shuffle=False, num_workers=4,
    )
    ckpt_model = WORKING_DIR / f'tft{target}DailyCheckpoint.ckpt'
    model = tft_with_ignore.load_from_checkpoint(ckpt_model)
    model.eval()
    raw = model.predict(loader, mode='raw', return_index=True, return_x=True)
    pred_start = raw.index['time_idx'].max()
    y_pred = raw.output.prediction[:, :, forecast_q].numpy().reshape(-1)
    dt_pred = data['datetime'].iloc[pred_start:len(data)]
    return pd.DataFrame({'datetime': dt_pred, target: y_pred})


def run_daily(update: bool = True) -> None:
    print(f'Daily forecast started at {datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")}')
    data = _prepare_daily(update=update)
    df_out = pd.DataFrame()
    for target in _dcfg.targets:
        df_t = _predict_daily_target(data, target)
        df_out = df_t if df_out.empty else df_out.merge(df_t, on='datetime', how='outer')
    df_out.to_csv(WORKING_DIR / 'daily_speed_predictions.csv', index=False)
    df_out.to_json(WORKING_DIR / 'daily_speed_predictions.json', orient='records', lines=True, date_format='iso')
    print(f'Daily forecast complete at {datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")}')


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Generate wind forecasts')
    p.add_argument('--mode', choices=['hourly', 'daily', 'both'], default='both',
                   help='Which forecast to run (default: both)')
    args = p.parse_args()

    if args.mode == 'both':
        update_db()                    # update DB once, then skip in each forecast
        run_hourly(update=False)
        run_daily(update=False)
    elif args.mode == 'hourly':
        run_hourly()
    else:
        run_daily()
