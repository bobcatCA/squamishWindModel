"""Generate the hourly wind forecast.

    python forecast.py
"""

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

from config import HourlyConfig
from tft_model import tft_with_ignore
from update_data import get_conditions_table_hourly

load_dotenv()
WORKING_DIR = Path(os.getenv('WORKING_DIRECTORY'))
TZ = pytz.timezone('America/Vancouver')

_hcfg = HourlyConfig.from_yaml()


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
    ckpt_model   = WORKING_DIR / 'models' / f'tft{target}HourlyCheckpoint.ckpt'
    ckpt_dataset = WORKING_DIR / 'models' / f'{target}_training_dataset_hourly.pkl'
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
    return pd.DataFrame({'datetime': dt_pred, target: y_mid})


def run_hourly(update: bool = True) -> None:
    print(f'Hourly forecast started at {datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")}')
    data = _prepare_hourly(update=update)
    df_out = pd.DataFrame()
    for target in _hcfg.targets:
        df_t = _predict_hourly_target(data, target)
        df_out = df_t if df_out.empty else df_out.merge(df_t, on='datetime', how='outer')
    df_out = df_out[['datetime'] + _hcfg.targets]
    df_out.to_csv(WORKING_DIR / 'forecasts' / 'hourly_speed_predictions.csv', index=False)
    df_out.to_json(WORKING_DIR / 'forecasts' / 'hourly_speed_predictions.json', orient='records', lines=True, date_format='iso')
    print(f'Hourly forecast complete at {datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    run_hourly()
