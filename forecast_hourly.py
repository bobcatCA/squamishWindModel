import os

import numpy as np
import pandas as pd
import pytz
import torch.serialization
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from pytorch_forecasting import TimeSeriesDataSet

from config import HourlyConfig
from tft_common import tft_with_ignore
from updateWeatherData import get_conditions_table_hourly

load_dotenv()
WORKING_DIR = Path(os.getenv('WORKING_DIRECTORY'))

_cfg = HourlyConfig.from_yaml()
ENCODER_LENGTH = _cfg.encoder_length
PREDICTION_LENGTH = _cfg.prediction_length
REAL_KNOWN = _cfg.real_known
REAL_UNKNOWN = _cfg.real_unknown
TARGETS = _cfg.targets


def prepare_data() -> pd.DataFrame:
    data = get_conditions_table_hourly(
        encoder_length=ENCODER_LENGTH,
        prediction_length=PREDICTION_LENGTH,
    )
    data['hour'] = data['datetime'].dt.hour
    data[TARGETS] = data[TARGETS].fillna(0)
    data[REAL_UNKNOWN] = data[REAL_UNKNOWN].interpolate(method='linear', limit=2).ffill()
    data[REAL_KNOWN] = data[REAL_KNOWN].interpolate(method='linear')
    data.reset_index(drop=True, inplace=True)
    data['static'] = 'S'
    data['time_idx'] = np.arange(data.shape[0])
    return data


def load_model_and_predict(data: pd.DataFrame, target: str,
                           forecast_q: int = 3) -> pd.DataFrame:
    checkpoint_model = WORKING_DIR / f'tft{target}HourlyCheckpoint.ckpt'
    checkpoint_dataset = WORKING_DIR / f'{target}_training_dataset_hourly.pkl'

    with torch.serialization.safe_globals([TimeSeriesDataSet]):
        training_dataset = torch.load(checkpoint_dataset, weights_only=False)

    inference_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, data, predict=True, stop_randomization=True,
    )
    batch = inference_dataset.to_dataloader(
        train=False, batch_size=len(inference_dataset),
        shuffle=False, num_workers=4,
    )
    model = tft_with_ignore.load_from_checkpoint(checkpoint_model)

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


def compute_quality_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df['sailing_window'] = df['speed'] > 15

    df['speed_score'] = (df['gust'] - df['lull']) / df['speed']
    df['speed_score'] = (df['speed_score'] * -2.22) + 5.444
    df['speed_score'] = np.clip(round(df['speed_score']), 1, 5)

    df['direction_score'] = df['direction_Q7'] - df['direction_Q1']
    df['direction_score'] = (df['direction_score'] / -22) + 12
    df['direction_score'] = np.clip(round(df['direction_score']), 1, 5)

    df.loc[~df['sailing_window'], ['speed_score', 'direction_score']] = 0
    return df


def main():
    data = prepare_data()
    df_out = pd.DataFrame()

    for target in TARGETS:
        df_target = load_model_and_predict(data, target, forecast_q=4)
        df_out = df_target if df_out.empty else df_out.merge(df_target, on='datetime', how='outer')

    df_out = compute_quality_metrics(df_out)
    df_out = df_out[['datetime', 'speed', 'speed_score', 'direction_score']]
    df_out.to_csv(WORKING_DIR / 'hourly_speed_predictions.csv', index=False)
    df_out.to_json(WORKING_DIR / 'hourly_speed_predictions.json',
                   orient='records', lines=True)


if __name__ == '__main__':
    tz = pytz.timezone('America/Vancouver')
    print(f'Hourly forecast started at {datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")}')
    main()
    print(f'Hourly forecast complete at {datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")}')
