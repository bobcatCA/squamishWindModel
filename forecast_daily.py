import os

import numpy as np
import pandas as pd
import pytz
import torch
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from pytorch_forecasting import TimeSeriesDataSet

from config import DailyConfig
from tft_common import tft_with_ignore
from updateWeatherData import get_conditions_table_daily

load_dotenv()
WORKING_DIR = Path(os.getenv('WORKING_DIRECTORY'))

_cfg = DailyConfig.from_yaml()
ENCODER_LENGTH = _cfg.encoder_length
PREDICTION_LENGTH = _cfg.prediction_length
REAL_KNOWN = _cfg.real_known
REAL_UNKNOWN = _cfg.real_unknown
TARGETS = _cfg.targets


def prepare_data() -> pd.DataFrame:
    data = get_conditions_table_daily(
        encoder_length=ENCODER_LENGTH,
        prediction_length=PREDICTION_LENGTH,
    )
    data[REAL_UNKNOWN] = data[REAL_UNKNOWN].interpolate(method='linear', limit=2).ffill()
    data[REAL_KNOWN] = data[REAL_KNOWN].interpolate(method='linear', limit=2).ffill(limit=1)
    data[TARGETS] = data[TARGETS].fillna(0)
    data.reset_index(drop=True, inplace=True)
    data['static'] = 'S'
    data['time_idx'] = np.arange(data.shape[0])
    return data


def build_prediction_dataset(data: pd.DataFrame, target: str) -> TimeSeriesDataSet:
    checkpoint_dataset = WORKING_DIR / f'{target}_training_dataset_daily.pkl'
    with torch.serialization.safe_globals([TimeSeriesDataSet]):
        training_dataset = torch.load(checkpoint_dataset, weights_only=False)
    return TimeSeriesDataSet.from_dataset(
        training_dataset, data, predict=True, stop_randomization=True,
    )


def predict_and_store(data: pd.DataFrame, dataset: TimeSeriesDataSet,
                      target: str, forecast_q: int = 3) -> pd.DataFrame:
    checkpoint = WORKING_DIR / f'tft{target}DailyCheckpoint.ckpt'
    model = tft_with_ignore.load_from_checkpoint(checkpoint)
    model.eval()

    loader = dataset.to_dataloader(
        train=False, batch_size=len(dataset), shuffle=False, num_workers=4,
    )
    raw = model.predict(loader, mode='raw', return_index=True, return_x=True)

    pred_start = raw.index['time_idx'].max()
    y_pred = raw.output.prediction[:, :, forecast_q].numpy().reshape(-1)
    dt_pred = data['datetime'].iloc[pred_start:len(data)]

    return pd.DataFrame({'datetime': dt_pred, target: y_pred})


def main():
    output_csv = WORKING_DIR / 'daily_speed_predictions.csv'
    data = prepare_data()
    df_out = pd.DataFrame()

    for target in TARGETS:
        dataset = build_prediction_dataset(data, target)
        df_target = predict_and_store(data, dataset, target, forecast_q=3)
        df_out = df_target if df_out.empty else df_out.merge(df_target, on='datetime', how='outer')

    df_out.to_csv(output_csv, index=False)
    df_out.to_json(WORKING_DIR / 'daily_speed_predictions.json',
                   orient='records', lines=True)


if __name__ == '__main__':
    tz = pytz.timezone('America/Vancouver')
    print(f'Daily forecast started at {datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")}')
    main()
    print(f'Daily forecast complete at {datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")}')
