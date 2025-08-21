import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import psutil
import sqlite3
import time
import torch.serialization
from pathlib import Path
from dotenv import load_dotenv
from pytorch_forecasting import (TimeSeriesDataSet, TemporalFusionTransformer)
from transformDataDaily import add_scores_to_df


# This class was necessary to ignore the loss loading from the Checkpoint (apparently can cause problems)
class TftWithIgnore(TemporalFusionTransformer):
    def __init__(self, *args, **kwargs):
        self.save_hyperparameters(ignore=['loss'])  # Now this works as expected
        super().__init__(*args, **kwargs)

# Load environment and global variables
load_dotenv()
WORKING_DIRECTORY = Path(os.getenv('WORKING_DIRECTORY'))
MAX_ENCODER_LENGTH = 12  # Number of past observations to feed in
MAX_PREDICTION_LENGTH = 8  # Number of future steps to predict

# Model architecture features
# CATEGORICAL_FEATURES = ['comoxSky', 'vancouverSky', 'victoriaSky', 'whistlerSky', 'is_daytime']
CATEGORICAL_FEATURES = []
REAL_KNOWN_FEATURES = [
    'lillooetDegC', 'pembertonDegC', 'vancouverDegC', 'whistlerDegC',
    'sin_hour'
                       ]
REAL_UNKNOWN_FEATURES = [
    'comoxKPa', 'pamKPa'
                         ]
TARGET_VARIABLES = ['speed', 'gust', 'lull', 'direction']  # Each will have a separate model

def monitor_resources(interval=1, log_file='hourly_forecast_resource_log.txt'):
    """
    :param interval: (Integer), log per n seconds
    :param log_file: (Str), what to name the log file
    :return: None
    """
    pid = os.getpid()
    process = psutil.Process(pid)

    with open(log_file, 'w') as f:
        f.write('timestamp,cpu_percent,memory_mb\n')
        while True:
            try:
                cpu = process.cpu_percent(interval=interval)  # % of single core
                mem = process.memory_info().rss / 1024**2  # in MB
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                f.write(f'{timestamp},{cpu:.2f},{mem:.2f}\n')
                f.flush()
            except psutil.NoSuchProcess:
                break


def prepare_data(data):
    """
    Fetches and prepares hourly weather data for inference.
    :return: pd.DataFrame: Cleaned and formatted dataframe ready for prediction.
    """
    #
    # # Get raw data from local database
    # sql_database_path = Path(os.getenv('WORKING_DIRECTORY')) / 'weather_data_hourly.db'
    # conn = sqlite3.connect(sql_database_path)
    #
    # # Get existing table column names, keep only those matching, and commit.
    # data = pd.read_sql_query('SELECT * FROM weather WHERE datetime > ?', conn, params=(beginning_timestamp.timestamp(), ))
    # conn.close()

    # Add calculated columns
    data['datetime'] = pd.to_datetime(data['datetime'], unit='s', utc=True)
    data['datetime'] = data['datetime'].dt.tz_convert('America/Vancouver')
    data['hour'] = data['datetime'].dt.hour
    data['sin_hour'] = np.sin(2 * np.pi * data['datetime'].dt.hour / 24)
    data['year_fraction'] = ((data['datetime'].dt.month - 1) * 30.416 + data['datetime'].dt.day - 1) / 365
    data['is_daytime'] = data['datetime'].dt.hour.between(10, 17).astype(str)
    data['is_thermal'] = ((data['lillooetDegC'] - data['vancouverDegC']) > 5).astype(str)

    # Pre-process data (fill missing, re-index)
    data[REAL_UNKNOWN_FEATURES] = data[REAL_UNKNOWN_FEATURES].ffill()
    data[TARGET_VARIABLES] = data[TARGET_VARIABLES].ffill()
    data[REAL_KNOWN_FEATURES] = data[REAL_KNOWN_FEATURES].ffill(limit=1)
    data[CATEGORICAL_FEATURES] = data[CATEGORICAL_FEATURES].bfill(limit=1)
    data[REAL_KNOWN_FEATURES] = data[REAL_KNOWN_FEATURES].bfill(limit=1)
    data[REAL_UNKNOWN_FEATURES] = data[REAL_UNKNOWN_FEATURES].bfill(limit=1)
    data.reset_index(drop=True, inplace=True)
    # data['static'] = 'S'  # Required static group identifier
    data['static'] = 'S'
    data['time_idx'] = np.arange(data.shape[0])
    return data


def load_model_and_predict(data, target, forecast_q=4):
    """
    :param data: Pandas dataframe, pre-processed
    :param target: Str, name of target (label) variable
    :return: pd.DataFrame: DataFrame of forecast results for the given target.
    """

    # Load pre-trained checkpoint and generate PyTorch dataset object
    checkpoint_model = WORKING_DIRECTORY / f'tft{target}HourlyCheckpoint.ckpt'
    checkpoint_training_dataset = WORKING_DIRECTORY / f'{target}_training_dataset_hourly.pkl'
    data = data.reset_index(drop=True)
    data['time_idx'] = data.index
    data.loc[(data.shape[0] - MAX_PREDICTION_LENGTH):, REAL_UNKNOWN_FEATURES] = 0
    data.loc[(data.shape[0] - MAX_PREDICTION_LENGTH):, TARGET_VARIABLES] = 0

    # Load training dataset for parameters to pass into inference batch
    with torch.serialization.safe_globals([TimeSeriesDataSet]):
        training_dataset = torch.load(checkpoint_training_dataset, weights_only=False)

    # Create inference dataset
    inference_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        data,
        predict=True,
        stop_randomization=True
    )

    # Generate a dataloader batch (entire dataset for inference pass)
    batch = inference_dataset.to_dataloader(
        train=False,
        batch_size=len(inference_dataset),
        shuffle=False,
        num_workers=4
    )
    model = TftWithIgnore.load_from_checkpoint(checkpoint_model)

    # Generate raw predictions, and extract from output
    raw_predictions = model.predict(batch, mode='raw', return_index=True, return_x=True)
    datetime_measured = data['datetime']
    pred_datetime_idx = raw_predictions.index['time_idx'].max()
    y_pred_mean = raw_predictions.output.prediction[:, :, forecast_q].numpy().reshape(-1)  # For Quantile
    y_pred = raw_predictions.output.prediction[:, :, :].numpy().reshape(-1)
    datetime_pred = data['datetime'].iloc[pred_datetime_idx:len(data)]

    # Pull out Q1 and Q7 for direction, lull, and gust (used later in ratings)
    if any(name in target for name in ['direction', 'lull', 'gust']):
        y_pred_q1 = raw_predictions.output.prediction[:, :, 0].numpy().reshape(-1)
        y_pred_q7 = raw_predictions.output.prediction[:, :, 6].numpy().reshape(-1)
        result_df = pd.DataFrame({
            'datetime': datetime_pred,
            target: y_pred_mean,
            f'{target}_Q1': y_pred_q1,
            f'{target}_Q7': y_pred_q7
        })
    else:
        result_df = pd.DataFrame({'datetime': datetime_pred, target: y_pred_mean})
        pass

    return result_df


def compute_quality_metrics(df):
    """
    :param df: Pandas dataframe with raw predictions
    :return: Pandas dataframe with quality ratings
    """

    # Sailing window: Only True if the speed is above a certain value
    df['sailingWindow'] = df['speed'] > 13

    # Gust/Lull index: 1 to 5 rating for the relative magnitude of gusts/lulls
    df['speed_score'] = (df['gust_Q7'] - df['gust_Q1'] + df['lull_Q7'] - df['lull_Q1'])
    df['speed_score'] = 4 * ((df['speed_score'] - 22) / 14)
    df['speed_score'] = np.clip(round(df['speed_score']), 1, 5)

    # Direction index:1 to 5 rating for the relative direction variability
    df['direction_score'] = df['direction_Q7'] - df['direction_Q1']
    df['direction_score'] = 4 * ((df['direction_score'] - 40) / 50)
    df['direction_score'] = np.clip(round(df['direction_score']), 1, 5)

    # df = df.groupby('datetime').mean()
    df.loc[df['sailingWindow'] == False, ['speed_score', 'direction_score']] = 0
    return df

def plot_measured_forecast(df_measured, df_forecast):
    date_measured = df_measured['datetime']
    speed_measured = df_measured['speed']

    date_forecast = df_forecast['datetime']
    speed_forecast = df_forecast['speed']

    fig, ax = plt.subplots()
    ax.plot(date_measured, speed_measured, label='measured')
    ax.plot(date_forecast, speed_forecast, label='forecast')
    ax.legend()
    plt.legend()
    plt.show()
    pass


def main():
    start_date = pd.Timestamp('June 20, 2025', tz='America/Vancouver')

    # Get raw data from local database
    sql_path = Path(os.getenv('WORKING_DIRECTORY')) / 'weather_data_hourly.db'
    conn = sqlite3.connect(sql_path)

    # Get existing table column names, keep only those matching, and commit.
    df_measured = pd.read_sql_query('SELECT * FROM weather WHERE datetime > ?', conn, params=(start_date.timestamp(), ))
    df_measured = prepare_data(df_measured)
    conn.close()

    df_predicted = pd.DataFrame()  # Initialize empty df to compile all the predictions

    # while start_date < pd.Timestamp.today(tz='America/Vancouver'):
    count = 0
    while count < 100:
        end_date = start_date + pd.Timedelta(
            f'{MAX_ENCODER_LENGTH + MAX_PREDICTION_LENGTH} hours')
        data = df_measured[(df_measured['datetime'] >= start_date)
                                        & (df_measured['datetime'] <= end_date)]

        df_forecast = pd.DataFrame()
        if data.shape[0] > 18:
            for target in TARGET_VARIABLES:
                # print(target)
                df_forecast_target = load_model_and_predict(data, target, forecast_q=3)
                if df_forecast.empty:
                    df_forecast = df_forecast_target
                else:
                    df_forecast = df_forecast.merge(df_forecast_target, on='datetime', how='outer')
                    pass
                pass

            else: pass

        if df_predicted.empty:
            df_predicted = df_forecast
        else:
            df_predicted = pd.concat([df_predicted, df_forecast])

        start_date = start_date + pd.to_timedelta(f'8 hours')
        count+=1
        print(count)
        pass
    df_predicted = compute_quality_metrics(df_predicted)
    plt.plot(df_measured['datetime'], df_measured['speed'])
    plt.plot(df_predicted['datetime'], df_predicted['speed'])
    plt.show()

    print('done')

if __name__ == '__main__':
    # Run the main function
    main()
