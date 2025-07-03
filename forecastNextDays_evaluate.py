import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import psutil
import random
import sqlite3
import threading
import time
import torch
import pytz
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from transformDataDaily import add_scores_to_df
from updateWeatherData import get_conditions_table_daily


# This class was necessary to ignore the loss loading from the Checkpoint (apparently can cause problems)
class tft_with_ignore(TemporalFusionTransformer):
    def __init__(self, *args, **kwargs):
        self.save_hyperparameters(ignore=['loss'])  # Now this works as expected
        super().__init__(*args, **kwargs)


# Load environment and global variables
load_dotenv()
WORKING_DIRECTORY = Path(os.getenv('WORKING_DIRECTORY'))
MAX_ENCODER_LENGTH = 8  # Number of past observations to feed in
MAX_PREDICTION_LENGTH = 5  # Number of future steps to predict

# Model architecture features
CATEGORICAL_FEATURES = ['comoxSky', 'vancouverSky', 'victoriaSky', 'whistlerSky']
REAL_KNOWN_FEATURES = ['year_fraction', 'comoxDegC', 'lillooetDegC',
                                     'pembertonDegC', 'vancouverDegC', 'victoriaDegC', 'whistlerDegC']
REAL_UNKNOWN_FEATURES = ['comoxKPa', 'vancouverKPa', 'lillooetKPa', 'pamKPa', 'ballenasKPa']
TARGET_VARIABLES = ['speed', 'speed_variability', 'direction_variability']  # Multiple targets - have to make a model for each


def monitor_resources(interval=1, log_file='daily_forecast_resource_log.txt'):
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


def prepare_data(start_time, encoder_length=MAX_ENCODER_LENGTH, prediction_length=MAX_PREDICTION_LENGTH):
    """
    Fetch and preprocess the weather data for forecasting.

    :return: DataFrame: A prepared dataframe ready for model ingestion.
    """
    start_time_14 = pd.to_datetime(start_time) + pd.to_timedelta(14, 'hours')
    query_start_14 = start_time_14 - timedelta(days=encoder_length)
    end_time_14 = start_time_14 + timedelta(days=prediction_length)
    time_values = pd.date_range(start=query_start_14, end=end_time_14, freq='d')

    # Get corresponding recent data from SQL server
    sql_database_path = WORKING_DIRECTORY / 'weather_data_hourly.db'
    conn = sqlite3.connect(sql_database_path)
    # df_query = pd.read_sql_query('SELECT * FROM weather WHERE datetime >= ?', conn, params=(query_start_14.timestamp(), ))
    df_query = pd.read_sql_query('SELECT * FROM weather', conn)
    df_query['datetime'] = df_query['datetime'].astype('datetime64[s]')
    df_query['datetime'] = df_query['datetime'].dt.tz_localize('America/Vancouver')
    conn.close()

    # Merge SQL data with desired date range
    df_encoder = pd.DataFrame()
    df_encoder['datetime'] = time_values
    df_encoder = df_encoder.merge(df_query, on='datetime', how='left')

    # Add in the Quality scores, these are daily labels to predict
    df_ratings = add_scores_to_df(df_encoder)
    df_encoder = df_encoder.merge(df_ratings, on='datetime', how='left')
    df_encoder['year_fraction'] = ((df_encoder['datetime'].dt.month - 1) * 30.416 + df_encoder['datetime'].dt.day - 1) / 365

    # Pre-process data (fill missing, re-index)
    df_measured_targets = df_encoder.loc[encoder_length:(encoder_length + prediction_length), TARGET_VARIABLES + ['datetime']]
    df_encoder.loc[encoder_length:(encoder_length + prediction_length), REAL_UNKNOWN_FEATURES] = 0
    df_encoder.loc[encoder_length:(encoder_length + prediction_length), TARGET_VARIABLES] = 0
    df_encoder[REAL_KNOWN_FEATURES] = df_encoder[REAL_KNOWN_FEATURES].ffill(limit=2)
    df_encoder[CATEGORICAL_FEATURES] = df_encoder[CATEGORICAL_FEATURES].bfill(limit=2)
    df_encoder[REAL_KNOWN_FEATURES] = df_encoder[REAL_KNOWN_FEATURES].bfill(limit=2)
    df_encoder[REAL_UNKNOWN_FEATURES] = df_encoder[REAL_UNKNOWN_FEATURES].bfill(limit=2)
    df_encoder.reset_index(drop=True, inplace=True)
    df_encoder['static'] = 'S'  # Required static group identifier
    df_encoder['time_idx'] = np.arange(df_encoder.shape[0])
    return df_encoder, df_measured_targets


def build_prediction_dataset(input_dataframe, target_variable):
    """
    Build the TimeSeriesDataSet for the model prediction.

    :param input_dataframe: (pd.DataFrame): Pre-processed dataframe.
    :param target_variable: (str): Target variable to predict.
    :return: TimeSeriesDataSet: Dataset ready for prediction.
    """

    # Load training dataset for parameters to pass into inference batch
    training_dataset_checkpoint = WORKING_DIRECTORY / f'{target_variable}_training_dataset_daily.pkl'
    with torch.serialization.safe_globals([TimeSeriesDataSet]):
        training_dataset = torch.load(training_dataset_checkpoint, weights_only=False)

    # Create inference dataset
    inference_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        input_dataframe,
        predict=True,
        stop_randomization=True
    )
    return inference_dataset


def predict_and_store(input_dataset, pytorch_dataset, target, forecast_q=3):
    """
    :param input_dataset: (pd.DataFrame): Original data.
    :param pytorch_dataset:  (PyTorch TimeSeriesDataSet): Dataset for prediction.
    :param target: (str): Target variable name.
    :param forecast_n: Forecast step index (0...n, 0=present).
    :param forecast_q:  Quantile index (0...n).
    :return: pd.DataFrame: DataFrame with datetime and prediction columns.
    """

    # Load pre-trained checkpoint
    checkpoint = WORKING_DIRECTORY / f'tft{target}DailyCheckpoint.ckpt'
    tft_model = tft_with_ignore.load_from_checkpoint(checkpoint)
    tft_model.eval()

    # Create a dataloader batch and generate raw_predictions
    pytorch_dataloader = pytorch_dataset.to_dataloader(train=False, batch_size=len(pytorch_dataset),
                                                       shuffle=False, num_workers=4)
    raw_predictions = tft_model.predict(pytorch_dataloader, mode='raw', return_index=True, return_x=True)

    # Extract the predictions into usable format
    datetime_measured = input_dataset['datetime']
    pred_datetime_idx = raw_predictions.index['time_idx'].max()
    y_range_pred = raw_predictions.output.prediction[:, :, forecast_q].numpy().reshape(-1)
    datetime_pred = input_dataset['datetime'].iloc[pred_datetime_idx:len(input_dataset)]

    # Compile multiple output series into dataframe along with corresponding label and return
    df_target = pd.DataFrame()
    df_target['datetime'] = datetime_pred
    df_target[f'{target}'] = y_range_pred
    # df_target = df_target.groupby('datetime')[f'{target}'].mean().reset_index()
    return df_target


def save_forecast(df_transmit, output_path):
    """
    Takes the most recent 6 points of the forecast and save to csv.

    :param df_forecast: (pd.DataFrame): The dataframe containing forecast results.
    :param output_path: (Path): Path to save the CSV file.
    :return:
    """
    df_transmit.to_csv(output_path, index=False)
    return df_transmit


def plot_measured_forecast(df_observed, df_forecast):
    date_measured = df_observed['datetime']
    speed_measured = df_observed['speed']

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
    begin_time = pd.Timestamp('2025-Jun-21', tz='America/Vancouver')

    while(True):
        begin_time = begin_time + timedelta(days=1)
        df_encoder, df_measured = prepare_data(begin_time)
        df_predict = pd.DataFrame()

        # Loop through targets and predict for each
        for target in TARGET_VARIABLES:
            prediction_dataset = build_prediction_dataset(df_encoder, target)
            df_target = predict_and_store(df_encoder, prediction_dataset, target, forecast_q=3)
            if df_predict.empty:
                df_predict = df_target
            else:
                df_predict = df_predict.merge(df_target, on='datetime', how='outer')

        plot_measured_forecast(df_measured, df_predict)
        pass

if __name__ == '__main__':
    main()
    pass
