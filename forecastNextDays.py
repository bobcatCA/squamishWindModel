import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import psutil
import random
import threading
import time
import torch
import pytz
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
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


def prepare_data():
    """
    Fetch and preprocess the weather data for forecasting.

    :return: DataFrame: A prepared dataframe ready for model ingestion.
    """
    # Compile HTML-scraped weather data and pre-process
    data = get_conditions_table_daily()

    ##### For testing only ######
    # data = pd.read_csv('mergedOnSpeed_daily.csv')
    # # data.rename(columns={'time': 'datetime'}, inplace=True)
    # data = data.loc[55:70].reset_index(drop=True)
    # data.dropna(thresh=14, inplace=True)
    # data.loc[len(data) - MAX_PREDICTION_LENGTH:, REAL_UNKNOWN_FEATURES] = np.nan
    # data.loc[len(data) - MAX_PREDICTION_LENGTH:, TARGET_VARIABLES] = np.nan
    # ##############################

    data[REAL_UNKNOWN_FEATURES] = data[REAL_UNKNOWN_FEATURES].ffill()
    data[TARGET_VARIABLES] = data[TARGET_VARIABLES].ffill()
    data.reset_index(drop=True, inplace=True)
    data['static'] = 'S'  # Required static group
    data['time_idx'] = np.arange(data.shape[0])
    return data


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
    # Load data from HTML scraper
    output_csv = WORKING_DIRECTORY / 'daily_speed_predictions.csv'
    df_encoder = prepare_data()
    df_predict = pd.DataFrame()

    # Loop through targets and predict for each
    for target in TARGET_VARIABLES:
        prediction_dataset = build_prediction_dataset(df_encoder, target)
        df_target = predict_and_store(df_encoder, prediction_dataset, target, forecast_q=3)
        if df_predict.empty:
            df_predict = df_target
        else:
            df_predict = df_predict.merge(df_target, on='datetime', how='outer')

    # Save to CSV and finish script
    df_save = save_forecast(df_predict, output_csv)
    df_save.to_json(WORKING_DIRECTORY / f'daily_speed_predictions.json', orient='records', lines=True)
    # plot_measured_forecast(df_encoder, df_predict)

    # # Save as HTML table
    # html_table_daily = df_predict.to_html()
    # with open('df_forecast_daily.html', 'w') as f:
    #     f.write(html_table_daily)

if __name__ == '__main__':
    # Start monitoring in a background thread
    # monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    # monitor_thread.start()

    logging.getLogger('lightning.pytorch').setLevel(logging.WARNING)  # To suppress INFO level messages
    local_tz = pytz.timezone('America/Vancouver')
    start_time = datetime.now(local_tz).strftime('%Y-%m-%d %H:%M:%S')
    print(f'Daily wind prediction task started at {start_time}')

    main()

    end_time = datetime.now(local_tz).strftime('%Y-%m-%d %H:%M:%S')
    print(f'Daily prediction task complete at {end_time}')

    # Sleep 1ms to let the logger finish the last write
    # time.sleep(0.5)