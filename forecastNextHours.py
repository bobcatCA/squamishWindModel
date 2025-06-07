import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import psutil
# import threading
import time
from pathlib import Path
from dotenv import load_dotenv
from updateWeatherData import get_conditions_table_hourly
from pytorch_forecasting import (TimeSeriesDataSet, TemporalFusionTransformer, GroupNormalizer)


# This class was necessary to ignore the loss loading from the Checkpoint (apparently can cause problems)
class tft_with_ignore(TemporalFusionTransformer):
    def __init__(self, *args, **kwargs):
        self.save_hyperparameters(ignore=['loss'])  # Now this works as expected
        super().__init__(*args, **kwargs)

# Load environment and global variables
load_dotenv()
WORKING_DIRECTORY = Path(os.getenv('WORKING_DIRECTORY'))
MAX_ENCODER_LENGTH = 50  # Number of past observations to feed in
MAX_PREDICTION_LENGTH = 8  # Number of future steps to predict

# Model architecture features
CATEGORICAL_FEATURES = ['comoxSky', 'vancouverSky', 'victoriaSky', 'whistlerSky']
REAL_KNOWN_FEATURES = ['sin_hour', 'year_fraction', 'comoxDegC', 'lillooetDegC',
                       'pembertonDegC', 'vancouverDegC', 'victoriaDegC', 'whistlerDegC']
REAL_UNKNOWN_FEATURES = ['comoxKPa', 'vancouverKPa', 'lillooetKPa', 'pamKPa', 'ballenasKPa']
TARGET_VARIABLES = ['speed', 'gust', 'lull', 'direction']  # Each will have a separate model
# TARGET_VARIABLES = ['speed']  # Each will have a separate model


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


def prepare_data():
    """
    Fetches and prepares hourly weather data for inference.
    :return: pd.DataFrame: Cleaned and formatted dataframe ready for prediction.
    """

    # Get raw data from HTML scrapers and local database
    data = get_conditions_table_hourly(
        encoder_length=MAX_ENCODER_LENGTH,
        prediction_length=MAX_PREDICTION_LENGTH
    )

    ###### For testing only ######
    # data = pd.read_csv('mergedOnSpeed_hourly.csv')
    # data.rename(columns={'time': 'datetime'}, inplace=True)
    # data = data.loc[23794:25791].reset_index(drop=True)
    # data.loc[len(data) - 8:, REAL_UNKNOWN_FEATURES] = np.nan
    # data.loc[len(data) - 8:, TARGET_VARIABLES] = np.nan
    ##############################

    # Pre-process data (fill missing, re-index)
    data[REAL_UNKNOWN_FEATURES] = data[REAL_UNKNOWN_FEATURES].ffill()
    data[CATEGORICAL_FEATURES] = data[CATEGORICAL_FEATURES].bfill(limit=1)
    data[TARGET_VARIABLES] = data[TARGET_VARIABLES].fillna(0)
    data.reset_index(drop=True, inplace=True)
    data['static'] = 'S'  # Required static group identifier
    data['time_idx'] = np.arange(data.shape[0])
    return data


def load_model_and_predict(data, target, forecast_n=7, forecast_q=3):
    """
    :param data: Pandas dataframe, pre-processed
    :param target: Str, name of target (label) variable
    :return: pd.DataFrame: DataFrame of forecast results for the given target.
    """

    # Load pre-trained checkpoint and generate PyTorch dataset object
    checkpoint_path = WORKING_DIRECTORY / f'tft{target}HourlyCheckpoint.ckpt'
    dataset = TimeSeriesDataSet(
        data,
        time_idx='time_idx',
        target=target,
        group_ids=['static'],
        static_categoricals=['static'],
        time_varying_known_categoricals=CATEGORICAL_FEATURES,
        time_varying_known_reals=REAL_KNOWN_FEATURES,
        time_varying_unknown_reals=[target] + REAL_UNKNOWN_FEATURES,
        min_encoder_length=8,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        target_normalizer=GroupNormalizer(groups=['static']),
        add_relative_time_idx=True,
        add_target_scales=True,
        randomize_length=None
    )

    # Generate a dataloader batch (entire dataset for inference pass)
    batch = dataset.to_dataloader(train=False, batch_size=len(dataset), shuffle=False, num_workers=7)
    model = tft_with_ignore.load_from_checkpoint(checkpoint_path)

    # Generate raw predictions, and extract from output
    raw_predictions = model.predict(batch, mode='raw', return_index=True, return_x=True)
    datetime_idx = data['datetime']
    pred_datetime = datetime_idx[raw_predictions.index['time_idx']].reset_index(drop=True)

    # Predicted mean (3rd quantile out of 0-7)
    y_pred_mean = pd.Series(
        raw_predictions.output.prediction[:, forecast_n, forecast_q]
    ).shift(periods=forecast_n)

    # Pull out Q1 and Q7 for direction, lull, and gust (used later in ratings)
    if any(name in target for name in ['direction', 'lull', 'gust']):
        y_pred_Q1 = pd.Series(
            raw_predictions.output.prediction[:, forecast_n, 0]
        ).shift(periods=forecast_n)
        y_pred_Q7 = pd.Series(
            raw_predictions.output.prediction[:, forecast_n, 6]
        ).shift(periods=forecast_n)
        result_df = pd.DataFrame({
            'datetime': pred_datetime,
            target: y_pred_mean,
            f'{target}_Q1': y_pred_Q1,
            f'{target}_Q7': y_pred_Q7
        })
    else:
        result_df = pd.DataFrame({'datetime': pred_datetime, target: y_pred_mean})
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
    df['speed_variability'] = (df['gust_Q7'] - df['gust_Q1'] + df['lull_Q7'] - df['lull_Q1'])
    df['speed_variability'] = 5 - 4 * ((df['speed_variability'] - 22) / 14)
    df['speed_variability'] = np.clip(round(df['speed_variability']), 1, 5)

    # Direction index:1 to 5 rating for the relative direction variability
    df['direction_variability'] = df['direction_Q7'] - df['direction_Q1']
    df['direction_variability'] = 5 - 4 * ((df['direction_variability'] - 40) / 50)
    df['direction_variability'] = np.clip(round(df['direction_variability']), 1, 5)

    df = df.groupby('datetime').mean()
    df.loc[df['sailingWindow'] == False, ['speed_variability', 'direction_variability']] = 0
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
    # logging.getLogger('lightning.pytorch').setLevel(logging.WARNING)  # To suppress INFO level messages
    data = prepare_data()
    df_transmit = pd.DataFrame()

    for target in TARGET_VARIABLES:
        df_forecast = load_model_and_predict(data, target, forecast_n=0, forecast_q=3)
        if df_transmit.empty:
            df_transmit = df_forecast
        else:
            df_transmit = df_transmit.merge(df_forecast, on='datetime', how='outer')

    df_transmit = compute_quality_metrics(df_transmit)
    df_transmit = df_transmit[['speed', 'speed_variability', 'direction_variability']].iloc[-8:].reset_index(drop=False)
    # plot_measured_forecast(data, df_transmit.reset_index())

    # Save to file (csv, json...)
    df_transmit.to_csv(WORKING_DIRECTORY / f'hourly_speed_predictions.csv', index=False)
    df_transmit.to_json(WORKING_DIRECTORY / f'hourly_speed_predictions.json', orient='records', lines=True)
    # html_table_hourly = df_transmit.to_html()
    # with open('df_forecast_hourly.html', 'w') as f:
    #     f.write(html_table_hourly)

if __name__ == '__main__':
    # Start monitoring in a background thread, if desired to monitore resource load
    # monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    # monitor_thread.start()

    # Run the main function
    main()

    # Sleep 1ms to let the logger finish the last write (uncomment if using monitor_resources)
    # time.sleep(0.5)
