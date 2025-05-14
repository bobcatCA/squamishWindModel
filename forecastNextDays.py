import numpy as np
import os
import pandas as pd
import random
import torch
from pathlib import Path
from dotenv import load_dotenv
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from updateWeatherData import get_conditions_table_daily


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
# TARGET_VARIABLES = ['speed']  # Multiple targets - have to make a model for each


def prepare_data():
    """
    Fetch and preprocess the weather data for forecasting.

    :return: DataFrame: A prepared dataframe ready for model ingestion.
    """
    # Compile HTML-scraped weather data and pre-process
    data = get_conditions_table_daily()
    data[REAL_UNKNOWN_FEATURES] = data[REAL_UNKNOWN_FEATURES].ffill()
    data[TARGET_VARIABLES] = data[TARGET_VARIABLES].fillna(0)
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

    dataset = TimeSeriesDataSet(
        input_dataframe,
        time_idx='time_idx',
        target=target_variable,
        group_ids=['static'],
        static_categoricals=['static'],
        time_varying_known_categoricals=CATEGORICAL_FEATURES,
        time_varying_known_reals=REAL_KNOWN_FEATURES,
        time_varying_unknown_reals=[target_variable] + REAL_UNKNOWN_FEATURES,
        min_encoder_length=1,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        target_normalizer=GroupNormalizer(groups=['static']),
        add_relative_time_idx=True,
        add_target_scales=True,
        randomize_length=None
    )
    return dataset


def predict_and_store(input_dataset, pytorch_dataset, target, forecast_n=3, forecast_q=4):
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
    tft_model = TemporalFusionTransformer.load_from_checkpoint(checkpoint)
    tft_model.eval()

    # Create a dataloader batch and generate raw_predictions
    pytorch_dataloader = pytorch_dataset.to_dataloader(train=False, batch_size=len(pytorch_dataset), shuffle=False)
    raw_predictions = tft_model.predict(pytorch_dataloader, mode='raw', return_index=True, return_x=True)

    # Extract the predictions into usable format
    x_range_pred = input_dataset['datetime'][raw_predictions.index['time_idx']]
    y_range_pred = (
        pd.Series(raw_predictions.output.prediction[:, forecast_n, forecast_q])
        .shift(periods=forecast_n)
    )

    # Compile multiple output series into dataframe along with corresponding label and return
    df_target = pd.DataFrame()
    df_target['datetime'] = x_range_pred
    df_target[f'{target}'] = y_range_pred
    df_target = df_target.groupby('datetime')[f'{target}'].mean().reset_index()
    return df_target


def save_forecast(df_forecast, output_path):
    """
    Takes the most recent 6 points of the forecast and save to csv.

    :param df_forecast: (pd.DataFrame): The dataframe containing forecast results.
    :param output_path: (Path): Path to save the CSV file.
    :return:
    """
    df_transmit = df_forecast.iloc[-5:].reset_index(drop=True)
    df_transmit['datetime'] = df_transmit['datetime'].dt.date
    df_transmit.to_csv(output_path, index=False)


def main():
    # Attempt to get repeatability/deterministic behaviour during inferece
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)

    # Load data from HTML scraper
    output_csv = WORKING_DIRECTORY / 'daily_speed_predictions.csv'
    preprocessed_dataset = prepare_data()
    df_predict = pd.DataFrame()

    # Loop through targets and predict for each
    for target in TARGET_VARIABLES:
        prediction_dataset = build_prediction_dataset(preprocessed_dataset, target)
        df_target = predict_and_store(preprocessed_dataset, prediction_dataset, target)
        if df_predict.empty:
            df_predict = df_target
        else:
            df_predict = df_predict.merge(df_target, on='datetime', how='outer')

    # Save to CSV and finish script
    save_forecast(df_predict, output_csv)

    # # Save as HTML table TODO: Update for API call
    # html_table_daily = df_transmit.to_html()
    # with open('df_forecast_daily.html', 'w') as f:
    #     f.write(html_table_daily)

if __name__ == '__main__':
    main()
