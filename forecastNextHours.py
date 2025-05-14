import numpy as np
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from updateWeatherData import get_conditions_table_hourly
from pytorch_forecasting import (
    TimeSeriesDataSet, TemporalFusionTransformer,
    GroupNormalizer
)


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

    # Pre-process data (fill missing, re-index)
    data[REAL_UNKNOWN_FEATURES] = data[REAL_UNKNOWN_FEATURES].ffill()
    data[CATEGORICAL_FEATURES] = data[CATEGORICAL_FEATURES].bfill(limit=1)
    data[TARGET_VARIABLES] = data[TARGET_VARIABLES].fillna(0)
    data.reset_index(drop=True, inplace=True)
    data['static'] = 'S'  # Required static group identifier
    data['time_idx'] = np.arange(data.shape[0])
    return data


def load_model_and_predict(data, target):
    """
    :param data: Pandas dataframe, pre-processed
    :param target: Str, name of target (label) variable
    :return: pd.DataFrame: DataFrame of forecast results for the given target.
    """

    # Load pre-trained checkpoint and generate PyTorch dataset object
    checkpoint_path = WORKING_DIRECTORY / f'tft{target}HourlyCheckpoint1.ckpt'
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
    batch = dataset.to_dataloader(train=False, batch_size=len(dataset), shuffle=False)
    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)

    # Generate raw predictions, and extract from output
    raw_predictions = model.predict(batch, mode='raw', return_index=True, return_x=True)

    forecast_step = 7  # n hours ahead prediction
    datetime_idx = data['datetime']
    pred_datetime = datetime_idx[raw_predictions.index['time_idx']].reset_index(drop=True)

    # Predicted mean (3rd quantile out of 0-7)
    y_pred_mean = pd.Series(
        raw_predictions.output.prediction[:, forecast_step, 3]
    ).shift(periods=forecast_step)

    # Pull out Q1 and Q7 for direction, lull, and gust (used later in ratings)
    if any(name in target for name in ['direction', 'lull', 'gust']):
        y_pred_Q1 = pd.Series(
            raw_predictions.output.prediction[:, forecast_step, 0]
        ).shift(periods=forecast_step)
        y_pred_Q7 = pd.Series(
            raw_predictions.output.prediction[:, forecast_step, 6]
        ).shift(periods=forecast_step)
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

def main():
    data = prepare_data()
    df_transmit = pd.DataFrame()

    for target in TARGET_VARIABLES:
        df_forecast = load_model_and_predict(data, target)
        if df_transmit.empty:
            df_transmit = df_forecast
        else:
            df_transmit = df_transmit.merge(df_forecast, on='datetime', how='outer')

    df_transmit = compute_quality_metrics(df_transmit)
    df_transmit = df_transmit[['speed', 'speed_variability', 'direction_variability']].iloc[-8:].reset_index(drop=False)
    df_transmit.to_csv(WORKING_DIRECTORY / f'hourly_speed_predictions.csv', index=False)

if __name__ == '__main__':
    main()



# working_directory = '/home/bobcat/PycharmProjects/squamishWindModel/'
#
# # Set encoder/decoder lengths
# max_encoder_length = 50  # Number of past observations
# max_prediction_length = 8  # Number of future steps you want to predict
#
# # Build the variables that form the basis of the model architecture
# training_features_categorical = ['comoxSky', 'vancouverSky', 'victoriaSky', 'whistlerSky']
# # training_features_reals_known = ['day_fraction', 'year_fraction']
# training_features_reals_known = ['sin_hour', 'year_fraction', 'comoxDegC', 'lillooetDegC',
#                                  'pembertonDegC', 'vancouverDegC', 'victoriaDegC', 'whistlerDegC']
# training_features_reals_unknown = ['comoxKPa', 'vancouverKPa', 'lillooetKPa', 'pamKPa', 'ballenasKPa']
# training_labels = ['speed', 'gust', 'lull', 'direction']  # Multiple targets - have to make a model for each
#
# # Option 1: Fetch data using HTML scrapers
# data = get_conditions_table_hourly(encoder_length=max_encoder_length, prediction_length=max_prediction_length)
#
# # Option 2: Use previous data
# # data = pd.read_csv('mergedOnSpeed_hourly.csv')
# # data['time'] = pd.to_datetime(data['time'])
# # data = data.sort_values('time')  # Sort chronologically (if not already)
# # data = data.iloc[27000:28000]  # Narrow down the dataset to speed it up (for demonstration)
# # data.reset_index(inplace=True, drop=True)
# # data.loc[data.index[-8:], training_features_reals_unknown] = np.nan
# # data.loc[data.index[-8:], training_labels] = np.nan
# data[training_features_reals_unknown] = data[training_features_reals_unknown].ffill()
# data[training_features_categorical] = data[training_features_categorical].bfill(limit=1)  # Only fill 1hr past/future gap
# data[training_labels] = data[training_labels].fillna(0)
# data.reset_index(drop=True, inplace=True)
#
# # Now move on to feeding it into the Inference pass
# data['static'] = 'S'  # Put a static data column into the df (required for training)
# data['time_idx'] = np.arange(data.shape[0])  # Add index for model - requires time = 0, 1, 2, ..... , n
#
# df_forecast = pd.DataFrame()  # Store the forecasts in the loop as columns in a df
# fig, ax = plt.subplots(4, 1, sharex=True)
#
# # Loop through each target variable and make a model for each
# for count, training_label in enumerate(training_labels):
#     tft_checkpoint_filename = working_directory + 'tft' + training_label + 'HourlyCheckpoint1.ckpt'
#
#     # Define the TimeSeriesDataSet
#     prediction_dataset = TimeSeriesDataSet(
#         data,
#         time_idx='time_idx',  # Must be index = 0, 1, 2, ..... , n
#         target=training_label,
#         group_ids=['static'],  # Still not entirely sure how this feeds into the model
#         static_categoricals=['static'],  # Just a dummy set to have one static
#         time_varying_known_categoricals=training_features_categorical,
#         time_varying_known_reals=training_features_reals_known,  # Real Inputs: temperature, presssure, humidity, etc.
#         time_varying_unknown_reals=([training_label] + training_features_reals_unknown),  # Target variable: speed, gust, lull, or direction
#         min_encoder_length=8,  # Based on PyTorch example
#         max_encoder_length=max_encoder_length,
#         min_prediction_length=1,
#         max_prediction_length=max_prediction_length,
#         target_normalizer=GroupNormalizer(groups=['static']),  # groups argument only required if multiple categoricals
#         # target_normalizer=NaNLabelEncoder(add_nan=True),  # Handles NaN for future prediction
#         add_relative_time_idx=True,  # This may or may not affect much
#         add_target_scales=True,
#         randomize_length=None
#     )
#
#     # Make batches and load a pre-trained model
#     batch = prediction_dataset.to_dataloader(train=False, batch_size=len(prediction_dataset), shuffle=False)
#     tft = TemporalFusionTransformer.load_from_checkpoint(tft_checkpoint_filename)
#
#     # Predict using the pre-trained model
#     rawPredictions = tft.predict(batch, mode='raw', return_index=True, return_x=True)
#     forecast_n = 7  # Plot the n hours ahead prediction
#
#     # tft.plot_prediction(rawPredictions.x, rawPredictions.output,
#     #                     idx=496, show_future_observed=True)
#     # plt.show()
#
#     # Determine date ranges for measured, predicted, forecast
#     x_range_meas = data['datetime']
#     x_range_pred = data['datetime'][rawPredictions.index['time_idx']]
#
#     # Determine the predicted/forecast (Mean = 3/7 Quantile)
#     y_range_meas = data[training_label]
#     # y_range_pred = pd.Series(rawPredictions.output.prediction[:, 0, 3])  # Appears to be format [n, tau, quant]
#     y_range_pred = pd.Series(rawPredictions.output.prediction[:, forecast_n, 3]).shift(axis=0, periods=forecast_n)  # Appears to be format [n, tau, quant]
#
#     # Plot the measured range for all variables
#     if any(name in training_label for name in ['direction', 'lull', 'gust']):
#         # Add to DataFrame
#         y_range_pred_Q1 = pd.Series(rawPredictions.output.prediction[:, forecast_n, 0]).shift(axis=0,
#                                                                                               periods=forecast_n)
#         y_range_pred_Q7 = pd.Series(rawPredictions.output.prediction[:, forecast_n, 6]).shift(axis=0,
#                                                                                               periods=forecast_n)
#         df_forecast[f'{training_label}_Q1'] = y_range_pred_Q1
#         df_forecast[f'{training_label}_Q7'] = y_range_pred_Q7
#         ax[count].plot(x_range_meas, y_range_meas, label='Measured')
#         ax[count].fill_between(x_range_pred.sort_index(), y_range_pred_Q1.sort_index(), y_range_pred_Q7.sort_index(), color='orange', alpha=0.2)
#     else:
#         # Add to DataFrame
#         df_forecast['datetime'] = x_range_pred
#         df_forecast[f'{training_label}'] = y_range_pred
#
#         # Plot on the predicted mean for speed, gust, lull
#         ax[count].plot(x_range_meas, y_range_meas, label='Measured')
#         ax[count].plot(x_range_pred, y_range_pred, '.', label='Predicted')
#         pass
#
#     # Remaining formatting for each chart
#     ax[count].set_ylabel(training_label)
#     ax[count].legend()
# pass
#
# # Calculate the Quality Ratings based on the predictions
# minimum_speed = 13
# df_forecast['sailingWindow'] = df_forecast['speed'] > minimum_speed
# df_forecast['speed_variability'] = (df_forecast['gust_Q7'] - df_forecast['gust_Q1'] +
#                                  df_forecast['lull_Q7'] - df_forecast['lull_Q1'])
# df_forecast['speed_variability'] = 5 - 4 * ((df_forecast['speed_variability'] - 22) / 14)
# df_forecast['speed_variability'] = np.clip(round(df_forecast['speed_variability']), 1, 5)
# df_forecast['direction_variability'] = df_forecast['direction_Q7'] - df_forecast['direction_Q1']
# df_forecast['direction_variability'] = 5 - 4 * ((df_forecast['direction_variability'] - 40) / 50)
# df_forecast['direction_variability'] = np.clip(round(df_forecast['direction_variability']), 1, 5)
# df_forecast = df_forecast.groupby('datetime').mean()  # TFT outputs multiple predictions per stamp, take mean() for now.
# df_forecast.loc[df_forecast['sailingWindow'] == False, 'speed_variability'] = 0
# df_forecast.loc[df_forecast['sailingWindow'] == False, 'direction_variability'] = 0
#
# # Compile subset and send
# df_transmit = df_forecast[['speed', 'speed_variability', 'direction_variability']].iloc[-8:].reset_index(drop=False)
#
# # # Save as HTML table TODO: Update for API call
# # html_table_hourly = df_transmit.to_html()
# # with open('df_forecast_hourly.html', 'w') as f:
# #     f.write(html_table_hourly)
#
# # Save to csv
# df_transmit.to_csv(working_directory + 'hourly_speed_predictions.csv')
#

# print('done')
#
