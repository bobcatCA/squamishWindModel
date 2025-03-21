import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from updateWeatherData import get_conditions_table_hourly
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline, QuantileLoss, NaNLabelEncoder
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer

# Set encoder/decoder lengths
max_encoder_length = 50  # Number of past observations
max_prediction_length = 8  # Number of future steps you want to predict

# Build the variables that form the basis of the model architecture
training_features_categorical = ['comoxSky', 'vancouverSky', 'victoriaSky', 'whistlerSky']
# training_features_reals_known = ['day_fraction', 'year_fraction']
training_features_reals_known = ['sin_hour', 'year_fraction', 'comoxDegC', 'lillooetDegC',
                                 'pembertonDegC', 'vancouverDegC', 'victoriaDegC', 'whistlerDegC']
training_features_reals_unknown = ['comoxKPa', 'vancouverKPa', 'lillooetKPa', 'pamKPa', 'ballenasKPa']
training_labels = ['speed', 'gust', 'lull', 'direction']  # Multiple targets - have to make a model for each

# Option 1: Fetch data using HTML scrapers
data = get_conditions_table_hourly(encoder_length=max_encoder_length, prediction_length=max_prediction_length)

# Option 2: Use previous data
# data = pd.read_csv('mergedOnSpeed_hourly.csv')
# data['time'] = pd.to_datetime(data['time'])
# data = data.sort_values('time')  # Sort chronologically (if not already)
# data = data.iloc[27000:28000]  # Narrow down the dataset to speed it up (for demonstration)
# data.reset_index(inplace=True, drop=True)
# data.loc[data.index[-8:], training_features_reals_unknown] = np.nan
# data.loc[data.index[-8:], training_labels] = np.nan
data[training_features_reals_unknown] = data[training_features_reals_unknown].ffill()
# data[training_labels] = data[training_labels].fillna(data[training_labels].mean())  # Replace NaNs with mean (or 0)
data[training_labels] = data[training_labels].fillna(0)
data.dropna(axis=0, inplace=True)
data.reset_index(drop=True, inplace=True)

# Now move on to feeding it into the Inference pass
data['static'] = 'S'  # Put a static data column into the df (required for training)
data['time_idx'] = np.arange(data.shape[0])  # Add index for model - requires time = 0, 1, 2, ..... , n

df_forecast = pd.DataFrame()  # Store the forecasts in the loop as columns in a df
fig, ax = plt.subplots(4, 1, sharex=True)

# Loop through each target variable and make a model for each
for count, training_label in enumerate(training_labels):
    tft_checkpoint_filename = 'tft' + training_label + 'HourlyCheckpoint1.ckpt'

    # Define the TimeSeriesDataSet
    prediction_dataset = TimeSeriesDataSet(
        data,
        time_idx='time_idx',  # Must be index = 0, 1, 2, ..... , n
        target=training_label,
        group_ids=['static'],  # Still not entirely sure how this feeds into the model
        static_categoricals=['static'],  # Just a dummy set to have one static
        time_varying_known_categoricals=training_features_categorical,
        time_varying_known_reals=training_features_reals_known,  # Real Inputs: temperature, presssure, humidity, etc.
        time_varying_unknown_reals=([training_label] + training_features_reals_unknown),  # Target variable: speed, gust, lull, or direction
        min_encoder_length=8,  # Based on PyTorch example
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        target_normalizer=GroupNormalizer(groups=['static']),  # groups argument only required if multiple categoricals
        # target_normalizer=NaNLabelEncoder(add_nan=True),  # Handles NaN for future prediction
        add_relative_time_idx=True,  # This may or may not affect much
        add_target_scales=True,
        randomize_length=None
    )

    # Make batches and load a pre-trained model
    batch = prediction_dataset.to_dataloader(train=False, batch_size=len(prediction_dataset), shuffle=False)
    tft = TemporalFusionTransformer.load_from_checkpoint(tft_checkpoint_filename)

    # Predict using the pre-trained model
    rawPredictions = tft.predict(batch, mode='raw', return_index=True, return_x=True)
    forecast_n = 7  # Plot the n hours ahead prediction

    # tft.plot_prediction(rawPredictions.x, rawPredictions.output,
    #                     idx=496, show_future_observed=True)
    # plt.show()

    # Determine date ranges for measured, predicted, forecast
    x_range_meas = data['datetime']
    x_range_pred = data['datetime'][rawPredictions.index['time_idx']]

    # Determine the predicted/forecast (Mean = 3/7 Quantile)
    y_range_meas = data[training_label]
    # y_range_pred = pd.Series(rawPredictions.output.prediction[:, 0, 3])  # Appears to be format [n, tau, quant]
    y_range_pred = pd.Series(rawPredictions.output.prediction[:, forecast_n, 3]).shift(axis=0, periods=forecast_n)  # Appears to be format [n, tau, quant]

    # Plot the measured range for all variables
    if any(name in training_label for name in ['direction', 'lull', 'gust']):
        # Add to DataFrame
        y_range_pred_Q1 = pd.Series(rawPredictions.output.prediction[:, forecast_n, 0]).shift(axis=0,
                                                                                              periods=forecast_n)
        y_range_pred_Q7 = pd.Series(rawPredictions.output.prediction[:, forecast_n, 6]).shift(axis=0,
                                                                                              periods=forecast_n)
        df_forecast[f'{training_label}_Q1'] = y_range_pred_Q1
        df_forecast[f'{training_label}_Q7'] = y_range_pred_Q7
        ax[count].plot(x_range_meas, y_range_meas, label='Measured')
        ax[count].fill_between(x_range_pred.sort_index(), y_range_pred_Q1.sort_index(), y_range_pred_Q7.sort_index(), color='orange', alpha=0.2)
    else:
        # Add to DataFrame
        df_forecast['datetime'] = x_range_pred
        df_forecast[f'{training_label}'] = y_range_pred

        # Plot on the predicted mean for speed, gust, lull
        ax[count].plot(x_range_meas, y_range_meas, label='Measured')
        ax[count].plot(x_range_pred, y_range_pred, '.', label='Predicted')
        pass

    # Remaining formatting for each chart
    ax[count].set_ylabel(training_label)
    ax[count].legend()
pass

# Calculate the Quality Ratings based on the predictions
df_forecast['sailingWindow'] = df_forecast['speed'] > 15
df_forecast['gustLullRating'] = (df_forecast['gust_Q7'] - df_forecast['gust_Q1'] +
                                 df_forecast['lull_Q7'] - df_forecast['lull_Q1'])
df_forecast['gustLullRating'] = 5 - 4 * ((df_forecast['gustLullRating'] - 22) / 14)
df_forecast['gustLullRating'] = np.clip(round(df_forecast['gustLullRating']), 1, 5)
df_forecast['directionRating'] = df_forecast['direction_Q7'] - df_forecast['direction_Q1']
df_forecast['directionRating'] = 5 - 4 * ((df_forecast['directionRating'] - 40) / 50)
df_forecast['directionRating'] = np.clip(round(df_forecast['directionRating']), 1, 5)

df_forecast = df_forecast.groupby('datetime').mean()  # TFT outputs multiple predictions per stamp, take mean() for now.
print('done')