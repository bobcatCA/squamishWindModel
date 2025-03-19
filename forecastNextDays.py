import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from updateWeatherData import get_conditions_table
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline, QuantileLoss, NaNLabelEncoder
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer

# Set encoder/decoder lengths
max_encoder_length = 8  # Number of past observations
max_prediction_length = 5  # Number of future steps you want to predict

# Build the variables that form the basis of the model architecture
training_features_categorical = ['comoxSky', 'vancouverSky', 'victoriaSky', 'whistlerSky']
# training_features_reals_known = ['day_fraction', 'year_fraction']
training_features_reals_known = ['sin_hour', 'year_fraction', 'comoxDegC', 'lillooetDegC',
                                 'pembertonDegC', 'vancouverDegC', 'victoriaDegC', 'whistlerDegC']
training_features_reals_unknown = ['comoxKPa', 'vancouverKPa', 'lillooetKPa', 'pamKPa', 'ballenasKPa']
training_labels = ['speed', 'speed_variability', 'dir_score']  # Multiple targets - have to make a model for each

# Option 1: Fetch data using HTML scrapers
# data = get_conditions_table(training_labels, [*training_features_categorical, *training_features_reals_known])

# Option 2: Use previous data
data = pd.read_csv('mergedOnSpeed_daily.csv')
data['time'] = pd.to_datetime(data['time'])
data = data.sort_values('time')  # Sort chronologically (if not already)
data = data.iloc[1000:1030]  # Narrow down the dataset to speed it up (for demonstration)
data.reset_index(inplace=True, drop=True)
data.loc[data.index[-5:], training_features_reals_unknown] = np.nan
data.loc[data.index[-5:], training_labels] = np.nan
data[training_features_reals_unknown] = data[training_features_reals_unknown].ffill()
data[training_labels] = data[training_labels].fillna(0)

# Now move on to feeding it into the Inference pass
data['static'] = 'S'  # Put a static data column into the df (required for training)
data['time_idx'] = np.arange(data.shape[0])  # Add index for model - requires time = 0, 1, 2, ..... , n

df_forecast = pd.DataFrame()  # Store the forecasts in the loop as columns in a df
df_forecast['date'] = data['date']

# Loop through each target variable and make a model for each
for count, training_label in enumerate(training_labels):
    tft_checkpoint_filename = 'tft' + training_label + 'DailyCheckpoint.ckpt'

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
        min_encoder_length=1,  # Based on PyTorch example
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
    forecast_n = 4  # Plot the n hours ahead prediction
    forecast_q = 4

    # tft.plot_prediction(rawPredictions.x, rawPredictions.output,
    #                     idx=496, show_future_observed=True)
    # plt.show()

    # Determine date ranges for measured, predicted, forecast
    x_range_meas = data['date']
    x_range_pred = data['date'][rawPredictions.index['time_idx']]

    # Determine the predicted/forecast (Mean = 3/7 Quantile)
    y_range_meas = data[training_label]
    # y_range_pred = pd.Series(rawPredictions.output.prediction[:, 0, 3])  # Appears to be format [n, tau, quant]
    y_range_pred = pd.Series(rawPredictions.output.prediction[:, forecast_n, forecast_q]).shift(axis=0, periods=forecast_n)  # Appears to be format [n, tau, quant]

    # Add to DataFrame
    df_target = pd.DataFrame()
    df_target['date'] = x_range_pred
    df_target[f'{training_label}'] = y_range_pred
    df_target = df_target.groupby('date')[f'{training_label}'].mean()
    df_forecast = df_forecast.merge(df_target, on='date', how='right')
pass

# fig, ax = plt.subplots(3, 1, sharex=True)
#
# data = pd.read_csv('mergedOnSpeed_daily.csv').iloc[1000:1030]
# data['time'] = pd.to_datetime(data['time'])
#
# # Plot on the predicted mean for speed, variability, direction
# ax[0].plot(data['date'], data['speed'], label='Measured')
# ax[1].plot(data['date'], data['speed_variability'], label='Measured')
# ax[2].plot(data['date'], data['dir_score'], label='Measured')
#
# ax[0].plot(df_forecast['date'], df_forecast['speed'], label='Predicted')
# ax[1].plot(df_forecast['date'], df_forecast['speed_variability'], label='Predicted')
# ax[2].plot(df_forecast['date'], df_forecast['dir_score'], label='Predicted')
#
# # Remaining plot settings
# ax[0].legend()
# ax[1].legend()
# ax[2].legend()

plt.show()
print('done')