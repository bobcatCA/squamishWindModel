import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline, QuantileLoss, NaNLabelEncoder
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from updateWeatherData import get_conditions_table_daily

# Set encoder/decoder lengths
max_encoder_length = 8  # Number of past observations
max_prediction_length = 5  # Number of future steps you want to predict

# Build the variables that form the basis of the model architecture
training_features_categorical = ['comoxSky', 'vancouverSky', 'victoriaSky', 'whistlerSky']
# training_features_reals_known = ['day_fraction', 'year_fraction']
training_features_reals_known = ['year_fraction', 'comoxDegC', 'lillooetDegC',
                                 'pembertonDegC', 'vancouverDegC', 'victoriaDegC', 'whistlerDegC']
training_features_reals_unknown = ['comoxKPa', 'vancouverKPa', 'lillooetKPa', 'pamKPa', 'ballenasKPa']
training_labels = ['speed', 'speed_variability', 'dir_score']  # Multiple targets - have to make a model for each

# Option 1: Fetch data using live database and HTML scrapers
data = get_conditions_table_daily()

# Option 2: Use previous data
# data = pd.read_csv('mergedOnSpeed_daily.csv')
# data['time'] = pd.to_datetime(data['time'])
# data = data.sort_values('time')  # Sort chronologically (if not already)
# data = data.iloc[1000:1030]  # Narrow down the dataset to speed it up (for demonstration)
# data.reset_index(inplace=True, drop=True)
# data.loc[data.index[-5:], training_features_reals_unknown] = np.nan
# data.loc[data.index[-5:], training_labels] = np.nan
data[training_features_reals_unknown] = data[training_features_reals_unknown].ffill()
data[training_labels] = data[training_labels].fillna(0)
data.reset_index(drop=True, inplace=True)

# Now move on to feeding it into the Inference pass
data['static'] = 'S'  # Put a static data column into the df (required for training)
data['time_idx'] = np.arange(data.shape[0])  # Add index for model - requires time = 0, 1, 2, ..... , n

df_forecast = pd.DataFrame()  # Store the forecasts in the loop as columns in a df
df_forecast['datetime'] = data['datetime']

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
    forecast_n = 4  # n hours ahead prediction
    forecast_q = 4  # q quantile (out of 0-7)

    # tft.plot_prediction(rawPredictions.x, rawPredictions.output,
    #                     idx=496, show_future_observed=True)
    # plt.show()

    # Determine date ranges for measured, predicted, forecast
    x_range_meas = data['datetime']
    x_range_pred = data['datetime'][rawPredictions.index['time_idx']]

    # Determine the predicted/forecast (Mean = 3/7 Quantile)
    y_range_meas = data[training_label]
    # y_range_pred = pd.Series(rawPredictions.output.prediction[:, 0, 3])  # Appears to be format [n, tau, quant]
    y_range_pred = pd.Series(rawPredictions.output.prediction[:, forecast_n, forecast_q]).shift(axis=0, periods=forecast_n)  # Appears to be format [n, tau, quant]

    # Add to DataFrame
    df_target = pd.DataFrame()
    df_target['datetime'] = x_range_pred
    df_target[f'{training_label}'] = y_range_pred
    df_target = df_target.groupby('datetime')[f'{training_label}'].mean()
    df_forecast = df_forecast.merge(df_target, on='datetime', how='right')
pass

# Narrow it down to the forecast days
df_transmit = df_forecast.iloc[-5:].reset_index(drop=True)
df_transmit['datetime'] = df_transmit['datetime'].dt.date

# Save as HTML table TODO: Update for API call
html_table_daily = df_transmit.to_html()
with open('df_forecast_daily.html', 'w') as f:
    f.write(html_table_daily)

print('done')