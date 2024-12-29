import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
# import torch
# from torchmetrics import Metric
#
# # Define custom weighted Loss Function
# class WeightedMSELoss(Metric):
#     def __init__(self, weights_func):
#         super().__init__()
#         self.weights_func = weights_func
#         self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
#         self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
#
#     def update(self, y_pred, y_true):
#         weights = self.weights_func(y_true)
#         loss = torch.mean(weights * (y_pred - y_true) ** 2)
#         self.sum_loss += loss * len(y_true)
#         self.total += len(y_true)
#
#     def compute(self):
#         return self.sum_loss / self.total
#
# def custom_weights(targets):
#     # Higher weights for high wind speeds
#     return 1 + 1.0 * (targets > 20) - 1.0 * (targets < 10)

# Load and pre-process dataset
data = pd.read_csv('mergedOnSpeed_hourly.csv')

# Process the timestamps.
data['time'] = pd.to_datetime(data['time'])
data = data.sort_values('time')  # Sort chronologically (if not already)
data = data.iloc[26600:]  # Narrow down the dataset to speed it up (for demonstration)
time_series = data['time'].reset_index(drop=True)  # Save for later, so we have a real time index to plot against
data['static'] = 'S'  # Put a static data column into the df (required for training)
data['time_idx'] = np.arange(data.shape[0])  # Add index for model - requires time = 0, 1, 2, ..... , n
data = data.drop(columns=['time'])  # Drop for feeding into training model (TODO: is this necessary?)

# Set encoder/decoder lengths
max_encoder_length = 30  # Number of past observations
max_prediction_length = 10  # Number of future steps you want to predict

# Build the variables that form the basis of the model architecture
training_features_categorical = ['comoxSky', 'vancouverSky', 'victoriaSky', 'whistlerSky']
training_features_reals_known = ['day_fraction', 'year_fraction']
training_features_reals_unknown = ['comoxDegC', 'comoxKPa', 'vancouverDegC', 'vancouverKPa', 'whistlerDegC', 'pembertonDegC',
                           'lillooetDegC', 'lillooetKPa', 'pamDegC', 'pamKPa', 'ballenasDegC', 'ballenasKPa']
training_labels = ['speed', 'gust_relative', 'lull_relative', 'direction']  # Multiple targets - have to make a model for each

df_predictions = pd.DataFrame()  # Store the predictions in the loop as columns in a df

# Loop through each target variable and make a model for each
for training_label in training_labels:
    tft_checkpoint_filename = 'tft' + training_label + 'HourlyCheckpoint.ckpt'

    # Define the TimeSeriesDataSet
    prediction_dataset = TimeSeriesDataSet(
        data,
        time_idx='time_idx',  # Must be index = 0, 1, 2, ..... , n
        target=training_label,
        group_ids=['static'],  # Still not entirely sure how this feeds into the model
        static_categoricals=['static'],  # Just a dummy set to have one static
        time_varying_unknown_categoricals=training_features_categorical,
        time_varying_known_reals=training_features_reals_known,  # Real Inputs: temperature, presssure, humidity, etc.
        time_varying_unknown_reals=([training_label] + training_features_reals_unknown),
        min_encoder_length=max_encoder_length // 2,  # Based on PyTorch example
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        target_normalizer=GroupNormalizer(groups=['static']),  # groups argument only required if multiple categoricals
        # allow_missing_timesteps=True,  # Comment out if not using groups
        add_relative_time_idx=True,  # This may or may not affect much
        add_target_scales=True,
        randomize_length=None
    )

    # Make batches and load a pre-trained model
    batch = prediction_dataset.to_dataloader(train=False, batch_size=len(prediction_dataset), shuffle=False)
    tft = TemporalFusionTransformer.load_from_checkpoint(tft_checkpoint_filename)

    # Predict using the pre-trained model
    raw_predictions = tft.predict(batch, mode='raw', return_x=True)
    df_predictions[training_label] = raw_predictions.output.prediction[:, 0, 0]  # Appears to be format [n, tau, quant]
    training_label_forecast = training_label + '_forecast'
    prediction_timeDelta = 4  # Plot the n hours ahead prediction
    df_predictions[training_label_forecast] = raw_predictions.output.prediction[:, prediction_timeDelta, 0]  # Predicted n steps from present

    pass

# Get x and y series for each variable ready to plot
x_pred = time_series[raw_predictions.x['decoder_time_idx'][:, 0].numpy()]
x_meas = time_series[data['time_idx']]
y_speed_meas = data['speed']
y_speed_pred = df_predictions['speed']
y_speed_forecast = df_predictions['speed_forecast']
y_gust_meas = data['gust_relative']
y_gust_pred = df_predictions['gust_relative']
y_gust_forecast = df_predictions['gust_relative_forecast']
y_lull_meas = data['lull_relative']
y_lull_pred = df_predictions['lull_relative']
y_lull_forecast = df_predictions['lull_relative_forecast']
y_direction_meas = data['direction']
y_direction_pred = df_predictions['direction']
y_direction_forecast = df_predictions['direction_forecast']

# Build plots
fig, ax = plt.subplots(4, 1, sharex=True)
ax[0].plot(x_meas, y_speed_meas, label='Measured')
ax[0].plot(x_pred, y_speed_pred, '.', label='Predicted')
ax[0].plot(x_pred + pd.Timedelta(hours=prediction_timeDelta), y_speed_forecast, '.', label='Forecast')
ax[0].set_ylabel('Windspeed')
ax[1].plot(x_meas, y_gust_meas, label='Measured')
ax[1].plot(x_pred, y_gust_pred, '.', label='Predicted')
ax[1].plot(x_pred + pd.Timedelta(hours=prediction_timeDelta), y_gust_forecast, '.', label='Forecast')
ax[1].set_ylabel('Gust')
ax[2].plot(x_meas, y_lull_meas, label='Measured')
ax[2].plot(x_pred, y_lull_pred, '.', label='Predicted')
ax[2].plot(x_pred + pd.Timedelta(hours=prediction_timeDelta), y_lull_forecast, '.', label='Forecast')
ax[2].set_ylabel('Lull')
ax[3].plot(x_meas, y_direction_meas, label='Measured')
ax[3].plot(x_pred, y_direction_pred, '.', label='Predicted')
ax[3].plot(x_pred + pd.Timedelta(hours=prediction_timeDelta), y_direction_forecast, '.', label='Forecast')
ax[3].set_ylabel('Direction')
plt.legend()
plt.show()

# Additional plots to audit the raw data
# fig, ax = plt.subplots()
# ax.plot(time_series, data['vancouverDegC'], label='vancouver Temp')
# ax.plot(time_series, data['whistlerDegC'], label='Whistler Temp')
# ax.plot(time_series, data['pembertonDegC'], label='pemberton Temp')
# ax.plot(time_series, data['lillooetDegC'], label='lillooet Temp')
# plt.legend()
# plt.show()

print('complete')