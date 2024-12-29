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
data = data.iloc[26700:29000]  # Narrow down the dataset to speed it up (for demonstration)
data = data.reset_index(drop=True)  # Reset for indexing dates later
# time_series = data['time'].reset_index(drop=True)  # Save for later, so we have a real time index to plot against
data['static'] = 'S'  # Put a static data column into the df (required for training)
data['time_idx'] = np.arange(data.shape[0])  # Add index for model - requires time = 0, 1, 2, ..... , n
# data = data.drop(columns=['time'])  # Drop for feeding into training model (TODO: is this necessary?)

# Set encoder/decoder lengths
max_encoder_length = 10  # Number of past observations
max_prediction_length = 5  # Number of future steps you want to predict

# Build the variables that form the basis of the model architecture
training_features_categorical = ['comoxSky', 'vancouverSky', 'victoriaSky', 'whistlerSky']
training_features_reals_known = ['day_fraction', 'year_fraction']
training_features_reals_unknown = ['comoxDegC', 'comoxKPa', 'vancouverDegC', 'vancouverKPa', 'whistlerDegC', 'pembertonDegC',
                                   'lillooetDegC', 'lillooetKPa', 'pamDegC', 'pamKPa', 'ballenasDegC', 'ballenasKPa', 'temperature']
training_labels = ['speed', 'gust_relative', 'lull_relative', 'direction']  # Multiple targets - have to make a model for each

df_predictions = pd.DataFrame()  # Store the predictions in the loop as columns in a df
df_forecast = pd.DataFrame()  # Store the forecasts in the loop as columns in a df
fig, ax = plt.subplots(4, 1, sharex=True)

# Loop through each target variable and make a model for each
for count, training_label in enumerate(training_labels):
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
    forecast_n = 4  # Plot the n hours ahead prediction

    # Determine date ranges for measures, predicted, forecast
    x_range_meas = data['time']
    x_range_pred = data['time'][raw_predictions.x['decoder_time_idx'][:, 0].numpy()]
    x_range_forecast = data['time'][raw_predictions.x['decoder_time_idx'][:, forecast_n].numpy()]

    if training_label == 'speed':
        y_range_pred = pd.Series(raw_predictions.output.prediction[:, 0, 3].numpy(), index=x_range_pred.index)  # Appears to be format [n, tau, quant]
        y_range_forecast = pd.Series(raw_predictions.output.prediction[:, forecast_n, 3].numpy(), index=x_range_forecast.index)  # Predicted n steps from present

        y_range = {
            'Measured speed' : data[training_label],
            'Mean Speed Predicted' : y_range_pred,
            'Mean Speed Forecast' : y_range_forecast
        }

        for series in y_range:
            if 'Measured' in series:
                x_range = x_range_meas
            elif 'Predicted' in series:
                    x_range = x_range_pred
            elif 'Forecast' in series:
                x_range = x_range_forecast
            else: x_range = np.nan

            ax[count].plot(x_range.sort_index(), y_range[series].sort_index(), label=series)
            ax[count].set_ylabel(training_label)
            ax[count].legend()
            pass
        pass

    else:
        y_range_pred_Q1 = pd.Series(raw_predictions.output.prediction[:, 0, 1].numpy(), index=x_range_pred.index)
        y_range_pred_Q5 = pd.Series(raw_predictions.output.prediction[:, 0, 5].numpy(), index=x_range_pred.index)
        y_range_forecast_Q1 = pd.Series(raw_predictions.output.prediction[:, forecast_n, 1], index=x_range_forecast.index)
        y_range_forecast_Q5 = pd.Series(raw_predictions.output.prediction[:, forecast_n, 5], index=x_range_forecast.index)

        # y_range = {
        #     'Measured' : data[training_label],
        #     'Predicted-Q1' : y_range_pred_Q1,
        #     'Predicted-Q5' : y_range_pred_Q5,
        #     'Forecast-Q1' : y_range_forecast_Q1,
        #     'Forecast-Q5' : y_range_forecast_Q5
        # }
        ax[count].plot(x_range_meas, data[training_label], label='Measured')
        ax[count].fill_between(x_range_pred.sort_index(), y_range_pred_Q1.sort_index(), y_range_pred_Q5.sort_index(), color='brown', alpha=0.2)
        ax[count].set_ylabel(training_label)
        ax[count].legend()
    pass

pass
#
# # For plotting, shorten dataFrame according to the decoder/encoder length
# df_predictions['x_pred'] = data['time'][raw_predictions.x['decoder_time_idx'][:, 0].numpy()]
# df_forecast['x_forecast'] = data['time'][raw_predictions.x['decoder_time_idx'][:, prediction_timeDelta].numpy()]
#
# # Get x and y series for each variable ready to plot
# x_pred = df_predictions['x_pred']
# x_forecast = df_predictions['x_forecast']
# x_meas = data['time']
# # x_meas = time_series[data['time_idx']]
# y_speed_meas = data['speed']
# y_speed_pred = df_predictions['speed']
# y_speed_forecast = df_predictions['speed_forecast']
# y_gust_relative_meas = data['gust_relative']
# y_gust_relative_pred_Q1 = df_predictions['gust_relative_Q1']
# y_gust_relative_pred_Q5 = df_predictions['gust_relative_Q5']
# # y_gust_relative_forecast = df_predictions['gust_relative_forecast']
# y_gust_relative_forecast_Q1 = df_predictions['gust_relative_forecast_Q1']
# y_gust_relative_forecast_Q5 = df_predictions['gust_relative_forecast_Q5']
# # y_lull_relative_meas = data['lull']
# # y_lull_relative_pred = df_predictions['lull']
# # y_lull_relative_forecast = df_predictions['lull_relative_forecast']
# # y_direction_meas = data['direction']
# # y_direction_pred = df_predictions['direction']
# # y_direction_forecast = df_predictions['direction_forecast']

# Build plots
# fig, ax = plt.subplots(4, 1, sharex=True)
# ax[0].plot(x_meas, y_speed_meas, label='Measured')
# ax[0].plot(x_pred, y_speed_pred, '.', label='Predicted')
# ax[0].plot(x_pred + pd.Timedelta(hours=prediction_timeDelta), y_speed_forecast, '.', label='Forecast')
# ax[0].set_ylabel('Windspeed')
# ax[1].plot(x_meas, y_gust_relative_meas, label='Measured')
# ax[1].fill_between(x_pred, y_gust_relative_pred_Q1, y_gust_relative_pred_Q5, color='brown', alpha=0.2, label='Predicted')
# ax[1].plot(x_pred, y_gust_relative_pred_Q1, color='orange', label='Q1')
# ax[1].plot(x_pred, y_gust_relative_pred_Q5, color='red', label='Q5')
# ax[1].plot(x_pred, y_gust_relative_pred, '.', label='Predicted')
# ax[1].plot(x_pred + pd.Timedelta(hours=prediction_timeDelta), y_gust_relative_forecast, '.', label='Forecast')
# ax[1].fill_between(x_pred + pd.Timedelta(hours=prediction_timeDelta),
#                    y_gust_relative_forecast_Q1, y_gust_relative_forecast_Q5, '.', label='Forecast')
# ax[1].set_ylabel('Gust')
# ax[2].plot(x_meas, y_lull_relative_meas, label='Measured')
# ax[2].plot(x_pred, y_lull_relative_pred, '.', label='Predicted')
# ax[2].plot(x_pred + pd.Timedelta(hours=prediction_timeDelta), y_lull_relative_forecast, '.', label='Forecast')
# ax[2].set_ylabel('Lull')
# ax[3].plot(x_meas, y_direction_meas, label='Measured')
# ax[3].plot(x_pred, y_direction_pred, '.', label='Predicted')
# ax[3].plot(x_pred + pd.Timedelta(hours=prediction_timeDelta), y_direction_forecast, '.', label='Forecast')
# ax[3].set_ylabel('Direction')
# plt.legend()
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
