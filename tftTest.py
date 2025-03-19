import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer

# Load and pre-process dataset
data = pd.read_csv('mergedOnSpeed_daily.csv')  # Assuming you have your data in a CSV
# data = data[20000:26599]  # Subset to reduce compute time
data['time'] = pd.to_datetime(data['time'])  # Ensure it's in DateTime format
data = data.sort_values('time')  # Sort chronologically (if not already)

# Put in extra required columns for TFT
data['static'] = 'S'  # Put a static data column into the df (required for training)
data['time_idx'] = np.arange(data.shape[0])  # Add index for model - requires time = 0, 1, 2, ..... , n
data = data.reset_index(drop=True)  # Reset for indexing dates later

# Set encoder/decoder lengths
max_encoder_length = 8  # Number of past observations
max_prediction_length = 5  # Number of future steps you want to predict

# Build the variables that form the basis of the model architecture
training_features_categorical = ['comoxSky', 'vancouverSky', 'victoriaSky', 'whistlerSky']
# training_features_reals_known = ['day_fraction', 'year_fraction']
training_features_reals_known = ['year_fraction', 'comoxDegC', 'lillooetDegC',
                                 'pembertonDegC', 'vancouverDegC', 'victoriaDegC', 'whistlerDegC']
# training_features_reals_unknown = ['comoxDegC', 'comoxKPa', 'vancouverDegC', 'vancouverKPa', 'whistlerDegC', 'pembertonDegC',
#                            'lillooetDegC', 'lillooetKPa', 'pamDegC', 'pamKPa', 'ballenasDegC', 'ballenasKPa']
training_features_reals_unknown = ['comoxKPa', 'vancouverKPa', 'lillooetKPa', 'pamKPa', 'ballenasKPa']
training_labels = ['speed', 'speed_variability', 'dir_score']  # Multiple targets - have to make a model for each
# data[training_features_reals_unknown] = np.nan

df_predictions = pd.DataFrame()  # Store the predictions in the loop as columns in a df
df_forecast = pd.DataFrame()  # Store the forecasts in the loop as columns in a df
# fig, ax = plt.subplots(4, 1, sharex=True)

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
        # allow_missing_timesteps=True,  # Comment out if not using groups
        add_relative_time_idx=True,  # This may or may not affect much
        add_target_scales=True,
        randomize_length=None
    )

    # Make batches and load a pre-trained model
    batch = prediction_dataset.to_dataloader(train=False, batch_size=len(prediction_dataset), shuffle=False)
    tft = TemporalFusionTransformer.load_from_checkpoint(tft_checkpoint_filename)

    # Predict using the pre-trained model
    rawPredictions = tft.predict(batch, mode='raw', return_x=True)
    forecast_n = 4  # Plot the n hours ahead prediction

    for idx in range(0, 200, 2):  # plot some examples
        tft.plot_prediction(
            rawPredictions.x,
            rawPredictions.output,
            idx=idx,
            show_future_observed=True,
        )
        plt.show()
    # tft.plot_prediction(rawPredictions.x, rawPredictions.output, idx=80)
    # plt.show()
    # break

    # # Determine date ranges for measured, predicted, forecast
    # x_range_meas = data['time']
    # x_range_pred = data['time'][raw_predictions.x['decoder_time_idx'][:, 0].numpy()]
    # x_range_forecast = data['time'][raw_predictions.x['decoder_time_idx'][:, forecast_n].numpy()]
    #
    # if 'speed' in training_label:
    #     quantile_of_interest = 3  # Mean for speed
    # elif 'gust' in training_label:
    #     quantile_of_interest = 4  # Near high-end for gust
    # elif 'lull in training_label':
    #     quantile_of_interest = 2  # Near low-end for lull
    # else: quantile_of_interest = np.nan
    #
    # # Determine the predicted/forecast (Mean = 3/7 Quantile)
    # y_range_meas = data[training_label]
    # y_range_pred = pd.Series(raw_predictions.output.prediction[:, 0, quantile_of_interest].numpy(),
    #                          index=x_range_pred.index)  # Appears to be format [n, tau, quant]
    # y_range_forecast = pd.Series(raw_predictions.output.prediction[:, forecast_n, quantile_of_interest].numpy(),
    #                              index=x_range_forecast.index)  # Predicted n steps from present
    #
    # # Plot the measured range for all variables
    # ax[count].plot(x_range_meas, y_range_meas, label='Measured')
    #
    # # Get quantile ranges for gust, lull, direction
    # if 'direction' in training_label:
    #     y_range_pred_Q1 = pd.Series(raw_predictions.output.prediction[:, 0, 1].numpy(), index=x_range_pred.index)
    #     y_range_pred_Q5 = pd.Series(raw_predictions.output.prediction[:, 0, 5].numpy(), index=x_range_pred.index)
    #     y_range_forecast_Q1 = pd.Series(raw_predictions.output.prediction[:, forecast_n, 1], index=x_range_forecast.index)
    #     y_range_forecast_Q5 = pd.Series(raw_predictions.output.prediction[:, forecast_n, 5], index=x_range_forecast.index)
    #     ax[count].fill_between(x_range_pred.sort_index(), y_range_pred_Q1.sort_index(), y_range_pred_Q5.sort_index(), color='orange', alpha=0.2)
    # else:
    #
    #     # Plot on the predicted mean for speed, gust, lull
    #     ax[count].plot(x_range_pred.sort_index(), y_range_pred.sort_index(), label='Predicted')
    #     ax[count].plot(x_range_forecast.sort_index(), y_range_forecast.sort_index(), label='Forecast')
    #     pass
    #
    # # Remaining formatting for each chart
    # ax[count].set_ylabel(training_label)
    # ax[count].legend()
pass

# plt.show()
