import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from updateWeatherData import get_conditions_table
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer

# Fetch data using HTML scrapers
data = get_conditions_table()

# Process the timestamps.
data['time'] = pd.to_datetime(data['time'])
data = data.sort_values('time')  # Sort chronologically (if not already)
data = data.iloc[26700:28500]  # Narrow down the dataset to speed it up (for demonstration)
data = data.reset_index(drop=True)  # Reset for indexing dates later
data['static'] = 'S'  # Put a static data column into the df (required for training)
data['time_idx'] = np.arange(data.shape[0])  # Add index for model - requires time = 0, 1, 2, ..... , n

# Set encoder/decoder lengths
max_encoder_length = 50  # Number of past observations
max_prediction_length = 8  # Number of future steps you want to predict

# Build the variables that form the basis of the model architecture
training_features_categorical = ['comoxSky', 'vancouverSky', 'victoriaSky', 'whistlerSky']
# training_features_reals_known = ['day_fraction', 'year_fraction']
training_features_reals_known = ['sin_hour', 'year_fraction', 'comoxDegC', 'lillooetDegC',
                                 'pembertonDegC', 'vancouverDegC', 'victoriaDegC', 'whistlerDegC']
# training_features_reals_unknown = ['comoxDegC', 'comoxKPa', 'vancouverDegC', 'vancouverKPa', 'whistlerDegC', 'pembertonDegC',
#                            'lillooetDegC', 'lillooetKPa', 'pamDegC', 'pamKPa', 'ballenasDegC', 'ballenasKPa']
training_features_reals_unknown = ['comoxKPa', 'vancouverKPa', 'lillooetKPa', 'pamKPa', 'ballenasKPa']
training_labels = ['speed', 'gust', 'lull', 'direction']  # Multiple targets - have to make a model for each
data[training_features_reals_unknown] = np.nan

df_predictions = pd.DataFrame()  # Store the predictions in the loop as columns in a df
df_forecast = pd.DataFrame()  # Store the forecasts in the loop as columns in a df
# fig, ax = plt.subplots(4, 1, sharex=True)

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
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        target_normalizer=GroupNormalizer(groups=['static']),  # groups argument only required if multiple categoricals
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

    for idx in range(0, 2000, 24):  # plot some examples
        tft.plot_prediction(
            rawPredictions.x,
            rawPredictions.output,
            idx=idx,
            show_future_observed=True,
        )
        plt.show()
    tft.plot_prediction(rawPredictions.x, rawPredictions.output, idx=80)
    plt.show()
    break

pass

