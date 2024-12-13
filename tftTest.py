import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer

# Load dataset
df = pd.read_csv('mergedOnSpeed.csv')  # Assuming you have your data in a CSV
data = df.iloc[308000:312000]  # Narrow down the dataset to speed it up (for demonstration)
data['time'] = pd.to_datetime(data['time'])
data = data.sort_values('time')
date_series = data['time'].reset_index(drop=True)  # Save for later to use on X axis
data = data.drop(columns=['time'])
data['static'] = 'S'

# Add a time index column required for TimeSeriesDataSet
data['time_idx'] = np.arange(data.shape[0])
# data['weights'] = 1.0

# 2. Split the data into training and validation
max_encoder_length = 10  # Number of past observations
max_prediction_length = 5  # Number of future steps you want to predict
training_cutoff = data['time_idx'].max() - max_prediction_length

training_features = ['vancouverDegC', 'whistlerDegC', 'pembertonDegC', 'lillooetDegC']
# training_labels = ['speed', 'gust', 'lull', 'direction']  # Multiple targets
training_labels = ['speed', 'gust', 'lull', 'direction']  # Multiple targets
df_predictions = pd.DataFrame()

for training_label in training_labels:
    tft_checkpoint_filename = 'tft' + training_label + 'Checkpoint.ckpt'

    # Define the TimeSeriesDataSet
    prediction_dataset = TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        # target=training_labels,
        target=training_label,
        group_ids=["static"],  # 'static' column for grouping
        static_categoricals=["static"],  # Encoding the 'static' column
        time_varying_known_reals=training_features,
        time_varying_unknown_reals=[training_label],  # Our target variable 'speed'
        # time_varying_unknown_reals=training_labels,
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        # target_normalizer=MultiNormalizer([GroupNormalizer(groups=["static"])] * len(training_labels)),
        target_normalizer=GroupNormalizer(groups=['static']),
        add_relative_time_idx=False,
        add_target_scales=True,
        randomize_length=None,
        # weights='weights'
    )

    batch = prediction_dataset.to_dataloader(train=False, batch_size=len(prediction_dataset), shuffle=False)

    # Load the pre-trained model
    tft = TemporalFusionTransformer.load_from_checkpoint(tft_checkpoint_filename)

    # Predict using the model
    # raw_predictions, x = tft.predict(batch, mode="raw", return_x=True)
    raw_predictions = tft.predict(batch, mode='raw', return_x=True)
    df_predictions[training_label] = raw_predictions.output.prediction[:, 0, 0]

    pass
# y_pred_gust = raw_predictions.output.prediction[1][:, 0, 0]
# y_pred_lull = raw_predictions.output.prediction[2][:, 0, 0]
# y_pred_direction = raw_predictions.output.prediction[3][:, 0, 0]
x_pred = raw_predictions.x['decoder_time_idx'][:, 0]
x_meas = data['time_idx']
y_meas_speed = data['speed']
y_pred_speed = df_predictions['speed']
y_meas_gust = data['gust']
y_pred_gust = df_predictions['gust']
y_meas_lull = data['lull']
y_pred_lull = df_predictions['lull']
y_meas_direction = data['direction']
y_pred_direction = df_predictions['direction']

fig, ax = plt.subplots(4, 1, sharex=True)
ax[0].plot(x_meas, y_meas_speed, label='Measured')
ax[0].plot(x_pred, y_pred_speed, label='Predicted')
ax[1].plot(x_meas, y_meas_gust, label='Measured')
ax[1].plot(x_pred, y_pred_gust, label='Predicted')
ax[2].plot(x_meas, y_meas_lull, label='Measured')
ax[2].plot(x_pred, y_pred_lull, label='Predicted')
ax[3].plot(x_meas, y_meas_direction, label='Measured')
ax[3].plot(x_pred, y_pred_direction, label='Predicted')
# plt.ylabel('Wind Speed (knots)')
plt.legend()
plt.show()