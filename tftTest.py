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

# 2. Split the data into training and validation
max_encoder_length = 30  # Number of past observations
max_prediction_length = 20  # Number of future steps you want to predict
training_cutoff = data['time_idx'].max() - max_prediction_length

# Define the features and labels
model_features = ['comoxDegC', 'comoxKPa','lillooetDegC', 'lillooetKPa',
                  'pamDegC', 'pamKPa', 'pembertonDegC', 'pembertonKPa',
                  'vancouverDegC', 'vancouverKPa', 'whistlerDegC', 'whistlerKPa']
model_labels = ['speed', 'gust', 'lull', 'direction']

# Define the normalizer for all the different variables:
target_normalizer = MultiNormalizer([
    GroupNormalizer(groups=["static"], transformation="softplus"),  # For 'speed'
    GroupNormalizer(groups=["static"], transformation="softplus"),  # For 'gust'
    GroupNormalizer(groups=["static"], transformation="softplus"),  # For 'lull'
    GroupNormalizer(groups=["static"], transformation="softplus"),  # For 'direction'
])

# 3. Define the TimeSeriesDataSet
prediction_dataset = TimeSeriesDataSet(
    data,
    time_idx="time_idx",
    target=model_labels,
    group_ids=["static"],  # 'static' column for grouping
    static_categoricals=["static"],  # Encoding the 'static' column
    time_varying_known_reals=model_features,
    time_varying_unknown_reals=model_labels,  # Our target variable 'speed'
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    target_normalizer=target_normalizer,  # Normalize target (speed)
    add_relative_time_idx=False,
    add_target_scales=True,
    randomize_length=None,
)

batch = prediction_dataset.to_dataloader(train=False, batch_size=len(prediction_dataset), shuffle=False)

# Step 3: Load the pre-trained model
tft = TemporalFusionTransformer.load_from_checkpoint('tftSquamish_gen2.ckpt')

# Predict using the model
# raw_predictions, x = tft.predict(batch, mode="raw", return_x=True)
raw_predictions = tft.predict(batch, mode='raw', return_x=True)

y_pred_speed = raw_predictions.output.prediction[0][:, 0, 0]  # You can get a Gaussian distr. of points for each stamp by indexing
y_pred_gust = raw_predictions.output.prediction[1][:, 0, 0]
y_pred_lull = raw_predictions.output.prediction[2][:, 0, 0]
y_pred_direction = raw_predictions.output.prediction[3]
x_pred = raw_predictions.x['decoder_time_idx'][:, 0]
x_meas = data['time_idx']
y_meas_speed = data['speed']
y_meas_gust = data['gust']
y_meas_lull = data['lull']
y_meas_direction = data['direction']

fig, axs = plt.subplots(4, 1, sharex=True)
# ax.plot(x_meas, y_meas, label='Measured')
# ax.plot(x_pred, y_pred, label='Predicted')
axs[0].plot(date_series, y_meas_speed, label='Measured')
axs[0].plot(date_series.loc[max_encoder_length:len(data) - max_prediction_length], y_pred_speed, label='Speed-Pred')
axs[1].plot(date_series, y_meas_gust, label='Measured')
axs[1].plot(date_series.loc[max_encoder_length:len(data) - max_prediction_length], y_pred_gust, label='Gust-Pred')
axs[2].plot(date_series, y_meas_lull, label='Measured')
axs[2].plot(date_series.loc[max_encoder_length:len(data) - max_prediction_length], y_pred_lull, label='Lull-Pred')
axs[3].plot(date_series, y_meas_direction, label='Measured')
axs[3].plot(date_series.loc[max_encoder_length:len(data) - max_prediction_length], y_pred_lull, label='Direction-Pred')
# plt.ylabel('Wind Speed (knots)')
plt.legend()
plt.show()

######################################
# # Now show the Vanilla Net result
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.models import load_model
#
# # Split data into features and target
# y = data['speed']
# X = data.drop(columns=['speed', 'static', 'time_idx'])
#
# # Load the pre-trained model
# model = load_model('daytimeTempsHumSkyPressSpeed.keras')
#
# # Normalize features
# scaler = StandardScaler()
# X_test_scaled = scaler.fit_transform(X)
#
# # Run model to predict
# y_pred = model.predict(X_test_scaled)
#
# # Now do the plotting
# ax.plot(date_series, y_pred, label='Predicted_NN-Vanilla')
# plt.ylabel('Wind Speed (knots)')
# plt.legend()
# plt.show()

