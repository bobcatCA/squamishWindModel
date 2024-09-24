import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer


# 1. Load your dataset
df = pd.read_csv('mergedOnSpeed.csv')  # Assuming you have your data in a CSV
data = df.iloc[10000:11000]
print(data.shape)
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.sort_values('datetime')
data = data.drop(columns=['datetime'])
data['static'] = 'S'

# Add a time index column required for TimeSeriesDataSet
data['time_idx'] = np.arange(data.shape[0])
# data['weights'] = 1.0

# 2. Split the data into training and validation
max_encoder_length = 60  # Number of past observations
max_prediction_length = 20  # Number of future steps you want to predict
training_cutoff = data['time_idx'].max() - max_prediction_length

# 3. Define the TimeSeriesDataSet
prediction_dataset = TimeSeriesDataSet(
    data,
    time_idx="time_idx",
    target="speed",
    group_ids=["static"],  # 'static' column for grouping
    static_categoricals=["static"],  # Encoding the 'static' column
    time_varying_known_reals=['vancouverDegC', 'whistlerDegC', 'pembertonDegC', 'lillooetDegC'],
    time_varying_unknown_reals=["speed"],  # Our target variable 'speed'
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    target_normalizer=GroupNormalizer(groups=["static"]),  # Normalize target (speed)
    add_relative_time_idx=False,
    add_target_scales=True,
    randomize_length=None,
    # weights='weights'
)

batch = prediction_dataset.to_dataloader(train=False, batch_size=len(prediction_dataset), shuffle=False)

# Step 3: Load the pre-trained model
tft = TemporalFusionTransformer.load_from_checkpoint('tftCheckpoint.ckpt')

# Predict using the model
# raw_predictions, x = tft.predict(batch, mode="raw", return_x=True)
raw_predictions = tft.predict(batch, mode='raw', return_x=True)
y_pred = raw_predictions.output.prediction[:, 0, 3]
x_pred = raw_predictions.x['decoder_time_idx'][:, 0]
x_meas = data['time_idx']
y_meas = data['speed']

fig, ax = plt.subplots()
ax.plot(x_meas, y_meas, label='Measured')
ax.plot(x_pred, y_pred, label='Predicted')
plt.ylabel('Wind Speed (knots)')
plt.legend()
plt.show()
