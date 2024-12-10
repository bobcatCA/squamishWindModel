import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from lightning.pytorch import Trainer
from torch.nn import MSELoss

# Load your dataset
df = pd.read_csv('mergedOnSpeed.csv')  # Assuming you have your data in a CSV
data = df
data['time'] = pd.to_datetime(data['time'])
data = data.sort_values('time')
data = data.drop(columns=['time'])
data['static'] = 'S'

# Interpolate for missing data (only for those missing just a few points)
data["pamKPa"] = data["pamKPa"].interpolate()
data["pamKPa"] = data["pamKPa"].bfill()

# Add a time index column required for TimeSeriesDataSet
data['time_idx'] = pd.Series(range(len(data)))

# Split the data into training and validation
model_features = ['comoxDegC', 'comoxKPa','lillooetDegC', 'lillooetKPa',
                  'pamDegC', 'pamKPa', 'pembertonDegC', 'pembertonKPa',
                  'vancouverDegC', 'vancouverKPa', 'whistlerDegC', 'whistlerKPa']
model_labels = ['speed', 'gust', 'lull', 'direction']
max_encoder_length = 15  # Number of past observations
max_prediction_length = 5  # Number of future steps you want to predict
training_cutoff = data['time_idx'].max() - max_prediction_length

# Define the normalizer for all the different variables:
target_normalizer = MultiNormalizer([
    GroupNormalizer(groups=["static"], transformation="softplus"),  # For 'speed'
    GroupNormalizer(groups=["static"], transformation="softplus"),  # For 'gust'
    GroupNormalizer(groups=["static"], transformation="softplus"),  # For 'lull'
    GroupNormalizer(groups=["static"], transformation="softplus"),  # For 'direction'
])

# Define the TimeSeriesDataSet
training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
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

# Create a validation dataset
validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training_cutoff + 1)

# Create PyTorch DataLoader for training and validation
batch_size = 64
# train_dataloader = DataLoader(training, batch_size=batch_size, shuffle=True, num_workers=0)
# val_dataloader = DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=0)
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=7)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=7)

# Define the Temporal Fusion Transformer model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=32,  # Size of the hidden layer
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=[1, 1, 1, 1],
    loss=MSELoss(),
    log_interval=10,
    reduce_on_plateau_patience=4
)

# Wrap the model in a PyTorch Lightning Trainer
trainer = Trainer(
    accelerator='cpu',
    max_epochs=10,
    gradient_clip_val=0.1
)

# Train the model
trainer.fit(tft, train_dataloader, val_dataloader)

# Save the model after training
trainer.save_checkpoint('tftCheckpoint.ckpt')

# # 10. Load the trained model and use it for predictions
# best_model = TemporalFusionTransformer.load_from_checkpoint('tftCheckpoint.ckpt')
#
# # Predicting on the validation set
# predictions, index = best_model.predict(val_dataloader, return_index=True)
#
# # 11. Extract predicted speeds
# predicted_speeds = predictions[..., 0]  # Assuming speed is the first quantile
# print(predicted_speeds)
