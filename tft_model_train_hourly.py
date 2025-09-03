import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, RMSE, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import MAE

class tft_with_ignore(TemporalFusionTransformer):
    def __init__(self, *args, loss=None, **kwargs):
        # Save hyperparameters, except loss and
        self.save_hyperparameters(ignore=['loss'])

        # Call parent class
        super().__init__(*args, loss=loss, **kwargs)

data = pd.read_csv('hourly_database.csv')  # Assuming you have your data in a CSV
# data = data[20000:26599]  # Subset to reduce compute time
gpu_or_cpu = 'cpu'

# Process the timestamps: Sort, format, re-index, introduce static column
data['datetime'] = pd.to_datetime(data['datetime'])  # Ensure it's in DateTime format
data['speed_missing'] = data['speed_missing'].astype(str)  # Needs to be type str to be a gategorical
# data['is_thermal'] = data['is_thermal'].astype(str)
data['hour'] = data['hour'].astype(str)
data = data.sort_values('datetime')  # Sort chronologically (if not already)
time_series = data['datetime'].reset_index(drop=True)  # Save for later, so we have a real time index to plot against
# data = data.drop(columns=['time'])  # Drop for feeding into training model (TODO: is this necessary?)
data['static'] = 'S'  # Put a static data column into the df (required for training)
data['time_idx'] = np.arange(data.shape[0])  # Add index for model - requires time = 0, 1, 2, ..... , n

# Split the data into training and validation
max_encoder_length = 12  # Number of past observations
max_prediction_length = 8  # Number of future steps you want to predict
training_cutoff = data['time_idx'].max() - 2 * (max_encoder_length + max_prediction_length)

# Build the variables that form the basis of the model architecture
# training_groups = ['static', 'is_daytime', 'is_thermal']
training_groups = ['static']
training_features_categorical = [
    'speed_missing'
]

training_features_reals_known = [
    'lillooetDegC', 'pembertonDegC','sin_hour', 'vancouverDegC', 'victoriaDegC',
    'whistlerDegC', 'year_fraction'
]

training_features_reals_unknown = [
    'comoxKPa', 'lillooetKPa', 'pamKPa', 'temperature', 'vancouverKPa', 'victoriaKPa'
]

# training_features_reals_unknown = ['comoxKPa', 'vancouverKPa', 'lillooetKPa', 'pamKPa', 'ballenasKPa']
training_labels = ['speed', 'gust', 'lull', 'direction']  # Multiple targets - have to make a model for each

# Loop through each label and make a model for each
for training_label in training_labels:
    # print(f'now training {training_label}')
    # Define the TimeSeriesDataSet
    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx='time_idx',  # Must be index = 0, 1, 2, ..... , n
        target=training_label,
        group_ids=training_groups,
        static_categoricals=['static'],  # Just a dummy set to have one static
        time_varying_known_categoricals=training_features_categorical,
        time_varying_known_reals=training_features_reals_known,  # Real Inputs: temperature, presssure, humidity, etc.
        time_varying_unknown_reals=([training_label] + training_features_reals_unknown),  # Target variable: speed, gust, lull, or direction
        min_encoder_length=10,  # Based on PyTorch example
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        target_normalizer=GroupNormalizer(groups=training_groups),  # groups argument only required if multiple categoricals
        allow_missing_timesteps=True,  # Comment out if not using groups
        add_relative_time_idx=True,  # This may or may not affect much
        add_target_scales=True,
        randomize_length=None
    )

    # Save the training dataset (properties will be used later during inference)
    training.save(f'{training_label}_training_dataset_hourly.pkl')

    # Create a validation dataset
    validation = TimeSeriesDataSet.from_dataset(
        training, data[data.time_idx > training_cutoff], predict=False, stop_randomization=True)

    # Create PyTorch DataLoader for training and validation
    batch_size = 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=4)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=4)

    # Configure the model
    pl.seed_everything(42)

    # Find optimal learning rate
    trainer = pl.Trainer(
        accelerator=gpu_or_cpu,
        gradient_clip_val=0.1,
        max_epochs=1,
        enable_checkpointing=False,
        logger=False,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        output_size=1,  # 1 with RMSE, 7 with Quantile
        loss=RMSE(),
        # loss=QuantileLoss(),
        # optimizer='Ranger',
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # Find optimal learning rate
    res = Tuner(trainer).lr_find(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        max_lr=10.0,
        min_lr=1e-6,
    )

    optimal_lr = res.suggestion()
    print(f'Optimal learning rate: {optimal_lr}')

    # Define the Temporal Fusion Transformer model
    loss_func = QuantileLoss()
    tft = tft_with_ignore.from_dataset(
        training,
        learning_rate=optimal_lr,
        hidden_size=64,  # Size of the hidden layer
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        # output_size=1,  # Will be 1 (not using quintiles)
        output_size=7,
        # loss=loss_func,
        logging_metrics=[],
        log_interval=10,
        reduce_on_plateau_patience=4,
        optimizer='adam'
    )
    tft.loss = loss_func

    # Wrap the model in a PyTorch Lightning Trainer
    trainer = Trainer(
        accelerator=gpu_or_cpu,
        max_epochs=10,
        gradient_clip_val=0.1
    )

    # Train the model
    trainer.fit(tft, train_dataloader, val_dataloader)

    # Save the model after training
    checkpoint_filename = 'tft' + training_label + 'HourlyCheckpoint.ckpt'
    trainer.save_checkpoint(checkpoint_filename)

    pass

print('done')





















