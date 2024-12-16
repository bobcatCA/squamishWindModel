import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, RMSE
from pytorch_forecasting.data import GroupNormalizer
from lightning.pytorch import Trainer
import numpy as np

# Load the dataset
data = pd.read_csv('mergedOnSpeed.csv')  # Assuming you have your data in a CSV
data = data[25720:30000]  # Drop the first bit, Pam rocks data missing and important

# Pre-Processing
# data['pamKPa'] = data['pamKPa'].fillna()  # Pam Rocks was missing a few points at the start

# Add date-features. NOTE: year_fraction approximates assuming a month has 30.416 days. Should be close enough
data['time'] = pd.to_datetime(data['time'])  # Ensure it's in DateTime format
data['day_fraction'] = (data['time'] - data['time'].dt.normalize()).dt.total_seconds() / 86400  # Add as a categorical feature
data['month'] = data['time'].dt.month
data['year_fraction'] = (pd.to_timedelta(data['month'] * 30.416, unit='D')).dt.days / 365

# Process the timestamps: Sort, format, re-index, introduce static column
data = data.sort_values('time')  # Sort chronologically (if not already)
time_series = data['time'].reset_index(drop=True)  # Save for later, so we have a real time index to plot against
data = data.drop(columns=['time'])  # Drop for feeding into training model (TODO: is this necessary?)
data['static'] = 'S'  # Put a static data column into the df (required for training)
data['time_idx'] = np.arange(data.shape[0])  # Add index for model - requires time = 0, 1, 2, ..... , n

# Split the data into training and validation
max_encoder_length = 10  # Number of past observations
max_prediction_length = 5  # Number of future steps you want to predict
training_cutoff = data['time_idx'].max() - max_prediction_length

# Build the variables that form the basis of the model architecture
training_features_categorical = ['comoxSky', 'vancouverSky', 'victoriaSky', 'whistlerSky']
training_features_reals = ['comoxDegC', 'comoxKPa', 'vancouverDegC', 'vancouverKPa', 'whistlerDegC', 'pembertonDegC',
                           'lillooetDegC', 'lillooetKPa', 'pamDegC', 'pamKPa', 'victoriaDegC', 'victoriaKPa',
                           'day_fraction', 'year_fraction']
training_labels = ['speed', 'gust', 'lull', 'direction']  # Multiple targets - have to make a model for each

# TODO: deterimine if the loop is absolutely necessary. I haven't been able to make good predictions in a single model
# model, it seems like all the target parameters are just averaging together.
for training_label in training_labels:
    print(f'now training {training_label}')
    # Define the TimeSeriesDataSet
    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx='time_idx',  # Must be index = 0, 1, 2, ..... , n
        target=training_label,
        group_ids=['static'],  # Just a dummy for now - might add Month, Year, or some other categories
        static_categoricals=['static'],  # TODO: is this required since the model doesn't depend on categoricals?
        time_varying_unknown_categoricals=training_features_categorical,
        time_varying_known_reals=training_features_reals,  # Real Inputs: temperature, presssure, humidity, etc.
        time_varying_unknown_reals=[training_label],  # Target variable: speed, gust, lull, or direction
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        target_normalizer=GroupNormalizer(groups=['static']),  # groups argument only required if multiple categoricals
        add_relative_time_idx=True,  # This may or may not affect much
        add_target_scales=True,
        randomize_length=None
    )

    # Create a validation dataset
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

    # Create PyTorch DataLoader for training and validation
    batch_size = 32  # Probably should be higher than 32
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=7)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=7)
    loss_func = RMSE()  # TODO: determine if this is the best loss funciton or not

    # Define the Temporal Fusion Transformer model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,  # Size of the hidden layer
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=4,
        output_size=1,  # Will be 1 (not using quintiles)
        loss=loss_func,
        log_interval=10,
        reduce_on_plateau_patience=4
    )

    # Wrap the model in a PyTorch Lightning Trainer
    trainer = Trainer(
        accelerator='cpu',
        max_epochs=7,
        gradient_clip_val=0.1
    )

    # Train the model
    trainer.fit(tft, train_dataloader, val_dataloader)

    # Save the model after training
    checkpoint_filename = 'tft' + training_label + 'Checkpoint.ckpt'
    trainer.save_checkpoint(checkpoint_filename)

    pass

print('done')