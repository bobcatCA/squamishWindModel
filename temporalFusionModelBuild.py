import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, RMSE, MultiLoss
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from lightning.pytorch import Trainer
from torch.nn import MSELoss

# Load your dataset
data = pd.read_csv('mergedOnSpeed.csv')  # Assuming you have your data in a CSV
data = data[:10000]
data['time'] = pd.to_datetime(data['time'])
data = data.sort_values('time')
data = data.drop(columns=['time'])
data['static'] = 'S'

# Add a time index column required for TimeSeriesDataSet
data['time_idx'] = pd.Series(range(len(data)))

# Make 2 lists, 1 of the features to train on , and 1 to loop through to make a model for each label
training_features = ['vancouverDegC', 'whistlerDegC', 'pembertonDegC', 'lillooetDegC']
training_labels = ['speed', 'gust', 'lull', 'direction']  # Multiple targets

# Hyperparameter setting
max_encoder_length = 10  # Number of past observations
max_prediction_length = 5  # Number of future steps you want to predict
training_cutoff = data['time_idx'].max() - max_prediction_length

# TODO: deterimine if the loop is absolutely necessary. I haven't been able to make good predictions in a single
# model, it seems like all the target parameters are just averaging together.
for training_label in training_labels:
    print(f'now training {training_label}')
    # Define the TimeSeriesDataSet
    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx='time_idx',
        target=training_label,
        group_ids=['static'],  # 'static' column for grouping
        static_categoricals=['static'],  # Encoding the 'static' column
        time_varying_known_reals=training_features,
        time_varying_unknown_reals=[training_label],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        target_normalizer=GroupNormalizer(groups=['static']),
        add_relative_time_idx=False,
        add_target_scales=True,
        randomize_length=None,
    )

    # Create a validation dataset
    validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training_cutoff + 1)

    # Create PyTorch DataLoader for training and validation
    batch_size = 32
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=7)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=7, shuffle=False)
    loss_func = RMSE()  # TODO: determine if this is the best loss funciton or not

    # Define the Temporal Fusion Transformer model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,  # Size of the hidden layer
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=4,
        output_size=1,  # 1 targets
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
