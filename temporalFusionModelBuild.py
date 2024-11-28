import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
from lightning.pytorch import Trainer

# 1. Load your dataset
df = pd.read_csv('mergedOnSpeed.csv')  # Assuming you have your data in a CSV
data = df
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.sort_values('datetime')
data = data.drop(columns=['datetime'])
data['static'] = 'S'

# Add a time index column required for TimeSeriesDataSet
data['time_idx'] = pd.Series(range(len(data)))
# data['weights'] = 1.0

# 2. Split the data into training and validation
max_encoder_length = 15  # Number of past observations
max_prediction_length = 5  # Number of future steps you want to predict
training_cutoff = data['time_idx'].max() - max_prediction_length

# 3. Define the TimeSeriesDataSet
training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
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

# 4. Create a validation dataset
validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training_cutoff + 1)

# 5. Create PyTorch DataLoader for training and validation
batch_size = 64
# train_dataloader = DataLoader(training, batch_size=batch_size, shuffle=True, num_workers=0)
# val_dataloader = DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=0)
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=7)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=7)

# 6. Define the Temporal Fusion Transformer model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=32,  # Size of the hidden layer
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # Quantile predictions by default
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4
)

# 7. Wrap the model in a PyTorch Lightning Trainer
trainer = Trainer(
    accelerator='cpu',
    max_epochs=10,
    gradient_clip_val=0.1
)

# 8. Train the model
trainer.fit(tft, train_dataloader, val_dataloader)

# 9. Save the model after training
trainer.save_checkpoint('tftCheckpoint.ckpt')

# 10. Load the trained model and use it for predictions
best_model = TemporalFusionTransformer.load_from_checkpoint('tftCheckpoint.ckpt')

# Predicting on the validation set
predictions, index = best_model.predict(val_dataloader, return_index=True)

# 11. Extract predicted speeds
predicted_speeds = predictions[..., 0]  # Assuming speed is the first quantile
print(predicted_speeds)

# import warnings
#
# import lightning.pytorch as pl
# from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
# import pandas as pd
# # from pandas.core.common import SettingWithCopyWarning
# import torch
#
# from pytorch_forecasting import GroupNormalizer, TimeSeriesDataSet
# from pytorch_forecasting.data import NaNLabelEncoder
# from pytorch_forecasting.data.examples import generate_ar_data
# from pytorch_forecasting.metrics import NormalDistributionLoss
# from pytorch_forecasting.models.deepar import DeepAR
#
# # warnings.simplefilter("error", category=SettingWithCopyWarning)
#
#
# data = generate_ar_data(seasonality=10.0, timesteps=400, n_series=100)
# data["static"] = "2"
# data["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(data.time_idx, "D")
# validation = data.series.sample(20)
# max_encoder_length = 60
# max_prediction_length = 20
#
# training_cutoff = data["time_idx"].max() - max_prediction_length
#
# training = TimeSeriesDataSet(
#     data[lambda x: ~x.series.isin(validation)],
#     time_idx="time_idx",
#     target="value",
#     categorical_encoders={"series": NaNLabelEncoder().fit(data.series)},
#     group_ids=["series"],
#     static_categoricals=["static"],
#     min_encoder_length=max_encoder_length,
#     max_encoder_length=max_encoder_length,
#     min_prediction_length=max_prediction_length,
#     max_prediction_length=max_prediction_length,
#     time_varying_unknown_reals=["value"],
#     time_varying_known_reals=["time_idx"],
#     target_normalizer=GroupNormalizer(groups=["series"]),
#     add_relative_time_idx=False,
#     add_target_scales=True,
#     randomize_length=None,
# )
#
# validation = TimeSeriesDataSet.from_dataset(
#     training,
#     data[lambda x: x.series.isin(validation)],
#     # predict=True,
#     stop_randomization=True,
# )
# batch_size = 64
# train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
# val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
#
# early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")
# lr_logger = LearningRateMonitor()
#
# trainer = pl.Trainer(
#     max_epochs=10,
#     accelerator="cpu",
#     devices="auto",
#     gradient_clip_val=0.1,
#     limit_train_batches=30,
#     limit_val_batches=3,
#     # fast_dev_run=True,
#     # logger=logger,
#     # profiler=True,
#     callbacks=[lr_logger, early_stop_callback],
# )
#
#
# deepar = DeepAR.from_dataset(
#     training,
#     learning_rate=0.1,
#     hidden_size=32,
#     dropout=0.1,
#     loss=NormalDistributionLoss(),
#     log_interval=10,
#     log_val_interval=3,
#     # reduce_on_plateau_patience=3,
# )
# print(f"Number of parameters in network: {deepar.size()/1e3:.1f}k")
#
# # # find optimal learning rate
# # deepar.hparams.log_interval = -1
# # deepar.hparams.log_val_interval = -1
# # trainer.limit_train_batches = 1.0
# # res = Tuner(trainer).lr_find(
# #     deepar, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, min_lr=1e-5, max_lr=1e2
# # )
#
# # print(f"suggested learning rate: {res.suggestion()}")
# # fig = res.plot(show=True, suggest=True)
# # fig.show()
# # deepar.hparams.learning_rate = res.suggestion()
#
# torch.set_num_threads(10)
# trainer.fit(
#     deepar,
#     train_dataloaders=train_dataloader,
#     val_dataloaders=val_dataloader,
# )
#
# # calcualte mean absolute error on validation set
# actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
# predictions = deepar.predict(val_dataloader)
# print(f"Mean absolute error of model: {(actuals - predictions).abs().mean()}")
#
# # # plot actual vs. predictions
# # raw_predictions, x = deepar.predict(val_dataloader, mode="raw", return_x=True)
# # for idx in range(10):  # plot 10 examples
# #     deepar.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)
# print('done')