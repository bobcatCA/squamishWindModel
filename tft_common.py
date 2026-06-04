import yaml

import lightning.pytorch as pl
import numpy as np
import pandas as pd
from lightning.pytorch import Trainer
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting import (
    QuantileLoss, RMSE, TemporalFusionTransformer, TimeSeriesDataSet,
)
from pytorch_forecasting.data import GroupNormalizer

from config import CONFIG_YAML


class tft_with_ignore(TemporalFusionTransformer):
    """TFT subclass that skips serialising loss and logging_metrics into hparams.

    Without this, loading a checkpoint with a custom loss object fails.
    """

    def __init__(self, *args, loss=None, logging_metrics=None, **kwargs):
        self.save_hyperparameters(ignore=['loss', 'logging_metrics'])
        super().__init__(*args, loss=loss, logging_metrics=logging_metrics, **kwargs)


def _build_dataset(data: pd.DataFrame, target: str, config,
                   cutoff: int) -> TimeSeriesDataSet:
    required = set(config.real_known + config.real_unknown + config.categorical + [target])
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f'Feature columns missing from data: {missing}')

    kwargs = dict(
        time_idx='time_idx',
        target=target,
        group_ids=['static'],
        static_categoricals=['static'],
        time_varying_known_categoricals=config.categorical,
        time_varying_known_reals=config.real_known,
        time_varying_unknown_reals=[target] + config.real_unknown,
        min_encoder_length=config.min_encoder_length,
        max_encoder_length=config.encoder_length,
        min_prediction_length=config.min_prediction_length,
        max_prediction_length=config.prediction_length,
        target_normalizer=GroupNormalizer(groups=['static']),
        add_relative_time_idx=True,
        add_target_scales=True,
        randomize_length=None,
    )
    if config.allow_missing_timesteps:
        kwargs['allow_missing_timesteps'] = True
    if 'weight' in data.columns:
        kwargs['weight'] = 'weight'
    return TimeSeriesDataSet(data[data.time_idx <= cutoff], **kwargs)


def _apply_mask_intervals(data: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose date falls within any mask_interval in train_config.yaml."""
    with open(CONFIG_YAML) as f:
        intervals = yaml.safe_load(f).get('data', {}).get('mask_intervals', [])
    for iv in intervals:
        start = pd.Timestamp(iv['start']).date()
        end = pd.Timestamp(iv['end']).date()
        mask = (data['datetime'].dt.date >= start) & (data['datetime'].dt.date <= end)
        n = int(mask.sum())
        data = data[~mask].reset_index(drop=True)
        print(f'  Masked {n:,} rows  ({iv["start"]} → {iv["end"]})')
    return data


def train_model(config) -> None:
    """Train one TFT per target variable and save checkpoints + dataset PKLs."""
    data = pd.read_csv(config.data_path)
    data.dropna(thresh=14, inplace=True)

    if 'datetime' in data.columns:
        data['datetime'] = pd.to_datetime(data['datetime'], utc=True)
        data = data.sort_values('datetime')
    for col in config.categorical:
        data[col] = data[col].astype(str)

    num_cols = data.select_dtypes(include='number').columns
    data[num_cols] = data[num_cols].ffill().bfill()

    data = _apply_mask_intervals(data)
    data['static'] = 'S'
    data['time_idx'] = np.arange(len(data))
    data = data.reset_index(drop=True)

    if config.sample_weight_boost > 1.0:
        hour = data['datetime'].dt.hour
        in_window = (hour >= config.weight_target_start_hour) & (hour < config.weight_target_end_hour)
        data['weight'] = np.where(in_window, float(config.sample_weight_boost), 1.0)
        n_boosted = int(in_window.sum())
        print(f'  Sample weighting: {config.sample_weight_boost}× for hours '
              f'{config.weight_target_start_hour}–{config.weight_target_end_hour} '
              f'({n_boosted:,} of {len(data):,} rows)')

    cutoff = config.training_cutoff(data['time_idx'].max())
    loss_fn = QuantileLoss()

    # PKL name includes checkpoint_prefix only when non-default, so the
    # production inference pipeline (which uses the bare name) is unaffected.
    pkl_prefix = '' if config.checkpoint_prefix == 'tft' else f'{config.checkpoint_prefix}_'

    for target in config.targets:
        print(f'\nTraining {target} ({config.checkpoint_suffix})')

        training = _build_dataset(data, target, config, cutoff)
        training.save(f'{pkl_prefix}{target}_training_dataset_{config.checkpoint_suffix.lower()}.pkl')

        val_data = data if config.val_full_data else data[data.time_idx > cutoff]
        validation = TimeSeriesDataSet.from_dataset(
            training, val_data,
            predict=config.val_predict_mode,
            stop_randomization=True,
        )

        train_dl = training.to_dataloader(train=True, batch_size=config.batch_size, num_workers=4)
        val_dl = validation.to_dataloader(train=False, batch_size=config.batch_size, num_workers=4)

        pl.seed_everything(42)

        lr = config.learning_rate
        if config.find_lr:
            lr_probe = TemporalFusionTransformer.from_dataset(
                training,
                learning_rate=lr,
                hidden_size=config.hidden_size,
                attention_head_size=config.attention_head_size,
                dropout=config.dropout,
                hidden_continuous_size=config.hidden_continuous_size,
                output_size=1,
                loss=RMSE(),
                log_interval=10,
                reduce_on_plateau_patience=4,
            )
            lr_trainer = Trainer(
                accelerator=config.accelerator,
                gradient_clip_val=config.gradient_clip_val,
                max_epochs=1,
                enable_checkpointing=False,
                logger=False,
            )
            res = Tuner(lr_trainer).lr_find(
                lr_probe,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl,
                max_lr=10.0,
                min_lr=1e-6,
            )
            lr = res.suggestion()
            print(f'  Optimal LR: {lr:.2e}')

        model_kwargs = dict(
            learning_rate=lr,
            hidden_size=config.hidden_size,
            attention_head_size=config.attention_head_size,
            dropout=config.dropout,
            hidden_continuous_size=config.hidden_continuous_size,
            output_size=config.output_size,
            loss=loss_fn,
            logging_metrics=[],
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        if config.optimizer:
            model_kwargs['optimizer'] = config.optimizer

        tft = tft_with_ignore.from_dataset(training, **model_kwargs)

        trainer = Trainer(
            accelerator=config.accelerator,
            max_epochs=config.max_epochs,
            gradient_clip_val=config.gradient_clip_val,
        )
        trainer.fit(tft, train_dl, val_dl)
        trainer.save_checkpoint(
            f'{config.checkpoint_prefix}{target}{config.checkpoint_suffix}Checkpoint.ckpt'
        )
