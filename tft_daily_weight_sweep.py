"""Test whether training-time deweighting of calm (non-windy) days helps the
daily TFT. DAILY ONLY — hourly's train() never creates a 'weight' column, so
this has no effect there; see _build_dataset()'s opt-in weight support in
tft_model.py.

For each candidate deweight value, rows with squamishSpeed below
CALM_THRESHOLD get their per-row loss scaled down during TRAINING only. The
VALIDATION split always uses weight=1.0 for every row regardless of which
value is being tested — otherwise the reported val_loss would itself be
computed differently per trial (calm days scored less in the metric, not
just in training), which would make comparing across trials meaningless
rather than a fair, apples-to-apples check.

    python tft_daily_weight_sweep.py
"""

import lightning.pytorch as pl
import pandas as pd
from lightning.pytorch import Trainer
from pytorch_forecasting import QuantileLoss, TimeSeriesDataSet

from config import DailyConfig
from tft_model import _apply_mask_intervals, _build_dataset, tft_with_ignore

CALM_THRESHOLD = 12.0
DEWEIGHT_VALUES = [1.0, 0.5, 0.25, 0.1, 0.0]  # 1.0 = baseline/off
SCREEN_EPOCHS = 15
SEED = 0


def _load_data():
    cfg = DailyConfig.from_yaml()
    target = cfg.targets[0]
    features = list(cfg.real_known or []) + list(cfg.real_unknown or [])
    data = pd.read_csv(cfg.data_path)
    data.dropna(subset=features + [target], inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'], utc=True)
    data = data.sort_values('datetime')
    num_cols = data.select_dtypes(include='number').columns
    data[num_cols] = data[num_cols].ffill().bfill()
    data = _apply_mask_intervals(data)
    data = data.reset_index(drop=True)
    data['static'] = 'S'
    data['time_idx'] = range(len(data))
    return cfg, data, target


def _fit_eval(cfg: DailyConfig, data: pd.DataFrame, target: str, deweight: float,
              epochs: int = SCREEN_EPOCHS, seed: int = SEED) -> float:
    calm = data[target] < CALM_THRESHOLD

    d_train = data.copy()
    d_train['weight'] = 1.0
    d_train.loc[calm, 'weight'] = deweight

    d_val = data.copy()
    d_val['weight'] = 1.0  # always unweighted — keeps val_loss comparable across trials

    trial = DailyConfig(**cfg.__dict__)
    trial.max_epochs = epochs

    cutoff = trial.training_cutoff(d_train['time_idx'].max())
    training = _build_dataset(d_train, target, trial, cutoff)
    val_data = d_val if trial.val_full_data else d_val[d_val.time_idx > cutoff]
    validation = TimeSeriesDataSet.from_dataset(
        training, val_data, predict=trial.val_predict_mode, stop_randomization=True,
    )
    train_dl = training.to_dataloader(train=True, batch_size=trial.batch_size, num_workers=0)
    val_dl = validation.to_dataloader(train=False, batch_size=trial.batch_size, num_workers=0)

    pl.seed_everything(seed)
    model = tft_with_ignore.from_dataset(
        training,
        learning_rate=trial.learning_rate,
        hidden_size=trial.hidden_size,
        attention_head_size=trial.attention_head_size,
        dropout=trial.dropout,
        hidden_continuous_size=trial.hidden_continuous_size,
        output_size=trial.output_size,
        loss=QuantileLoss(),
        logging_metrics=[],
        log_interval=0,
        reduce_on_plateau_patience=4,
    )
    trainer = Trainer(
        accelerator=trial.accelerator, max_epochs=trial.max_epochs,
        gradient_clip_val=trial.gradient_clip_val, enable_checkpointing=False,
        logger=False, enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(model, train_dl, val_dl)
    val_loss = trainer.callback_metrics.get('val_loss')
    return float(val_loss) if val_loss is not None else float('inf')


def main() -> None:
    cfg, data, target = _load_data()
    n_calm = int((data[target] < CALM_THRESHOLD).sum())
    print(f'{len(data):,} rows post-mask  |  {n_calm:,} below {CALM_THRESHOLD}kt '
          f'({n_calm / len(data):.0%})  |  epochs={SCREEN_EPOCHS}\n')

    results = []
    for dw in DEWEIGHT_VALUES:
        loss = _fit_eval(cfg, data, target, dw)
        label = 'baseline (off)' if dw == 1.0 else f'deweight={dw}'
        results.append((dw, loss))
        print(f'  {label:20s} val_loss={loss:.4f}')

    base_loss = dict(results)[1.0]
    print(f'\nBaseline val_loss={base_loss:.4f}')
    best_dw, best_loss = min(results, key=lambda x: x[1])
    if best_dw != 1.0 and best_loss < base_loss - 1e-3:
        print(f'Best: deweight={best_dw} -> val_loss={best_loss:.4f}  (delta={base_loss - best_loss:+.4f})')
    else:
        print('No deweight value improved on the baseline.')


if __name__ == '__main__':
    main()
