"""Feature sweep for tft_daily_model.py, using actual TFT training runs (not
the cheaper MLP proxy simple_feature_sweep.py uses).

Base = train_config.yaml's daily: real_known (the 4 temperature columns) +
real_unknown, held fixed. Every other station DegC/KPa/Hum column (plus
squamishDegC) is screened as a single addition on top of that base — DegC
candidates go into real_known (EC forecasts temperature), everything else
into real_unknown (EC doesn't forecast pressure/humidity; squamishDegC is
the local sensor, never forecastable). Current hyperparameters (hidden_size
etc.) are used as-is — this sweep is feature selection only.

Trials run at SCREEN_EPOCHS with find_lr disabled and a fixed LR, for speed
and comparability across many runs — not meant to produce a final model,
just to rank candidates. Report both the univariate screen (every candidate
added alone, so you can see which are individually good/bad) and one round
of greedy combination on top of that.

    python tft_daily_sweep.py
"""

import lightning.pytorch as pl
import pandas as pd
from lightning.pytorch import Trainer
from pytorch_forecasting import QuantileLoss, TimeSeriesDataSet

from config import DailyConfig
from tft_model import _apply_mask_intervals, _build_dataset, tft_with_ignore

STATIONS = ['vancouver', 'whistler', 'comox', 'victoria', 'pemberton', 'lillooet', 'pam', 'ballenas']

SCREEN_EPOCHS = 15
FIXED_LR = 2.14e-5  # from the last find_lr run at the current hyperparameters
SEED = 0


def _load_data_and_base():
    cfg = DailyConfig.from_yaml()
    data = pd.read_csv(cfg.data_path)
    data.dropna(thresh=14, inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'], utc=True)
    data = data.sort_values('datetime')
    num_cols = data.select_dtypes(include='number').columns
    data[num_cols] = data[num_cols].ffill().bfill()
    data = _apply_mask_intervals(data)
    data = data.reset_index(drop=True)
    base_known = list(cfg.real_known or [])
    base_unknown = list(cfg.real_unknown or [])
    return cfg, data, base_known, base_unknown


def _candidates(data: pd.DataFrame, base_known: list[str], base_unknown: list[str]):
    """Return (column, slot) pairs not already in the base — DegC goes to
    real_known (forecastable), everything else to real_unknown."""
    used = set(base_known) | set(base_unknown)
    cands = []
    for s in STATIONS:
        for suf, slot in (('DegC', 'known'), ('KPa', 'unknown'), ('Hum', 'unknown')):
            col = f'{s}{suf}'
            if col in data.columns and col not in used:
                cands.append((col, slot))
    if 'squamishDegC' in data.columns and 'squamishDegC' not in used:
        cands.append(('squamishDegC', 'unknown'))
    return cands


def _fit_eval(data: pd.DataFrame, cfg: DailyConfig, real_known: list[str], real_unknown: list[str],
              epochs: int = SCREEN_EPOCHS, seed: int = SEED) -> float:
    target = cfg.targets[0]
    d = data.dropna(subset=real_known + real_unknown + [target]).reset_index(drop=True)
    d['static'] = 'S'
    d['time_idx'] = range(len(d))

    trial = DailyConfig(**cfg.__dict__)
    trial.real_known = real_known
    trial.real_unknown = real_unknown
    trial.max_epochs = epochs
    trial.learning_rate = FIXED_LR

    cutoff = trial.training_cutoff(d['time_idx'].max())
    training = _build_dataset(d, target, trial, cutoff)
    val_data = d if trial.val_full_data else d[d.time_idx > cutoff]
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
    cfg, data, base_known, base_unknown = _load_data_and_base()
    print(f'Base real_known={base_known}')
    print(f'Base real_unknown={base_unknown}')
    print(f'{len(data):,} rows post-mask  |  hidden_size={cfg.hidden_size}  epochs={SCREEN_EPOCHS}  lr={FIXED_LR}\n')

    base_loss = _fit_eval(data, cfg, base_known, base_unknown)
    print(f'Base only:  val_loss={base_loss:.4f}\n')

    candidates = _candidates(data, base_known, base_unknown)
    print(f'-- Univariate screen ({len(candidates)} candidates, base + one each) --')
    results = []
    for col, slot in candidates:
        trial_known = base_known + [col] if slot == 'known' else base_known
        trial_unknown = base_unknown + [col] if slot == 'unknown' else base_unknown
        loss = _fit_eval(data, cfg, trial_known, trial_unknown)
        results.append((col, slot, loss, base_loss - loss))
        print(f'  {col:16s} ({slot:7s}) val_loss={loss:.4f}  delta={base_loss - loss:+.4f}')

    results.sort(key=lambda x: x[2])
    print('\n-- Ranked (best first) --')
    for col, slot, loss, delta in results:
        print(f'  {col:16s} ({slot:7s}) val_loss={loss:.4f}  delta={delta:+.4f}')

    print('\n-- Greedy: add the single best candidate on top of the base --')
    best_col, best_slot, best_loss, best_delta = results[0]
    if best_delta > 1e-3:
        print(f'  + {best_col} ({best_slot}) -> val_loss={best_loss:.4f}  (base was {base_loss:.4f})')
    else:
        print(f'  no candidate improved on the base (best: {best_col} -> {best_loss:.4f}, base={base_loss:.4f})')


if __name__ == '__main__':
    main()
