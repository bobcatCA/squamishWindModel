"""Feature selection sweep for hourly and daily TFT models.

Trains each feature configuration for a small number of epochs and reports
validation loss per target. Results are written to feature_selection_results.json.

    python feature_selection.py --mode hourly --epochs 2
    python feature_selection.py --mode daily  --epochs 3
"""

import argparse
import copy
import json

import lightning.pytorch as pl
import pandas as pd
from lightning.pytorch import Trainer
from pytorch_forecasting import QuantileLoss, TimeSeriesDataSet

from config import DailyConfig, HourlyConfig
from tft_common import _apply_mask_intervals, _build_dataset, tft_with_ignore

# ── Hourly feature configurations ─────────────────────────────────────────────
# Pressure NaN rates (after ffill): whistlerKPa 33.6%, pembertonKPa 19.6%,
# pamKPa 11.3%, ballenasKPa 8.6%, lillooetKPa 4.0%.

_TEMPORAL     = ['year_fraction']
_PRESS_BASE   = ['comoxKPa', 'lillooetKPa', 'pamKPa', 'vancouverKPa', 'victoriaKPa']
_PRESS_NO_VIC = ['comoxKPa', 'lillooetKPa', 'pamKPa', 'vancouverKPa']

HOURLY_CONFIGS = {
    'p_base': dict(
        real_known=_TEMPORAL,
        real_unknown=_PRESS_BASE,
    ),
    'p_plus_inland_2t': dict(
        real_known=_TEMPORAL + ['lillooetDegC', 'pembertonDegC'],
        real_unknown=_PRESS_BASE,
    ),
    'current': dict(
        real_known=_TEMPORAL + ['lillooetDegC', 'pembertonDegC', 'vancouverDegC',
                                  'victoriaDegC', 'whistlerDegC'],
        real_unknown=_PRESS_BASE + ['temperature'],
    ),
    'p_no_victoria': dict(
        real_known=_TEMPORAL,
        real_unknown=_PRESS_NO_VIC,
    ),
}

# ── Daily feature configurations ──────────────────────────────────────────────

_YEAR_FRAC     = ['year_fraction']
_ALL_TEMPS     = ['lillooetDegC', 'pembertonDegC', 'vancouverDegC', 'victoriaDegC', 'whistlerDegC']
_INLAND_TEMPS  = ['lillooetDegC', 'pembertonDegC', 'whistlerDegC']
_COASTAL_TEMPS = ['vancouverDegC', 'victoriaDegC']

DAILY_CONFIGS = {
    'current': dict(
        real_known=_YEAR_FRAC + _ALL_TEMPS,
        real_unknown=_PRESS_BASE,
    ),
    'press_only': dict(
        real_known=_YEAR_FRAC,
        real_unknown=_PRESS_BASE,
    ),
    'inland_temps': dict(
        real_known=_YEAR_FRAC + _INLAND_TEMPS,
        real_unknown=_PRESS_BASE,
    ),
    'coastal_temps': dict(
        real_known=_YEAR_FRAC + _COASTAL_TEMPS,
        real_unknown=_PRESS_BASE,
    ),
    'temps_no_press': dict(
        real_known=_YEAR_FRAC + _ALL_TEMPS,
        real_unknown=[],
    ),
}


# ── Training helpers ──────────────────────────────────────────────────────────

def _load_data(config) -> pd.DataFrame:
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
    data['static']   = 'S'
    data['time_idx'] = range(len(data))
    return data.reset_index(drop=True)


def _train_one(config, data: pd.DataFrame, cutoff: int,
               target: str, epochs: int) -> float:
    training = _build_dataset(data, target, config, cutoff)
    val_data = data if config.val_full_data else data[data.time_idx > cutoff]
    validation = TimeSeriesDataSet.from_dataset(
        training, val_data, predict=config.val_predict_mode, stop_randomization=True,
    )
    train_dl = training.to_dataloader(
        train=True, batch_size=config.batch_size, num_workers=2, persistent_workers=True,
    )
    val_dl = validation.to_dataloader(
        train=False, batch_size=config.batch_size, num_workers=2, persistent_workers=True,
    )
    pl.seed_everything(42)
    tft = tft_with_ignore.from_dataset(
        training,
        learning_rate=config.learning_rate,
        hidden_size=config.hidden_size,
        attention_head_size=config.attention_head_size,
        dropout=config.dropout,
        hidden_continuous_size=config.hidden_continuous_size,
        output_size=config.output_size,
        loss=QuantileLoss(),
        logging_metrics=[],
        log_interval=-1,
        reduce_on_plateau_patience=4,
    )
    trainer = Trainer(
        accelerator=config.accelerator,
        max_epochs=epochs,
        gradient_clip_val=config.gradient_clip_val,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
    )
    trainer.fit(tft, train_dl, val_dl)
    val_loss = trainer.callback_metrics.get('val_loss', float('nan'))
    return float(val_loss.item() if hasattr(val_loss, 'item') else val_loss)


def run(mode: str, epochs: int) -> dict:
    if mode == 'hourly':
        base = HourlyConfig.from_yaml()
        feature_configs = HOURLY_CONFIGS
        out_file = 'feature_selection_results.json'
    else:
        base = DailyConfig.from_yaml()
        feature_configs = DAILY_CONFIGS
        out_file = 'feature_selection_daily_results.json'

    data   = _load_data(base)
    cutoff = base.training_cutoff(data['time_idx'].max())
    print(f'Mode: {mode}  |  Data: {len(data):,} rows  |  '
          f'cutoff idx={cutoff}  |  {epochs} epochs per run\n')

    results = {}
    for cfg_name, feature_override in feature_configs.items():
        print(f'\n{"="*64}')
        print(f'Config: {cfg_name}')
        print(f'  known:   {feature_override["real_known"]}')
        print(f'  unknown: {feature_override["real_unknown"]}')

        config = copy.deepcopy(base)
        config.real_known   = feature_override['real_known']
        config.real_unknown = feature_override['real_unknown']
        config.find_lr      = False

        results[cfg_name] = {}
        for target in config.targets:
            try:
                val_loss = _train_one(config, data, cutoff, target, epochs)
                results[cfg_name][target] = round(val_loss, 4)
                print(f'  {target:>18}: val_loss = {val_loss:.4f}')
            except Exception as exc:
                print(f'  {target:>18}: FAILED — {exc}')
                results[cfg_name][target] = None

    targets = base.targets
    col_w = 16
    print(f'\n{"="*64}')
    print(f'RESULTS  (val_loss, lower = better)  — {epochs} epochs  [{mode}]')
    print(f'{"="*64}')
    print(f'{"Config":<22}' + ''.join(f'{t:>{col_w}}' for t in targets))
    print('-' * (22 + col_w * len(targets)))
    for cfg_name, res in results.items():
        row = f'{cfg_name:<22}'
        for t in targets:
            v = res.get(t)
            row += f'{v:>{col_w}.4f}' if v is not None else f'{"ERR":>{col_w}}'
        print(row)

    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nFull results saved to {out_file}')
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Feature selection sweep')
    p.add_argument('--mode', choices=['hourly', 'daily'], required=True)
    p.add_argument('--epochs', type=int, default=2,
                   help='Training epochs per configuration (default: 2)')
    args = p.parse_args()
    run(args.mode, args.epochs)
