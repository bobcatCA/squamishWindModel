"""Generate hourly and/or daily wind forecasts.

    python forecast.py                  # run both hourly and daily
    python forecast.py --mode hourly
    python forecast.py --mode daily
    python forecast.py --skip-features comoxHum,pamHum,vancouverHum
        # Temporary bridging override for named columns that don't have enough
        # history yet (e.g. a feature whose live capture just started). Each
        # named column gets an explicit, loudly-logged 0 placeholder for any
        # encoder-window gap instead of failing — scoped ONLY to the columns
        # you name; anything else with a real gap still fails loudly.
"""

import argparse
import os

import numpy as np
import pandas as pd
import pytz
import torch
import torch.serialization
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from pytorch_forecasting import TimeSeriesDataSet

from config import DailyConfig, HourlyConfig
from tft_model import tft_with_ignore
from update_data import _DB_COL_RENAME, get_conditions_table_daily, get_conditions_table_hourly

load_dotenv()
WORKING_DIR = Path(os.getenv('WORKING_DIRECTORY'))
TZ = pytz.timezone('America/Vancouver')

# The website consuming forecasts/hourly_speed_predictions.{csv,json} expects
# the pre-squamish*-rename column names (e.g. `speed`, not `squamishSpeed`) —
# the exact inverse of update_data.py's _DB_COL_RENAME, so build it from that
# single source of truth rather than hardcoding a second copy.
_OUTPUT_COL_RENAME = {v: k for k, v in _DB_COL_RENAME.items()}

# Same idea for forecasts/daily_speed_predictions.{csv,json} — the website
# expects the pipeline's pre-rebuild field names. speed_steadiness/
# direction_steadiness are the same underlying scores as the old speed_score/
# direction_score, just renamed across a pipeline rebuild; hours_above_20 has
# no current equivalent at all and is simply absent from `cols` below.
_DAILY_OUTPUT_COL_RENAME = {
    **_OUTPUT_COL_RENAME,
    'speed_steadiness': 'speed_score',
    'direction_steadiness': 'direction_score',
}

_hcfg = HourlyConfig.from_yaml()
_dcfg = DailyConfig.from_yaml()


def _warn_missing(data: pd.DataFrame, features: list, label: str, prediction_length: int,
                   skip_features: frozenset = frozenset()) -> None:
    """Print a warning for any feature columns with a NaN in the encoder
    window — the only kind of missing data forecast.py doesn't fill, unless
    the column is named in skip_features (see --skip-features).

    Prediction-horizon NaN aren't reported here: they're structural, not a
    data gap (the target/real_unknown future is unknown by definition, and
    never read by the model there), and get an inert placeholder elsewhere.
    A NaN in the encoder window has no such fallback and will fail at
    TimeSeriesDataSet construction — this warning is the only thing that
    explains why, before that failure happens — unless the caller explicitly
    opted the column into a placeholder via --skip-features, in which case
    this says so instead of "will fail".
    """
    horizon_start = len(data) - prediction_length
    for col in features:
        na = data[col].isna().iloc[:horizon_start]
        n_na = int(na.sum())
        if n_na == 0:
            continue
        total = len(na)
        groups = na.ne(na.shift()).cumsum()
        max_gap = int(na.groupby(groups).sum().max())
        if col in skip_features:
            print(f'Warning [{label}] {col}: {n_na}/{total} encoder-window values missing '
                  f'(max consecutive gap: {max_gap}) — PLACEHOLDER 0 (explicitly skipped via --skip-features)')
        elif n_na == total:
            print(f'Warning [{label}] {col}: ALL {total} encoder-window values missing — check scraper or DB')
        else:
            print(f'Warning [{label}] {col}: {n_na}/{total} encoder-window values missing '
                  f'(max consecutive gap: {max_gap}) — will fail at model construction, not filled')


# ── Hourly forecast ───────────────────────────────────────────────────────────

def _prepare_hourly(update: bool = True, skip_features: frozenset = frozenset()) -> pd.DataFrame:
    data = get_conditions_table_hourly(
        encoder_length=_hcfg.encoder_length,
        prediction_length=_hcfg.prediction_length,
        update=update,
    )
    data['hour'] = data['datetime'].dt.hour
    _warn_missing(data, _hcfg.targets, 'hourly', _hcfg.prediction_length, skip_features)
    data[_hcfg.targets] = data[_hcfg.targets].astype(float)
    # The prediction-horizon rows are unavoidably NaN for the target — that's
    # what's being forecast, there's no real value to retrieve. TimeSeriesDataSet
    # still needs a numeric placeholder there even in predict=True mode, so fill
    # ONLY those rows; a NaN anywhere in the encoder window is a genuine data
    # gap and should still fail loudly like everything else here now does.
    horizon = data.index[-_hcfg.prediction_length:]
    data.loc[horizon, _hcfg.targets] = data.loc[horizon, _hcfg.targets].fillna(0)
    if _hcfg.real_unknown:
        _warn_missing(data, _hcfg.real_unknown, 'hourly', _hcfg.prediction_length, skip_features)
        # real_unknown means "unknown in the future" by definition — TFT's
        # encoder-only inputs are never read for decoder/prediction-horizon
        # steps, but TimeSeriesDataSet still requires a numeric placeholder
        # there. Same rule as the target: fill ONLY the horizon.
        data.loc[horizon, _hcfg.real_unknown] = data.loc[horizon, _hcfg.real_unknown].fillna(0)
    if _hcfg.real_known:
        _warn_missing(data, _hcfg.real_known, 'hourly', _hcfg.prediction_length, skip_features)
    # Explicit, named bridging override — fills the WHOLE window (encoder
    # included) for exactly the columns named via --skip-features, since
    # _warn_missing already logged these as an accepted exception above
    # rather than letting them fail. Nothing else gets this treatment.
    skip_cols = [c for c in skip_features if c in data.columns]
    if skip_cols:
        data[skip_cols] = data[skip_cols].fillna(0)
    data.reset_index(drop=True, inplace=True)
    data['static']   = 'S'
    data['time_idx'] = np.arange(data.shape[0])
    return data


def _predict_hourly_target(data: pd.DataFrame, target: str,
                            forecast_q: int = 4) -> pd.DataFrame:
    ckpt_model   = WORKING_DIR / 'models' / f'tft{target}HourlyCheckpoint.ckpt'
    ckpt_dataset = WORKING_DIR / 'models' / f'{target}_training_dataset_hourly.pkl'
    with torch.serialization.safe_globals([TimeSeriesDataSet]):
        training_dataset = torch.load(ckpt_dataset, weights_only=False)
    inference_ds = TimeSeriesDataSet.from_dataset(
        training_dataset, data, predict=True, stop_randomization=True,
    )
    batch = inference_ds.to_dataloader(
        train=False, batch_size=len(inference_ds), shuffle=False, num_workers=4,
    )
    model = tft_with_ignore.load_from_checkpoint(ckpt_model)
    raw = model.predict(batch, mode='raw', return_index=True, return_x=True)
    pred_start = raw.index['time_idx'].max()
    y_mid = raw.output.prediction[:, :, forecast_q].numpy().reshape(-1)
    dt_pred = data['datetime'].iloc[pred_start:len(data)]
    return pd.DataFrame({'datetime': dt_pred, target: y_mid})


def run_hourly(update: bool = True, skip_features: frozenset = frozenset()) -> None:
    print(f'Hourly forecast started at {datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")}')
    data = _prepare_hourly(update=update, skip_features=skip_features)
    df_out = pd.DataFrame()
    for target in _hcfg.targets:
        df_t = _predict_hourly_target(data, target)
        df_out = df_t if df_out.empty else df_out.merge(df_t, on='datetime', how='outer')
    df_out = df_out[['datetime'] + _hcfg.targets]
    df_out = df_out.rename(columns=_OUTPUT_COL_RENAME)
    df_out.to_csv(WORKING_DIR / 'forecasts' / 'hourly_speed_predictions.csv', index=False)
    df_out.to_json(WORKING_DIR / 'forecasts' / 'hourly_speed_predictions.json', orient='records', lines=True, date_format='iso')
    print(f'Hourly forecast complete at {datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")}')


# ── Daily forecast ────────────────────────────────────────────────────────────

def _prepare_daily(update: bool = True, skip_features: frozenset = frozenset()) -> pd.DataFrame:
    data = get_conditions_table_daily(
        encoder_length=_dcfg.encoder_length,
        prediction_length=_dcfg.prediction_length,
        update=update,
    )
    horizon = data.index[-_dcfg.prediction_length:]
    if _dcfg.real_unknown:
        _warn_missing(data, _dcfg.real_unknown, 'daily', _dcfg.prediction_length, skip_features)
        # Same reasoning as hourly: real_unknown is unavoidably NaN for the
        # prediction horizon by definition, and unused there by the model —
        # placeholder only that range, not the encoder window.
        data.loc[horizon, _dcfg.real_unknown] = data.loc[horizon, _dcfg.real_unknown].fillna(0)
    if _dcfg.real_known:
        _warn_missing(data, _dcfg.real_known, 'daily', _dcfg.prediction_length, skip_features)
    for target in _dcfg.targets:
        if target in data.columns:
            _warn_missing(data, [target], 'daily', _dcfg.prediction_length, skip_features)
            data[target] = data[target].astype(float)
            # Same reasoning as hourly: only the prediction-horizon rows get a
            # placeholder — a NaN target in the encoder window is a real gap.
            data.loc[horizon, target] = data.loc[horizon, target].fillna(0)
    # Explicit, named bridging override — see _prepare_hourly's comment.
    skip_cols = [c for c in skip_features if c in data.columns]
    if skip_cols:
        data[skip_cols] = data[skip_cols].fillna(0)
    data.reset_index(drop=True, inplace=True)
    data['static']   = 'S'
    data['time_idx'] = np.arange(data.shape[0])
    return data


def _predict_daily_target(data: pd.DataFrame, target: str,
                           forecast_q: int = 3) -> pd.DataFrame:
    ckpt_model   = WORKING_DIR / 'models' / f'tft{target}DailyCheckpoint.ckpt'
    ckpt_dataset = WORKING_DIR / 'models' / f'{target}_training_dataset_daily.pkl'
    with torch.serialization.safe_globals([TimeSeriesDataSet]):
        training_dataset = torch.load(ckpt_dataset, weights_only=False)
    inference_ds = TimeSeriesDataSet.from_dataset(
        training_dataset, data, predict=True, stop_randomization=True,
    )
    batch = inference_ds.to_dataloader(
        train=False, batch_size=len(inference_ds), shuffle=False, num_workers=4,
    )
    model = tft_with_ignore.load_from_checkpoint(ckpt_model)
    raw = model.predict(batch, mode='raw', return_index=True, return_x=True)
    pred_start = raw.index['time_idx'].max()
    y_mid = raw.output.prediction[:, :, forecast_q].numpy().reshape(-1)
    dt_pred = data['datetime'].iloc[pred_start:len(data)]
    return pd.DataFrame({'datetime': dt_pred, target: y_mid})


def run_daily(update: bool = True, skip_features: frozenset = frozenset()) -> None:
    """Forecast every target in daily: targets (squamishSpeed, speed_steadiness,
    direction_steadiness) — the steadiness scores are computed live from full
    hourly (not just 2pm) squamish sensor history by
    get_conditions_table_daily(), reusing build_dataset.py's
    add_speed_steadiness()/add_direction_steadiness() so the live and training
    definitions can't drift apart (see CLAUDE.md's Daily section)."""
    print(f'Daily forecast started at {datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")}')
    data = _prepare_daily(update=update, skip_features=skip_features)
    df_out = pd.DataFrame()
    for target in _dcfg.targets:
        if target not in data.columns:
            print(f'Skipping {target}: get_conditions_table_daily() does not produce this column')
            continue
        df_t = _predict_daily_target(data, target)
        df_out = df_t if df_out.empty else df_out.merge(df_t, on='datetime', how='outer')
    if df_out.empty:
        print('Daily forecast: no targets available for live inference.')
        return
    cols = [c for c in _dcfg.targets if c in df_out.columns]
    df_out = df_out[['datetime'] + cols]
    df_out = df_out.rename(columns=_DAILY_OUTPUT_COL_RENAME)
    df_out.to_csv(WORKING_DIR / 'forecasts' / 'daily_speed_predictions.csv', index=False)
    df_out.to_json(WORKING_DIR / 'forecasts' / 'daily_speed_predictions.json', orient='records', lines=True, date_format='iso')
    print(f'Daily forecast complete at {datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")}')


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Generate wind forecasts')
    p.add_argument('--mode', choices=['hourly', 'daily', 'both'], default='both',
                    help='Which forecast to run (default: both)')
    p.add_argument('--skip-features', default='',
                    help='Comma-separated column names (e.g. comoxHum,pamHum) to placeholder '
                         'with 0 instead of failing on an encoder-window gap — temporary bridging '
                         'override for a feature that does not have enough history yet. Scoped '
                         'ONLY to the named columns; everything else still fails loudly on a gap.')
    args = p.parse_args()
    skip_features = frozenset(f.strip() for f in args.skip_features.split(',') if f.strip())

    if args.mode in ('hourly', 'both'):
        run_hourly(skip_features=skip_features)
    if args.mode in ('daily', 'both'):
        run_daily(skip_features=skip_features)
