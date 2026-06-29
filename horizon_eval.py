"""
horizon_eval.py — compare model predictions at multiple horizons to actual measurements.

Hourly mode (default): 1h and 4h ahead, one subplot per target (speed/gust/lull/direction).
Daily mode:            1d, 3d, and 5d ahead, one subplot per target (speed/hours_above_20/…).

Usage:
    python horizon_eval.py
    python horizon_eval.py --mode daily
    python horizon_eval.py --start 2025-06-01 --stride 4
    python horizon_eval.py --target speed
    python horizon_eval.py --save
"""

import argparse
import os
import sqlite3

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    import mplcursors
    _has_mplcursors = True
except ImportError:
    _has_mplcursors = False
    print('mplcursors not found — install with: pip install mplcursors')

import numpy as np
import pandas as pd
import torch.serialization
from dotenv import load_dotenv
from pathlib import Path
from pytorch_forecasting import TimeSeriesDataSet

from build_dataset import add_scores_to_df
from config import DailyConfig, HourlyConfig
from tft_common import tft_with_ignore

load_dotenv()
WORKING_DIR = Path(os.getenv('WORKING_DIRECTORY'))
DB_PATH = WORKING_DIR / 'weather_data_hourly.db'

_hcfg = HourlyConfig.from_yaml()
_dcfg = DailyConfig.from_yaml()

HORIZON_COLORS = {1: '#1f77b4', 3: '#2ca02c', 4: '#ff7f0e', 5: '#d62728'}


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_db_data_hourly(start: pd.Timestamp, cfg: HourlyConfig) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            'SELECT * FROM weather WHERE datetime > ?',
            conn, params=(start.timestamp(),),
        )
    df['datetime'] = (
        pd.to_datetime(df['datetime'], unit='s', utc=True)
        .dt.tz_convert('America/Vancouver')
    )
    df['sin_hour']     = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['year_fraction'] = (df['datetime'].dt.month * 30.416 + df['datetime'].dt.day) / 365
    _h = df['datetime'].dt.hour
    df['wind_hour'] = np.where(
        (_h >= 10) & (_h <= 13), 0.5 * (1 - np.cos(np.pi * (_h - 10) / 3)),
        np.where((_h > 13) & (_h < 18), 0.5 * (1 + np.cos(np.pi * (_h - 13) / 5)), 0.0)
    )
    if cfg.real_unknown:
        df[cfg.real_unknown] = df[cfg.real_unknown].ffill().bfill()
    if cfg.real_known:
        df[cfg.real_known]   = df[cfg.real_known].ffill().bfill()
    df[cfg.targets] = df[cfg.targets].fillna(0)
    df.reset_index(drop=True, inplace=True)
    df['static']   = 'S'
    df['time_idx'] = np.arange(df.shape[0])
    return df


def _load_db_data_daily(start: pd.Timestamp, cfg: DailyConfig) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df_hourly = pd.read_sql_query(
            'SELECT * FROM weather WHERE datetime > ?',
            conn, params=(start.timestamp(),),
        )
    df_hourly['datetime'] = (
        pd.to_datetime(df_hourly['datetime'], unit='s', utc=True)
        .dt.tz_convert('America/Vancouver')
    )
    df = add_scores_to_df(df_hourly)
    df['year_fraction'] = (df['datetime'].dt.month * 30.416 + df['datetime'].dt.day) / 365
    feats = (cfg.real_known or []) + (cfg.real_unknown or [])
    existing = [c for c in feats if c in df.columns]
    if existing:
        df[existing] = df[existing].ffill().bfill()
    df[cfg.targets] = df[cfg.targets].fillna(0)
    df.reset_index(drop=True, inplace=True)
    df['static']   = 'S'
    df['time_idx'] = np.arange(df.shape[0])
    return df


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model_and_dataset(target: str, mode: str):
    suffix = 'Hourly' if mode == 'hourly' else 'Daily'
    ckpt = WORKING_DIR / f'tft{target}{suffix}Checkpoint.ckpt'
    pkl  = WORKING_DIR / f'{target}_training_dataset_{mode}.pkl'
    with torch.serialization.safe_globals([TimeSeriesDataSet]):
        training_dataset = torch.load(pkl, weights_only=False)
    model = tft_with_ignore.load_from_checkpoint(ckpt)
    model.eval()
    return model, training_dataset


# ── Inference ─────────────────────────────────────────────────────────────────

def _collect_horizon_preds(df: pd.DataFrame, model, training_dataset,
                            target: str, stride: int,
                            horizons: dict, window_size: int) -> dict[int, pd.DataFrame]:
    """Slide windows over df, collect median predictions at each horizon step."""
    records = {h: [] for h in horizons}
    n_windows = max(0, (len(df) - window_size) // stride + 1)

    for w_idx, start_row in enumerate(range(0, len(df) - window_size + 1, stride)):
        window = df.iloc[start_row: start_row + window_size].copy().reset_index(drop=True)
        window['time_idx'] = window.index
        if training_dataset.weight is not None and training_dataset.weight not in window.columns:
            window[training_dataset.weight] = 1.0

        try:
            inference_ds = TimeSeriesDataSet.from_dataset(
                training_dataset, window, predict=True, stop_randomization=True,
            )
            batch = inference_ds.to_dataloader(
                train=False, batch_size=len(inference_ds), shuffle=False, num_workers=0,
            )
            raw = model.predict(batch, mode='raw', return_index=True, return_x=True)
            pred_start = int(raw.index['time_idx'].max())

            for h, step in horizons.items():
                q25, med, q75 = sorted([
                    float(raw.output.prediction[0, step, 2]),
                    float(raw.output.prediction[0, step, 3]),
                    float(raw.output.prediction[0, step, 4]),
                ])
                dt  = window['datetime'].iloc[pred_start + step]
                records[h].append({'datetime': dt, target: med,
                                   f'{target}_q25': q25, f'{target}_q75': q75})

        except Exception:
            pass

        if (w_idx + 1) % 10 == 0 or (w_idx + 1) == n_windows:
            print(f'  {w_idx + 1}/{n_windows}', end='\r', flush=True)

    print()
    return {h: pd.DataFrame(rows) for h, rows in records.items()}


# ── Plot ──────────────────────────────────────────────────────────────────────

def run(start: pd.Timestamp, stride: int, save: bool,
        targets: list = None, mode: str = 'hourly') -> None:

    if mode == 'hourly':
        cfg         = _hcfg
        df          = _load_db_data_hourly(start, cfg)
        horizons    = {1: 0, 4: 3}
        unit        = 'h'
        date_fmt    = '%b %d\n%H:%M'
        title_sfx   = '1h / 4h horizons'
        show_band   = False
    else:
        cfg         = _dcfg
        df          = _load_db_data_daily(start, cfg)
        horizons    = {1: 0}
        unit        = 'd'
        date_fmt    = '%b %d'
        title_sfx   = '1d horizon'
        show_band   = True

    if targets is None:
        targets = cfg.targets

    window_size = cfg.encoder_length + cfg.prediction_length

    print(
        f'Loaded {len(df)} rows  '
        f'{df["datetime"].iloc[0].strftime("%Y-%m-%d")} → '
        f'{df["datetime"].iloc[-1].strftime("%Y-%m-%d")}'
    )
    n_windows = max(0, (len(df) - window_size) // stride + 1)
    print(f'{n_windows} windows × {len(targets)} targets  (stride = {stride}{unit})')

    fig, axes = plt.subplots(
        len(targets), 1, figsize=(16, 4 * len(targets)), sharex=True,
    )
    if len(targets) == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        print(f'\n{target}')
        model, training_dataset = _load_model_and_dataset(target, mode)
        preds = _collect_horizon_preds(
            df, model, training_dataset, target, stride, horizons, window_size,
        )

        ax.plot(df['datetime'], df[target],
                color='#333333', linewidth=0.8, alpha=0.45, label='actual')
        for h in horizons:
            if not preds[h].empty:
                p, color = preds[h], HORIZON_COLORS[h]
                if show_band:
                    ax.fill_between(p['datetime'], p[f'{target}_q25'], p[f'{target}_q75'],
                                    color=color, alpha=0.15)
                ax.plot(p['datetime'], p[target],
                        color=color, linewidth=1.2, alpha=0.85,
                        label=f'{h}{unit} ahead')

        ax.set_ylabel(target, fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_title(target, fontsize=11, loc='left', pad=4)

    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=14))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
    fig.autofmt_xdate(rotation=0)
    fig.suptitle(
        f'{mode.capitalize()} model — actual vs predicted at {title_sfx}',
        fontsize=13, y=1.005,
    )
    plt.tight_layout()

    if _has_mplcursors:
        import pytz
        _tz = pytz.timezone('America/Vancouver')
        cursor = mplcursors.cursor(axes, hover=True)

        @cursor.connect('add')
        def _on_hover(sel):
            x, y = sel.target
            fmt = '%Y-%m-%d %H:%M %Z' if mode == 'hourly' else '%Y-%m-%d %Z'
            dt = mdates.num2date(x).astimezone(_tz).strftime(fmt)
            sel.annotation.set_text(f'{sel.artist.get_label()}\n{dt}\n{y:.1f}')
            sel.annotation.get_bbox_patch().set(alpha=0.85)

    if save:
        out = WORKING_DIR / f'horizon_eval_{mode}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'\nSaved {out}')

    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Evaluate TFT models at multiple forecast horizons')
    p.add_argument('--mode', choices=['hourly', 'daily'], default='hourly',
                   help='Which model pipeline to evaluate (default: hourly)')
    p.add_argument('--start', default=None,
                   help='Start date YYYY-MM-DD (default: all DB data)')
    p.add_argument('--stride', type=int, default=None,
                   help='Steps between inference windows (default: 4 for hourly, 1 for daily)')
    p.add_argument('--target',
                   help='Evaluate a single target (default: all for the chosen mode)')
    p.add_argument('--save', action='store_true',
                   help='Save figure to horizon_eval_{mode}.png')
    args = p.parse_args()

    start_ts = (
        pd.Timestamp(args.start, tz='America/Vancouver')
        if args.start
        else pd.Timestamp('2000-01-01', tz='America/Vancouver')
    )
    stride   = args.stride if args.stride is not None else (4 if args.mode == 'hourly' else 1)
    targets  = [args.target] if args.target else None

    run(start_ts, stride=stride, save=args.save, targets=targets, mode=args.mode)
