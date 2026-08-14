"""Plain feedforward NN for daily squamishSpeed (the 2pm snapshot).

Same shape as simple_model.py: reads real_known/real_unknown/categorical from
train_config.yaml's daily: section (see simple_feature_sweep.py --mode daily
for how to search out what belongs there).

    python simple_daily_model.py --train                    # train + save models/simple_nn_daily.pt
    python simple_daily_model.py --test [--source csv|db]    # predict + plot
    python simple_daily_model.py                             # train then test
"""

import argparse
import os
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv

from config import DailyConfig
from ec_scrape import normalize_sky_series
from simple_model import MLP, SKY_CATEGORIES, _one_hot_categorical
from tft_model import _apply_mask_intervals

load_dotenv()
WORKING_DIR = Path(os.getenv('WORKING_DIRECTORY', '.'))
DB_PATH    = WORKING_DIR / 'web_data' / 'weather_data_hourly.db'
MODEL_PATH = WORKING_DIR / 'models' / 'simple_nn_daily.pt'

DAILY_HOUR = 14  # matches build_dataset.py's DAILY_HOUR


def _features(cfg: DailyConfig) -> list[str]:
    feats = list(cfg.real_known or []) + list(cfg.real_unknown or [])
    return list(dict.fromkeys(feats))


def train(epochs: int = 300, lr: float = 1e-3, val_frac: float = 0.2) -> None:
    cfg = DailyConfig.from_yaml()
    target = cfg.targets[0]
    cont_features = _features(cfg)
    categorical = list(cfg.categorical or [])
    if not cont_features and not categorical:
        raise ValueError('No real_known/real_unknown/categorical features configured in '
                          "train_config.yaml's daily: section — run simple_feature_sweep.py --mode daily")

    data = pd.read_csv(cfg.data_path)
    data['datetime'] = pd.to_datetime(data['datetime'], utc=True)
    data = data.sort_values('datetime').reset_index(drop=True)
    data = _apply_mask_intervals(data)
    if categorical:
        raw_coverage = data[categorical].notna().mean()
        for col in categorical:
            cov = raw_coverage[col]
            flag = '  <-- mostly forward-filled, barely any real readings' if cov < 0.3 else ''
            print(f'  {col}: {cov:.0%} raw (pre-fill) coverage{flag}')
        data[categorical] = data[categorical].ffill().bfill()
    data = data.dropna(subset=cont_features + categorical + [target]).reset_index(drop=True)

    for col in categorical:
        data[col] = normalize_sky_series(data[col])
    cat_df = _one_hot_categorical(data, categorical)
    data = pd.concat([data, cat_df], axis=1)
    features = cont_features + list(cat_df.columns)

    n_val = max(1, int(len(data) * val_frac))
    train_df, val_df = data.iloc[:-n_val], data.iloc[-n_val:]

    mean = train_df[features].mean()
    std  = train_df[features].std().replace(0, 1)
    mean[cat_df.columns] = 0.0
    std[cat_df.columns]  = 1.0

    def to_tensor(df):
        X = torch.tensor(((df[features] - mean) / std).values, dtype=torch.float32)
        y = torch.tensor(df[target].values, dtype=torch.float32)
        return X, y

    X_train, y_train = to_tensor(train_df)
    X_val, y_val     = to_tensor(val_df)

    model = MLP(len(features))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    val_loss_fn = nn.MSELoss()

    print(f'Training on {len(train_df):,} rows, validating on {len(val_df):,} rows')
    print(f'Features: {features}  →  target: {target}')
    if categorical:
        print(f'Categorical (one-hot): {categorical}  ({SKY_CATEGORIES})')

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(X_train)
        loss = val_loss_fn(pred, y_train)
        loss.backward()
        opt.step()

        if epoch % 20 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_loss = val_loss_fn(model(X_val), y_val)
            print(f'  epoch {epoch:4d}  train MSE {loss.item():6.2f}  '
                  f'val MSE {val_loss.item():6.2f}  (RMSE {val_loss.item() ** 0.5:.2f})')

    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'features': features,
        'cont_features': cont_features,
        'categorical': categorical,
        'target': target,
        'mean': mean,
        'std': std,
    }, MODEL_PATH)
    print(f'Saved {MODEL_PATH}')


def _load_test_df(cfg: DailyConfig, start_ts: pd.Timestamp, source: str) -> pd.DataFrame:
    if source == 'db':
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(
                'SELECT * FROM weather WHERE datetime > ?', conn, params=(start_ts.timestamp(),),
            )
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s', utc=True).dt.tz_convert('America/Vancouver')
        df = df[df['datetime'].dt.hour == DAILY_HOUR]
    else:
        df = pd.read_csv(cfg.data_path)
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert('America/Vancouver')
        df = df[df['datetime'] > start_ts]
    return df.sort_values('datetime').reset_index(drop=True)


def test(start: str = None, save: bool = False, source: str = 'csv') -> None:
    ckpt = torch.load(MODEL_PATH, weights_only=False)
    features, target = ckpt['features'], ckpt['target']
    cont_features = ckpt['cont_features']
    categorical = ckpt.get('categorical', [])
    mean, std = ckpt['mean'], ckpt['std']

    model = MLP(len(features))
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    cfg = DailyConfig.from_yaml()
    start_ts = (
        pd.Timestamp(start, tz='America/Vancouver') if start
        else pd.Timestamp('2000-01-01', tz='America/Vancouver')
    )
    df = _load_test_df(cfg, start_ts, source)

    df[cont_features] = df[cont_features].ffill().bfill()
    for col in categorical:
        df[col] = normalize_sky_series(df[col].ffill().bfill())
    df = df.dropna(subset=cont_features + categorical + [target]).reset_index(drop=True)
    if df.empty:
        print(f'No rows available for source={source} — '
              f'{"the live DB may be missing one of the selected features (e.g. Hum)" if source == "db" else ""}')
        return
    if categorical:
        df = pd.concat([df, _one_hot_categorical(df, categorical)], axis=1)

    X = torch.tensor(((df[features] - mean) / std).values, dtype=torch.float32)
    with torch.no_grad():
        pred = model(X).numpy()

    actual = df[target].values
    mae = np.mean(np.abs(pred - actual))
    rmse = np.sqrt(np.mean((pred - actual) ** 2))
    print(f'source={source}  {len(df):,} rows from {df["datetime"].min()} → {df["datetime"].max()}')
    print(f'MAE={mae:.2f}  RMSE={rmse:.2f}  (actual mean={actual.mean():.2f}, max={actual.max():.2f})')

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df['datetime'], actual, color='#333333', linewidth=1, alpha=0.6, marker='o', markersize=3, label='actual')
    ax.plot(df['datetime'], pred, color='#1f77b4', linewidth=1.2, alpha=0.9, marker='o', markersize=3, label='predicted')
    ax.set_ylabel(target)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.set_title(f'Daily NN — predicted vs actual {target} (2pm)  [source={source}]')
    plt.tight_layout()

    if save:
        out = WORKING_DIR / 'forecasts' / f'simple_daily_test_{source}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved {out}')
    else:
        plt.show()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Plain feedforward NN for daily squamishSpeed')
    p.add_argument('--train', action='store_true', help='Train and save the model')
    p.add_argument('--test',  action='store_true', help='Predict against CSV or DB and plot')
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--start', help='Test start date YYYY-MM-DD (default: all available data)')
    p.add_argument('--source', choices=['csv', 'db'], default='csv',
                   help='Where --test pulls data from: the full daily-built CSV (default; the '
                        'configured features may include Hum columns the live DB does not have), '
                        "or the live SQLite DB filtered to each day's 2pm row")
    p.add_argument('--save', action='store_true', help='Save plot to forecasts/ instead of showing it')
    args = p.parse_args()

    run_train = args.train or not (args.train or args.test)
    run_test  = args.test or not (args.train or args.test)

    if run_train:
        train(epochs=args.epochs, lr=args.lr)
    if run_test:
        test(start=args.start, save=args.save, source=args.source)
