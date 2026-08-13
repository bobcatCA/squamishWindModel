"""Plain feedforward NN baseline for squamishSpeed.

A same-timestep regression (given current known + unknown features, predict
current squamishSpeed) using the exact real_known/real_unknown feature set
from train_config.yaml's hourly section. No encoder/decoder windows, no
quantile loss, no time-of-day sample weighting — this is a simplicity
baseline to sanity-check whether those features explain the diurnal wind
pattern at all before returning to TFT.

    python simple_model.py --train                 # train + save models/simple_nn_hourly.pt
    python simple_model.py --test [--start DATE]    # load model, predict against the live DB
    python simple_model.py                          # train then test
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

from config import HourlyConfig
from ec_scrape import normalize_sky_series
from tft_model import _apply_mask_intervals

load_dotenv()
WORKING_DIR = Path(os.getenv('WORKING_DIRECTORY', '.'))
DB_PATH    = WORKING_DIR / 'web_data' / 'weather_data_hourly.db'
MODEL_PATH = WORKING_DIR / 'models' / 'simple_nn_hourly.pt'

# normalize_sky_series buckets every raw EC condition string into exactly these.
SKY_CATEGORIES = ['Fair', 'Mostly Cloudy', 'Cloudy', 'Other']


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden=(64, 32)):
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _features(cfg: HourlyConfig) -> list[str]:
    feats = list(cfg.real_known or []) + list(cfg.real_unknown or [])
    return list(dict.fromkeys(feats))


def _one_hot_categorical(df: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    """One-hot encode each categorical column against the fixed sky-condition vocabulary.

    Columns are always emitted in the same {col}_{category} order regardless of
    which categories are actually present, so train- and test-time encodings
    line up even if the DB test window never sees e.g. 'Cloudy'.
    """
    parts = {
        f'{col}_{cat}': (df[col] == cat).astype(float)
        for col in categorical for cat in SKY_CATEGORIES
    }
    return pd.DataFrame(parts, index=df.index)


def _spread_weighted_mse(pred: torch.Tensor, actual: torch.Tensor, spread_weight: float) -> torch.Tensor:
    """MSE where each row's squared error is scaled up in proportion to how far
    the TRUE value sits from the typical (mean) reading — in EITHER direction.
    A row right at the mean -> x1; the most extreme peak or trough in the
    training set -> x(1+spread_weight). Unlike a magnitude-only weighting,
    this pulls on both ends symmetrically: overpredicted troughs get the same
    emphasis as underpredicted peaks. Every row still contributes >=1x — this
    only adds emphasis, never drops rows — so it can't reproduce the
    zero-gradient-window failure the TFT's hard exclusion hit (there's no
    encoder/decoder window here, each row is scored independently).
    """
    center = actual.mean()
    max_dev = (actual - center).abs().max().clamp(min=1e-6)
    weight = 1.0 + spread_weight * (actual - center).abs() / max_dev
    return (weight * (pred - actual) ** 2).mean()


def _strength_weighted_mse(pred: torch.Tensor, actual: torch.Tensor, strength_weight: float) -> torch.Tensor:
    """MSE weighted purely by how strong the TRUE wind was — NOT symmetric
    like spread-weighting. A calm row (0 kt) gets ~0 weight (errors there are
    effectively ignored); the strongest reading in the training set gets full
    weight 1.0. `strength_weight` is an EXPONENT on that ramp:
      1.0 -> weight scales linearly with speed
      >1  -> sharper, cares almost only about the very strongest readings
      <1  -> gentler, still gives some credit to mid-range speeds
    A small floor keeps calm rows from being *exactly* zero (avoids fully
    dead gradients for the huge calm-hour majority), but they're still
    effectively down-weighted to near nothing relative to peak rows.
    """
    frac = (actual / actual.max()).clamp(min=1e-3)
    weight = frac ** strength_weight
    return (weight * (pred - actual) ** 2).mean()


def train(epochs: int = 300, lr: float = 1e-3, val_frac: float = 0.2,
          spread_weight: float = None, strength_weight: float = None) -> None:
    cfg = HourlyConfig.from_yaml()
    if spread_weight is None:
        spread_weight = cfg.spread_weight
    if strength_weight is None:
        strength_weight = cfg.strength_weight
    target = cfg.targets[0]
    cont_features = _features(cfg)
    categorical = list(cfg.categorical or [])
    if not cont_features and not categorical:
        raise ValueError('No real_known/real_unknown/categorical features configured in train_config.yaml')

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
    # one-hot indicator columns are already 0/1 — don't standardize them
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
    val_loss_fn = nn.MSELoss()  # plain, unweighted — comparable across weighting settings

    print(f'Training on {len(train_df):,} rows, validating on {len(val_df):,} rows')
    print(f'Features: {features}  →  target: {target}')
    if categorical:
        print(f'Categorical (one-hot): {categorical}  ({SKY_CATEGORIES})')
    if strength_weight > 0:
        print(f'Strength-weighted loss: weight ∝ (actual/max)^{strength_weight} '
              f'— calm rows ~ignored, only strong readings matter')
    elif spread_weight > 0:
        print(f'Spread-weighted loss: up to {1 + spread_weight:.1f}x on the most extreme '
              f'readings (peaks AND troughs)')

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(X_train)
        if strength_weight > 0:
            loss = _strength_weighted_mse(pred, y_train, strength_weight)
        elif spread_weight > 0:
            loss = _spread_weighted_mse(pred, y_train, spread_weight)
        else:
            loss = val_loss_fn(pred, y_train)
        loss.backward()
        opt.step()

        if epoch % 20 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_loss = val_loss_fn(model(X_val), y_val)
            print(f'  epoch {epoch:4d}  train loss {loss.item():6.2f}  '
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


def _load_test_df_db(start_ts: pd.Timestamp) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            'SELECT * FROM weather WHERE datetime > ?', conn, params=(start_ts.timestamp(),),
        )
    df['datetime'] = pd.to_datetime(df['datetime'], unit='s', utc=True).dt.tz_convert('America/Vancouver')
    return df.sort_values('datetime').reset_index(drop=True)


def _load_test_df_csv(start_ts: pd.Timestamp) -> pd.DataFrame:
    """Load the FULL training CSV (not just the masked/trained-on subset) —
    useful for features like humidity that exist in the training data but
    were never wired into the live DB/scraper, so `--source db` can't see them.
    """
    cfg = HourlyConfig.from_yaml()
    df = pd.read_csv(cfg.data_path)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert('America/Vancouver')
    return df[df['datetime'] > start_ts].sort_values('datetime').reset_index(drop=True)


def test(start: str = None, save: bool = False, source: str = 'db') -> None:
    ckpt = torch.load(MODEL_PATH, weights_only=False)
    features, target = ckpt['features'], ckpt['target']
    cont_features = ckpt['cont_features']
    categorical = ckpt.get('categorical', [])
    mean, std = ckpt['mean'], ckpt['std']

    model = MLP(len(features))
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    start_ts = (
        pd.Timestamp(start, tz='America/Vancouver') if start
        else pd.Timestamp('2000-01-01', tz='America/Vancouver')
    )
    df = _load_test_df_csv(start_ts) if source == 'csv' else _load_test_df_db(start_ts)

    df[cont_features] = df[cont_features].ffill().bfill()
    for col in categorical:
        df[col] = normalize_sky_series(df[col].ffill().bfill())
    df = df.dropna(subset=cont_features + categorical + [target]).reset_index(drop=True)
    if categorical:
        df = pd.concat([df, _one_hot_categorical(df, categorical)], axis=1)

    X = torch.tensor(((df[features] - mean) / std).values, dtype=torch.float32)
    with torch.no_grad():
        pred = model(X).numpy()

    actual = df[target].values
    mae  = np.mean(np.abs(pred - actual))
    rmse = np.sqrt(np.mean((pred - actual) ** 2))
    print(f'source={source}  {len(df):,} rows from {df["datetime"].min()} → {df["datetime"].max()}')
    print(f'MAE={mae:.2f}  RMSE={rmse:.2f}  (actual mean={actual.mean():.2f}, max={actual.max():.2f})')

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(df['datetime'], actual, color='#333333', linewidth=0.8, alpha=0.5, label='actual')
    ax.plot(df['datetime'], pred, color='#1f77b4', linewidth=1.1, alpha=0.9, label='predicted')
    ax.set_ylabel(target)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.set_title(f'Plain NN (nowcast) — predicted vs actual {target}  [source={source}]')
    plt.tight_layout()

    if save:
        out = WORKING_DIR / 'forecasts' / f'simple_nn_test_{source}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved {out}')
    else:
        plt.show()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Plain feedforward NN baseline for squamishSpeed')
    p.add_argument('--train', action='store_true', help='Train and save the model')
    p.add_argument('--test',  action='store_true', help='Predict against the live DB and plot')
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--spread-weight', type=float, default=None, dest='spread_weight',
                   help='Override spread_weight from train_config.yaml for this run '
                        '(0.0=off/plain MSE, e.g. 2.0 for 3x max weight on the most extreme readings)')
    p.add_argument('--strength-weight', type=float, default=None, dest='strength_weight',
                   help='Override strength_weight from train_config.yaml for this run '
                        '(0.0=off; one-sided — ignores calm-reading error, weights toward strong readings; '
                        'takes priority over --spread-weight if both set)')
    p.add_argument('--start', help='Test start date YYYY-MM-DD (default: all available data)')
    p.add_argument('--source', choices=['db', 'csv'], default='db',
                   help='Where --test pulls data from: live SQLite DB (default), or the full '
                        'training CSV (training_data/hourly_database.csv) — use csv for features '
                        "like humidity that aren't in the live DB")
    p.add_argument('--save', action='store_true', help='Save plot to forecasts/ instead of showing it')
    args = p.parse_args()

    run_train = args.train or not (args.train or args.test)
    run_test  = args.test or not (args.train or args.test)

    if run_train:
        train(epochs=args.epochs, lr=args.lr,
              spread_weight=args.spread_weight, strength_weight=args.strength_weight)
    if run_test:
        test(start=args.start, save=args.save, source=args.source)
