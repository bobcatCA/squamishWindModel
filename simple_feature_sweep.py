"""Feature selection for the plain-vanilla nowcast NNs (simple_model.py,
simple_daily_model.py).

    python simple_feature_sweep.py                  # hourly (default)
    python simple_feature_sweep.py --mode daily
    python simple_feature_sweep.py --mode daily --max-additions 10

Base features = whatever's currently active in train_config.yaml's
{mode}: real_known + real_unknown (read fresh each run, so this always
tracks your current config — empty base is fine, e.g. daily starts from
nothing). Every other station DegC/KPa/Hum column (plus squamishDegC) is a
candidate to add ON TOP of that fixed base — screened one at a time, then
greedily combined up to --max-additions extra features (default: 2 for
hourly, since it already has an established base to extend; effectively
unbounded for daily, to search out a starting set from scratch).

Methodology: all candidate columns are forward/back-filled once up front so
every combination is compared on the exact same rows and train/val split —
otherwise per-combo dropna would let sparser combos silently train (and get
scored) on a different, easier subset.
"""

import argparse

import pandas as pd
import torch
import torch.nn as nn

from config import DailyConfig, HourlyConfig
from simple_model import MLP
from tft_model import _apply_mask_intervals

STATIONS = ['vancouver', 'whistler', 'comox', 'victoria', 'pemberton', 'lillooet', 'pam', 'ballenas']
SUFFIXES = ['DegC', 'KPa', 'Hum']
DEFAULT_MAX_ADDITIONS = {'hourly': 2, 'daily': 99}
SCREEN_EPOCHS = 150
FINAL_EPOCHS = 300
SEEDS = (0, 1, 2)


def _load_data_and_base(mode: str):
    cfg = (HourlyConfig if mode == 'hourly' else DailyConfig).from_yaml()
    target = cfg.targets[0]
    base_features = list(dict.fromkeys(list(cfg.real_known or []) + list(cfg.real_unknown or [])))
    data = pd.read_csv(cfg.data_path)
    data['datetime'] = pd.to_datetime(data['datetime'], utc=True)
    data = data.sort_values('datetime').reset_index(drop=True)
    data = _apply_mask_intervals(data)
    return data, target, base_features


def _candidates(data: pd.DataFrame, base_features: list[str]) -> tuple[list[str], list[str]]:
    all_cols = [f'{s}{suf}' for s in STATIONS for suf in SUFFIXES if f'{s}{suf}' in data.columns]
    if 'squamishDegC' in data.columns:
        all_cols.append('squamishDegC')
    candidates = [c for c in all_cols if c not in base_features]
    coverage = data[candidates].notna().mean()
    for c in candidates:
        print(f'  {c:16s} {coverage[c]:.0%} raw coverage')
    return candidates, all_cols


def _fit_eval(data: pd.DataFrame, target: str, features: list[str],
              epochs: int, val_frac: float = 0.2, seeds=(0,)) -> tuple[float, int]:
    d = data.dropna(subset=features + [target]).reset_index(drop=True)
    n_val = max(1, int(len(d) * val_frac))
    train_df, val_df = d.iloc[:-n_val], d.iloc[-n_val:]

    mean = train_df[features].mean()
    std  = train_df[features].std().replace(0, 1)

    def to_tensor(df):
        X = torch.tensor(((df[features] - mean) / std).values, dtype=torch.float32)
        y = torch.tensor(df[target].values, dtype=torch.float32)
        return X, y

    X_train, y_train = to_tensor(train_df)
    X_val, y_val = to_tensor(val_df)
    loss_fn = nn.MSELoss()

    rmses = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = MLP(len(features))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            loss = loss_fn(model(X_train), y_train)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            rmses.append(loss_fn(model(X_val), y_val).item() ** 0.5)

    return sum(rmses) / len(rmses), len(d)


def main(mode: str, max_additions: int) -> None:
    data, target, base_features = _load_data_and_base(mode)
    print(f'[{mode}] Base features (from train_config.yaml real_known+real_unknown): {base_features}')

    print('\nCandidate coverage (post mask_intervals, pre-fill):')
    candidates, all_cols = _candidates(data, base_features)

    # fill everything once so every combo is scored on the same rows/split
    fill_cols = list(dict.fromkeys(base_features + candidates))
    data[fill_cols] = data[fill_cols].ffill().bfill()

    base_rmse, n = _fit_eval(data, target, base_features, SCREEN_EPOCHS, seeds=SEEDS)
    print(f'\nBase only {base_features}:  val RMSE={base_rmse:.3f}  (n={n})')

    print('\n-- Univariate screen (base + one additional feature) --')
    uni_results = []
    for feat in candidates:
        rmse, _ = _fit_eval(data, target, base_features + [feat], SCREEN_EPOCHS, seeds=SEEDS)
        uni_results.append((feat, rmse, base_rmse - rmse))
    uni_results.sort(key=lambda x: x[1])
    for feat, rmse, delta in uni_results:
        print(f'  {feat:16s} val RMSE={rmse:.3f}  delta={delta:+.3f}')

    print(f'\n-- Greedy addition (up to {max_additions} extra features) --')
    selected = list(base_features)
    added = []
    remaining = [f for f, _, _ in uni_results]
    best_rmse = base_rmse
    while remaining and len(added) < max_additions:
        scored = [(feat, _fit_eval(data, target, selected + [feat], SCREEN_EPOCHS, seeds=SEEDS)[0])
                   for feat in remaining]
        scored.sort(key=lambda x: x[1])
        best_feat, best_feat_rmse = scored[0]
        if best_feat_rmse < best_rmse - 1e-3:
            selected.append(best_feat)
            added.append(best_feat)
            remaining.remove(best_feat)
            best_rmse = best_feat_rmse
            print(f'  + {best_feat:16s} -> val RMSE={best_rmse:.3f}   added so far={added}')
        else:
            print(f'  no further improvement (best candidate {best_feat} -> {best_feat_rmse:.3f}, '
                  f'current best {best_rmse:.3f})')
            break

    print('\n=== Result ===')
    print(f'Base:     {base_features}  val RMSE={base_rmse:.3f}')
    print(f'Added:    {added or "(nothing improved on the base)"}')
    final_rmse, n = _fit_eval(data, target, selected, FINAL_EPOCHS, seeds=SEEDS)
    print(f'Final:    {selected}  val RMSE={final_rmse:.3f}  (n={n})')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Greedy feature selection for simple_model.py / simple_daily_model.py')
    p.add_argument('--mode', choices=['hourly', 'daily'], default='hourly')
    p.add_argument('--max-additions', type=int, default=None, dest='max_additions',
                   help='Cap on greedily-added features (default: 2 for hourly, 99 for daily)')
    args = p.parse_args()
    main(args.mode, args.max_additions or DEFAULT_MAX_ADDITIONS[args.mode])
