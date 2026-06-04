"""Train hourly or daily TFT wind models.

Usage:
    python train.py --mode hourly [options]
    python train.py --mode daily  [options]
"""

import argparse

from config import DailyConfig, HourlyConfig
from tft_common import train_model


def parse_args():
    p = argparse.ArgumentParser(description='Train TFT wind models')
    p.add_argument('--mode', choices=['hourly', 'daily'], required=True,
                   help='Which model pipeline to train')
    p.add_argument('--epochs', type=int, help='Override max_epochs')
    p.add_argument('--dropout', type=float, help='Override dropout')
    p.add_argument('--hidden-size', type=int, dest='hidden_size',
                   help='Override hidden_size')
    p.add_argument('--lr', type=float,
                   help='Set a fixed learning rate (skips LR finder)')
    p.add_argument('--targets', nargs='+',
                   help='Train a subset of targets, e.g. --targets speed gust')
    p.add_argument('--no-lr-find', action='store_true',
                   help='Skip LR finder and use --lr or the config default')
    p.add_argument('--weight-boost', type=float, dest='weight_boost',
                   help='Up-weight target-hour samples by this factor (e.g. 3.0)')
    p.add_argument('--checkpoint-prefix', dest='checkpoint_prefix',
                   help='Prefix for checkpoint and PKL filenames (default: "tft")')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    config = HourlyConfig.from_yaml() if args.mode == 'hourly' else DailyConfig.from_yaml()

    if args.epochs is not None:
        config.max_epochs = args.epochs
    if args.dropout is not None:
        config.dropout = args.dropout
    if args.hidden_size is not None:
        config.hidden_size = args.hidden_size
    if args.targets:
        config.targets = args.targets
    if args.lr is not None:
        config.learning_rate = args.lr
        config.find_lr = False
    if args.no_lr_find:
        config.find_lr = False
    if args.weight_boost is not None:
        config.sample_weight_boost = args.weight_boost
    if args.checkpoint_prefix is not None:
        config.checkpoint_prefix = args.checkpoint_prefix

    train_model(config)
