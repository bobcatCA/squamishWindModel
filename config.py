from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

CONFIG_YAML = Path(__file__).parent / 'train_config.yaml'


def _load_section(section: str, yaml_path: Path = CONFIG_YAML) -> dict:
    with open(yaml_path) as f:
        return yaml.safe_load(f).get(section, {})


@dataclass
class _BaseConfig:
    """Shared hyperparameters for hourly and daily TFT configs.

    Feature lists (targets, real_known, real_unknown, categorical) and all
    sequence/training parameters are loaded from train_config.yaml by
    from_yaml(). The Python defaults below are intentionally empty — edit
    train_config.yaml to change any of these values.
    """

    data_path: str = ''
    checkpoint_suffix: str = ''

    encoder_length: int = 0
    prediction_length: int = 0
    min_encoder_length: int = 0
    min_prediction_length: int = 0

    # Feature categories — populated from train_config.yaml by from_yaml()
    targets: List[str] = field(default_factory=list)
    real_known: List[str] = field(default_factory=list)
    real_unknown: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    allow_missing_timesteps: bool = False

    hidden_size: int = 64
    attention_head_size: int = 4
    dropout: float = 0.1
    hidden_continuous_size: int = 32
    output_size: int = 7  # number of quantile outputs

    max_epochs: int = 10
    batch_size: int = 128
    gradient_clip_val: float = 0.1
    accelerator: str = 'cpu'
    optimizer: Optional[str] = None
    find_lr: bool = False
    learning_rate: float = 1e-3

    val_full_data: bool = False
    val_predict_mode: bool = False

    # Checkpoint/PKL filename prefix — override via --checkpoint-prefix in train.py
    # to save named variants without disturbing the production checkpoints.
    checkpoint_prefix: str = 'tft'

    # Symmetric peak/trough loss weighting for simple_model.py's plain NN
    # ONLY (not read by the TFT pipeline). Squared error on rows far from the
    # training target's mean — in EITHER direction — is scaled up to
    # (1+spread_weight)x. 0.0 = plain unweighted MSE. Override with
    # `python simple_model.py --spread-weight N`.
    spread_weight: float = 0.0

    # One-sided strength weighting for simple_model.py ONLY. Unlike
    # spread_weight, this does NOT care about troughs — weight ~0 for calm
    # readings, ramping up to 1.0 at the strongest reading, as
    # (actual/max)^strength_weight. Takes priority over spread_weight if both
    # are set. 0.0 = off. Override with `python simple_model.py --strength-weight N`.
    strength_weight: float = 0.0

    @classmethod
    def _from_yaml(cls, section: str, yaml_path: Path = CONFIG_YAML):
        instance = cls()
        for key, value in _load_section(section, yaml_path).items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        return instance

    def training_cutoff(self, max_idx: int) -> int:
        raise NotImplementedError


@dataclass
class HourlyConfig(_BaseConfig):
    data_path: str = 'training_data/hourly_database.csv'
    checkpoint_suffix: str = 'Hourly'

    encoder_length: int = 12
    prediction_length: int = 8
    min_encoder_length: int = 10
    min_prediction_length: int = 8

    allow_missing_timesteps: bool = True

    dropout: float = 0.1
    hidden_continuous_size: int = 32

    max_epochs: int = 10
    batch_size: int = 128
    gradient_clip_val: float = 0.1
    optimizer: Optional[str] = 'adam'
    find_lr: bool = True
    learning_rate: float = 1e-3

    @classmethod
    def from_yaml(cls, yaml_path: Path = CONFIG_YAML) -> 'HourlyConfig':
        return cls._from_yaml('hourly', yaml_path)

    def training_cutoff(self, max_idx: int) -> int:
        return max_idx - 2 * (self.encoder_length + self.prediction_length)


@dataclass
class DailyConfig(_BaseConfig):
    data_path: str = 'training_data/daily_database.csv'
    checkpoint_suffix: str = 'Daily'

    encoder_length: int = 5
    prediction_length: int = 5
    min_encoder_length: int = 2
    min_prediction_length: int = 1

    dropout: float = 0.2
    hidden_continuous_size: int = 4

    max_epochs: int = 8
    batch_size: int = 1024
    gradient_clip_val: float = 0.2
    optimizer: Optional[str] = None  # TFT default
    find_lr: bool = False
    learning_rate: float = 8e-4

    val_full_data: bool = True
    val_predict_mode: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: Path = CONFIG_YAML) -> 'DailyConfig':
        return cls._from_yaml('daily', yaml_path)

    def training_cutoff(self, max_idx: int) -> int:
        return max_idx - self.prediction_length
