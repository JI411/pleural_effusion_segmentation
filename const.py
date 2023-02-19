"""
Main constants: paths, params and etc.
"""
import os
import typing as tp
from dataclasses import dataclass

from pathlib import Path

ROOT_DIR = Path(__file__).resolve(strict=True).parent

SRC_DIR: tp.Final[Path] = ROOT_DIR / 'src'
DATA_DIR: tp.Final[Path] = ROOT_DIR / 'data'
INPUT_DIR: tp.Final[Path] = DATA_DIR / 'input'
OUTPUT_DIR: tp.Final[Path] = DATA_DIR / 'output'
LOG_DIR: tp.Final[Path] = OUTPUT_DIR / 'logs'

DATASET_LINK: tp.Final[str] = 'lekomtsev/pleural_effusion_segmentation/subset_split.zip:latest'

SEED: tp.Final[int] = int(os.environ.get('SEED', 411))
DEFAULT_NUM_WORKERS: tp.Final[int] = int(os.environ.get('DEFAULT_NUM_WORKERS', 1))
DEFAULT_VALID_FRACTION: tp.Final[float] = float(os.environ.get('DEFAULT_VALID_FRACTION', 0.2))

PathType = tp.Union[Path, str]

@dataclass(frozen=True)
class DatasetPathConfig:
    """Paths to train/valid dataset."""
    train_images: Path = INPUT_DIR / 'subset_split' / 'subset_train' / 'subset_img' / 'subset_img' / 'subset'
    train_masks: Path = INPUT_DIR / 'subset_split' / 'subset_train' / 'subset_masks' / 'subset_masks'
    valid_images: Path = INPUT_DIR / 'subset_split' / 'subset_test' / 'subset_img' / 'subset_img' / 'subset'
    valid_masks: Path = INPUT_DIR / 'subset_split' / 'subset_test' / 'subset_masks' / 'subset_masks'
