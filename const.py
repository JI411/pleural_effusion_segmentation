"""
Main constants: paths, params and etc
"""
import os
import typing as tp

from pathlib import Path

ROOT_DIR = Path(__file__).resolve(strict=True).parent

SRC_DIR: tp.Final[Path] = ROOT_DIR / 'src'
INPUT_DIR: tp.Final[Path] = ROOT_DIR / 'input'
IMAGES_DIR: tp.Final[Path] = INPUT_DIR / 'subset' / 'subset' / 'subset_img' / 'subset'
MASKS_DIR: tp.Final[Path] = INPUT_DIR / 'subset' / 'subset' / 'subset_masks' / 'subset_masks'
OUTPUT_DIR: tp.Final[Path] = ROOT_DIR / 'output'

DATASET_LINK: tp.Final[str] = 'lekomtsev/pleural_effusion_segmentation/subset.zip:latest'

SEED: tp.Final[int] = int(os.environ.get('SEED', 411))
DEFAULT_NUM_WORKERS: tp.Final[int] = int(os.environ.get('DEFAULT_NUM_WORKERS', 4))
DEFAULT_VALID_FRACTION: tp.Final[float] = float(os.environ.get('DEFAULT_VALID_FRACTION', 0.2))

PathType = tp.Union[Path, str]
