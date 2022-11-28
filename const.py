"""
Main constants: paths, params and etc
"""

import typing as tp

from pathlib import Path

ROOT_DIR = Path(__file__).resolve(strict=True).parent

SRC_DIR: tp.Final[Path] = ROOT_DIR / 'src'
INPUT_DIR: tp.Final[Path] = ROOT_DIR / 'input'
IMAGES_DIR: tp.Final[Path] = INPUT_DIR / 'subset' / 'subset' / 'subset_img' / 'subset'
MASKS_DIR: tp.Final[Path] = INPUT_DIR / 'subset' / 'subset' / 'subset_masks' / 'subset_masks'
OUTPUT_DIR: tp.Final[Path] = ROOT_DIR / 'output'

SEED: tp.Final[int] = 411
DEFAULT_BATCH_SIZE: tp.Final[int] = 2
DEFAULT_NUM_WORKERS: tp.Final[int] = 4
DEFAULT_VALID_FRACTION: tp.Final[float] = 0.2
