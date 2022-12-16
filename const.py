"""
Main constants: paths, params and etc
"""
import os
import typing as tp

from pathlib import Path

ROOT_DIR = Path(__file__).resolve(strict=True).parent

SRC_DIR: tp.Final[Path] = ROOT_DIR / 'src'
DATA_DIR: tp.Final[Path] = ROOT_DIR / 'data'
INPUT_DIR: tp.Final[Path] = DATA_DIR / 'input'
IMAGES_DIR: tp.Final[Path] = INPUT_DIR / 'subset' / 'subset' / 'subset_img' / 'subset'
MASKS_DIR: tp.Final[Path] = INPUT_DIR / 'subset' / 'subset' / 'subset_masks' / 'subset_masks'
OUTPUT_DIR: tp.Final[Path] = DATA_DIR / 'output'
LOG_DIR: tp.Final[Path] = OUTPUT_DIR / 'logs'

DATASET_LINK: tp.Final[str] = 'lekomtsev/pleural_effusion_segmentation/subset.zip:latest'

SEED: tp.Final[int] = int(os.environ.get('SEED', 411))
DEFAULT_NUM_WORKERS: tp.Final[int] = int(os.environ.get('DEFAULT_NUM_WORKERS', 1))
DEFAULT_VALID_FRACTION: tp.Final[float] = float(os.environ.get('DEFAULT_VALID_FRACTION', 0.2))
DEFAULT_CROP_SHAPE_IN_3D: tp.Final[tp.Tuple[int, int, int]] = (32, 256, 256)

PathType = tp.Union[Path, str]
