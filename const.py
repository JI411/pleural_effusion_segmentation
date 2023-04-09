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

DATASET_LINKS: tp.Final[tp.Tuple[str, ...]] = (
    'lekomtsev/pleural_effusion_segmentation/train-relabeled.tar:latest',
    'lekomtsev/pleural_effusion_segmentation/valid-relabeled.tar:latest'
)

SEED: tp.Final[int] = int(os.environ.get('SEED', 411))
DEFAULT_NUM_WORKERS: tp.Final[int] = int(os.environ.get('DEFAULT_NUM_WORKERS', 1))
PATCH_SIZE: tp.Final[tp.Tuple[int, int, int]] = (128, 128, 64)

PathType = tp.Union[Path, str]

@dataclass(frozen=True)
class DatasetPathConfig:
    """Paths to train/valid dataset."""
    train_dir: Path = INPUT_DIR / 'train-relabeled' / 'train'
    valid_dir: Path = INPUT_DIR / 'valid-relabeled' / 'valid'
    images_dir_name: str = 'volume'
    masks_dir_name: str = 'mask'
    mask_file_name: str = 'semantic_segmentation.nrrd'
