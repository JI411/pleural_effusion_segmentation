"""
Main constants: paths, params and etc
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve(strict=True).parent

SRC_DIR = ROOT_DIR / 'src'
INPUT_DIR = ROOT_DIR / 'input'
IMAGES_DIR = INPUT_DIR / 'subset' / 'subset' / 'subset_img' / 'subset'
MASKS_DIR = INPUT_DIR / 'subset' / 'subset' / 'subset_masks' / 'subset_masks'
OUTPUT_DIR = ROOT_DIR / 'output'

SEED = 411
