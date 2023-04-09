"""
Preprocessing for images.
"""
import volumentations as vol

from const import PATCH_SIZE


def train_augmentation() -> vol.Compose:
    """Train augmentations."""
    return vol.Compose([
        vol.Rotate((-5, 5), (-2, 2), (-2, 2), p=0.3),
        vol.RandomCropFromBorders(crop_value=0.05, p=0.05),
        vol.Resize(
            PATCH_SIZE,
            interpolation=1,
            resize_type=0,
            always_apply=True,
            p=1.
        ),
        vol.Flip(0, p=0.01),
        vol.GaussianNoise(var_limit=(10, 40), p=0.4),
        vol.Normalize(range_norm=False, p=1.),
    ], p=1.)

def valid_augmentation() -> vol.Compose:
    """Validation augmentations."""
    return vol.Compose([
        vol.Resize(
            PATCH_SIZE,
            interpolation=1,
            resize_type=0,
            always_apply=True,
            p=1.
        ),
        vol.Normalize(range_norm=False, p=1.),
    ], p=1.)
