"""
Preprocessing for images.
"""
import volumentations as vol

from const import PATCH_SIZE


def train_augmentation() -> vol.Compose:
    """Train augmentations."""

    return vol.Compose([
        # vol.RandomCropFromBorders(crop_value=0.1, p=0.1),
        # vol.ElasticTransform((0, 0.2), interpolation=2, p=0.1),
        # vol.Rotate((-10, 10), (-5, 5), (-5, 5), p=0.1),
        vol.Resize(
            PATCH_SIZE,
            interpolation=1,
            resize_type=1,
            always_apply=True,
            p=1.
        ),
        vol.GaussianNoise(var_limit=(0, 10), p=0.3),
        # vol.Flip(0, p=0.2),
        vol.Flip(1, p=0.2),
        # vol.OneOrOther([]),
        # vol.Flip(2, p=0.1),
        vol.Normalize(range_norm=False, p=1.),
    ], p=1.)

def valid_augmentation() -> vol.Compose:
    """Validation augmentations."""
    return vol.Compose([
        vol.Resize(
            PATCH_SIZE,
            interpolation=1,
            resize_type=1,
            always_apply=True,
            p=1.
        ),
        vol.Normalize(range_norm=False, p=1.),
    ], p=1.)
