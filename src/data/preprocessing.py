"""
Preprocessing for images.
"""
import volumentations as vol

from const import PATCH_SIZE


def train_augmentation() -> vol.Compose:
    """Train augmentations."""
    return vol.Compose([
        vol.Normalize(range_norm=False, p=1.),
        vol.Rotate((-5, 5), (-3, 3), (-3, 3), border_mode='nearest', p=0.05),
        vol.RandomCropFromBorders(crop_value=0.05, p=0.08),
        vol.Resize(
            PATCH_SIZE,
            interpolation=1,
            resize_type=0,
            p=1.,
        ),
        vol.Flip(0, p=0.01),
        vol.RandomScale(scale_limit=[0.995, 1.005], p=0.05),
        vol.GaussianNoise(var_limit=(0., 0.14), p=0.2),
        vol.Normalize(range_norm=False, p=1.),
    ], p=1.)

def valid_augmentation() -> vol.Compose:
    """Validation augmentations."""
    return vol.Compose([
        vol.Normalize(range_norm=False, p=1.),
        vol.Resize(
            PATCH_SIZE,
            interpolation=1,
            resize_type=0,
            always_apply=True,
            p=1.
        ),
        vol.Normalize(range_norm=False, p=1.),
    ], p=1.)
