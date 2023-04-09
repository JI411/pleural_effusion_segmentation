"""
Preprocessing for images.
"""
import volumentations as vol

from const import PATCH_SIZE


def train_augmentation() -> vol.Compose:
    """Train augmentations."""
    return vol.Compose([
        vol.Resize(
            PATCH_SIZE,
            interpolation=1,
            resize_type=1,
            always_apply=True,
            p=1.
        ),
        # TODO: noise, rotate, flip, resize+crop, ...  # pylint: disable=fixme
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
