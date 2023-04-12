import torch


def test__augmentations():
    """Test augmentations."""
    from src.data import preprocessing

    train_aug, valid_aug = preprocessing.train_augmentation(), preprocessing.valid_augmentation()
    for _ in range(10):
        x = torch.rand(1, 1, 32, 64, 64)
        assert (
            train_aug(x).shape == valid_aug(x).shape,
            'Train and valid augmentations should have same shape, '
            f'but train_aug(x).shape == {train_aug(x).shape} and valid_aug(x).shape == {valid_aug(x).shape}'
        )
