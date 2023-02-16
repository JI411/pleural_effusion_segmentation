"""
Dataset and Dataloader classes
"""
import typing as tp

import numpy as np

import const
from src.data import read_data, preprocessing
from src.data.base import BaseDataset, Batch

T = tp.TypeVar('T')

def split_channels(array: np.ndarray) -> np.ndarray:
    """ Split array to channels (axis=0) """
    return np.split(array, array.shape[0], axis=0)

def flatten(list_of_lists: tp.List[tp.List[T]]) -> tp.List[T]:
    """Convert list of lists to list"""
    return [item for sublist in list_of_lists for item in sublist]

class PleuralEffusionDataset2D(BaseDataset):
    """ Pleural Effusion Dataset """

    split_channels = True
    def __init__(
            self,
            images_dir: const.PathType = const.IMAGES_DIR,
            masks_dir: const.PathType = const.MASKS_DIR,
    ) -> None:
        """
        Create dataset class.

        Images read and caching in memory, because reading from disk is too slow for 2D variant.
        :param images_dir: dir with dirs with .dcm images; default const.IMAGES_DIR
        :param masks_dir: dir with dirs with .nii.gz masks; default const.MASKS_DIR
        """
        super().__init__(images_dir, masks_dir)
        self.images = self.get_images_cache()
        self.masks = self.get_masks_cache()
        self.save_channels()

    def save_channels(self) -> None:
        """ Save only channels with more than 1 target pixel in mask"""
        if not self.split_channels:
            return
        save_channels = [np.sum(mask) > 1 for mask in self.masks]
        self.images = [img for img, save in zip(self.images, save_channels) if save]
        self.masks = [mask for mask, save in zip(self.masks, save_channels) if save]

    def get_images_cache(self) -> tp.List[np.ndarray]:
        """ Read images from disk, preprocess they and cache in memory """
        images = [list(read_data.load_dicom_recursive(p))[0] for p in self.image_dir_paths]
        images = [preprocessing.rotate_array(img) for img in images]
        images = [preprocessing.normalize(img) for img in images]
        if self.split_channels:
            images = [split_channels(img) for img in images]
            images = flatten(images)
        return images

    def get_masks_cache(self) -> tp.List[np.ndarray]:
        """ Read masks from disk and cache in memory """
        masks = [read_data.load_mask_from_dir(p) for p in self.masks_dir_paths]
        if self.split_channels:
            masks = [split_channels(mask) for mask in masks]
            masks = flatten(masks)
        return masks

    def __len__(self) -> int:
        """ Len of dataset """
        return len(self.images)

    def __getitem__(self, idx: int) -> Batch:
        """ Get (1 x W x H ) lung image and mask for it """
        image = self.images[idx].astype('float32')
        mask = self.masks[idx].astype(int)
        if not self.split_channels:
            channel_idx = np.random.randint(image.shape[0])
            image, mask = image[channel_idx][None], mask[channel_idx][None]
        return Batch(image=image, mask=mask)
