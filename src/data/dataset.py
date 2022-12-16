"""
Dataset and Dataloader classes
"""
import numpy as np

import const
from src.data import read_data, preprocessing
from src.data.base import BaseDataset, Batch


class PleuralEffusionDataset2D(BaseDataset):
    """ Pleural Effusion Dataset """

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

    def get_images_cache(self):
        """ Read images from disk, preprocess they and cache in memory """
        images = [list(read_data.load_dicom_recursive(p))[0] for p in self.image_dir_paths]
        images = [preprocessing.rotate_array(img) for img in images]
        return [preprocessing.normalize(img) for img in images]

    def get_masks_cache(self):
        """ Read masks from disk and cache in memory """
        return [read_data.load_mask_from_dir(p) for p in self.masks_dir_paths]

    def __getitem__(self, idx: int) -> Batch:
        """ Get (1 x W x H ) lung image and mask for it """
        image = self.images[idx].astype('float32')
        mask = self.masks[idx].astype(int)
        # TODO: add channel_idx random sampler, use fraction of 1  # pylint: disable=fixme
        channel_idx = np.random.randint(image.shape[0])
        batch = Batch(image=image[channel_idx][None], mask=mask[channel_idx][None])
        if self.transforms is not None:
            batch = self.transforms(batch)
        return batch

class PleuralEffusionDataset3D(BaseDataset):
    """ Pleural Effusion Dataset """

    def __getitem__(self, idx: int) -> Batch:
        """ Get (1 x C x W x H ) lung image and mask for it """
        image = list(read_data.load_dicom_recursive(self.image_dir_paths[idx]))[0]
        image = preprocessing.rotate_array(image)
        mask = read_data.load_mask_from_dir(self.masks_dir_paths[idx])
        image = preprocessing.normalize(image)
        batch = Batch(image=image.astype('float32'), mask=mask.astype(int))
        if self.transforms is not None:
            batch = self.transforms(batch)
        return Batch(image=batch['image'].astype('float32')[None], mask=batch['mask'].astype(int)[None])
