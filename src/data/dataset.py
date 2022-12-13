"""
Dataset and Dataloader classes
"""

from src.data import read_data, preprocessing
from src.data.base import PleuralEffusionDataset, Batch

class PleuralEffusionDataset3D(PleuralEffusionDataset):
    """ Pleural Effusion Dataset """

    def __getitem__(self, idx: int) -> Batch:
        """ Get lung image and mask for it """
        image = list(read_data.load_dicom_recursive(self.image_dir_paths[idx]))[0]
        image = preprocessing.rotate_array(image)
        mask = read_data.load_mask_from_dir(self.masks_dir_paths[idx])
        image = preprocessing.normalize(image)
        return Batch(image=image.astype('float32')[:3, ...], mask=mask.astype(int)[:1, ...])  # todo remove
