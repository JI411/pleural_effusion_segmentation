"""
Read data from disk
"""

import typing as tp
from pathlib import Path

import SimpleITK as sitk
import nibabel as nib
import numpy as np


def load_dicom(directory: tp.Union[Path, str]) -> np.ndarray:
    """
    Read image from dir with files dicom format, have same dimensions with mask from load_mask
    :param directory: path to directory contains .dcm files
    :return: 3d image
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(directory))
    reader.SetFileNames(dicom_names)
    image_itk = reader.Execute()

    image_zyx = sitk.GetArrayFromImage(image_itk)
    return image_zyx.astype(np.int16)

def load_dicom_recursive(directory: tp.Union[Path, str]) -> tp.Generator[np.ndarray, None, None]:
    """
    Load all dicom images from dir, can be used to read all dataset
    :param directory: path to directory contains directories with .dcm files
    :return: 3d images
    """
    for dicom_dir in Path(directory).glob('*'):
        if not dicom_dir.is_dir():
            continue
        if list(dicom_dir.glob('*.dcm')):
            yield load_dicom(directory=dicom_dir)
        yield from load_dicom_recursive(dicom_dir)


def load_mask(nii: tp.Union[Path, str]) -> np.ndarray:
    """
    Read mask in .nii format, have same dimensions with image from load_dicom
    :param nii: path to .nii.gz file
    :return: 3d mask
    """
    mask = nib.load(nii)
    mask = mask.get_fdata()
    return mask.transpose(2, 0, 1)

def load_mask_from_dir(nii: tp.Union[Path, str]) -> np.ndarray:
    """
    Read mask in .nii format, have same dimensions with image from load_dicom
    :param nii: path to .nii.gz file or dir with the only one .nii file
    :return: 3d mask
    """
    if nii.is_dir():
        nii_files = list(nii.rglob('*.nii.gz'))
        if len(nii_files) != 1:
            raise FileNotFoundError(f'Wrong path to mask file: {nii}')
        nii = nii_files[0]
    return load_mask(nii=nii)
