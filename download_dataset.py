"""
Download dataset
"""

from pathlib import Path
from zipfile import ZipFile

import wandb

import const


def unzip(archives_dir: Path) -> None:
    """
    Unzip archives fromm dir recursively
    :param archives_dir: dir with one or more archives
    :return:
    """
    extracted = set()
    while (archives := set(archives_dir.rglob('*.zip'))) != extracted:
        for zip_file in (archives - extracted):
            with ZipFile(zip_file, 'r') as zip_archive:
                extracted.add(zip_file)
                zip_archive.extractall(zip_file.with_suffix(''))


def load_data() -> None:
    """
    Load dataset from W&B
    :return:
    """
    api = wandb.Api()
    artifact = api.artifact('lekomtsev/pleural_effusion_segmentation/subset.zip:latest')
    artifact.download(root=const.INPUT_DIR)
    unzip(const.INPUT_DIR)


if __name__ == '__main__':
    load_data()
