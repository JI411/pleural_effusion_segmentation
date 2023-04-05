"""
Download dataset.
"""

from pathlib import Path
import tarfile
import typing as tp

import wandb

import const


def uncompress_tar(archives_dir: const.PathType) -> None:
    """
    Uncompress tar archives fromm dir recursively.

    :param archives_dir: dir with one or more archives
    :return:
    """
    archives_dir = Path(archives_dir)
    for path in archives_dir.glob('*.tar*'):
        if path.suffix == ".tar.gz":
            mode = "r:gz"
        elif path.suffix == ".tar":
            mode = "r:"
        else:
            continue
        target_dir = path.with_suffix('').with_suffix('')
        target_dir.mkdir(exist_ok=True)
        with tarfile.open(path, mode) as tar:
            tar.extractall(path=target_dir)


def load_data(wandb_links: tp.Iterable[str]) -> None:
    """Load dataset from W&B."""
    api = wandb.Api()
    for link in wandb_links:
        artifact = api.artifact(link)
        artifact.download(root=const.INPUT_DIR)
    uncompress_tar(const.INPUT_DIR)


if __name__ == '__main__':
    load_data(const.DATASET_LINKS)
