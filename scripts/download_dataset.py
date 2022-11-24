from pathlib import Path

import gdown

from const import DATA_DIR

try:
    from zipfile import ZipFile
except ImportError:
    from zipfile36 import ZipFile


def unzip(archives_dir: Path) -> None:
    for zip_file in archives_dir.glob('*.zip'):
        with ZipFile(zip_file, 'r') as zip_archive:
            if not zip_file.with_suffix('').is_dir():
                zip_archive.extractall(zip_file.with_suffix(''))


def load_data() -> None:
    gdown.cached_download(
        id='1tYTSAYKMF_gHV542tSwKighLIoNImE1g',
        path=str(DATA_DIR / 'subset_masks.zip'),
        md5='140515cb3f4942e3baf77645115e88b8'
    )
    gdown.cached_download(
        id='1tYTSAYKMF_gHV542tSwKighLIoNImE1g',
        path=str(DATA_DIR / 'subset_img.zip'),
        md5='cd247ca63549f7b281fb01a718f89fc5',
    )
    unzip(archives_dir=DATA_DIR)


if __name__ == '__main__':
    load_data()
