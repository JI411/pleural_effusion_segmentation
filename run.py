"""
Run main training script
"""

from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

import const
from src.model import wrappers
from src.train import PleuralSegmentationModule


def main(args):
    from pytorch_lightning.callbacks import DeviceStatsMonitor
    """Run training pipline."""
    seed_everything(const.SEED, workers=True)

    batch_size = args.batch
    if not batch_size:
        model = PleuralSegmentationModule(model=wrappers.UnetSMPWrapper(), batch_size=1)
        trainer = Trainer(auto_scale_batch_size='power')
        batch_size = trainer.tune(model)['scale_batch_size']

    model = PleuralSegmentationModule(model=wrappers.UnetSMPWrapper(), batch_size=batch_size)
    wandb_logger = WandbLogger(project="pleural_effusion_segmentation", save_dir=const.LOG_DIR)
    wandb_logger.watch(model)
    trainer = Trainer.from_argparse_args(args, logger=wandb_logger, profiler="simple", callbacks=[DeviceStatsMonitor()])  # todo
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--batch', type=int, action='store', default=None, help='Batch size. Default: find batch size with Trainer'
    )
    parser = Trainer.add_argparse_args(parser)
    arguments = parser.parse_args()

    main(arguments)
