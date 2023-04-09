"""
Run main training script.
"""

from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import const
from src.model.base import BaseModelWrapper
from src.model.wrappers import ModelsZoo
from src.train import PleuralSegmentationModule


def main(args) -> None:
    """Run training pipeline."""
    seed_everything(const.SEED, workers=True)

    net: BaseModelWrapper = ModelsZoo[args.model].value

    batch_size = args.batch
    if not batch_size:
        model = PleuralSegmentationModule(model=net(), batch_size=1)
        trainer = Trainer(auto_scale_batch_size='power')
        batch_size = trainer.tune(model)['scale_batch_size']

    model = PleuralSegmentationModule(model=net(), batch_size=batch_size)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    wandb_logger = WandbLogger(project="pleural_effusion_segmentation", save_dir=const.LOG_DIR, name=args.name)
    wandb_logger.watch(model)
    trainer = Trainer.from_argparse_args(
        args, logger=wandb_logger, default_root_dir=const.LOG_DIR, callbacks=[lr_monitor]
    )
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--batch', type=int, action='store', default=None, help='Batch size. Default: find batch size with Trainer.'
    )
    parser.add_argument(
        '--name', type=str, action='store', default=None, help='Name to wandb run. Default: random name.'
    )
    parser.add_argument(
        '--model', type=str, action='store', default='unet3d',
        help='Name of model from src/model/wrappers/ModelZoo. Default: unet3d.'
    )
    parser = Trainer.add_argparse_args(parser)
    arguments = parser.parse_args()

    main(arguments)
