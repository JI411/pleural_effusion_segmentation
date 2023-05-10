"""
This script converts a PyTorch model to ONNX.
"""
import typing as tp
from argparse import Namespace, ArgumentParser

import torch

import const
from src.model.base import BaseModelWrapper
from src.model.wrappers import ModelsZoo


def parse_args() -> Namespace:
    """Get arguments for training script. Add params to set of params from lightning Trainer."""
    parser = ArgumentParser()
    parser.add_argument(
        'checkpoint', type=str, action='store',
        help='Path to model checkpoint.'
    )
    parser.add_argument(
        '--model', '-m', type=str, action='store', default='unet3d',
        help='Name of model from src/model/wrappers/ModelZoo. Default: unet3d.'
    )
    parser.add_argument(
        '--shape', '-s', type=int, action='store', nargs='+',
        default=(1, const.PATCH_SIZE[2], const.PATCH_SIZE[0], const.PATCH_SIZE[1]),
        help='Shape of model input in (batch, depth, height, width). Without num channels. '
             'Default: (1, const.PATCH_SIZE[2], const.PATCH_SIZE[0], const.PATCH_SIZE[1]).'
    )
    parser.add_argument(
        '--output_onnx_path', '--output', '-o', type=str, action='store', default=const.OUTPUT_DIR / 'model.onnx',
        help='Path to save ONNX model. Default: const.OUTPUT_DIR / model.onnx.'
    )
    parser.add_argument('--device', type=str, action='store', default='cpu', help='Device to use. Default: cpu.')
    parser.add_argument('--opset', type=int, action='store', default=11, help='ONNX opset version. Default: 11.')

    parsed = parser.parse_args()
    if parsed.shape is not None and len(parsed.shape) != 4:
        raise ValueError('Shape must be in (batch, depth, height, width) format.')
    return parsed

def torch2onnx(
        model: BaseModelWrapper,
        input_shape: tp.Tuple[int, int, int, int],
        device: torch.device,
        output_onnx_path: const.PathType,
        opset_version: int,
) -> None:
    """
    Convert PyTorch model to ONNX.
    :param model: model to convert
    :param input_shape: model input image shape in (batch, depth, height, width) format
    :param device: device to use
    :param output_onnx_path: path to save ONNX model
    :param opset_version: ONNX opset version
    :return:
    """
    input_shape = (input_shape[0], 1, *input_shape[1:])
    batch = torch.randn(*input_shape, requires_grad=True, device=device)
    model = model.to(device)
    out = model(batch)
    torch.onnx.export(
        model,
        out,
        output_onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    )


if __name__ == '__main__':
    args = parse_args()
    wrapper: BaseModelWrapper = ModelsZoo[args.model].value()
    state_dict = torch.load(args.checkpoint)['state_dict']
    state_dict = {key.replace('model.model.', 'model.'): value for key, value in state_dict.items()}
    wrapper.load_state_dict(state_dict)
    torch2onnx(
        model=wrapper,
        input_shape=args.shape,
        device=torch.device(args.device),
        output_onnx_path=args.output_onnx_path,
        opset_version=args.opset,
    )
