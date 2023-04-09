"""
Copied and modified from https://github.com/black0017/MedicalZooPytorch/blob/master/lib/medzoo/Unet3D.py (MIT License).
Original repository have many dependencies, that are not needed for this project.
"""
# pylint: disable=too-many-statements, too-many-locals, too-many-instance-attributes
import torch
from torch import nn

from src.model.model_zoo.base import BaseModel


def norm_lrelu_upscale_conv_norm_lrelu(feat_in, feat_out):
    """Instance Normalization -> Leaky ReLU -> Upsample -> Convolution -> Instance Normalization -> Leaky ReLU."""
    return nn.Sequential(
        nn.InstanceNorm3d(feat_in),
        nn.LeakyReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        # should be feat_in * 2 or feat_in
        nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
        nn.InstanceNorm3d(feat_out),
        nn.LeakyReLU()
    )


def lrelu_conv(feat_in, feat_out):
    """Leaky ReLU -> Convolution."""
    return nn.Sequential(
        nn.LeakyReLU(),
        nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False)
    )


def norm_lrelu_conv(feat_in, feat_out):
    """Instance Normalization -> Leaky ReLU -> Convolution."""
    return nn.Sequential(
        nn.InstanceNorm3d(feat_in),
        nn.LeakyReLU(),
        nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False)
    )


def conv_norm_lrelu(feat_in, feat_out):
    """Convolution -> Instance Normalization -> Leaky ReLU."""
    return nn.Sequential(
        nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
        nn.InstanceNorm3d(feat_out),
        nn.LeakyReLU()
    )


class UNet3D(BaseModel):
    """
    Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
    """

    def __init__(self, in_channels: int, n_classes: int, base_n_filter: int):
        """Initialize the model."""
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

        self.conv3d_c1_1 = nn.Conv3d(
            self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.conv3d_c1_2 = nn.Conv3d(
            self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.lrelu_conv_c1 = lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)

        self.conv3d_c2 = nn.Conv3d(
            self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=2, padding=1, bias=False,
        )
        self.norm_lrelu_conv_c2 = norm_lrelu_conv(self.base_n_filter * 2, self.base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2)

        self.conv3d_c3 = nn.Conv3d(
            self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=2, padding=1, bias=False,
        )
        self.norm_lrelu_conv_c3 = norm_lrelu_conv(self.base_n_filter * 4, self.base_n_filter * 4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter * 4)

        self.conv3d_c4 = nn.Conv3d(
            self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=3, stride=2, padding=1, bias=False,
        )
        self.norm_lrelu_conv_c4 = norm_lrelu_conv(self.base_n_filter * 8, self.base_n_filter * 8)
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter * 8)

        self.conv3d_c5 = nn.Conv3d(
            self.base_n_filter * 8, self.base_n_filter * 16, kernel_size=3, stride=2, padding=1, bias=False,
        )
        self.norm_lrelu_conv_c5 = norm_lrelu_conv(self.base_n_filter * 16, self.base_n_filter * 16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = norm_lrelu_upscale_conv_norm_lrelu(
            self.base_n_filter * 16, self.base_n_filter * 8,
        )

        self.conv3d_l0 = nn.Conv3d(
            self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0, bias=False,
        )
        self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 8)

        self.conv_norm_lrelu_l1 = conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 16)
        self.conv3d_l1 = nn.Conv3d(
            self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0, bias=False,
        )
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = norm_lrelu_upscale_conv_norm_lrelu(
            self.base_n_filter * 8, self.base_n_filter * 4,
        )

        self.conv_norm_lrelu_l2 = conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 8)
        self.conv3d_l2 = nn.Conv3d(
            self.base_n_filter * 8, self.base_n_filter * 4, kernel_size=1, stride=1, padding=0, bias=False,
        )
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = norm_lrelu_upscale_conv_norm_lrelu(
            self.base_n_filter * 4, self.base_n_filter * 2,
        )

        self.conv_norm_lrelu_l3 = conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3d(
            self.base_n_filter * 4, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0, bias=False,
        )
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = norm_lrelu_upscale_conv_norm_lrelu(
            self.base_n_filter * 2, self.base_n_filter,
        )

        self.conv_norm_lrelu_l4 = conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter * 2)
        self.conv3d_l4 = nn.Conv3d(
            self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False,
        )

        self.ds2_1x1_conv3d = nn.Conv3d(
            self.base_n_filter * 8, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False,
        )
        self.ds3_1x1_conv3d = nn.Conv3d(
            self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""
        #  Level 1 context pathway
        out = self.conv3d_c1_1(image)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)

        # Level 2 context pathway
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out

        # Level 3 context pathway
        out = self.conv3d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out

        # Level 4 context pathway
        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)
        context_4 = out

        # Level 5
        out = self.conv3d_c5(out)
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        out += residual_5
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)

        out = self.conv3d_l0(out)
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)

        # Level 1 localization pathway
        out = torch.cat([out, context_4], dim=1)
        out = self.conv_norm_lrelu_l1(out)
        out = self.conv3d_l1(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)

        # Level 2 localization pathway
        # print(out.shape)
        # print(context_3.shape)
        out = torch.cat([out, context_3], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)

        # Level 3 localization pathway
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_lrelu_l3(out)
        ds3 = out
        out = self.conv3d_l3(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)

        # Level 4 localization pathway
        out = torch.cat([out, context_1], dim=1)
        out = self.conv_norm_lrelu_l4(out)
        out_pred = self.conv3d_l4(out)

        ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        ds1_ds2_sum_upscale = self.upsample(ds2_1x1_conv)
        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsample(ds1_ds2_sum_upscale_ds3_sum)
        return out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
