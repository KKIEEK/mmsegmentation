# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmseg.models.backbones.unet import BasicConvBlock
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import UpConvBlock
from mmseg.registry import MODELS


@MODELS.register_module()
class UNetHead(BaseDecodeHead):
    """UNet Head for utilizing the UNet architecture with various backbones.

    This head is also the implementation of `U-Net: Convolutional Networks
    for Biomedical Image Segmentation <https://arxiv.org/abs/1505.04597>`_.

    Args:
        num_convs (Sequence[int]): Number of convolutional layers in the
            convolution block of the correspondence decoder stage.
            Default: (2, 2, 2, 2).
        upsamples (Sequence[int]): Whether upsample the feature map after
            each decoder layer. Default: (True, True, True, True).
        dilations (Sequence[int]): Dilation rate of each stage in decoder.
            Default: (1, 1, 1, 1).
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv').
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 num_convs=(2, 2, 2, 2),
                 upsamples=(True, True, True, True),
                 dilations=(1, 1, 1, 1),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='InterpConv'),
                 dcn=None,
                 plugins=None,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        if self.init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='Conv2d'),
                dict(
                    type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
            ]

        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        num_stages = len(self.in_channels) - 1
        assert len(num_convs) == (num_stages-1), \
            'The length of num_convs should be equal to (num_stages-1), '\
            f'while the num_convs is {num_convs}, the length of '\
            f'num_convs is {len(num_convs)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(dilations) == (num_stages-1), \
            'The length of dilations should be equal to (num_stages-1), '\
            f'while the dilations is {dilations}, the length of '\
            f'dilations is {len(dilations)}, and the num_stages is '\
            f'{num_stages}.'

        self.decoder = nn.ModuleList()
        for i in range(num_stages):
            in_channels = self.channels * 2**i
            if i == num_stages - 1:
                in_channels = self.in_channels[-1]
            upsample = upsamples[i]
            self.decoder.append(
                UpConvBlock(
                    conv_block=BasicConvBlock,
                    in_channels=in_channels,
                    skip_channels=self.in_channels[i + 1],
                    out_channels=self.channels * 2**(i - 1),
                    num_convs=num_convs[i - 1],
                    stride=1,
                    dilation=dilations[i - 1],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    upsample_cfg=upsample_cfg if upsample else None,
                    dcn=dcn,
                    plugins=plugins))

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        output = inputs[-1]
        for i in reversed(range(len(self.decoder))):
            output = self.decoder[i](inputs[i], output)
        output = self.cls_seg(output)
        return output


@MODELS.register_module()
class SiameseUNetHead(UNetHead):
    pass
