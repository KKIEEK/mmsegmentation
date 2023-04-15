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
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv').
    """

    def __init__(self,
                 num_convs=(2, 2, 2, 2),
                 upsamples=(True, True, True, True),
                 dilations=(1, 1, 1, 1),
                 upsample_cfg=dict(type='InterpConv'),
                 init_cfg=[
                     dict(type='Kaiming', layer='Conv2d'),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm']),
                     dict(
                         type='Normal',
                         std=0.01,
                         override=dict(name='conv_seg'))
                 ],
                 **kwargs):
        super().__init__(
            input_transform='multiple_select', init_cfg=init_cfg, **kwargs)

        num_stages = len(self.in_channels) - 1
        assert len(num_convs) == num_stages, \
            'The length of num_convs should be equal to num_stages, '\
            f'while the num_convs is {num_convs}, the length of '\
            f'num_convs is {len(num_convs)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(dilations) == num_stages, \
            'The length of dilations should be equal to num_stages, '\
            f'while the dilations is {dilations}, the length of '\
            f'dilations is {len(dilations)}, and the num_stages is '\
            f'{num_stages}.'

        self.decoder = nn.ModuleList()
        for i in range(num_stages):
            in_channels = self.channels * 2**(i + 2)
            if i == num_stages - 1:
                in_channels = self.in_channels[-1]
            upsample = upsamples[i]
            self.decoder.append(
                UpConvBlock(
                    conv_block=BasicConvBlock,
                    in_channels=in_channels,
                    skip_channels=self.in_channels[i],
                    out_channels=self.channels * 2**(i + 1),
                    num_convs=num_convs[i],
                    stride=1,
                    dilation=dilations[i],
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    upsample_cfg=upsample_cfg if upsample else None))

        self.upsample = UpConvBlock(
            conv_block=BasicConvBlock,
            in_channels=self.channels * 2,
            skip_channels=0,
            out_channels=self.channels,
            num_convs=num_convs[0],
            stride=1,
            dilation=dilations[0],
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            upsample_cfg=upsample_cfg)

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        output = inputs[-1]
        for i in reversed(range(len(self.decoder))):
            output = self.decoder[i](inputs[i], output)
        output = self.upsample(output.new_empty(0), output)
        output = self.cls_seg(output)
        return output
