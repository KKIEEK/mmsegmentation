# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from torch import Tensor

from mmseg.models import EncoderDecoder
from mmseg.registry import MODELS


@MODELS.register_module()
class ChangeDetector(EncoderDecoder):
    """Segmentor for Change Detection.

    ChangeDetector almost the same as EncoderDecoder except for extracting of
    image pair.

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        xs = super().extract_feat(
            # Convert (b, c, h, w) to (2*b, c/2, h, w)
            torch.cat(torch.chunk(inputs, 2, dim=1), dim=0))

        outs = list()
        for x in xs:
            # Convert (2*b, c', h', w') to (b, c'*2, h', w')
            outs.append(torch.cat(torch.chunk(x, 2, dim=0), dim=1))
        return outs
