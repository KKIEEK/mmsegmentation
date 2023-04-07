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
        pair = []
        for x in torch.chunk(inputs, 2, dim=1):
            x = super().extract_feat(x)
            pair.append(x)
        return [torch.cat((x1, x2), dim=1) for x1, x2 in zip(*pair)]
