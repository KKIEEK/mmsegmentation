# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basecddataset import BaseCDDataset


@DATASETS.register_module()
class LevirCDDataset(BaseCDDataset):
    """LevirCD dataset.

    In segmentation map annotation for LevirCD, 0 stands for background, which
    is included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` and ``seg_map_suffix`` is fixed to '.png'.
    """
    METAINFO = dict(
        classes=('background', 'changed'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,
                 pair_keys=('A', 'B'),
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            pair_keys=pair_keys,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
