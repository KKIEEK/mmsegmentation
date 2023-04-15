# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np
from mmcv.transforms import LoadImageFromFile

from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImagePairFromFile(LoadImageFromFile):
    """Load an image pair.

    Required Keys:

    - img_paths

    Modified Keys:

    - img
    - img_path
    - img_paths
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def _load_img(self, results: dict, img_path: str) -> dict:
        """Private function to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseCDDataset``.

        Returns:
            dict: The dict contains loaded image.
        """
        results = copy.copy(results)
        results['img_path'] = img_path
        return super().transform(results)

    def transform(self, results: dict) -> dict:
        """Transform function to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        imgs = []
        for img_path in results['img_paths'][1:]:
            imgs.append(self._load_img(results, img_path)['img'])

        results = self._load_img(results, results['img_paths'][0])
        results['img_path'] = None
        results['img'] = np.concatenate((results['img'], *imgs), axis=2)
        return results
