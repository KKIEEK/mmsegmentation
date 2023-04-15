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
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def _load_img(self, results: dict, img_path: str) -> dict:
        """Private function to load image.

        Args:
            results (dict): Result dict from :obj:``BaseCDDataset``.

        Returns:
            dict: The dict contains loaded image.
        """
        results = copy.copy(results)
        results['img_path'] = img_path
        return super().transform(results)

    def transform(self, results: dict) -> dict:
        """Functions to load image pair.

        Args:
            results (dict): Result dict from :class:`BaseCDDataset`.

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
