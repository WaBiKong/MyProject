# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         image_list.py
# Description:  关联图片和resize后的尺寸
# Author:       WaBiKong
# Date:         2021/11/18
# -------------------------------------------------------------------------------

from torch.jit.annotations import List, Tuple
from torch import Tensor


class ImageList(object):
    def __init__(self, tensors, image_sizes):
        # type: (Tensor, List[Tuple[int, int]]) -> None
        """
        Args:
            tensors: padding后的图像数据
            image_sizes: padding前的图像尺寸
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        # type: (Device) -> ImageList
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)