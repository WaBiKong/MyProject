# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         transform.py
# Description:  transform in dataset
# Author:       WaBiKong
# Date:         2021/11/16
# -------------------------------------------------------------------------------

import random
from torchvision.transforms import functional as F


# 对数据集进行预处理
class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """将PIL图像转换为Tensor"""

    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:  # 随机生成的数小于prob则翻转，即随机翻转
            height, width = image.shape[-2:]  # 获取图片最后两维的大小，即高度和宽度
            image = image.flip(-1)  # 水平翻转图片
            bbox = target['boxes']
            # bbox = xmin, ymin, xmax, ymax
            # y坐标没变, xmin = width - xmax, xmax = width - min
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应的bbox坐标信息
            target['boxes'] = bbox
        return image, target