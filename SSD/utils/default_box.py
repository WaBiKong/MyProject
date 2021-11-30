# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# name          : default_box.py
# Description   :
# Author        : WaBiKong
# Date          : 2021/11/30
# -------------------------------------------------------------------------------
import itertools
from math import sqrt

import numpy as np
import torch


def dboxes300():
    figsize = 300  # 输入网络的图像大小
    feat_size = [38, 19, 10, 5, 3, 1]  # 每个预测特征层的feature map尺寸
    steps = [8, 16, 32, 64, 100, 300]  # 每个预测特征层上一个cell在原图上的跨度
    scales = [21, 45, 99, 153, 207, 261, 315]  # 每个特征层上预测的default box的scale
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]  # 每个特征层上预测的default box的ratios
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes


class DefaultBoxes(object):
    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios, scale_xy=0.1, scale_wh=0.2):
        self.fig_size = fig_size  # 输入网络的图像大小 300×300
        # [38, 19, 10, 5, 3, 1]
        self.feat_size = feat_size  # 每个预测层的feature map尺寸

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        # [8, 16, 32, 64, 100, 300]
        self.steps = steps  # 每个特征层上的一个cell在原图上的跨度

        # [21, 45, 99, 153, 207, 261, 315]
        self.scales = scales  # 每个预测特征层上预测的default box的scale

        fk = fig_size / np.array(steps)  # 计算每层特征层的fk
        # [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.aspect_ratios = aspect_ratios  # 每个预测特征层上预测的default box的ratios

        self.default_boxes = []
        # size of feature amd number of feature
        # 遍历每层特征层，计算default box
        for idx, sfeat in enumerate(self.feat_size):
            sk1 = scales[idx] / fig_size  # scale转为相对值[0-1]
            sk2 = scales[idx + 1] / fig_size
            sk3 = sqrt(sk1 * sk2)
            # 先添加两个1:1比例的default box的高宽
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            # 再将剩下不同比例的default box宽和高添加到all_szie中
            for alpha in aspect_ratios[idx]:
                w, h = sk1 * sqrt(alpha), sk1 / sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))

            # 计算当前特征层对应原图的所有default box
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):  # i->行(y), j->列(x)
                    # 计算每个default box的中心坐标(范围在0-1之间)
                    cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        # 将default box转为Tensor格式
        self.dboxes = torch.as_tensor(self.default_boxes, dtype=torch.float32)

        self.dboxes.clamp_(min=0, max=1)  # 将坐标(x, y, w, h)都限制在0-1之间

        # 将坐标(x, y, w, h)转换为(xmin, ymin, xmax, ymax)，方便后续正负样本匹配时计算IoU
        self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]  # xmin
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]  # xmin
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]  # xmin
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]  # xmin

    def __call__(self, order='ltrb'):
        # 根据需求返回相应格式的default box
        if order == 'ltrb':
            return self.dboxes_ltrb
        if order == 'xywh':
            return self.dboxes

    @property
    def scale_xy(self):
        return self.scale_xy_

    @property
    def scale_wh(self):
        return self.scale_wh_
