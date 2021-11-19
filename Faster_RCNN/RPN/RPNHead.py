# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         RPNHead.py
# Description:  先对特征图进行3*3卷积，再将得到的特征层分别进行1*1卷积进行目标分数预测和四个坐标回归
# Author:       WaBiKong
# Date:         2021/11/19
# -------------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn


class RPNHead(nn.Module):
    """
    add a RPN head with classification and regression
    通过滑动窗口计算预测目标概率与bbox regression参数
    Arguments:
        in_channels: number of channels of the input feature
        num_anchors: number of anchors to be predicted
    """

    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        # 3x3 滑动窗口（这里输出的通道数等于输入的通道数）
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        # 目标分数预测器，计算预测的目标分数（这里的目标只是指前景或者背景）
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=(1, 1), stride=(1, 1))
        # 目标边界框预测器，计算预测的目标bbox regression参数
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=(1, 1), stride=(1, 1))

        for layer in self.children():  # 对三个卷积层进行参数初始化
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):  # x: 预测特征层
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        logits = []
        bbox_reg = []
        for i, feature in enumerate(x):  # 对于每个预测特征层
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg
