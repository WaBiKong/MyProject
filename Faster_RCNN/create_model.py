# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         create_model.py
# Description:  创建模型
# Author:       WaBiKong
# Date:         2021/11/19
# -------------------------------------------------------------------------------

import torch
import torchvision

from RPN.AnchorsGenerator import AnchorsGenerator
from backbone.mobilenetv2_model import MobileNetV2
from backbone.vgg_model import vgg
from faster_rcnn_farmework import FasterRCNN


def create_model(backbone_name='vgg', num_classes=0):
    global backbone
    if backbone_name == 'mobilenetv2':
        # https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
        backbone = MobileNetV2(weights_path="./backbone/mobilenet_v2.pth").features
        backbone.out_channels = 1280  # 设置对应backbone输出特征矩阵的channels
    elif backbone_name == 'vgg':
        # https://download.pytorch.org/models/vgg16-397923af.pth
        vgg_feature = vgg(model_name="vgg16", weights_path="./backbone/vgg16.pth").features
        backbone = torch.nn.Sequential(*list(vgg_feature._modules.values())[:-1])  # 删除features中最后一个Maxpool层
        backbone.out_channels = 512

    anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],  # 在哪些特征层上进行roi pooling
                                                    output_size=[7, 7],  # roi_pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率

    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model