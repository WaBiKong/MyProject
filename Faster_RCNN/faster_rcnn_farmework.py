# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         faster_rcnn_farmework.py
# Description:  faste_rcnn基本框架：包括FasterRCNNBase类和继承自FasterRCNNBase的FasterRCNN类
# Author:       WaBiKong
# Date:         2021/11/17
# -------------------------------------------------------------------------------

import warnings
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional

from Faster_RCNN.RPN.AnchorsGenerator import AnchorsGenerator
from Faster_RCNN.RPN.RPNHead import RPNHead
from Faster_RCNN.RPN.RegionProposalNetwork import RegionProposalNetwork

from transform import GeneralizedRCNNTransform


class FasterRCNNBase(nn.Module):
    """
    Main class for Generalized R-CNN

    Args:
        backbone: 特征提取网络部分
        rpn: 区域建议生成网络部分
        roi_heads: takes the features + the proposals from the RPN and computes detections / masks from it.
        transform: performs the data transformation from the inputs to feed into the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(FasterRCNNBase, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        note: 这里输入的图片大小是不相同的，后面会进行预处理将这些图片放到相同大小的Tensor中打包成一个batch
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """

        if self.training and targets is None:  # 训练模式必须有targets
            raise ValueError("In training mode, targets should be passed")

        if self.training:  # 对传入的targets进一步检查
            assert targets is not None
            for target in targets:
                boxes = target['boxes']
                if isinstance(boxes, torch.Tensor):  # 判断boxes是不是torch.Tensor的格式
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:  # 判断boxse.shape是否为[N, 4]
                        raise ValueError(f"Expected target boxes to be"
                                         f" a tensor of shape [N, 4], got {boxes.shape}.")
                else:
                    raise ValueError(f"Expected target boxes to be of type Tensor, got {type(boxes)}")

        original_image_sizes: List[Tuple[int, int]] = []  # List[Tuple[int, int]]: 对变量进行类型申明
        for img in images:
            val = img.shape[-2:]  # 获取图片后两个维度, pytorch中维度排列顺序为[channel, height, width]
            assert len(val) == 2  # 防止输入的是一维向量
            original_image_sizes.append((val[0], val[1]))  # 记录原始图片size

        # 对图像进行预处理
        images, targets = self.transform(images, targets)

        # 将图片输入backbone得到特征图
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):  # 若特征图只有一层，将feature放入有序字典中并编号为'0'
            features = OrderedDict([('0', features)])  # 若在多层特征图上预测，传入的就是一个有序字典

        # 将图片、特征层以及标注信息targets传入rpn中
        proposals, proposal_losses = self.rpn(images, features, targets)

        # 将rpn生成的数据以及标注信息targets信息传入fast_rcnn后半部分
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        # 对网络的预测结果进行后处理（主要将bboxes还原到尺度图上）
        # images.image_sizes: 预处理后的图像尺寸
        # original_image_sizes: 处理前的图像尺寸
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)


# class TwoMLPHead(nn.Module):
#     """
#     Standard heads for FPN-based models
#     Arguments:
#         in_channels (int): number of input channels
#         representation_size (int): size of the intermediate representation
#     """
#
#     def __init__(self, in_channels, representation_size):
#         super(TwoMLPHead, self).__init__()
#
#         self.fc6 = nn.Linear(in_channels, representation_size)
#         self.fc7 = nn.Linear(representation_size, representation_size)
#
#     def forward(self, x):
#         x = x.flatten(start_dim=1)
#
#         x = F.relu(self.fc6(x))
#         x = F.relu(self.fc7(x))
#
#         return x
#
#
# class FastRCNNPredictor(nn.Module):
#     """
#     Standard classification + bounding box regression layers
#     for Fast R-CNN.
#     Arguments:
#         in_channels (int): number of input channels
#         num_classes (int): number of output classes (including background)
#     """
#
#     def __init__(self, in_channels, num_classes):
#         super(FastRCNNPredictor, self).__init__()
#         self.cls_score = nn.Linear(in_channels, num_classes)
#         self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
#
#     def forward(self, x):
#         if x.dim() == 4:
#             assert list(x.shape[2:]) == [1, 1]
#         x = x.flatten(start_dim=1)
#         scores = self.cls_score(x)
#         bbox_deltas = self.bbox_pred(x)
#
#         return scores, bbox_deltas


# 定义FasterRCNN类继承自FasterRCNNBase类
class FasterRCNN(FasterRCNNBase):
    """
    Implements Faster R-CNN.
    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.
    The behavior of the model changes depending if it is in training or evaluation mode.
    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values
          between 0 and H and 0 and W
        - labels (Int64Tensor[N]): the class label for each ground-truth box
    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN.
    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values between
          0 and H and 0 and W
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        rpn_score_thresh (float): during inference, only return proposals with a classification score
            greater than rpn_score_thresh
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
    """

    def __init__(self, backbone, num_classes=None,  # num_classes: 类别数 + 1
                 # transform parameter
                 min_size=800, max_size=1333,  # 预处理resize时限制的最大尺寸
                 image_mean=None, image_std=None,  # 预处理normalize时使用的均值和方差
                 # RPN parameters
                 rpn_anchor_generator=None,  # anchor生成器
                 rpn_head=None,  # RPN预测网络
                 # rpn中在nms处理前保留的proposal数（根据score）
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 # rpn中在nms处理后保留的proposal数
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,  # rpn中进行nms处理时使用的iou阈值
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,  # rpn计算损失时，采集正负样本设置的阈值
                 rpn_batch_size_per_image=256,  # rpn计算损失时采样的样本数
                 rpn_positive_fraction=0.5,  # rpn计算损失时。正样本占总样本的比例
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05,  # 移除低目标概率
                 box_nms_thresh=0.5,  # fast rcnn中进行nms的阈值
                 box_detections_per_img=100,  # 对预测结果根据score排序取前100个
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,  # fast rcnn计算误差时，采集正负样本设置的阈值
                 box_batch_size_per_image=512, box_positive_fraction=0.25,  # fast rcnn计算误差时采样的样本数，以及正样本占所有样本的比例
                 bbox_reg_weights=None):
        if not hasattr(backbone, 'out_channels'):  # 如果backbone没有out_channels属性
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the nummber of output channels (assumed "
                "to be the same for all the levels"
            )

        # 判断rpn_anchor_geneerator是不是我们自己定义的AnchorsGenerator类或者None
        assert isinstance(rpn_anchor_generator, (AnchorsGenerator, type(None)))
        # 判断box_roi_pool是不是我们自己定义的MultiScaleRoIAlign类或者None
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor "
                                 "is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        # 预测特种层的channels
        out_channels = backbone.out_channels

        # 若anchor生成器为空，则自动生成针对resnet50_fpn的anchor生成器
        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            anchor_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorsGenerator(
                anchor_sizes, anchor_ratios
            )

        # 生成RPN通过滑动窗口进行预测的网络部分
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        # 默认rpm_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000
        # 默认rpm_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        # 定义整个RPN框架
        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh
        )

        # fast rcnn中的roi pooling层
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],  # 在哪些特征层进行roi pooling
                output_size=[7, 7],
                sampling_ratios=2
            )

        # fast rcnn中roi pooling后的展平处理，即两个全连接层部分
        if box_head is None:
            resolution = box_roi_pool.output_size[0]  # 默认等于7
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size
            )

        # 在box_head的输出上做预测的部分
        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        # 将roi pooling, box_head以及box_predictor结合在一起
        roi_heads = RoIHeads(
            # box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,  # 0.5  0.5
            box_batch_size_per_image, box_positive_fraction,  # 512  0.25
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)  # 0.05  0.5  100

        # 预处理的图片均值和方差
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        # 对数据进行标准化，缩放，打包成batch等处理部分
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)