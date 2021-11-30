# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# name          : post_process.py
# Description   : 
# Author        : WaBiKong
# Date          : 2021/11/30
# -------------------------------------------------------------------------------
from torch.jit.annotations import Tuple, List

import torch.jit
from torch import nn, Tensor
from torch.nn import functional as F

from nms import batched_nms


class PostProcess(nn.Module):
    def __init__(self, dboxes):
        super(PostProcess, self).__init__()
        # [num_anchors, 4] -> [1, num_anchors, 4]
        # dboxes表示生成的默认框
        self.dboxes_xywh = nn.Parameter(dboxes(order='xywh').unsqueeze(dim=0),
                                        requires_grad=False)
        self.scale_xy = dboxes.scale_xy  # 0.1
        self.scale_wh = dboxes.scale_wh  # 0.2

        self.criteria = 0.5
        self.max_output = 100

    def forward(self, bboxes_in, scores_in):
        # 通过预测的bbox_in回归参数得到最终的预测坐标bboxes
        # 将预测目标score通过softmax处理得到预测分数probs
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)

        outputs = torch.jit.annotate(List[Tuple[Tensor, Tensor, Tensor]], [])
        # 遍历一个batch中的每张图片image数据
        # bboxes: [batch, 8732, 4]
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):  #split_szie, split_dim
            # bbox: [1, 8732, 4]
            # squeeze(0): 消去第0维得到[8732, 4]和[8732, label_num]
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            outputs.append(self.decode_single_new(bbox, prob, self.criteria, self.max_output))
        return outputs

    def scale_back_batch(self, bboxes_in, scores_in):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        """
            1）通过预测的boxes回归参数得到最终预测坐标
            2）将box格式从xywh转换回ltrb
            3）将预测目标score通过softmax处理

            Do scale and transform from xywh to ltrb
            suppose input N x 4 x num_bbox | N x label_num x num_bbox

            bboxes_in: [N, 4, 8732]是网络预测的xywh回归参数
            scores_in: [N, label_num, 8732]是预测的每个default box的各目标概率
        """
        # Returns a view of the original tensor with its dimensions permuted
        # [batch, 4, 8732] -> [batch, 8732, 4]
        bboxes_in = bboxes_in.permute(0, 2, 1)
        # [batch, label_num, 8732] -> [batch, 8732, label]
        scores_in = scores_in.oermute(0, 2, 1)

        # 分别获取缩放后预测的回归参数
        bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2]  # 预测的x, y回归参数
        bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]  # 预测的w, h回归参数

        # 将预测的回归参数叠加到default box上得到最终的预测边界框
        bboxes_in[:, :, :2] = bboxes_in[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]

        # transform format to ltrb
        l = bboxes_in[:, :, 0] - 0.5 * bboxes_in[:, :, 2]
        t = bboxes_in[:, :, 1] - 0.5 * bboxes_in[:, :, 3]
        r = bboxes_in[:, :, 0] + 0.5 * bboxes_in[:, :, 2]
        b = bboxes_in[:, :, 1] + 0.5 * bboxes_in[:, :, 2]

        bboxes_in[:, :, 0] = l  # xmin
        bboxes_in[:, :, 1] = t  # ymin
        bboxes_in[:, :, 2] = r  # xmax
        bboxes_in[:, :, 3] = b  # ymax

        # scores_in: [batch, 8732, label_num]
        return bboxes_in, F.softmax(scores_in, -1)

    def decode_single_new(self, bboxes_in, scores_in, criteria, max_output):
        # type: (Tensor, Tensor, float, int) -> Tuple[Tensor, Tensor, Tensor]
        """
        decode:
            input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x label_num)
            output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
            criteria : IoU threshold of bboexes
            max_output : maximum number of output bboxes
        """
        device = bboxes_in.device
        num_classes = scores_in.shape[-1]

        # 对越界的bbox进行裁剪
        bboxes_in = bboxes_in.clamp(min=0, max=1)

        # [8732, 4] -> [8732, 21, 4]  拷贝到和类别数一样
        bboxes_in = bboxes_in.repeat(1, num_classes).reshape(scores_in.shape[0], -1, 4)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        # [num_classes] -> [8732, num_classes]
        labels = labels.view(1, -1).expand_as(scores_in)

        # 移除归为背景类别的概率信息
        bboxes_in = bboxes_in[:, 1:, :]  # [8732, 21, 4] -> [8732, 20, 4]
        scores_in = scores_in[:, 1:]  # [8732, 21] -> [8732, 20]
        labels = labels[:, 1:]  # [8732, 21] -> [8732, 20]

        # batch everything, by making every class prediction be a separate instance
        bboxes_in = bboxes_in.reshape(-1, 4)  # [8732, 20, 4] -> [8732x20, 4]
        scores_in = scores_in.reshape(-1)  # [8732, 20] -> [8732x20]
        labels = labels.reshape(-1)  # [8732, 20] -> [8732x20]

        # remove low scoring boxes
        # 移除低概率目标，self.scores_thresh=0.05
        inds = torch.where(torch.gt(scores_in, 0.05))[0]
        bboxes_in, scores_in, labels = bboxes_in[inds, :], scores_in[inds], labels[inds]

        # remove empty boxes
        ws, hs = bboxes_in[:, 2] - bboxes_in[:, 0], bboxes_in[:, 3] - bboxes_in[:, 1]
        keep = (ws >= 1 / 300) & (hs >= 1 / 300)
        keep = torch.where(keep)[0]
        bboxes_in, scores_in, labels = bboxes_in[keep], scores_in[keep], labels[keep]

        # non-maximum suppression
        keep = batched_nms(bboxes_in, scores_in, labels, iou_threshold=criteria)

        # 只取要求的前max_output个
        keep = keep[:max_output]
        bboxes_out = bboxes_in[keep, :]
        scores_out = scores_in[keep]
        labels_out = labels[keep]

        return bboxes_out, labels_out, scores_out
