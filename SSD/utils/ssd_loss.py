# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# name          : ssd_loss.py
# Description   : 
# Author        : WaBiKong
# Date          : 2021/11/30
# -------------------------------------------------------------------------------
import torch
from torch import nn


class Loss(nn.Module):
    def __init__(self, dboxes):
        super(Loss, self).__init__()
        # xy、wh的缩放因子
        self.scale_xy = 1.0 / dboxes.scale_xy  # 10
        self.scale_wh = 1.0 / dboxes.scale_wh  # 5

        self.location_loss = nn.SmoothL1Loss(reduction='none')

        # transpose(0, 1): [num_anchors, 4] -> [4, num_anchors]
        # unsqueeze(0): [4, num_anchors] -> [1, 4, num_anchors]
        # nn.Parameter: 转化为pytorch中的参数
        self.dboxes = nn.Parameter(dboxes(order='xywh').transpose(0, 1).unsqueeze(0),
                                   requires_grad=False)

        self.confidence_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, ploc, plabel, gloc, glabel):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
        """
                ploc, plabel: N x 4 x 8732, N x label x 8732
                            predicted location and labels
                gloc, glabel: N x 4 x 8732, N x 8732
                            ground truth location and labels
            """
        # 获取正样本的mask    Tensor: [N, 8732]
        # glabel>0则匹配到了类别(正样本)，=0则为背景(负样本)
        mask = torch.gt(glabel, 0)  # (gt: >)
        # 计算一个batch中每张图片的正样本个数  Tensor: [N]
        pos_num = mask.sum(dim=1)

        # 计算gt的location回归参数 Tensor: [N, 4, 8732]
        vec_gd = self._location_vec(gloc)

        # 计算定位损失(只有正样本)
        loc_loss = self.location_loss(ploc, vec_gd).sum(dim=1)  # Tensor: [N, 8732]
        loc_loss = (mask.float() * loc_loss).sum(dim=1)  # Tensor: [N]

        # 计算全部样本置信度损失   Tensor: [N, 8732]
        con = self.confidence_loss(plabel, glabel)

        # positive mask will never selected
        # 获取负样本
        con_neg = con.clone()
        con_neg[mask] = 0.0
        # 按照confidence_loss降序排列 con_idx(Tensor: [N, 8732])
        # descending=True: 降序
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)  # 这个步骤比较巧妙

        # number of negative three times positive
        # 用于损失计算的负样本是正样本的三倍(Hard negative mining部分)
        # 负样本总数不能超过8732
        neg_num = torch.clamp(3 * pos_num, max=mask.size(1)).unsqueeze(-1)
        # 选取neg_num个负样本
        neg_mask = torch.lt(con_rank, neg_num)  # (lt: <)   Tensor: [N, 8732]

        # confidence最终loss使用选取的正样本loss+负样本loss
        con_loss = (con * (mask.float() + neg_mask.float())).sum(dim=1)  # Tensor: [N]

        # 避免出现图像中没有gtbox的情况
        total_loss = loc_loss + con_loss
        # eg. [15, 3, 5, 0] -> [1.0, 1.0, 1.0, 0.0]
        num_mask = torch.gt(pos_num, 0).float()  # 统计一个batch中的每张图像中是否存在正样本
        pos_num = pos_num.float().clamp(min=1e-6)  # 防止出现分母为零的情况
        ret = (total_loss * num_mask / pos_num).mean(dim=0)  # 只计算正样本的图像损失
        return ret

    def _location_vec(self, loc):
        # type: (Tensor) -> Tensor
        """
            计算ground truth相对的anchors的回归参数
            Args:
                loc: anchor匹配到的对应的gtbox  N x 4 x 8732
            """
        gxy = self.scale_xy * (loc[:, :2, :] - self.dboxes[:, :2, :]) / self.dboxes[:, 2:, :]  # N x 2 x 8732
        gwh = self.scale_wh * (loc[:, 2:, :] / self.dboxes[:, 2:, :]).log()  # N x 2 x 8732
        return torch.cat((gxy, gwh), dim=1).contiguous()