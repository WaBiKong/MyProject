# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# name          : Encoder.py
# Description   : 
# Author        : WaBiKong
# Date          : 2021/11/30
# -------------------------------------------------------------------------------
import torch
from torch.nn import functional as F
from SSD.utils.nms import batched_nms


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.
    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format
    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def calc_iou_tensor(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    #  When the shapes do not match,
    #  the shape of the returned output tensor follows the broadcasting rules
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


# This function is from https://github.com/kuangliu/pytorch-ssd.
class Encoder(object):
    """
        Inspired by https://github.com/kuangliu/pytorch-src
        Transform between (bboxes, lables) <-> SSD output
        dboxes: default boxes in size 8732 x 4,
            encoder: input ltrb format, output xywh format
            decoder: input xywh format, output ltrb format
        encode:
            input  : bboxes_in (Tensor nboxes x 4), labels_in (Tensor nboxes)
            output : bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
            criteria : IoU threshold of bboexes
        decode:
            input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
            output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
            criteria : IoU threshold of bboexes
            max_output : maximum number of output bboxes
    """
    def __init__(self, dboxes):
        self.dboxes = dboxes(order='ltrb')
        self.dboxes_xywh = dboxes(order='xywh').unsqueeze(dim=0)
        self.nboxes = self.dboxes.size(0)  # default boxes的数量
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh

    def encode(self, bboxes_in, labels_in, criteria=0.5):
        """
        encode:
            input  : bboxes_in (Tensor nboxes x 4), labels_in (Tensor nboxes)
            output : bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
            criteria : IoU threshold of bboexes
        """
        # [nboxes, 8732]
        ious = calc_iou_tensor(bboxes_in, self.dboxes)  # 计算每个GT与default box的iou
        # [8732,]
        best_dbox_ious, best_dbox_idx = ious.max(dim=0)  # 寻找每个default box匹配到的最大IoU
        # [nboxes,]
        best_bbox_ious, best_bbox_idx = ious.max(dim=1)  # 寻找每个GT匹配到的最大IoU

        # 将每个GT匹配到的最佳default box设置为正样本（对应论文中Matching strategy的第一条）
        # set best ious 2.0
        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)  # dim, index, value
        # 将相应default box匹配最大IOU的GT索引进行替换
        idx = torch.arange(0, best_bbox_idx.size(0), dtype=torch.int64)
        best_dbox_idx[best_bbox_idx[idx]] = idx

        # filter IoU > 0.5
        # 寻找与GT iou大于0.5的default box,对应论文中Matching strategy的第二条(这里包括了第一条匹配到的信息)
        masks = best_dbox_ious > criteria
        # [8732,]
        labels_out = torch.zeros(self.nboxes, dtype=torch.int64)
        labels_out[masks] = labels_in[best_dbox_idx[masks]]
        # 将default box匹配到正样本的位置设置成对应GT的box信息
        bboxes_out = self.dboxes.clone()
        bboxes_out[masks, :] = bboxes_in[best_dbox_idx[masks], :]

        # Transform format to xywh format
        x = 0.5 * (bboxes_out[:, 0] + bboxes_out[:, 2])  # x
        y = 0.5 * (bboxes_out[:, 1] + bboxes_out[:, 3])  # y
        w = bboxes_out[:, 2] - bboxes_out[:, 0]  # w
        h = bboxes_out[:, 3] - bboxes_out[:, 1]  # h
        bboxes_out[:, 0] = x
        bboxes_out[:, 1] = y
        bboxes_out[:, 2] = w
        bboxes_out[:, 3] = h
        return bboxes_out, labels_out

    def scale_back_batch(self, bboxes_in, scores_in):
        """
            将box格式从xywh转换回ltrb, 将预测目标score通过softmax处理
            Do scale and transform from xywh to ltrb
            suppose input N x 4 x num_bbox | N x label_num x num_bbox
            bboxes_in: 是网络预测的xywh回归参数
            scores_in: 是预测的每个default box的各目标概率
        """
        if bboxes_in.device == torch.device("cpu"):
            self.dboxes = self.dboxes.cpu()
            self.dboxes_xywh = self.dboxes_xywh.cpu()
        else:
            self.dboxes = self.dboxes.cuda()
            self.dboxes_xywh = self.dboxes_xywh.cuda()

        # Returns a view of the original tensor with its dimensions permuted.
        bboxes_in = bboxes_in.permute(0, 2, 1)
        scores_in = scores_in.permute(0, 2, 1)
        # print(bboxes_in.is_contiguous())

        bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2]   # 预测的x, y回归参数
        bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]   # 预测的w, h回归参数

        # 将预测的回归参数叠加到default box上得到最终的预测边界框
        bboxes_in[:, :, :2] = bboxes_in[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]

        # transform format to ltrb
        l = bboxes_in[:, :, 0] - 0.5 * bboxes_in[:, :, 2]
        t = bboxes_in[:, :, 1] - 0.5 * bboxes_in[:, :, 3]
        r = bboxes_in[:, :, 0] + 0.5 * bboxes_in[:, :, 2]
        b = bboxes_in[:, :, 1] + 0.5 * bboxes_in[:, :, 3]

        bboxes_in[:, :, 0] = l  # xmin
        bboxes_in[:, :, 1] = t  # ymin
        bboxes_in[:, :, 2] = r  # xmax
        bboxes_in[:, :, 3] = b  # ymax

        return bboxes_in, F.softmax(scores_in, dim=-1)

    def decode_batch(self, bboxes_in, scores_in, criteria=0.45, max_output=200):
        # 将box格式从xywh转换回ltrb（方便后面非极大值抑制时求iou）, 将预测目标score通过softmax处理
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)

        outputs = []
        # 遍历一个batch中的每张image数据
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            outputs.append(self.decode_single_new(bbox, prob, criteria, max_output))
        return outputs

    def decode_single_new(self, bboxes_in, scores_in, criteria, num_output=200):
        """
        decode:
            input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
            output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
            criteria : IoU threshold of bboexes
            max_output : maximum number of output bboxes
        """
        device = bboxes_in.device
        num_classes = scores_in.shape[-1]

        # 对越界的bbox进行裁剪
        bboxes_in = bboxes_in.clamp(min=0, max=1)

        # [8732, 4] -> [8732, 21, 4]
        bboxes_in = bboxes_in.repeat(1, num_classes).reshape(scores_in.shape[0], -1, 4)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores_in)

        # remove prediction with the background label
        # 移除归为背景类别的概率信息
        bboxes_in = bboxes_in[:, 1:, :]
        scores_in = scores_in[:, 1:]
        labels = labels[:, 1:]

        # batch everything, by making every class prediction be a separate instance
        bboxes_in = bboxes_in.reshape(-1, 4)
        scores_in = scores_in.reshape(-1)
        labels = labels.reshape(-1)

        # remove low scoring boxes
        # 移除低概率目标，self.scores_thresh=0.05
        inds = torch.nonzero(scores_in > 0.05, as_tuple=False).squeeze(1)
        bboxes_in, scores_in, labels = bboxes_in[inds], scores_in[inds], labels[inds]

        # remove empty boxes
        ws, hs = bboxes_in[:, 2] - bboxes_in[:, 0], bboxes_in[:, 3] - bboxes_in[:, 1]
        keep = (ws >= 0.1 / 300) & (hs >= 0.1 / 300)
        keep = keep.nonzero(as_tuple=False).squeeze(1)
        bboxes_in, scores_in, labels = bboxes_in[keep], scores_in[keep], labels[keep]

        # non-maximum suppression
        keep = batched_nms(bboxes_in, scores_in, labels, iou_threshold=criteria)

        # keep only topk scoring predictions
        keep = keep[:num_output]
        bboxes_out = bboxes_in[keep, :]
        scores_out = scores_in[keep]
        labels_out = labels[keep]

        return bboxes_out, labels_out, scores_out

    # perform non-maximum suppression
    def decode_single(self, bboxes_in, scores_in, criteria, max_output, max_num=200):
        """
        decode:
            input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
            output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
            criteria : IoU threshold of bboexes
            max_output : maximum number of output bboxes
        """
        # Reference to https://github.com/amdegroot/ssd.pytorch
        bboxes_out = []
        scores_out = []
        labels_out = []

        # 非极大值抑制算法
        # scores_in (Tensor 8732 x nitems), 遍历返回每一列数据，即8732个目标的同一类别的概率
        for i, score in enumerate(scores_in.split(1, 1)):
            # skip background
            if i == 0:
                continue

            # [8732, 1] -> [8732]
            score = score.squeeze(1)

            # 虑除预测概率小于0.05的目标
            mask = score > 0.05
            bboxes, score = bboxes_in[mask, :], score[mask]
            if score.size(0) == 0:
                continue

            # 按照分数从小到大排序
            score_sorted, score_idx_sorted = score.sort(dim=0)

            # select max_output indices
            score_idx_sorted = score_idx_sorted[-max_num:]
            candidates = []

            while score_idx_sorted.numel() > 0:
                idx = score_idx_sorted[-1].item()
                # 获取排名前score_idx_sorted名的bboxes信息 Tensor:[score_idx_sorted, 4]
                bboxes_sorted = bboxes[score_idx_sorted, :]
                # 获取排名第一的bboxes信息 Tensor:[4]
                bboxes_idx = bboxes[idx, :].unsqueeze(dim=0)
                # 计算前score_idx_sorted名的bboxes与第一名的bboxes的iou
                iou_sorted = calc_iou_tensor(bboxes_sorted, bboxes_idx).squeeze()

                # we only need iou < criteria
                # 丢弃与第一名iou > criteria的所有目标(包括自己本身)
                score_idx_sorted = score_idx_sorted[iou_sorted < criteria]
                # 保存第一名的索引信息
                candidates.append(idx)

            # 保存该类别通过非极大值抑制后的目标信息
            bboxes_out.append(bboxes[candidates, :])   # bbox坐标信息
            scores_out.append(score[candidates])       # score信息
            labels_out.extend([i] * len(candidates))   # 标签信息

        if not bboxes_out:  # 如果为空的话，返回空tensor，注意boxes对应的空tensor size，防止验证时出错
            return [torch.empty(size=(0, 4)), torch.empty(size=(0,), dtype=torch.int64), torch.empty(size=(0,))]

        bboxes_out = torch.cat(bboxes_out, dim=0).contiguous()
        scores_out = torch.cat(scores_out, dim=0).contiguous()
        labels_out = torch.as_tensor(labels_out, dtype=torch.long)

        # 对所有目标的概率进行排序（无论是什 么类别）,取前max_num个目标
        _, max_ids = scores_out.sort(dim=0)
        max_ids = max_ids[-max_output:]
        return bboxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]