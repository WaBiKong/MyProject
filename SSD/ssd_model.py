# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# name          : ssd_model.py
# Description:
# Author        : WaBiKong
# Date          : 2021/11/30
# -------------------------------------------------------------------------------

import torch
from torch import nn, Tensor
from torch.jit.annotations import List

from backbone.resnet50_backbone import resnet50
from utils.default_box import dboxes300


# 对backbone中resnet50的剪切和修改
class Backbone(nn.Module):
    def __init__(self, pretrain_path=None):
        super(Backbone, self).__init__()
        net = resnet50()  # 基础网络为resnet50
        # 6个生成default_box的特征层的输出通道
        # resnet最后一个卷积块conv4的输出通道和后面添加的4个卷积块的输出通道
        self.out_channels = [1024, 512, 512, 256, 256, 256]

        if pretrain_path is not None:  # 加载预训练模型
            net.load_state_dict(torch.load(pretrain_path))

        # 获取resnet的前7层，conv1、BN、ReLU、Maxpool、layer1、layer2、layer3
        self.feature_extractor = nn.Sequential(*list(net.children())[:7])

        # 获取第7层的第一个残差块，即layer3中conv4的第一个block
        conv4_block1 = self.feature_extractor[-1][0]
        # 修改conv4_block1的步距，从2->1
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class SSD300(nn.Module):
    def __init__(self, backbone=None, num_classes=21):
        super(SSD300, self).__init__()
        if backbone is None:
            raise Exception("backbone is None")
        if not hasattr(backbone, 'out_channels'):
            raise Exception("the backbone not has attribute: out_channels")

        self.feature_extractor = backbone
        self.num_classes = num_classes
        # out_channels = [1024, 512, 512, 256, 256, 256]
        # 在backbone(resnet50)后添加额外的一系列卷积层(5个)，得到相应的一系列特征提取器
        self._build_additional_features(self.feature_extractor.out_channels)
        # 每个用来生成default_box的特征层的每个像素生成的default_box数
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        location_extractors = []
        confidence_extractors = []

        # 每个box生成层后都跟着一个坐标回归卷积层和一个类别预测卷积层
        # out_channels = [1024, 512, 512, 256, 256, 256]
        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            # nd is number_default_boxes, oc is output_channel
            location_extractors.append(nn.Conv2d(oc, nd * 4, kernel_szie=3, padding=1))
            confidence_extractors.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(location_extractors)
        self.conf = nn.ModuleList(confidence_extractors)
        self._init_weights()

        default_box = dboxes300()
        self.compute_loss = Loss(default_box)
        self.encoder = Encoder(default_box)
        self.postprocess = PostProcess(default_box)

    def forward(self, image, targets=None):
        x = self.feature_extractor(image)

        # Feature Map 38x38x1024, 19x19x512, 10x10x512, 5x5x256, 3x3x256, 1x1x256
        # 定义detection_features,存储每个Feature Map
        # 类型为List，里面元素类型为Tensor，初始化为空列表[]
        detection_features = torch.jit.annotate(List[Tensor], [])
        detection_features.append(x)
        for layer in self.additional_blocks:
            x = layer(x)
            detection_features.append(x)

        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
        # 得到每一个预测特征层的坐标回归参数和分类预测置信度
        locs, confs = self.bbox_view(detection_features, self.loc, self.confs)

        # For SSD300, shall return nbatch x 8732 x {nlabels, nlocs} results
        # 38x38x4 + 19x19x6 + 10x10x6 + 5x5x6 + 3x3x4 + 1x1x4 = 8732

        if self.training:  # 训练模式下计算损失
            if targets is None:
                raise ValueError("In training mode, targets should be passed")

            # bboxes_out (Tensor 8732 × 4), labels_out (Tenosr 8732)
            bboxes_out = targets['boxes']
            bboxes_out = bboxes_out.transpose(1, 2).contiguous()
            labels_out = targets['labels']

            loss = self.compute_loss(locs, confs, bboxes_out, labels_out)
            return {"total_losses": loss}

        # 预测模式下进行后处理
        # 将预测回归参数叠加到default box上得到最终预测box，并执行非极大值抑制虑除重叠框
        # results = self.encoder.decode_batch(locs, confs)
        results = self.postprocess(locs, confs)
        return results

    def _build_additional_features(self, input_size):
        """
        在backbone(resnet50)后添加额外的一系列卷积层，得到相应的一系列特征提取器
        """
        additional_blocks = []
        # input_size = [1024, 512, 512, 256, 256, 256]
        middle_channels = [256, 256, 128, 128, 128]
        for i, (input_ch, output_ch, middle_ch) in enumerate(zip(input_size[:-1], input_size[1:], middle_channels)):
            # 前三个卷积层padding为1、stride为2，后两个padding为0、stride为1
            padding, stride = (1, 2) if i < 3 else (0, 1)
            layer = nn.Sequential(
                # 后面使用了BN层，所以偏置bias设置为Flase
                nn.Conv2d(input_ch, middle_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(middle_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(middle_ch, output_ch, kernel_size=3, padding=padding, stride=stride),
                nn.BatchNorm2d(output_ch),
                nn.ReLU(inplace=True)
            )
            additional_blocks.append(layer)
        self.additional_blocks = nn.ModuleList(additional_blocks)

    def _init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def bbox_view(self, features, loc_extractor, conf_extractor):
        """
        计算每个预测特征层的坐标回归参数和分类预测置信度参数
        """
        locs = []
        confs = []
        for f, l, c in zip(features, loc_extractor, conf_extractor):
            # view: [batch, n*4, feat_size, feat_size] -> [batch, 4, -1]
            locs.append(l(f).view(f.size(0), 4, -1))
            # view: [batch, n*classes, feat_size, feat_size] -> [batch, classes, -1]
            confs.append(c(f).view(f.size(0), self.num_classes, -1))

        # 将所有回归参数和置信度在dim=2上拼接, 即上面的-1表示的维度，就是把预测的所有box的参数拼接
        # contiguous(): 将数据调整为连续存储的方式(view方法不会改变存储方式)
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs


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
        return torch.cat((gxy, gwh), dim=1).contigous()
