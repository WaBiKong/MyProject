# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# name          : model.py
# Description:  :
# Author        : WaBiKong
# Date          : 2021/12/03  星期五
# -------------------------------------------------------------------------------

import math

import torch
from torch import nn

from utils.layers import *
from utils.parse_cfg import parse_model_cfg

ONNX_EXPORT = False


def create_modules(modules_defs: list, img_szie):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    :param modules_defs: 通过.cfg文件解析得到的每个层结构的列表,其中每个元素为一个字典
    :param img_size:
    """

    img_szie = [img_szie] * 2 if isinstance(img_szie, int) else img_szie
    # 删除解析cfg文件列表中的第一个配置(对应的[net]配置)
    modules_defs.pop(0)
    output_filters = [3]  # 存放每一层的filters，即输出通道，初始化为3表示输入的图片的channel为3
    module_list = nn.ModuleList()
    routs = []  # 记录哪些特征层的输出会被后续的层使用到(可能是特征融合，也可能是拼接)
    yolo_index = -1  # 记录是第几个yolo层

    # 遍历搭建每个层结构
    for i, mdef in enumerate(modules_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'convolutional':
            bn = mdef['batch_normalize']  # 1 or 0 / use or not
            filters = mdef['filters']
            k = mdef['size']  # kernel_size
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            if isinstance(k, int):
                modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                       out_channels=filters,
                                                       kernel_size=k,
                                                       stride=stride,
                                                       padding=k // 2 if mdef['pad'] else 0,
                                                       bias=not bn))
            else:
                raise TypeError("conv2d filter size must be int type.")

            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters))
            else:
                # 如果改卷积层没有bn层，意味着该层为yolo的predictor
                # 记录该层的索引
                routs.append(i)

            if mdef['activation'] == 'leaky':
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
            else:
                pass

        elif mdef['type'] == 'maxpool':
            k = mdef['size']  # pool's kernel_size
            stride = mdef['stride']
            modules = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)

        elif mdef['type'] == 'upsample':
            if ONNX_EXPORT:
                g = (yolo_index + 1) * 2 / 32
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_szie))
            else:
                modules = nn.Upsample(scale_factor=mdef['stride'])

        elif mdef['type'] == 'route':  # [-2], [-1, -3, -5, -6]
            layers = mdef['layers']
            # filters: 记录当前层输出特征矩阵的channel
            # 若layers只有一个元素，则表示layers所指向的层的输出的filters
            # 若layers有多个元素，则将多个层的filters求和，因为多层时是在深度上进行拼接
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            # 创建route模块
            modules = FeatureConcat(layers=layers)

        elif mdef['type'] == 'shortcut':  # Residual，残差块
            layers = mdef['from']
            filters = output_filters[-1]
            routs.append(i + layers[0])
            modules = WeightedFeatureFusion(layers=layers, weight='weights_type' in mdef)

        elif mdef['type'] == 'yolo':
            yolo_index += 1  # 记录是第几个yolo_layer [0, 1, 2]
            stride = [32, 16, 8]  # 预测特征层对应原图的缩放比例

            modules = YOLOLayer(anchors=mdef['anchors'][mdef['mask']],
                                nc=mdef['classes'],
                                img_szie=img_szie,
                                stride=stride[yolo_index])

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            try:  # 对每个predict的bias进行初始化
                j = -1  # -1表示YOLOLayer上一层
                # bias: shape(255,) 索引0对应Sequential中的Conv2d
                # view: shape(3, 85)
                b = module_list[j][0].bias.view(modules.na, -1)
                b.data[:, 4] += -4.5  # obj
                b.data[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
                module_list[j][0].bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            except Exception as e:
                print("WARNING: SMART BIAS INITIALIZATION FAILURE.", e)

        # Register module list and number of output filters
        module_list.append(modules)
        # 当模块为maxpool和uasample时，没有filters，此处会添加上一次循环(缓存)中的filters值
        output_filters.append(filters)

    routs_binary = [False] * len(modules_defs)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary


class YOLOLayer(nn.Module):
    """
    对YOLO的predict输出进行处理
    """

    def __init__(self, anchors, nc, img_size, stride):
        super(YOLOLayer, self).__init__
        self.anchors = torch.Tensor(anchors)
        self.stride = stride  # layer stride 特征图上对应原图的步距 [32, 16, 8]
        self.na = len(anchors)  # anchors的数量
        self.nc = nc  # 类别数
        self.no = nc + 5  # 每个anchor预测的输出的通道数(w, y, w, h, obj, clas1, clas2...)
        # self.nx, self.ny: 预测特征层的宽度和高度
        # self.ng: grid cell的size
        self.nx, self.ny, self.ng = 0, 0, (0, 0)  # 初始化
        # 将anchors大小缩放到grid尺寸(特征图尺寸)
        self.anchor_vec = self.anchors / self.stride
        # batch_size, na, grid_h, grid_w, wh
        # 值为1的维度对应的值不是固定值，后续操作可根据广播机制自动扩充
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        self.grid = None

        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))

    def create_grids(self, ng=(13, 13), device='cpu'):
        """
        更新grids信息并生成新的grids参数
        Args:
            ng: 特征图大小
        """

        self.nx, self.ny = ng
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offset 构建每个cell处的anchor的xy偏移量(在feature map上的)
        if not self.training:  # 训练模式不需要回归到最终的预测bboxes，只需要偏移量进行loss计算
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device),
                                     torch.arange(self.nx, device=device)])
            # batch_size, na, grid_h, grid_w, wh
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p):
        if ONNX_EXPORT:
            bs = 1  # batch_size
        else:
            bs, _, ny, nx = p.shape  # batch_size, predict_param, grid_h, grid_w
            if (self.nx, self.ny) != (nx, ny) or self.grid is None:  # fix no grid bug
                self.create_grids((nx, ny), p.device)

        # 3: 每个cell生成三个anchor
        # 85: coco数据集有80类，加上w, y, w, h, obj
        # view: (batch_size, 255, 13, 13) -> (batch_size, 3, 85, 13, 13)
        # permute: (batch_size, 3, 85, 13, 13) -> (batch_size, 3, 13, 13, 85)
        # [bs, anchor, grid, grid, xywh + obj + classes]
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:  # 训练模式直接返回改变形状后的预测参数
            return p
        elif ONNX_EXPORT:
            # Avoid broadcasting for ANE operations
            m = self.na * self.nx * self.ny
            ng = 1. / self.ng.repeat(m, 1)
            grid = self.grid.repeat(1, self.na, 1, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng

            p = p.view(m, self.no)
            # xy = torch.sigmoid(p[:, 0:2]) + grid  # x, y
            # wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            # p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else \
            #     torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])  # conf
            p[:, :2] = (torch.sigmoid(p[:, 0:2]) + grid) * ng  # x, y
            p[:, 2:4] = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            p[:, 4:] = torch.sigmoid(p[:, 4:])
            p[:, 5:] = p[:, 5:self.no] * p[:, 4:5]
            return p
        else:  # 预测模式和评估模式需要将预测的参数与grid上的anchors结合成bbox
            # [bs, anchor, grid, grid, xywh + obj + classes]
            io = p.clone()
            # 计算在feature map上预测的xy的坐标
            # io[..., :2]表示前面的所有维度的所有和最后一个维度的[ :2]
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid
            # 计算在feature map上预测的wh
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh
            # 将预测的bboxes映射回原图尺度
            io[..., :4] *= self.stride
            # 将最后一维剩下的参数映射到[0, 1]之间
            torch.sigmoid_(io[..., 4:])
            # p: [bs, anchor, grid, grid, xywh + obj + classes]
            # io: view: [bs, anchor, grid, grid, xywh + obj + classes] -> [bs, anchor * grid * grid, xywh + obj + classes]
            return io.view(bs, -1, self.no), p


class Darknet(nn.Module):
    """
    YOLOv3 spp object detection model
    """

    def __init__(self, cfg, img_size=(416, 416), verbose=False):
        super(Darknet, self).__init__()
        # 这里传入的imgz_size只在导出ONNX模型时起作用
        self.input_size = [img_size] * 2 if isinstance(img_size, int) else img_size
        # 解析网络对应的,.cfg文件
        self.modules_defs = parse_model_cfg(cfg)
        # 根据解析的网络结构一层一层去搭建
        self.module_list, self.module_routs = create_modules(self.modules_defs, img_size)
        # 获取所有YOLOlayer层的索引
        self.yolo_layers = get_yolo_layers(self)

        # 打印模型的信息，如果verbose为True则打印详细信息
        self.info(verbose) if not ONNX_EXPORT else None

    def forward(self, x, verbose=False):
        return self.forward_once(x, verbose=verbose)

    def forward_once(self, x, verbose=False):
        # yolo_out收集每个YOLOLayer的输出
        # out收集每个模块的输出
        yolo_out, out = [], []
        if verbose:
            print('0', x.shape)
            str = ''

        for i, module in enumerate(self.module_list):  # 遍历每一个module
            name = module.__class__.__name__  # 获取module名字
            if name in ['WeightedFeatureFusion', 'FeatureConcat']:
                if verbose:
                    l = [i - 1] + module.layers
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]
                    str = '>>' + '+'.join(['layer %g % s' % x for x in zip(l, sh)])
                x = module(x, out)
            elif name == 'YOLOLayer':
                yolo_out.append(module(x))
            else:
                x = module(x)

            out.append(x if self.routs[i] else [])
            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), str)
                str = ''

        if self.training:  # 训练模式直接返回三个YOLOLayer的输出结果
            return yolo_out
        elif ONNX_EXPORT:  # export
            # x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            # return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
            p = torch.cat(yolo_out, dim=0)

            # # 根据objectness虑除低概率目标
            # mask = torch.nonzero(torch.gt(p[:, 4], 0.1), as_tuple=False).squeeze(1)
            # # onnx不支持超过一维的索引（pytorch太灵活了）
            # # p = p[mask]
            # p = torch.index_select(p, dim=0, index=mask)
            #
            # # 虑除小面积目标，w > 2 and h > 2 pixel
            # # ONNX暂不支持bitwise_and和all操作
            # mask_s = torch.gt(p[:, 2], 2./self.input_size[0]) & torch.gt(p[:, 3], 2./self.input_size[1])
            # mask_s = torch.nonzero(mask_s, as_tuple=False).squeeze(1)
            # p = torch.index_select(p, dim=0, index=mask_s)  # width-height 虑除小目标
            #
            # if mask_s.numel() == 0:
            #     return torch.empty([0, 85])
            return p
        else:  # 验证或者预测模式
            # YOLOLayer返回的是: 预测结果io.view(bs, -1, self.no), 预测参数p
            # self.no: 每个anchor预测的输出的通道数(w, y, w, h, obj, clas1, clas2...)
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # 将预测结果在第一个维度上拼接，得到[bs, all_anchors_pre_image, self.no]

            return x, p


def get_yolo_layers(self):
    """
    获取整个网络中三个'YOLOLayer'模块对应的索引
    """
    return [i for i, m in enumerate(self.module_list) if m.__class__.__name__ == 'YOLOLayer']  # [89, 101, 113]