# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         VOC.py
# Description:
# Author:       WaBiKong
# Date:         2021/11/16
# -------------------------------------------------------------------------------

import torch
from torch.utils.data import Dataset
import json
import os.path
from PIL import Image
from lxml import etree


class VOC2012Dataset(Dataset):
    """读取解析PASCAL VOC2012数据集"""

    def __init__(self, voc_root, transforms=None, train_set=True):
        """
        Args:
            voc_root: 数据集所在根目录
            transforms: 预处理方法
            train_set: True则返回训练集，False返回验证集
        """
        # root: 数据集目录, img_root: 数据集下的图片目录, annotations_root: 标注信息目录
        self.transforms = transforms
        self.root = os.path.join(voc_root, 'VOCdevkit', 'VOC2012')
        self.img_root = os.path.join(self.root, 'JPEGImages')
        self.annotations_root = os.path.join(self.root, 'Annotations')

        # 读取标注信息文件(.xml)
        if train_set:
            txt_list = os.path.join(self.root, 'train.txt')
        else:
            txt_list = os.path.join(self.root, 'val.txt')
        with open(txt_list) as read:  # strip()删除字符串两端空格
            # xml_list: xml文件的位置列表
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + '.xml')
                             for line in read.readlines()]

        try:
            json_file = open("./VOCdevkit/VOC2012/pascal_voc_classes.json")
            self.class_dict = json.load(json_file)
        except Exception as e:
            print(e)
            exit(-1)

    def __len__(self):
        return len(self.xml_list)

    # 返回索引值idx所对应的图片的xml信息
    def __getitem__(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)

        # 将xml文件解析成多级字典形式
        data = self.parse_xml_to_dict(xml)['annotation']

        # 获取图片路径
        img_path = os.path.join(self.img_root, data['filename'])
        image = Image.open(img_path)
        if image.format != 'JPEG':
            raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowd = []
        for obj in data['object']:
            xmin = float(obj['bndbox']['xmin'])
            xmax = float(obj['bndbox']['xmax'])
            ymin = float(obj['bndbox']['ymin'])
            ymax = float(obj['bndbox']['ymax'])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj['name']])
            iscrowd.append(int(obj['difficult']))

        # 将数据转换为Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])

        # 计算面积
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {'boxes': boxes, 'labels': labels, 'iscrowd': iscrowd, 'image_id': image_id, 'area': area}

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)

        # 将xml文件解析成多级字典形式
        data = self.parse_xml_to_dict(xml)['annotation']

        data_height = int(data['size']['height'])
        data_wight = int(data['size']['width'])

        return data_height, data_wight

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Return:
            Python dictionary holding XML contents
        """
        # 遍历到底层，直接返回tag对应的信息
        if len(xml) == 0:  # 判断是否还有子目录
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历所有标签
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能存在多个，所以要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}


"""
使用Dataloader测试类VOC2012Dataset的情况
"""
import random
import transforms
from draw_box_utils import draw_box
import matplotlib.pyplot as plt
import torchvision.transforms as ts

# read class_indict
category_index = {}
try:
    json_file = open('./VOCdevkit/VOC2012/pascal_voc_classes.json', 'r')
    class_indict = json.load(json_file)
    category_index = {v: k for k, v in class_indict.items()}
except Exception as e:
    print(e)
    exit(-1)

data_transform = {
    'train': transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(0.5)]),
    'val': transforms.Compose([transforms.ToTensor()])
}
# load train data set
train_data_set = VOC2012Dataset(os.getcwd(), data_transform['train'], train_set=True)
print(len(train_data_set))
val_data_set = VOC2012Dataset(os.getcwd(), data_transform['train'], train_set=False)
print(len(val_data_set))
for index in random.sample(range(0, len(train_data_set)), k=5):  # k=5表示一共取5个
    img, target = train_data_set[index]
    img = ts.ToPILImage()(img)
    draw_box(img,
             target['boxes'].numpy(),
             target['labels'].numpy(),
             [1 for i in range(len(target['labels'].numpy()))],
             category_index,
             thresh=0.5,
             line_thickness=5)
    plt.imshow(img)
    plt.show()
