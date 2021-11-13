import torch
from matplotlib import pyplot as plt

import numpy
from PIL import Image

from mnist.Net.AlexNet import *


def get_mnist_labels(i):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    return text_labels[i]


# 模型加载
net_state_dict = torch.load('../Net/AlexNet.pt')
new_net = returnAlexNet()
new_net.load_state_dict(net_state_dict)
i = 1
while i < 12:
    # 要识别的图片
    myImage = f'../data/myImage/{i}.jpg'
    # 读取图片数据
    im = Image.open(myImage).resize((224, 224))
    # 改为灰度图
    im = im.convert('L')
    # 将图片转换为numpy数组
    im_data = numpy.array(im)
    # 将numpy数组转换为张量tensor
    im_data = torch.from_numpy(im_data).float()
    plt.imshow(im_data, cmap='gray')

    im_data = im_data.view(1, 1, 224, 224)

    preds = get_mnist_labels(new_net(im_data).argmax(axis=1))
    plt.title(f'The predict is: {preds}\n')
    plt.show()
    print('The predict is:', preds)
    i += 1


