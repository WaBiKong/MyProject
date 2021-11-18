# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         split_data.py
# Description:  生成 train.txt 和 eval.txt 文件
# Author:       WaBiKong
# Date:         2021/11/17
# -------------------------------------------------------------------------------

import os
import random


def main():
    random.seed(0)  # 设置随机种子，保证随机结果可复现

    # 获取拥有图片名称的文件路径
    files_path = "./data/VOCdevkit/VOC2012/JPEGImages"
    assert os.path.exists(files_path), f'path:{files_path} does not exist.'

    val_rate = 0.5  # 设置验证集和训练集的比例为0.5

    # 获取图片的名称并排序，例如 2007_000027.jpg 变为 2007_000027
    # os.listdir(path)读取path路径文件夹下的文件名
    files_name = sorted([file.split('.')[0] for file in os.listdir(files_path)])
    files_num = len(files_name)  # 图片总数量
    # 随机获取验证集中图片序号，数量为 files_num*val_rate
    val_index = random.sample(range(0, files_num), k=int(files_num * val_rate))

    train_files = []
    val_files = []

    # 遍历所有图片
    for index, file_name in enumerate(files_name):
        # 如果图片序号在随机获取的验证集序号中，则把图片名字写入val_files
        if index in val_index:
            val_files.append(file_name)
        else:
            train_files.append(file_name)

    try:
        # 创建文件 train.txt 和 val.txt
        train_f = open('./data/VOCdevkit/VOC2012/train.txt', 'w')
        eval_f = open('./data/VOCdevkit/VOC2012/val.txt', 'w')
        # 列表中的每个元素用'\n'拼接
        train_f.write('\n'.join(train_files))
        eval_f.write('\n'.join(val_files))
    except FileExistsError as e:
        print(e)
        exit(1)


if __name__ == '__main__':
    main()