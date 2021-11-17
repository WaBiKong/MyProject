# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         train_eval_utils.py
# Description:
# Author:       WaBiKong
# Date:         2021/11/17
# -------------------------------------------------------------------------------

# 对一个批量大小的图片进行打包，将原来的tuple(image, target)拆开，将相同的部分放在一起
# 将batch数量的image打包，batch数量的target打包
def collate_fn(batch):
    return tuple(zip(*batch))