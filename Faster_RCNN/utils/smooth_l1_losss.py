# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         smooth_l1_losss.py
# Description:  边界框回归损失函数smooth_l1_loss
# Author:       WaBiKong
# Date:         2021/11/22
# -------------------------------------------------------------------------------

import torch


def smooth_l1_loss(input, target, beta: float = 1. / 9, size_average: bool = True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    # cond = n < beta
    cond = torch.lt(n, beta)
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()