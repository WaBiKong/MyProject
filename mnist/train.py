import torch
from matplotlib import pyplot as plt
from torch import nn

import Animator
import Accumulator


def train(net, train_iter, test_iter, num_epochs, lr, device=None):
    """用GPU训练模型"""

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    # 使用小批量随机梯度下降法
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc'])
    num_batches = len(train_iter)
    for epoch in range(num_epochs):
        metric = Accumulator.Accumulator(3)
        net.train()
        # enumerate()函数，返回枚举对象，包括下标和值，此处为下标i、值(X, y)
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss{train_l:.3f}, tarin acc{train_acc:.3f},test acc{test_acc:.3f}')
    plt.show()


# 定义准确率计算函数
def accuracy(y_hat, y):
    """计算预测正确的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 使用argmax获得每行中最大元素的索引来获得预测类别
        y_hat = y_hat.argmax(axis=1)
    cmp = astype(y_hat, y.dtype) == y  # cmp.shape == y.shape == size(n)
    return float(reduce_sum(astype(cmp, y.dtype)))  # sum计算cmp中元素为1的总和，即正确数量


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模型
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator.Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            # BERT微调
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
