# 预测
import torch
from matplotlib import pyplot as plt

def get_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    return [text_labels[int(i)] for i in labels]


# 样本可视化函数
def show_images(imgs, num_rows, num_cols, titles=None, scale=3):
    """Plot a list of image."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def predict(net, test_iter, n, device=None):
    """预测标签。"""
    print('predicting on', device)
    net.to(device)
    for X, y in test_iter:
        if isinstance(X, list):
            # BERT微调
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        break
    trues = get_mnist_labels(y)
    preds = get_mnist_labels(net(X).argmax(axis=1))
    for i in range(n):
        print(f'true is {trues[i]}, pred is {preds[i]}')
    if device == 'cpu':
        titles = ['true= ' + true + '\n' + 'pre= ' + pred for true, pred in zip(trues, preds)]
        show_images(
            X[0:n].reshape((n, 224, 224)), 1, n, titles=titles[0:n]
        )
        plt.show()


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')






