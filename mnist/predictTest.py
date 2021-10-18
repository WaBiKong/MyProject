from matplotlib import pyplot as plt


def get_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    return [text_labels[int(i)] for i in labels]


def predict(net, test_iter, device=None):
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
    for i in range(X.shape[0]):
        plt.subplot(4, 7, i + 1)
        plt.tight_layout()
        plt.imshow(X[i][0], cmap='gray')
        plt.title(f"Truth: {trues[i]}\nPredict: {preds[i]}")
        plt.xticks([])
        plt.yticks([])
    plt.show()

