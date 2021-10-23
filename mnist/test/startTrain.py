from mnist.data.load_data_mnist import *
from mnist.Net.LeNet import *
from mnist.Net.AlexNet import *
from mnist.train import *
import matplotlib.pyplot as plt

# # LeNet.
# batch_size = 256
# train_iter, test_iter = load_data_mnist(batch_size, 28)
# net = returnAlexNet()
# lr, num_epochs = 0.9, 10

# AlexNet
batch_size = 28
train_iter, test_iter = load_data_mnist(batch_size, 28)
net = returnAlexNet()
lr, num_epochs = 0.01, 10

for i, (X, y) in enumerate(test_iter, start=1):
    for a in range(X.shape[0]):
        plt.subplot(4, 7, a + 1)
        plt.tight_layout()
        plt.imshow(X[a][0], cmap='gray')
        plt.title("Ground Truth: {}".format(y[a]))
        plt.xticks([])
        plt.yticks([])
    plt.show()
    break

# train(net, train_iter, test_iter, num_epochs, lr, try_gpu())
#
# # 模型保存
# torch.save(net.state_dict(), '../Net/AlexNet.pt')
