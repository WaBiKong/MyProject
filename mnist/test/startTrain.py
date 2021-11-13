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
train_iter, test_iter = load_data_mnist(batch_size, 224)
net = returnAlexNet()
lr, num_epochs = 0.01, 10

# # 查看数据集每个批量所含图片数量，图片大小，labels数量
# batch_size = 28
# train_iter, test_iter = load_data_mnist(batch_size)
# for i, (X, y) in enumerate(train_iter):
#     print(X.shape)
#     print(y.shape)
#     break

train(net, train_iter, test_iter, num_epochs, lr, try_gpu())


# 模型保存
torch.save(net.state_dict(), '../Net/AlexNet.pt')
