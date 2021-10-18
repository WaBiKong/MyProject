from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms


def get_dataloader_workers():
    return 0


# trans = transforms.ToTensor()
#
#
# # 下载MNIST数据集
# def download():
#     mnist_pre = torchvision.datasets.MNIST(
#         root="../data", train=False, transform=trans, download=True
#     )
#     mnist_train = torchvision.datasets.MNIST(
#         root="../data", train=True, transform=trans, download=True
#     )
#     mnist_test = torchvision.datasets.MNIST(
#         root="../data", train=False, transform=trans, download=True
#     )
#     return mnist_pre, mnist_train, mnist_test
#
#
# mnist_pre, mnist_train, mnist_test = download()
#
#
# def data_resize(resize=None):
#
#     mnist_pre[1][0].resize((1, resize, resize))
#     mnist_train[1][0].resize((1, resize, resize))
#     mnist_test[1][0].resize((1, resize, resize))
#
#
# # 将数据集加载到内存中
# def load_data_mnist(batch_size):
#     return (DataLoader(mnist_train, batch_size, shuffle=True,
#                        num_workers=get_dataloader_workers()),
#             DataLoader(mnist_test, batch_size, shuffle=False,
#                        num_workers=get_dataloader_workers()))
#
#
# def lada_pre_data(batch_size):
#     return DataLoader(mnist_pre, batch_size, shuffle=True,
#                       num_workers=get_dataloader_workers())


# 下载MNIST数据集，然后将其加载到内存中
def load_data_mnist(batch_size, resize=None):
    """下载MNIST数据集，然后将其加载到内存中。"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.MNIST(
        root="../data", train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.MNIST(
        root="../data", train=False, transform=trans, download=True
    )
    return (DataLoader(mnist_train, batch_size, shuffle=True,
                       num_workers=get_dataloader_workers()),
            DataLoader(mnist_test, batch_size, shuffle=False,
                       num_workers=get_dataloader_workers()))


def lada_pre_data(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_pre = torchvision.datasets.MNIST(
        root="../data", train=False, transform=trans, download=True
    )
    return DataLoader(mnist_pre, batch_size, shuffle=True,
                      num_workers=get_dataloader_workers())

