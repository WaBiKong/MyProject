from mnist import predictTest
from mnist.predict import *
from mnist.Net.LeNet import *
from mnist.Net.AlexNet import *
from mnist.data.load_data_mnist import *

pre_iter = lada_pre_data(28, resize=224)


# 模型加载
net_state_dict = torch.load('../Net/AlexNet.pt')
new_net = returnAlexNet()
new_net.load_state_dict(net_state_dict)

# predict(new_net, pre_iter, 6, 'cpu')
predictTest.predict(new_net, pre_iter, try_gpu())

# 测试
print("asd")