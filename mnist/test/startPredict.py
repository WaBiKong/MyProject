from predict import *
from Net.LeNet import *
from Net.AlexNet import *
from data.load_data_mnist import *
import predictTest

pre_iter = lada_pre_data(28, resize=224)


# 模型加载
net_state_dict = torch.load('../Net/AlexNet.pt')
new_net = returnAlexNet()
new_net.load_state_dict(net_state_dict)

# predict(new_net, pre_iter, 6, 'cpu')
predictTest.predict(new_net, pre_iter, try_gpu())