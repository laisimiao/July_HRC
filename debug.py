# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 05:27:58 2019

@author: laisimiao
"""

import os
import torch as t
import pandas as pd
from torchvision import models
from tqdm import tqdm  #这是什么，进度条库
from torch.autograd import Variable
from mnist_models import conv_net,to_image,fc_net,AlexNet

METHOD = 'conv'
TYPE = 'cla'
BATCH_SIZE = 500
use_gpu = t.cuda.is_available()

test = pd.read_csv('./data/test_july.csv')
test = test.drop('index',axis=1)
test_data = t.from_numpy(test.values).float()

# 初始化模型
if METHOD == 'conv':
    test_data = to_image(test_data)
    net = conv_net()
elif METHOD == 'fc':
    test_data = to_image(test_data)
    net = fc_net()
elif METHOD == 'res':
    # 使用resnet18进行迁移学习，微调参数，如果冻结参数，将resnet作为特征选择器的话，训练速度更快。
    # 因为resnet参数过多，不建议使用CPU运算，使用Xeon E5620一个EPOCH要训练三个小时
    # 没试过这样写res是不是可以
    test_data = to_image(test_data)
    net = models.resnet18(pretrained=True)
    # 固定参数
    for p in net.parameters():
        p.requires_grad = False

    # 因为MNIST图片是单通道，并且尺寸较小，所以需要对resnet进行一些细节修改
    net.conv1 = t.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=3,
                           bias=False)
    net.maxpool = t.nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
    net.avgpool = t.nn.AvgPool2d(5, stride=1)

    num_ftrs = net.fc.in_features
    net.fc = t.nn.Linear(num_ftrs,10)

elif METHOD == 'alex':
    test_data = to_image(test_data)
    net = AlexNet()

else:
    raise Exception("Wrong Method!")
    
    
if use_gpu:
    net = net.cuda()

    
if os.path.exists('./models/MNIST/%s.pth' % METHOD):
    try:
        net.load_state_dict(t.load('./models/MNIST/%s.pth' % METHOD))
    except Exception as e:
        print(e)
        print("Parameters Error")


print('=======Predicting========')
submission = pd.read_csv("./data/sample_submission.csv")
# 切换成验证模式，验证模式下DROPOUT将不起作用
net.eval()

if use_gpu:
    test_data = Variable(test_data.cuda())
else:
    test_data = Variable(test_data)

result = t.Tensor().cuda()

index = 0

# 分段进行预测，节省内存 10000/500=20
for i in tqdm(range(int(test_data.shape[0]/BATCH_SIZE)),total=int(test_data.shape[0]/BATCH_SIZE)):
    label_prediction = net(test_data[index:index+BATCH_SIZE])
    index += BATCH_SIZE
    result = t.cat((result,label_prediction),0)

# 结果处理
if TYPE == 'cla':
    _,submission['label'] = t.max(result.data.cpu(),1) # t.max返回一个元祖，第一个元素是最大元素值，第二个元素是最大元素位置
elif TYPE == 'reg':
    submission['label'] = submission['label'].astype('int')
    submission['label'] = submission['label'].apply(lambda x:9 if x>= 10 else x)


submission.to_csv("submission.csv",index=False)  #不保存行索引

print("\ncreate submission csv file successfully!")
