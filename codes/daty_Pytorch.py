import torch.nn as nn
import torch.nn

import torchvision.models.detection.faster_rcnn as fasterRCNN


# 搭建小丫自己的神经网络
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(3, 21, 3, 1, 0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 1)

        # 第二个卷积
        self.conv2 = nn.Conv2d(3, 21, 3, 1, 0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 1)

        self.dense1 = nn.Linear(in_features=21, out_features=128) # 线性变换
        self.dense2 = nn.Linear(in_features=128, out_features=10) # 无bias偏差 即y = kx

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.dense1(x)
        x = self.dense2(x)

        return x


model = MyCNN()
print(model)






