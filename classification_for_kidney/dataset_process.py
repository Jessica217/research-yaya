import pandas as pd
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets
from PIL import Image
import torch.nn as nn
import torch

# 读取csv文件,并将新结果写入txt文件
def read_csv():
    data = pd.read_csv('DMSA.csv')
    labels_left = data['左侧output']
    labels_right = data['右侧output']
    new_data = labels_left.astype('str')+','+labels_right.astype('str')
    picture_path = 'datasets/DMSA_new'
    files = os.listdir(picture_path)

    with open('datasets/label.txt', 'w') as f:
        for index in range(len(files)):
            f.writelines(files[index] +' ' + new_data[index] + '\n')


# 创建自己的dataset数据集
data_path_test = r'./datasets/test'
data_path_train = r'./datasets/train'

label_path_train = r'./datasets/train/label_train.txt'
label_path_test = r'./datasets/test/label_test.txt'


# 自定义Dataset的子类用于train
class My_train_Dataset(Dataset):
    def __init__(self, data_path_train, label_path_train, imgs = []):
        self.imgs = imgs
        self.data_path_train = data_path_train
        datainfo = open(label_path_train, 'r')
        for info in datainfo:
            info = info.strip('\n')
            words = info.split()
            self.imgs.append((words[0], words[1]))
            print(type(words[1]))
            #print(imgs)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img, label = self.imgs[index]
        img = Image.open(self.data_path_train+'/'+img)
        if img.mode != 'L':
            img = img.convert('L')
        #print(img.mode)
        img = transforms.ToTensor()(img)
        return img, label


# 自定义Dataset的子类用于test
class My_test_Dataset(Dataset):
    def __init__(self, data_path_test, label_path_test, imgs = []):
        self.imgs = imgs
        self.data_path_test = data_path_test
        datainfo = open(label_path_test, 'r')
        for info in datainfo:
            info = info.strip('\n')
            words = info.split()
            self.imgs.append((words[0], words[1]))
            #print(imgs)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img, label = self.imgs[index]
        img = Image.open(self.data_path_test+'/'+img)
        if img.mode != 'L':
            img = img.convert('L')
        #print(img.mode)
        img = transforms.ToTensor()(img)
        return img, label


# 训练集
dataset_train = My_train_Dataset(data_path_train, label_path_train)
dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=2, num_workers=0) # shuffle=True表示在每个epoch之后打乱数据
# 测试集
dataset_test = My_test_Dataset(data_path_test, label_path_test)
dataloader_test = DataLoader(dataset_test, shuffle=True, batch_size=2, num_workers=0)


'''# 查看自己的dataloader_train
for img, label in dataloader_train:
    print(img, label)
print("========================================================================================================")
for img, label in dataloader_test:
    print(img, label)'''

# 超参数设置
EPOCH = 100       # 前向后向传播迭代次数
LR = 0.001      # 学习率 learning rate


# 设计分类网络模型
class kidney_classification_CNN(nn.Module):
    def __init__(self):
        super(kidney_classification_CNN, self).__init__()
        # 第一层网络
        self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=1,  # 输入尺寸为1*300*300
                  out_channels=8,
                  kernel_size=5,
                  stride=1,
                  padding=0), # 输出尺寸为8*296*296
        nn.ReLU(),
        nn.MaxPool2d(2)) # 8*148*148

        # 第二层网络
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8,  # 输入尺寸为8*148*148
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=0),  # 输出尺寸为16*144*144
            nn.ReLU(),
            nn.MaxPool2d(2))  # 16*72*72

        # 第三层网络
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16,  # 输入尺寸为16*72*72
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=0),  # 输出尺寸为32*68*68
            nn.ReLU(),
            nn.MaxPool2d(2)) # 32*34*34

        # 第四层网络
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32,  # 输入尺寸为32*34*34
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=0),  # 输出尺寸为64*30*30
            nn.ReLU(),
            nn.MaxPool2d(2)) # 最后输出尺寸为64*15*15

        # 全连接层
        self.out = nn.Linear(64*15*15, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return out


kidney_cnn = kidney_classification_CNN()
optimizer = torch.optim.Adam(kidney_cnn.parameters(), lr=LR)
loss = nn.CrossEntropyLoss() # 定义损失函数

# 训练
for epoch in range(EPOCH):

    for step, (batch_x, batch_y) in enumerate(dataloader_train):
        pred_y = kidney_cnn(batch_x)
        batch_y = torch.tensor(batch_y)
        loss = loss(pred_y, batch_y)
        optimizer.zero_grad()  # 清空上一层梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新优化器的学习率，一般按照epoch为单位进行更新

        if step % 50 == 0:
            test_output = kidney_cnn(dataloader_test)
            pred_y = torch.max(test_output, 1)[1].numpy()  # torch.max(test_out,1)返回的是test_out中每一行最大的数)
            # 返回的形式为torch.return_types.max(
            #           values=tensor([0.7000, 0.9000]),
            #           indices=tensor([2, 2]))
            # 后面的[1]代表获取indices
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())




























