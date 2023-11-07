import pandas as pd
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
from torchvision import datasets
from PIL import Image
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torchvision.models import alexnet


# 设计分类网络
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
        self.fc1 = nn.Linear(64*15*15, out_features=4)
        #self.fc2 = nn.Linear(in_features=128, out_features=4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        #x = self.fc2(x)
        return x

# 重写AlexNet网络，以适应输入图像
class CustomAlexNet(nn.Module):
    def __init__(self, num_classes=4):
        super(CustomAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),  # 将输入通道数调整为1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 加深网络层数
class YeNetB(nn.Module):

    def __init__(self, num_classes=4):
        super(YeNetB,self).__init__()
        self.conv1 = nn.Conv2d(1,48,3)
        self.conv6 = nn.Conv2d(48,96,3)
        self.conv2 = nn.Conv2d(96,128,3)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(128,256,3)
        self.conv4 = nn.Conv2d(256,384,3)
        self.bn1 = nn.BatchNorm2d(384)
        self.conv5 = nn.Conv2d(384,256,3)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(256*3*3,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,num_classes)
        self.dropout = nn.Dropout2d(0.5)

    def forward(self,x):
        x = self.conv1(x)#30*30*48
        x = F.relu(x)
        x = self.conv6(x)#28*28*96
        x = F.relu(x)
        x = self.conv2(x)#26*26*128
        x = F.relu(x)
        x = self.pool1(x)#13*13*128

        x = self.conv3(x)#11*11*256
        x = F.relu(x)
        x = self.conv4(x)#9*9*384
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv5(x)#7*7*256
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)#3*3*256

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除去批处理维度的其他所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class YeNet(nn.Module):

    def __init__(self, num_classes=4):
        super(YeNet,self).__init__()
        self.conv1 = nn.Conv2d(1,16,5)
        self.conv2 = nn.Conv2d(16,32,3)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(32,64,5)
        self.conv4 = nn.Conv2d(64,96,3)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(470400,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,num_classes)

    def forward(self,x):
        x = self.conv1(x)#28*28*48
        x = F.relu(x)
        x = self.conv2(x)#26*26*96
        x = F.relu(x)
        x = self.pool1(x)#13*13*96

        x = self.conv3(x)#9*9*128
        x = F.relu(x)
        x = self.conv4(x)#7*7*256
        x = F.relu(x)
        x = self.pool2(x)#3*3*256

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除去批处理维度的其他所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# 创建自己的dataset数据集
data_path_test = r'./datasets_for_CNN/test'
data_path_train = r'./datasets_for_CNN/train'

label_path_train = r'./datasets_for_CNN/train/label_train.txt'
label_path_test = r'./datasets_for_CNN/test/label_test.txt'


# 自定义Dataset的子类用于train
class MyDataset(Dataset):
    def __init__(self, data_path, label_path, imgs = []):
        self.imgs = imgs
        self.data_path = data_path
        self.label_path = label_path
        datainfo = open(label_path, 'r')
        mapping_dict = {"0": 0, "1": 1} # 表示2个类别
        for info in datainfo:
            info = info.strip('\n')
            words = info.split()
            self.imgs.append((words[0], mapping_dict[words[1]]))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img, label = self.imgs[index]
        img = Image.open(self.data_path+'/'+img)
        if img.mode != 'L':
            img = img.convert('L')
        img = transforms.ToTensor()(img)
        img = transforms.RandomHorizontalFlip(p=0.5)(img) # 随机翻转图像
        return img, label


# 训练集
dataset_train = MyDataset(data_path_train, label_path_train)
dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=8, num_workers=0) # shuffle=True表示在每个epoch之后打乱数据

# 测试集
#dataset_test = My_Dataset(data_path_test, label_path_test)
#dataloader_test = DataLoader(dataset_test, shuffle=True, batch_size=2, num_workers=0)


'''# 查看自己的dataloader_train
for img, label in dataloader_train:
    print(img, label)
print("========================================================================================================")
for img, label in dataloader_test:
    print(img, label)'''

# 超参数设置
EPOCH = 100       # 前向后向传播迭代次数
LR = 0.005      # 学习率 learning rate

# 初始化自定义的AlexNet模型
custom_alexnet = CustomAlexNet(num_classes=2)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(custom_alexnet.parameters(), lr=LR, momentum=0.9)


# 使用CUDA加速以GPU进行训练，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 将模型加载到GPU上
custom_alexnet.cuda()

# 训练并计算损失、准确度
losses = []
accuracies = []
for epoch in range(EPOCH):
    epoch_loss = 0
    correct_predictions = 0
    total_samples = 0

    for step, (batch_x, batch_y) in enumerate(dataloader_train):
        # 清空上一层梯度
        optimizer.zero_grad()
        # 将输入数据加载到GPU上
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()

        pred_y = custom_alexnet(batch_x)
        batch_y = torch.tensor(batch_y)
        # 计算损失
        loss_current = criterion(pred_y, batch_y)
        loss_current.backward()  # 反向传播
        optimizer.step()  # 更新优化器的学习率，一般按照epoch为单位进行更新
        epoch_loss += loss_current.item()
        # 计算准确度
        predictions = torch.argmax(batch_y)
        correct_predictions += (predictions == batch_y).sum().item()
        total_samples += batch_y.size(0)

    # 计算平均损失
    epoch_loss /= len(dataloader_train)
    # 计算准确度
    accuracy = correct_predictions / total_samples  # 计算准确度

    losses.append(epoch_loss)
    accuracies.append(accuracy)

    '''#if step % 50 == 0:
            #test_output = kidney_cnn(dataloader_test)
            #pred_y = torch.max(test_output, 1)[1].numpy()  # torch.max(test_out,1)返回的是test_out中每一行最大的数)
            # 返回的形式为torch.return_types.max(
            #           values=tensor([0.7000, 0.9000]),
            #           indices=tensor([2, 2]))
            # 后面的[1]代表获取indices'''

    print(f'Epoch [{epoch + 1}/{EPOCH}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy * 100:.2f}%')


# 可视化损失曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(accuracies, marker='o', color='r')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')

plt.tight_layout()
plt.show()






























