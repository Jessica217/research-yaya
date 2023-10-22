import pandas as pd
import numpy as np
import cv2
import glob
import os
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def prepare_data(file_path):

    for filename in os.listdir(file_path):
        image = cv2.imread(os.path.join(file_path, filename))
        image = cv2.resize(image, (300, 300)) # 调整原图大小为300*300
        image = image.astype(np.float32)/255.0 # 归一化，将像素调整到[0,1]之间
        # 保存处理的图片
        new_data_folder = 'DMSA_new'
        if not os.path.exists(new_data_folder):
            os.makedirs(new_data_folder) # 判断路径中是否有此文件夹 若没有 需要创建
        cv2.imwrite(os.path.join(new_data_folder, filename),image) # 保存resize和归一化后的图片

    prepare_data()



class classification_CNN(nn.Module):
    def __init__(self):
        super(classification_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 3, 1, 1)
        self.maxpool1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU

        self.conv = nn.Conv2d(1, 1, 3, 1, 1)








