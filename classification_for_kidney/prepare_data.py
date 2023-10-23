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



# 读取csv文件,并将新结果写入txt文件
def read_csv():
    data = pd.read_csv('DMSA.csv')
    labels_left = data['左侧output']
    labels_right = data['右侧output']
    new_data = labels_left.astype('str')+''+labels_right.astype('str')

    picture_path = 'datasets/DMSA_new'
    files = os.listdir(picture_path)
    with open('datasets/label.txt', 'w') as f:
        for index in range(len(files)):
            f.writelines(files[index] +' ' + new_data[index] + '\n')








