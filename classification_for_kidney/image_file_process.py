import pandas as pd
import cv2
import glob
import os


def prepare_data(file_path):

    for filename in os.listdir(file_path):
        print(filename)
        image = cv2.imread(os.path.join(file_path, filename))
        #print(os.path.join(file_path, filename))

        color_image = cv2.resize(image, (300, 300))
        # 将彩色图像转换为灰度图像
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        # 将像素值归一化到[0, 1]范围内
        normalized_gray_image = gray_image.astype(float) / 255.0
        # 保存处理的图片
        new_data_folder = 'DMSA_new'
        if not os.path.exists(new_data_folder):
            os.makedirs(new_data_folder) # 判断路径中是否有此文件夹 若没有 需要创建
        cv2.imwrite(os.path.join(new_data_folder, filename), normalized_gray_image*255) # 保存resize和归一化后的图片


prepare_data('./DMSA')


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








