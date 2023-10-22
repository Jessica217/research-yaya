from PIL import Image
import os

import cv2


def read_path(file_path):
    for filename in os.listdir(file_path):

        img = cv2.imread(file_path+filename)
        print(img.size)
        # 获取图像长宽
        height, width, _ = img.shape

        left_x = width // 2
        top_y = 0
        right_x = width
        bottom_y = height
        new_picture = img[top_y:bottom_y, left_x:right_x]

        cv2.imwrite('D:/DMSA（EKYY）_new_right/'+filename, new_picture)

read_path('.DMSA（EKYY）')
