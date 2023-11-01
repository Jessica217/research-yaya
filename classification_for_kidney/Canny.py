# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('./DMSA/0001.jpg')
lenna_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 灰度化处理图像
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 高斯滤波降噪
gaussian = cv2.GaussianBlur(grayImage, (5, 5), 0)

# Canny算子
Canny = cv2.Canny(gaussian, 50, 150)

'''# 使用cv2.findContours函数寻找边缘
contours, hierarchy = cv2.findContours(Canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#
Contours = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)  # 最后两个参数分别是轮廓的颜色和线宽'''

#阈值分割
import cv2


# 阈值分割
ret, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
'''cv2.imshow('thresh', th)
cv2.waitKey(0)'''

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示图形
titles = [u'原始图像', u'Canny算子']
images = [lenna_img,th ]
for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()