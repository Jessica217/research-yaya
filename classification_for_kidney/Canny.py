# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
greyImage = cv2.imread('./DMSA_new/0001.jpg', cv2.IMREAD_GRAYSCALE)

# 直方图
def his(greyImage):
    # 计算直方图
    histogram = cv2.calcHist([greyImage], [0], None, [256], [0,256])
    # 直方图均衡化 但是试过了没用
    equ = cv2.equalizeHist(greyImage)
    plt.plot(histogram)
    plt.show()


# Canny算子
def Canny(greyImage):
    gaussian = cv2.GaussianBlur(greyImage, (5, 5), 0)#高斯滤波
    Canny = cv2.Canny(gaussian, 50, 150)
    # 使用cv2.findContours函数寻找边缘
    contours, hierarchy = cv2.findContours(Canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    Contours = cv2.drawContours(greyImage, contours, -1, (0, 255, 0), 2)  # 最后两个参数分别是轮廓的颜色和线宽'''
    cv2.imshow('Canny', Contours)
    cv2.waitKey(0)


# 阈值分割
def threshold(greyImage):
    # 自适应阈值分割
    th_adaptive = cv2.adaptiveThreshold(greyImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)
    # 固定阈值分割
    ret, th = cv2.threshold(greyImage, 140, 255, cv2.THRESH_BINARY )
    cv2.imwrite('his_result.jpg',th)

    #计算阈值分割后的面积
    foreground_area = cv2.countNonZero(th)
    print("前景面积：", foreground_area)

    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 显示图形
    titles = [u'原始图像', u'阈值分割']
    images = [greyImage, th]
    for i in range(2):
        plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

Canny(cv2.imread('his_result.jpg', cv2.IMREAD_GRAYSCALE))