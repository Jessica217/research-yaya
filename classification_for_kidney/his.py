import cv2

# 读取图像
image = cv2.imread('./DMSA_new/0001.jpg', cv2.IMREAD_GRAYSCALE)

# 应用直方图均衡化
equalized_image = cv2.equalizeHist(image)

# 保存均衡化后的图像
#cv2.imwrite('output.jpg', equalized_image)

# 显示原始图像和均衡化后的图像（可选）
cv2.imshow('Original Image', image)
cv2.imshow('Equalized Image', equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
