import cv2

# 读取图像
image = cv2.imread(r'E:\data\resize_test\08_resize_ori\Lane01\Cyc001\R001C001_A.tif',0)

# 进行均值滤波
filtered_image = cv2.blur(image, (3, 3))  # 使用5x5的滤波器
cv2.imwrite("scaleDownPro/filtered_image.tif",filtered_image)
# 显示滤波后的图像
# cv2.imshow('Filtered Image', filtered_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
