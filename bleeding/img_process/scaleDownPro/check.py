import cv2
import numpy as np

# 加载图像
image = cv2.imread('ori.tif', cv2.IMREAD_GRAYSCALE)

# 创建带有负号的拉普拉斯核
laplacian_kernel = np.array([[0.01134374 ,0.08381951, 0.01134374],
 [0.08381951, -1.31934703 ,0.08381951],
 [0.01134374, 0.08381951 ,0.01134374]])  #0.61934703



# 对图像应用拉普拉斯核
filtered_image = cv2.filter2D(image, -1, -laplacian_kernel)

# 显示原始图像和增强后的图像
cv2.imwrite("lap_inv.tif",filtered_image)