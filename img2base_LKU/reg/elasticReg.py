# 导包
import pyelastix

import matplotlib.pyplot as plt
import cv2

# 读取数据
im1 = cv2.imread('R001C001_A.tif')
im2 = cv2.imread('R001C001_T.tif')

# 选择一个通道，转为浮点型
im1 = im1[:,:,1].astype('float32')
im2 = im2[:,:,1].astype('float32')

# 调用pyelastix库中的get_default_params方法获取默认参数值，主要参数：
# NumberOfResolutions(int)
# MaximumNumberOfIterations (int)
# MaximumNumberOfIterations (int)等
# 通过 params.变量名来设置参数值
params = pyelastix.get_default_params()
params.NumberOfResolutions = 3
print(params)

# 开始配准，register(浮动图像，参考图像，上面设置的变量)
im3, field = pyelastix.register(im1, im2, params)
cv2.imwrite("result.tif",im3)

# 可视化结果
fig = plt.figure(1);
plt.clf()
plt.subplot(231); plt.imshow(im1)
plt.subplot(232); plt.imshow(im2)
plt.subplot(234); plt.imshow(im3)
plt.subplot(235); plt.imshow(field[0])
plt.subplot(236); plt.imshow(field[1])
# 保存结果
cv2.imwrite('filename.jpg',im3)
# Enter mainloop
if hasattr(plt, 'use'):
    plt.use().Run()  # visvis
else:
    plt.show()  # mpl
