import numpy as np
import matplotlib.pyplot as plt

# 生成随机数据
x = np.random.randn(1000)
y = np.random.randn(1000)

# 创建二维直方图
plt.hist2d(x, y, bins=20, cmap='Blues')

# 设置坐标轴标签和标题
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Histogram')

# 添加颜色条
plt.colorbar()

# 展示图像
plt.show()
