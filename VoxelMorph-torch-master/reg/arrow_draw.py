import numpy as np
import matplotlib.pyplot as plt

# 构造变形场
field = np.load(r"E:\code\python_PK\VoxelMorph-torch-master\Result\temp\bspl_final_result/flow_G.npy")

import numpy as np
import matplotlib.pyplot as plt

# 创建模拟变形场数据
# field = np.random.rand(2, 512, 512) * 10

# 提取每隔10像素的数据点
x_spacing = np.arange(0, 4096, 100)
y_spacing = np.arange(0, 2160, 100)
X, Y = np.meshgrid(x_spacing, y_spacing)
U = field[0, y_spacing, :][:, x_spacing]
V = field[1, y_spacing, :][:, x_spacing]

# 绘制矢量场
fig, ax = plt.subplots()
ax.quiver(X, Y, V, U, angles='xy', scale_units='xy', scale=0.1)

# 设置坐标轴范围
ax.set_xlim(0, 4096)
ax.set_ylim(0, 2190)

# 保存图像
plt.savefig("deformation_field_arrows_vector_field_G.png", dpi=1200)
