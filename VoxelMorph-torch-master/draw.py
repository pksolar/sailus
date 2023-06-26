import numpy as np
import matplotlib.pyplot as plt

# 加载.npy文件中的变形场
displacement_field = np.load(r'E:\code\python_PK\VoxelMorph-torch-master\Result\temp\bspl_final_result\flow_c.npy')

# 提取x方向和y方向的分量
displacement_field_x = displacement_field[0,:, :]
displacement_field_y = displacement_field[1,:, :]

# 创建网格
y, x = np.mgrid[0:displacement_field.shape[1], 0:displacement_field.shape[2]]

# 绘制变形场
plt.figure(figsize=(100, 100))
plt.quiver(x, y, displacement_field_x, displacement_field_y, angles='xy', scale_units='xy', scale=0.1)
plt.gca().invert_yaxis()
plt.title('Displacement Field with Vector Arrows')
# 保存图像
plt.savefig(r'E:\code\python_PK\VoxelMorph-torch-master\Result\temp\bspl_final_result\displacement_field_plot.png', dpi=300)
# plt.show()
