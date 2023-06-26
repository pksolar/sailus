import matplotlib.pyplot as plt

# 创建一个包含两个子图的图像
fig, axes = plt.subplots(nrows=1, ncols=2)

# 绘制第一个子图
axes[0].plot([1, 2, 3, 4], [1, 4, 2, 3])
axes[0].set_title('Subplot 1')

# 绘制第二个子图
axes[1].plot([1, 2, 3, 4], [3, 2, 4, 1])
axes[1].set_title('Subplot 2')

# 展示图像
plt.show()