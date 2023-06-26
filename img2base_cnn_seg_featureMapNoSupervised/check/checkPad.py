import numpy as np

# 假设你的数据为 data，维度为 (12, 2700, 5120)
data = np.zeros((12, 2700, 5120))  # 这里使用全零数据作为示例

# 指定填充宽度为 (0, 2)，即在宽度上左右各填充 0 和 2 个像素
pad_width = ((0, 0), (2, 2), (0, 0))

# 使用 np.pad 对数据进行填充
padded_data = np.pad(data, pad_width, mode='constant')

# 输出填充后的数据形状
print(padded_data.shape)