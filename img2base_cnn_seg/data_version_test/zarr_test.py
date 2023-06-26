import numpy as np
import  time



# 创建一个4x1000x1000的随机浮点数矩阵
matrix = np.random.randint(0,100,size =  (4, 2160, 4096))

# 将矩阵按行拼接成一个一维数组
matrix_flat = matrix.reshape((-1,))

# 将一维数组保存为二进制文件
with open('matrix.bin', 'wb') as f:
    f.write(matrix_flat.tobytes())

s = time.time()

# 读取二进制文件
with open('matrix.bin', 'rb') as f:
    data = f.read()

# 将二进制数据恢复成一维数组
matrix_flat = np.frombuffer(data, dtype=np.float64)

# 将一维数组恢复成矩阵
matrix = matrix_flat.reshape((4, 2160, 4096))

e = time.time()
print(100*(e-s))
