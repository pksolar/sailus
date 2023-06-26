import numpy as np

# 创建矩阵A和矩阵B
A = np.load(r"E:\data\resize_test\08_resize_ori\res_deep\Lane01\deepLearnData\Cyc010\label\R001C001_label.npy")
B = np.load(r"E:\data\resize_test\08_resize_ori\res\Lane01\deepLearnData\Cyc010\label\R001C001_label.npy")


dic_ = {1:'A',2:'C',3:"G",4:"T"}

# 交互循环
while True:
    # 获取用户输入
    input_str = input("请输入坐标位置（例如：1,2）：")

    # 解析用户输入
    coordinates = [int(coord) for coord in input_str.split(",")]

    # 根据坐标位置获取对应矩阵的值
    value_A = A[coordinates[0], coordinates[1]]
    value_B = B[coordinates[0], coordinates[1]]

    # 输出结果
    print("map矫正后的值为：{}".format( dic_[value_A]))
    print("raw的值为：{}".format( dic_[value_B]))
