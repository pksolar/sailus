import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import  cv2
#读取right和raw两个坐标
right = np.load(r"E:\data\resize_test\08_resize_ori\res_deep\Lane01\deepLearnData\Cyc010\label\R001C001_label.npy")
right_same = right.copy()
right_same[right_same == 0] = 10
raw = np.load(r"E:\data\resize_test\08_resize_ori\res\Lane01\deepLearnData\Cyc010\label\R001C001_label.npy")
raw_same = raw.copy()
raw_same[raw_same == 0] = 20
diff = np.where(right != raw )
same = np.where(right_same  == raw_same)
#读取配准后的4张图：
a_img  = cv2.imread(r"E:\data\resize_test\08_resize_ori\res_deep\Image\Lane01\Cyc010\R001C001_A.jpg",0)
c_img  = cv2.imread(r"E:\data\resize_test\08_resize_ori\res_deep\Image\Lane01\Cyc010\R001C001_C.jpg",0)
g_img  = cv2.imread(r"E:\data\resize_test\08_resize_ori\res_deep\Image\Lane01\Cyc010\R001C001_G.jpg",0)
t_img  = cv2.imread(r"E:\data\resize_test\08_resize_ori\res_deep\Image\Lane01\Cyc010\R001C001_T.jpg",0)
# 生成随机数据

a_mismatch_ins = a_img[diff[0],diff[1]]
c_mismatch_ins = c_img[diff[0],diff[1]]
g_mismatch_ins = g_img[diff[0],diff[1]]
t_mismatch_ins = t_img[diff[0],diff[1]]
print("hh")
# c =
# g =
# t =
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.hist2d(a_mismatch_ins,t_mismatch_ins,bins=65, norm=matplotlib.colors.LogNorm(), density=True, cmap="jet")



# 设置坐标轴标签和标题
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Histogram')

# 添加颜色条
# plt.colorbar()

# 展示图像
plt.show()
