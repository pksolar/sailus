import numpy as np
import cv2
import glob
def PCA(data, r):  # .参数r代表降低到r维
    data = np.float32(np.mat(data))  # 转化为矩阵，32bits单精度浮点数，
    rows, cols = np.shape(data)  # 读取矩阵形状
    # data_mean = np.mean(data, 0)  # 对列求平均值，0计算列，1计算行
    a = data - 0#np.tile(data_mean, (rows, 1))  # 优化目标，将所有样例减去对应均值得到a
    # 用np.tile()将data_mean矩阵沿行复制
    cov = np.dot(a.T, a)  # 得到协方差矩阵
    eig_val, eig_vec = np.linalg.eig(cov)  # 求协方差矩阵的特征值和特征向量
    v_r = eig_vec[:, 0:r]  # 按列取前r个特征向量，就是主成分分析的解
    #v_r = np.dot(a, v_r)  # 小矩阵特征向量向大矩阵特征向量过度
    for i in range(r):
        v_r[:, i] = v_r[:, i] / np.linalg.norm(v_r[:, i])  # 特征向量归一化，减少不同样本之间的差异性
        # norm求范数，默认2-范数
    final_data = np.dot(a, v_r)
    final_data = np.array(final_data)
    return final_data, v_r

paths = glob.glob("data/base_images/R001C001_*.jpg")
img_list = []
for path in paths:
    img = cv2.imread(path,0)[:,:,np.newaxis]
    img_list.append(img)
img_4 = np.concatenate(img_list,axis=2)
img_4_flatten = img_4.reshape(-1,4)
img_,v_r=PCA(img_4_flatten,3)
#数据归一化：
ymax = 255
ymin = 0
xmax = np.max(img_)
xmin = np.min(img_)
img_norm = ((ymax-ymin)*(img_-xmin)/(xmax-xmin))
img_final = img_norm.reshape(2160,4096,3).astype(np.uint8)
cv2.imshow("after",img_final)
cv2.imwrite("3channel.jpg",img_final)
cv2.waitKey(0)

print("img_4_flatten.shap")


