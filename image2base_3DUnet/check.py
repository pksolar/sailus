import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F

# 生成随机的 5x5 张量
tensor = torch.randn(1, 1, 4, 4)

# 进行最大池化操作，使用 2x2 的池化窗口和步长为 2
pooled_tensor = F.max_pool2d(tensor, kernel_size=2, stride=1)

print("原始张量:")
print(tensor)

print("\n池化后的张量:")
print(pooled_tensor)





# a = np.load(r"E:\code\python_PK\img2base_cnn_seg\debug\img_08_R001C001_Cyc048_A_img.tif.npy")
# label = np.load(r"E:\code\python_PK\img2base_cnn_seg\debug\label_08_R001C001_Cyc048_A_img.tif.npy")
# msk = np.load(r"E:\code\python_PK\img2base_cnn_seg\debug\msk_08_R001C001_Cyc048_A_img.tif.npy")
#
# # b = np.load(r"E:\code\python_PK\img2base_cnn_seg\debug\label_08_R001C001_Cyc020_A_img.tif.npy")
# # b_true = np.load(r"E:\data\deep\image2base\single_element\label\08_R001C001_Cyc020_label.npy")
# # if b.all() == b_true.all():
# #     print("yes too")
# # # c = np.load(r"E:\code\python_PK\img2base_cnn_seg\debug\msk_08_R001C001_Cyc018_A_img.tif.npy")
# print("hello world")