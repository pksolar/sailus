import cv2
import numpy as np
"G"
# kernel = np.array([[ 0.1256, -1.0250,  0.1383],
#           [-1.0273,  4.1715, -1.0278],
#           [ 0.1485, -1.0157,  0.1418]])
#
kernel_5 =  np.array([[ 0.0187,-0.6683,  -0.5481,-0.6683,0.0187],
                      [-0.2752,-1.0000, -1.0000,-1.0000,-0.2752],
                      [ -0.4813, -0.5458,  3.9332,-0.5458,-0.4813],
                      [-0.2752, -1.0000, -1.0000, -1.0000, -0.2752],
                      [ 0.0187,-0.6683,  -0.5481,-0.6683,0.0187]])
#
# kernel_my =  np.array([[ -0.1, -0.15,  -0.1],
#           [-0.15,  1.5, -0.15],
#           [ -0.1, -0.15,  -0.1]])
#
# far = -0.5
# near = (1.414) * far
# center = 0.35-4*(near+far)
#
# kernel_w_ad = np.array([[ far, near, far],
#           [near, center, near],
#           [ far, near, far]])
# print(kernel_w_ad)
#
# center_ = kernel_w_ad[1,1]
# near_ = -kernel_w_ad[1,0]
# far_ = -kernel_w_ad[0,0]
#
# kernel_w = np.array([[ -0.5, -0.75, -0.5],
#           [-0.75,  5.25, -0.75],
#           [ -0.5, -0.75, -0.5]])
#
#
# kernel_test = np.array([[ -0.1,-0.2,  -0.3, -0.2, -0.1],
#                         [ -0.2,-0.3,  -0.5, -0.3,-0.2],
#                         [-0.3, -0.5, 6.4, -0.5,-0.3],
#                         [ -0.2,-0.3,  -0.5, -0.3,-0.2],
#                         [ -0.1,-0.2,  -0.3, -0.2, -0.1]])
#
# print(np.sum(kernel_5))
#
# # print(np.sum(blockImage3))
# kernel_zero_5 = np.zeros((5,5))
#
# kernel_w_5 = kernel_zero_5.copy()
# kernel_w_5[1:-1,1:-1] = kernel_w_ad
kernelA = np.array([[ -0.5, -0.75, -0.5],
           [-0.75,  5.25, -0.75],
           [ -0.5, -0.75, -0.5]])

kernelC = np.array([[ -0.5, -0.75, -0.5],
           [-0.75,  5.25, -0.75],
           [ -0.5, -0.75, -0.5]])


kernelG = np.array([[ -0.5, -0.75, -0.5],
           [-0.75,  5.25, -0.75],
           [ -0.5, -0.75, -0.5]])


kernelT = np.array([[ -0.5, -0.75, -0.5],
           [-0.75,  5.25, -0.75],
           [ -0.5, -0.75, -0.5]])

a = kernelA.T.reshape(-1,1)
c = kernelC.T.reshape(-1,1)
g = kernelG.T.reshape(-1,1)
t = kernelT.T.reshape(-1,1)
np.savetxt("E:\software\sailu\kernelA.txt",a,fmt='%.4f')
np.savetxt("E:\software\sailu\kernelC.txt",c,fmt='%.4f')
np.savetxt("E:\software\sailu\kernelG.txt",g,fmt='%.4f')
np.savetxt("E:\software\sailu\kernelT.txt",t,fmt='%.4f')

# input = cv2.imread("imgtest/G1.tif",0)
# dst2 = cv2.filter2D(input,-1,kernel_w_5)
# dst3 = cv2.filter2D(input,-1,kernel_w)
# dst4 = cv2.filter2D(input,-1,kernel_my)
# # cv2.imshow("dst",dst2)
# cv2.imwrite("imgtest/dst_my_{:.3f}_{:.3f}_{:.3f}.jpg".format(center_,near_,far_),dst2)
#
# cv2.imwrite("imgtest/dst_winner_ori.jpg",dst3)
# cv2.imwrite("imgtest/1_total.jpg",dst4)
# cv2.waitKey(0)
