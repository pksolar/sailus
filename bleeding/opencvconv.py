import cv2
import numpy as np
"G"
kernel = np.array([[  -0.5, -0.5,   -0.5],
          [-0.5,  4, -0.5],
          [  -0.5, -0.5,   -0.5]])

#
# kernel_1 = np.array([[  -0.77, -0.55,   -0.77],
#                     [-0.55,  5.30, -0.55],
#                      [  -0.77, -0.55,   -0.77]])
kernel_1 = np.array([[  -0.6, -0.85,   -0.6],
                    [-0.85,  6.05, -0.85],
                     [  -0.6, -0.85,   -0.6]])





kernel_5 =  np.array([[ 0  ,-0.1,  -0.2,  -0.1,  0],
                      [-0.1,-0.5, -0.75,  -0.5,-0.1],
                      [-0.2,-0.75,  6.6  ,  -0.75,-0.2],
                      [-0.1,-0.5  , -0.75, -0.5,-0.1],
                      [ 0  ,-0.1,  -0.2,  -0.1,  0]])
kernel_5_zero =  np.array([[-0.1, -0.2, -0.3, -0.2, -0.1],
                      [-0.1,  -0.3,     -0.5,   -0.3,  -0.1 ],
                      [-0.3,  -0.5,  5.75,  -0.5,  -0.3],
                      [-0.1, -0.3,   -0.5,     -0.3,  -0.1],
                      [-0.1, -0.2, -0.3, -0.2, -0.1]])


kernel_5_2 =  np.array([[-0.1, -0.2, -0.3, -0.2, -0.1],
                      [-0.1,  -0.3,     -0.5,   -0.3,  -0.1 ],
                      [-0.3,  -0.5,5.5,  -0.5,  -0.5],
                      [-0.1, -0.3,   -0.5,     -0.3,  -0.1],
                      [-0.1, -0.2, -0.3, -0.2, -0.1]])

#
# kernel_7_seed =  np.array([[-0.1, -0.2,  -0.2,  -0.4],
# #                           [-0.15,-0.25, -0.3,-0.35],
# #                           [-0.2, -0.3, -0.45,  -0.5],
# #                           [-0.25,-0.35,-0.5,center],
#
#                            ])
#
# kernel_7 = np.zeros((7,7))[]



print(np.sum(kernel_5_2))

kernel_my =  np.array([[ -0.1, -0.15,  -0.1],
          [-0.15,  1.5, -0.15],
          [ -0.1, -0.15,  -0.1]])

far = -0.55
near = (1.414) * far
center = 0.35-4*(near+far)

kernel_w_ad = np.array([[ far, near, far],
          [near, center, near],
          [ far, near, far]])
print(kernel_w_ad)

center_ = kernel_w_ad[1,1]
near_ = -kernel_w_ad[1,0]
far_ = -kernel_w_ad[0,0]

kernel_w = np.array([[ -0.75, -0.5, -0.75],
          [-0.5,  5.25, -0.5],
          [ -0.75, -0.5, -0.75]])


kernel_test = np.array([[ -0.1,-0.2,  -0.3, -0.2, -0.1],
                        [ -0.2,-0.3,  -0.5, -0.3,-0.2],
                        [-0.3, -0.5,  6.4, -0.5,-0.3],
                        [ -0.2,-0.3,  -0.5, -0.3,-0.2],
                        [ -0.1,-0.2,  -0.3, -0.2, -0.1]])

print(np.sum(kernel_5_zero))

# print(np.sum(blockImage3))
kernel_zero_5 = np.zeros((5,5))

kernel_w_5 = kernel_zero_5.copy()
kernel_w_5[1:-1,1:-1] = kernel_w


a = kernel_w_5.T.reshape(-1,1)
np.savetxt("E:\software\sailu5.25_highDense_bleed5\kernel.txt",a,fmt='%.4f')




input = cv2.imread("imgtest/G1.tif",0)
input2 = cv2.imread("imgtest/G1_ori.tif",0)
dst2 = cv2.filter2D(input,-1,kernel_w_5)
dst3 = cv2.filter2D(input,-1,kernel_w)
dst4 = cv2.filter2D(input,-1,kernel_5_zero)
dst5 = cv2.filter2D(input2,-1,kernel_5_2).astype(float)

dst5 = cv2.GaussianBlur(dst5,(3,3),0.9).clip(0,255).astype(np.uint8)
# cv2.imshow("dst",dst2)


cv2.imwrite("imgtest/dst_my_{:.3f}_{:.3f}_{:.3f}.jpg".format(center_,near_,far_),dst2)

cv2.imwrite("imgtest/dst_winner_ori.jpg",dst3)
cv2.imwrite("imgtest/kernel_5_4_-0.4.jpg",dst4)
cv2.imwrite("imgtest/kernel_5_2.jpg",dst5)
cv2.waitKey(0)
