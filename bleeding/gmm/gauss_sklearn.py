from sklearn.mixture import GaussianMixture
import cv2
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
"1819,1448"
img = cv2.imread("R001C001_A.tif",0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel).astype(float)[1442:1453,1814:1828]
print(np.sum(img)*0.0475 * 0.48796890185524083) # 0.0247
data_list = []
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        for k in range(int(img[i,j])):
            data_list.append([i,j])
data = np.array(data_list)

# cv2.imwrite("img.jpg",img)
# cv2.imshow("img",img)
# cv2.waitKey(0)

g  = GaussianMixture(n_components = 2,covariance_type = 'full',tol = 1e-6,max_iter = 1000)
g.fit(data)
print("weight",g.weights_)
print("mean:",g.means_)
print("covariance:",g.covariances_)


x,y = multivariate_normal(mean=g.means_,cov = g.covariances_,size = 1000).T
plt.plot(x,y,'ro')
plt.show()
