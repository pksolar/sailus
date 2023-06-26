import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import  numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

weights  = [0.48796890185524083,1-0.48796890185524083]
means = np.array([[5.34422196 ,9.83771499], [5.75837464 ,5.0969383 ]]) #在图像上是：y,x

covs = np.array([[[ 4.62670435 ,-0.47305406],
  [-0.47305406  ,2.5968366 ]],
 [[ 4.86713455, -0.68891603],
  [-0.68891603 , 5.21620561]]])


x = np.linspace(14, 0, 100) #11
y = np.linspace(15, 0, 1000)#14
x, y = np.meshgrid(x, y)
xy = np.column_stack([x.flat, y.flat])
w0 = 1
gaussian = np.exp(-((pow(x, 2) + pow(y, 2)) / pow(w0, 2)))

fig = plt.figure()
ax = Axes3D(fig)
t = 0
for i in range(2):
    rv = multivariate_normal(means[i], covs[i])
    z = rv.pdf(xy)
    z = z.reshape(x.shape)
    t = t + z# * weights[i]


    # # 二维面振幅分布图
    # plt.figure()
    # plt.imshow(z)

    # 三维曲面振幅分布图



    ax.plot_surface(x, y, z, cmap='jet',color='black')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
plt.show()
