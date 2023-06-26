import  numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
"""
weight 0.48796890185524083
mean: [[5.34422196 9.83771499]
 [5.75837464 5.0969383 ]]
covariance: [[[ 4.62670435 -0.47305406]
  [-0.47305406  2.5968366 ]]

 [[ 4.86713455 -0.68891603]
  [-0.68891603  5.21620561]]]
  
  
  weight [0.4880647 0.5119353]
mean: [[5.34425242 9.83740087]
 [5.7584231  5.09635062]]
covariance: [[[ 4.62663899 -0.47309553]
  [-0.47309553  2.59730442]]

 [[ 4.8672338  -0.68881114]
  [-0.68881114  5.21488296]]]
"""
#x:width, y:height
weights  = [0.48796890185524083,1-0.48796890185524083]
means = np.array([[5.34422196 ,9.83771499], [5.75837464 ,5.0969383 ]]) #在图像上是：y,x

covs = np.array([[[ 4.62670435 ,-0.47305406],
  [-0.47305406  ,2.5968366 ]],
 [[ 4.86713455, -0.68891603],
  [-0.68891603 , 5.21620561]]])

# n_samples = 1000
# sample_labels = np.random.choice(len(weights), size=n_samples, p=weights)
# samples = np.zeros((n_samples, 2))
# for i in range(len(weights)):
#     mask = sample_labels == i
#     samples[mask] = np.random.multivariate_normal(means[i], covs[i], size=np.sum(mask))


x, y = np.meshgrid(np.linspace(0, 14, 100), np.linspace(0, 11, 100))
xy = np.column_stack([x.flat, y.flat])
fig, ax = plt.subplots()
# ax.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.5)


for i in range(2):
    rv = multivariate_normal(means[i], covs[i])
    z = rv.pdf(xy)
    z = z.reshape(x.shape)
    ax.contour(y, x, z, levels=10, colors='black', alpha=0.5)
    ax.text(means[i][0], means[i][1], f"Component {i}", fontsize=10)
plt.show()