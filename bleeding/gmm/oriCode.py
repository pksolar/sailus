import  numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
num_m = 10000
mu_m = 1.71
sigma_m = 0.056
rand_data_m = np.random.normal(mu_m,sigma_m,num_m)
y_m = np.ones(num_m)


num_w = 10000
mu_w = 1.58
sigma_w = 0.051
rand_data_w = np.random.normal(mu_w,sigma_w,num_w)
y_w = np.zeros(num_w)

data = np.append(rand_data_m,rand_data_w)
data = data.reshape(-1,1)
y = np.append(y_m,y_w)
print(data)
print(y)

from scipy.stats import multivariate_normal

num_iter = 1000
n,d = data.shape
mu1 = data.min(axis = 0)
mu2 = data.max(axis = 0)
sigma1 = np.identity(d)
sigma2 = np.identity(d)
pi = 0.5

for i in range(num_iter):
    norm1 = multivariate_normal(mu1,sigma1)
    norm2 = multivariate_normal(mu2, sigma2)
    tau1 = pi * norm1.pdf(data)
    tau2 = (1-pi) * norm2.pdf(data)
    gamma =tau1/(tau2+tau1)

    mu1 = np.dot(gamma,data)/np.sum(gamma)
    mu2 = np.dot(1-gamma,data)/np.sum(1-gamma)

    sigma1 = np.dot(gamma*(data-mu1).T,data-mu1)/np.sum(gamma)

    sigma2 = np.dot((1-gamma) * (data - mu2).T, data - mu2) / np.sum(1-gamma)

    pi = np.sum(gamma) / n

print("\n pi:",pi)
print("\n mu1:",mu1)
print("\n mu2:",mu2)
print("\n simgma:",sigma1," ",sigma2)

