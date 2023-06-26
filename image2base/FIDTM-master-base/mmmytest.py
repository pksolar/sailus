import numpy as np

C = np.array([[1,0.1,0.2],[0.2,1,0.15],[0.3,0.25,1]])
# S = np.array([100,100,100])
C_inv  = np.linalg.inv(C)
S = np.array([130,135,155])
print(np.dot(C*S))