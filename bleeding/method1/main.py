# This is a sample Python script.
import numpy as np

p = 101
cycle = 101 # cycle which is used for trainning.
m = 20 # m clusters in one block
t = np.zeros((m,m),dtype=float)
T = np.zeros((m,cycle,4),dtype=float) #  T is the final intensity matrix before bleeding correction.
S = np.zeros((m,cycle,4),dtype=float) #  S is the label,the nominal channel is one,otherwise :zero
C = np.eye(m) #bleeding coefficient
intensity = np.zeros((m,cycle,4),dtype=float) # lamda   intensity*S is the pure signal
T1 = np.expand_dims(T,0).repeat(m,axis=0)
T2 = np.expand_dims(T,1).repeat(m,axis=1)
pureT = T1-t*T2 # element_size

def impurity(T,cycle):
   for i in range(cycle):



