import numpy as np


def rec(m=10,n=2, p=0.5): #
    l = [0] * n
    l[-1] = 1


    for _ in range(n,m):
        tmp = float(1*sum(l))
        print(tmp)
        del l[0]
        l.append(tmp)
        print(l)
        print(len(l))
    return l[-1],l[-1]/(float(2**m))
a = rec()
print(a)
# print(99.5**32)
# print(/)

