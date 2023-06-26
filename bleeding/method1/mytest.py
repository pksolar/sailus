import  numpy as np
# a = np.array([4,3,2,5])
# a_ = np.expand_dims(a,0).repeat(4,0)
# print(a_)
# a2 = np.expand_dims(a,1).repeat(4,1)
# print(a2)
# print(a-0.1*a2)

def recurrence(m=4, n=2):
	# 通项公式的最开始几项
    if m < n:
        return 0
    if m == n:
        return 1
    l = [0]*n

    l[-1] = 1
    multi = 1
    # 不断向前递推
    for _ in range(n, m):
        multi *= 2
        print("mul:",multi)
        tmp = sum(l)+multi#  前面的和加起来，再加这次的情况数。
        print("tmp:",tmp)
        print("l_before: ",l)
        del l[0]
        l.append(tmp)
        print("l_after: ",l)
        print("-------------------------------")
    return l[-1], l[-1]/(2**m)
a = recurrence(10,2)
print(a)