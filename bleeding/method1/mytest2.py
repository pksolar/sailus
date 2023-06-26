def recurrence(m=10, n=2):
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
        tmp = sum(l)+multi
        del l[0]
        l.append(tmp)
    return l[-1], l[-1]/(2**m)
print(recurrence(3,2))