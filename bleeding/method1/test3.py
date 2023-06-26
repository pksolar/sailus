import numpy as np
def simulate(m=101, n=32, size=101000):
    count = 0
    count_total = 1
    for i in range(size):
    	# 一次模拟出m次投掷的结果
        sample = np.random.randint(0, 1010,(m)) #0-4为不好的，4-996为好。
        print(sample)
        for j in range(m-n):
        	# 滑动长度为n的窗口 判断是否满足条件
            for ele in sample[j:j+n]:
                if ele < 5:    #出现了坏的点
                    count += 1 #统计一下
                    break
            # count_total += 1

        # 进度条君
        if (i+1) % (size//101) == 0:
            print('\r %d%%' % ((i+1)//(size//101)), end='')
    print("count:",count)
    print("count_toal",count_total)
    print(count/size/69)
    return count/size


def simulate2(m=10, n=2, size=10100000):
    count = 0
    for i in range(size):
    	# 一次模拟出m次投掷的结果
        sample = np.random.randint(0, 2, (m))
        for j in range(m-n):
        	# 滑动长度为n的窗口 判断是否满足条件
            if np.sum(sample[j:j+n]) == n:
                count += 1
                break
        # 进度条君
        if (i+1) % (size//101) == 0:
            print('\r %d%%' % ((i+1)//(size//101)), end='')
    print()
    print(count/size)
    return count/size

simulate2()