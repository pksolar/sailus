import  numpy as np
def simulate(m, n, size=101000):
    count = 0
    x = np.array([1, 0])
    rate = [0.5,0.5]
    # x.extend([0]*5)

    for i in range(size):
        sample = np.random.choice(a=x, size=m, replace=True, p=rate)
        for j in range(m - n):
            if np.sum(sample[j:j + n]) == n:
                count += 1
                break
    print("  hahhdf  ")
    print(count / size)
simulate(10,2)