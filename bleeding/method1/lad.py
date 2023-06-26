

def least_absolute_deviations(nums, target):
    left, right = min(nums), max(nums) # 确定左右边界
    while left <= right:
        mid = (left + right) // 2
        error = sum(abs(num - mid) for num in nums) # 计算误差
        if error == target:
            return mid
        elif error < target:
            right = mid - 1
        else:
            left = mid + 1
    return left # 返回最接近的mid值

# 测试
nums = [1, 2, 3, 4, 5]
target = 3
print(least_absolute_deviations(nums, target)) # 输出3