import numpy as np

"""
数：	
a ： array_like

要排序的数组。

axis ： int或None，可选

Axis沿着它排序。如果为None，则在排序之前将数组展平。默认值为-1，它沿最后一个轴排序。

kind ： {‘quicksort’, ‘mergesort’, ‘heapsort’, ‘stable’}，可选

排序算法。默认为'quicksort'。

order ： str或str的列表，可选

当a是一个定义了字段的数组时，此参数指定要比较哪些字段的第一个，第二个等。可以将单个字段指定为字符串，并且不需要指定所有字段，但仍将使用未指定的字段，他们在dtype中出现的顺序，以打破关系。



"""
a =  np.array([[2,1,2],[4,3,2],[1,4,6]])
print(a[np.argsort(a[:,0])])
print(a)