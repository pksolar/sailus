def listtest(l):
    for i in range(4):
        l.append(i)
    return l.copy()

l =  []
a = listtest(l.copy())
b = listtest(l.copy())
print(a)
print(b)