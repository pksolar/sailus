import Levenshtein
import time

a = "abcaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
b = "bbbaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
s  = time.time()
for j in range(1,100000):
    n = sum(k != l for k, l in zip(a, b))
    # diff = Levenshtein.hamming(a, b)
e = time.time()

#

print(e -s)
print(len(a))

print(n)