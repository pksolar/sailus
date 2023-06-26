import numpy as np
baseTrans = {"A" : 1, "C" : 2, "G" : 3, "T" : 4, "N" : 5}
callResultOri=["ACGATCGAGCACACTCAAGGCAGACGTAGCGA","ACGATCGAGCACACTCAAGGCAGACGTAGCGA","ACGATCGAGCACACTCAAGGCAGACGTAGCGA"]
t = baseTrans[callResultOri[1][1]]
print(t)