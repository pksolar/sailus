import numpy as np
with open("R001C001noMap.txt","r") as f:
    a = f.readlines()
    b = a[:100]
with open("R001C001correctACGT.txt","r") as f:
    reads = f.readlines()
    nomappedreads = []
    for ele in b:
        number = int(ele.replace("\n",""))
        nomappedreads.append(reads[number])
        print(ele)

with open("nomapReads.txt",'w') as f:
    f.writelines(nomappedreads)

