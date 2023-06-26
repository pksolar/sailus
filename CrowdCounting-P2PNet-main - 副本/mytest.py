import torch
lista = ["aaa","ddd"]
with open("mytest.list","w")  as f:
    for i in lista:
        f.writelines(i)
with open("mytest.list") as fin:
    for line in fin:
        print(line)
