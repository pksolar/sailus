# li = ['aaa','bbb','ccc']
# with open(r'eng.txt', 'w') as f:
#     for i in li:
#         f.write(i + '\n')
import numpy as np
import json
name = ['joker','joe','nacy']
filename = 'name.json'
with open(filename,'w') as file_obj:
    json.dump(name,file_obj)
with open(filename) as f:
    names = json.load(f)
print(names[0])


# a = ["aaa","bbb"]
# astr = str(a)
# file = open("astr.txt",'w')
# file.write(astr)
# file.close()
# files = open("astr.txt","r")
# l = files.read()
# files.close()
# print(l[0])
# listastr = str(lista)
# file = open("lista.txt",'w')
# file.write(listastr)
# file.close()
# with open(r'eee.txt','w') as f:
#        for i in range(100):
#                a = np.random.rand(10)
#                # a = a.tolist()
#                stra = ["{:.4f}".format(x) for x in a]
#                if i==0:
#                     f.write("A B C D\n")
#                f.write(" ".join(stra)+'\n')
#                f.write("\n")



