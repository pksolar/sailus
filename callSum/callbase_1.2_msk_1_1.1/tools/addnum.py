import os

# coding: UTF-8  #设置编码
ff = open('../fast3.fq','w')  #打开一个文件，可写模式
with open('../fast2.fq','r') as f:  #打开一个文件只读模式
    lines = f.readlines() #读取文件中的每一行，放入line列表中
    for i,line in enumerate(lines):
        if (i+1) % 4 == 0:
            line_new =line.replace('\n','') #将换行符替换为空('')
            line_new=line_new+r'?'+'\n'  #行末尾加上"|",同时加上"\n"换行符
            #print(line_new)
        else:
            line_new = line

        ff.write(line_new) #写入一个新文件中
