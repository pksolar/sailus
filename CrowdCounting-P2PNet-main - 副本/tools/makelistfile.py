import glob
path_img = glob.glob(r"E:\code\python_PK\CrowdCounting-P2PNet-main\data\noname\img\*.jpg")
path_txt = glob.glob(r"E:\code\python_PK\CrowdCounting-P2PNet-main\data\noname\poslist\*.txt")


with open("data.list","w")  as f:
    for imgpath,txtpath in zip(path_img,path_txt):
        line =  imgpath+" "+txtpath
        print(line)
        f.writelines(line+"\n")




# lista = ["aaa","ddd"]
# with open("mytest.list","w")  as f:
#     for i in lista:
#         f.writelines(i)
# with open("mytest.list") as fin:
#     for line in fin:
#         print(line)
