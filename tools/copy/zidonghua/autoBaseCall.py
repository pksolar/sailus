import subprocess


dict_resize = {       1.2:[2592,4914],
                       1.25:[2700,5120],
                       1.3:[2808,4324],
                       1.35:[2916,5530],
                       1.4:[3024,5734]   }
fastqexe = r'E:\software\sailu\3.10SalusCall_offline_biggerImg.exe'
col = '3' # 有几列


for key,value in dict_resize.items():
    path_list = [rf'E:\code\python_PK\bleeding\img_process\1.9.39_resize{key}',rf'E:\code\python_PK\bleeding\img_process\1.9.1_resize{key}']
    res_list = [rf'E:\code\python_PK\bleeding\img_process\1.9.39_resize{key}\res',rf'E:\code\python_PK\bleeding\img_process\1.9.1_resize{key}\res']
    for path, resdir in zip(path_list, res_list):
        with open("E:\software\sailu\commen\config.ini") as f:
           res = f.readlines()
           res[6] = f"imWidth={value[1]}\n"
           res[7] = f"imHeight={value[0]}"

        with open("E:\software\sailu\commen\config.ini",'w') as fw:
           for line in res:
              fw.write(line)

        result = subprocess.run(['cmd', '/c', fastqexe,path,
                                 resdir,'1','100','1',col,'-m'], stdout=subprocess.PIPE)

        # 输出结果
        #print(result.stdout.decode('gbk'))