import glob
import subprocess


dict_resize = {       "_ori":[2160,4096],
                      1.15: [2484, 4710],
                        1.2:[2592,4914],
                       1.25:[2700,5120],
                       1.3:[2808,4324],
                       1.35:[2916,5530],
                       1.4:[3024,5734]   }
fastqexe = r'E:\software\sailu\3.10SalusCall_offline_biggerImg.exe'
col = '1' # 有几列
machineNames = ["30_resize"]
for machine_name in machineNames:
    #row and col is related to the machine:
    #read the last imgname
    imgname = glob.glob(rf"E:\data\resize_test\{machine_name}_ori\Lane01\Cyc001\*.tif")[-1].split("\\")[-1].split(".")[0]
    rows = str(int(imgname[1:4]))
    cols = str(int(imgname[5:8]))
    for key,value in dict_resize.items():
        # path_list = [rf'E:\code\python_PK\bleeding\img_process\1.9.39_resize{key}',rf'E:\code\python_PK\bleeding\img_process\1.9.1_resize{key}']
        # res_list = [rf'E:\code\python_PK\bleeding\img_process\1.9.39_resize{key}\res',rf'E:\code\python_PK\bleeding\img_process\1.9.1_resize{key}\res']
        path = rf'E:\data\resize_test\{machine_name}{key}'
        resdir = rf'E:\data\resize_test\{machine_name}{key}/res'

        with open("E:\software\sailu\commen\config.ini") as f:
           res = f.readlines()
           res[6] = f"imWidth={value[1]}\n"
           res[7] = f"imHeight={value[0]}"
        with open("E:\software\sailu\commen\config.ini",'w') as fw:
           for line in res:
              fw.write(line)

        result = subprocess.run(['cmd', '/c', fastqexe,path,
                                 resdir,'1','100',rows,cols,'-m'], stdout=subprocess.PIPE)

        # 输出结果
        #print(result.stdout.decode('gbk'))