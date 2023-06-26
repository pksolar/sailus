import glob
import subprocess


dict_resize = {       "_ori":[2160,4096],
                      1.15: [2484, 4710],
                        1.2:[2592,4916],
                       1.25:[2700,5120],
                       1.3:[2808,4324],
                       1.35:[2916,5530],
                       1.4:[3024,5734]   }
value = dict_resize["_ori"]
fastqexe = r'E:\software\sailu\3.10SalusCall_offline_biggerImg.exe'
col = '1' # 有几列
path = r"E:\code\python_PK\VoxelMorph-torch-master\reg\phase_imgRound_08_512_true"
rows =  '1'
cols = '1'
resdir = path + "/res"
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