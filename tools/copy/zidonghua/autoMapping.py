"""
E:\software\sailu\WinMappingScript\software\7z.exe x E:\code\python_PK\bleeding\img_process\08_resizex1.15\res\Lane01\Lane01_fastq.fq.gz -y -oE:\code\python_PK\bleeding\img_process\08_resizex1.15\res\Lane01\

E:\software\sailu\WinMappingScript\software\bowtie2-align-s.exe --very-fast -p 10 -q E:\code\python_PK\bleeding\img_process\08_resizex1.15\res\Lane01\Lane01_fastq.fq -S E:\code\python_PK\bleeding\img_process\08_resizex1.15\res\Lane01\Lane01_fastq.fq.sam  -x E:\software\sailu\WinMappingScript\reference\Ecoli.fa_index

python E:\software\sailu\WinMappingScript\samStaterThree.py E:\code\python_PK\bleeding\img_process\08_resizex1.15\res\Lane01\Lane01_fastq.fq.sam E:\code\python_PK\bleeding\img_process\08_resizex1.15\res\Lane01\Lane01_fastq.fq_total.out --splitFov E:\code\python_PK\bleeding\img_process\08_resizex1.15\res\Lane01\Lane01_fastq.fq_split.out

"""

dict_resize = {       "_ori":[2160,4096],
                      1.15: [2484, 4710],
                        1.2:[2592,4914],
                       1.25:[2700,5120],
                       1.3:[2808,4324],
                       1.35:[2916,5530],
                       1.4:[3024,5734]   }
exe1 = r"E:\software\sailu\WinMappingScript\software\7z.exe x"
exe2  = r"E:\software\sailu\WinMappingScript\software\bowtie2-align-s.exe --very-fast -p 10 -q"
exe3 = "python E:\software\sailu\WinMappingScript\samStaterThree.py"
machineNames = ["30_resize"]
rootDir = r"E:\data\resize_test"

import os

# 存储路径的变量
for machine_name  in machineNames:
    for key,value in dict_resize.items():
        run_version = fr"{machine_name}{key}"
        dir_path = fr'{rootDir}\{run_version}\res\Lane01\\'
        fastq_file = fr'{rootDir}\{run_version}\res\Lane01\Lane01_fastq.fq'
        gz_file =fr'{rootDir}\{run_version}\res\Lane01\Lane01_fastq.fq.gz'
        sam_file = fr'{rootDir}\{run_version}\res\Lane01\Lane01_fastq.fq.sam'

        stater_file =fr'{rootDir}\{run_version}\res\Lane01\Lane01_fastq.fq_total.out'
        split_file = fr'{rootDir}\{run_version}\res\Lane01\Lane01_fastq.fq_split.out'

        print("resize:",key)

        index_file = r'E:\software\sailu\WinMappingScript\reference\Ecoli.fa_index'
        # 第一条命令
        cmd1 = fr'E:\software\sailu\WinMappingScript\software\7z.exe x {gz_file} -y -o{dir_path}'
        os.system(cmd1)

        # 第二条命令
        cmd2 = fr'E:\software\sailu\WinMappingScript\software\bowtie2-align-s.exe --very-fast -p 10 -q {fastq_file} -S {sam_file} -x {index_file}'
        os.system(cmd2)

        # 第三条命令
        cmd3 = fr'python E:\software\sailu\WinMappingScript\samStaterThree.py {sam_file} {stater_file} --splitFov {split_file}'

        os.system(cmd3)
