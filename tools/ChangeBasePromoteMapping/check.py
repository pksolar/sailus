import os

def autoMap(ori_mappingdir,fastq_name):
    #run_version = fr"{machine_name}{key}"
    dir_path = fr'{ori_mappingdir}\\'
    fastq_file = fr'{ori_mappingdir}\\{fastq_name}.fq'
    gz_file = fr'{ori_mappingdir}\\{fastq_name}.fq.gz'
    sam_file = fr'{ori_mappingdir}\\{fastq_name}.fq.sam'

    stater_file = fr'{ori_mappingdir}\\\{fastq_name}.fq_total.out'
    split_file = fr'{ori_mappingdir}\\\{fastq_name}.fq_split.out'


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

autoMap("08h", "Lane01_fastq")