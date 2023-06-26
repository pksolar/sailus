import threading
import os
import subprocess
import fastQ
import threading
def autoMap(ori_mappingdir, fastq_name):
    # run_version = fr"{machine_name}{key}"
    dir_path = fr'{ori_mappingdir}\\'
    fastq_file = fr'{ori_mappingdir}\\{fastq_name}.fq'
    gz_file = fr'{ori_mappingdir}\\{fastq_name}.fq.gz'
    sam_file = fr'{ori_mappingdir}\\{fastq_name}.fq.sam'

    stater_file = fr'{ori_mappingdir}\\\{fastq_name}.fq_total.out'
    split_file = fr'{ori_mappingdir}\\\{fastq_name}.fq_split.out'

    index_file = r'C:\software\sailu\WinMappingScript\reference\Ecoli.fa_index'
    # 第一条命令
    cmd1 = fr'C:\software\sailu\WinMappingScript\software\7z.exe x {gz_file} -y -o{dir_path}'
    os.system(cmd1)

    # 第二条命令
    cmd2 = fr'C:\software\sailu\WinMappingScript\software\bowtie2-align-s.exe --very-fast -p 10 -q {fastq_file} -S {sam_file} -x {index_file}'
    os.system(cmd2)

    # 第二条命令
    cmd2 = fr'C:\software\sailu\WinMappingScript\software\bowtie2-align-s.exe --very-fast -p 10 -q {fastq_file} -S {sam_file} -x {index_file}'
    os.system(cmd2)
    process = subprocess.Popen(cmd2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    output, error = process.communicate()

    try:
        readsNum = float(error.split(" ")[0])
        if "100.00% overall" in error:
            # # result = error.split(" ")[-4][-6:-1]
            # result = float(result)
            print("!!!!!!!!!!!!!!!!!!!!mapping: ", result)
            return 1
        else:
            return 0
    except:
        print("!!!!may be can not detect any clusters!!!")
        return 0
        # result = 0
        # print("")

    # os.system(cmd3)


def change(baseType="A"):
    # 读取txt
    with open("nomapReads.txt", "r") as f:
        reads = f.readlines()
    # 改变txt
    for j, read in enumerate(reads):
        # 第j个read
        for i in range(100):
            print("reads:", j, "  cycle:", i)
            # 第i个基因
            if read[i] != baseType:
                new_read = read[:i] + baseType + read[i + 1:]
                # print(new_read)
                new_list = [new_read.replace("\n", "")]
                old_list = [read.replace("\n", "")]
                fastQ.writeFq('08h/' + 'Lane01_fastq.fq', new_list, 'R001C001')
                fastQ.writeFq('08h/' + 'old_Lane01_fastq.fq', old_list, 'R001C001')
                result = autoMap("08h", "Lane01_fastq")
                if result != 0:
                    print("i,j", i, " ", j)
                    string_result = rf"reads:{j},cycle:{i}"
                    with open("changeMapped.txt", "a") as f:
                        f.writelines(string_result + "\n")
                        f.writelines("ori:" + read + "\n")
                        f.writelines("new:" + new_read + "\n")
                    break
                # autoMap("08h", "old_Lane01_fastq")

threads = []
# create threads for each channel
threadA = threading.Thread(target=change, args=("A"))
threadC = threading.Thread(target=change, args=("C"))
threadG = threading.Thread(target=change, args=("G"))
threadT = threading.Thread(target=change, args=("T"))

# start threads
threadA.start()
threadC.start()
threadG.start()
threadT.start()

# add threads to the list
threads.append(threadA)
threads.append(threadC)
threads.append(threadG)
threads.append(threadT)

# wait for all threads to finish
for thread in threads:
    thread.join()
