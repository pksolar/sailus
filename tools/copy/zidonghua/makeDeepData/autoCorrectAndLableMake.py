import os
machineName = "R001C001"
namelsit = ["R002C064","R003C085"]
for name in namelsit:
    print(name)
    machineName = name
    rootdir = fr"E:\data\highDensity\dense0.6\{machineName}\\"
    resdir = rootdir+"res_deep"
    ori_mappingdir = fr"E:\data\highDensity\dense0.6\{machineName}\res"

    # # 步骤1，运行baseCall，为了生成用来mapping的文件fastq以及生成配准后的图像。
    step1_command = rf'E:\software\sailu\SalusCall_deeplearnTrue.exe {rootdir} {resdir} 1 97 1 1 -m'
    os.system(step1_command)

    # 步骤2，
    step2_command = rf'python E:\CrosstalkTool\fastqCorrect.py {ori_mappingdir}\Lane01\Lane01_fastq.fq.sam {ori_mappingdir}\Lane01\Lane01_fastq.fq.gz {ori_mappingdir}\Lane01\samResult'
    os.system(step2_command)

    # 步骤3
    step3_command = rf'python E:\CrosstalkTool\generData_PengKuanKuan.py {ori_mappingdir}\Lane01\sfile {ori_mappingdir}\Lane01\sfile {ori_mappingdir}\Lane01\samResult {resdir}\Lane01\deepLearnData'
    os.system(step3_command)

    # 步骤4
    step4_command = r'echo Step 4 is complete'
    os.system(step4_command)

#然后再用这个文件夹下的copy将copy到c盘。

