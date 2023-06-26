import fastQ
with open("readlone.txt",'r') as f:
    new_list = []
    listacgt = f.readlines()
    for i in listacgt:
        new_list.append(i.replace("\n",""))
save_path = "08h/"
fastQ.writeFq(save_path+'Lane01_fastq.fq', new_list, 'R001C001')