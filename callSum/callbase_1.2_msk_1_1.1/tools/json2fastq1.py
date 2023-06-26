import json
import fastQ
import  os
with open("../ACGT08.json","r") as fp:
    listacgt = json.load(fp)
fastQ.writeFq('../fastq/fast08_1.fq',listacgt,'ROO1C001')

