import json
import fastQ
import  os
with open("../ACGT08_3_onebyone.json","r") as fp:
    listacgt = json.load(fp)
fastQ.writeFq('../fastq/fast08_3_2013.fq',listacgt,'ROO1C001')

