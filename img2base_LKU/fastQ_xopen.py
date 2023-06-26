#!/usr/bin/env python
# coding=utf-8

#### import Moduules
import sys, os
import numpy as np
import gzip

#f = xopen.xopen(os.path.join(self.outputPath, barcodeInfo[0] + suffix1), "wb", threads=0, compresslevel=3) 写文件的时候，后缀直接写成.gz形式就可以了。


def writeFq(file, FovResult, FOV):
    ## 将call结果输出到fastQ文件中
    with open(file, "ab") as f:
        line_nums = 0
        for line in FovResult:
            f.write(("@_%s_%d\n" % (FOV, line_nums)).encode(encoding="utf-8"))
            f.write((FovResult[line_nums] + "\n").encode(encoding="utf-8"))
            f.write(("+" + "\n").encode(encoding="utf-8"))
            f.write(("?"*(len(FovResult[line_nums])) + "\n").encode(encoding="utf-8"))
            line_nums += 1


if __name__ == "__main__":
    removeAda(r"F:\data\20210923\Lane01_fastq.fq.gz")
