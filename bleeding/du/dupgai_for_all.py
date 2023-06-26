#!/usr/bin/env python3

#################################################################################
## Copyright: Salus-bio Corp 2022
## Author: Huihua Xia
## Date of creation: 6/15/2022
#
## Project: Quality estimation of sequencing data
## Description
## - Project: duplication
#
## - Goal: This script is to count the duplicated reads of all reads of input fastq file.
#
#################################################################################

import os
import gzip
import pandas as pd
import argparse


def dup(odir: str, sample_id: str, fqgz: str, n_num: int = 0):
    """
    Count numbers of duplicated reads.

    Parmas:
      odir: output dir for results of duplication.
      sample_id: sample id for the output file.
      fqgz: input fastq.gz file.
      n_num: input int, 0 indicates do not filter sequences containing N,
        1/2/3/... indicates filter to retain sequences containing < 1/2/3/... N.

    Return: None
    """
    # odir
machineNames = ["30_resize", "1.9.1_resize", "1.9.39_resize", "22_resize", "08_resize",
                "17_R1C78_resize"]  # "1.9.1_resize","1.9.39_resize",

dict_resize = {"_ori": [2160, 4096],
               1.15: [2484, 4710],
               1.2: [2592, 4914],
               1.25: [2700, 5120],
               1.3: [2808, 4324],
               1.35: [2916, 5530],
               1.4: [3024, 5734]}
odir  = r'E:\code\python_PK\bleeding\du'

    # remove N ?
for machine_name in machineNames:
    for key ,value in dict_resize.items():
        try:
            seq_f = fr"E:\data\resize_test\{machine_name}{key}\res\Lane01\Lane01_fastq.fq"
            sample_id = rf'{machine_name}{key}_dup'
            # count read numbers of each unique read
            seq_counts = {}
            total_read_num = 0
            with open(seq_f) as fh:
                    for i, line in enumerate(fh):
                        if i % 4 == 1:
                            total_read_num += 1
                            seq = line.strip()
                            seq_counts[seq] = seq_counts.get(seq, 0) + 1

            # extract duplicated reads
            dupseq_counts = {}
            dup_read_num = dup_read_nums = 0
            for k, v in seq_counts.items():
                if v > 1:
                    dup_read_num += 1
                    dup_read_nums += v
                    dupseq_counts[k] = v

            # duplication rate
            # total_read_num = total_read_num / 4
            dup_rate1 = dup_read_num / total_read_num * 100
            dup_rate2 = dup_read_nums / total_read_num * 100
            dup_rate_res = f"{odir}/{sample_id}.dup_rate.txt"
            os.system(f'echo "Sample_id: {sample_id}" > {dup_rate_res}')
            os.system(f'echo "Total_reads: {total_read_num}" >> {dup_rate_res}')
            os.system(f'echo "Unique_duplicated_reads: {dup_read_num}" >> {dup_rate_res}')
            os.system(f'echo "All_duplicated_reads: {dup_read_nums}" >> {dup_rate_res}')
            os.system(f'echo "Unique_duplication_rate(%): {dup_rate1}" >> {dup_rate_res}')
            os.system(f'echo "All_duplication_rate(%): {dup_rate2}" >> {dup_rate_res}')

            # save duplicated reads and their read numbers
            df = pd.DataFrame.from_dict(dupseq_counts, orient="index")
            df.columns = ["read_numbers"]
            df["read_numbers/all_dup_read_numbers(%)"] = (
                df["read_numbers"] / dup_read_nums * 100
            )
            ## keep 3 significant digits
            new_ratio = []
            for i in range(len(df)):
                new_r = float("{:.3}".format(df.iloc[i, 1]))
                new_ratio.append(new_r)
            df['read_numbers/all_dup_read_numbers(%)'] = new_ratio
            ## sort and save
            df.index.name = "read_seq"
            df = df.sort_values(["read_numbers"], ascending=False)
            dup_res = f"{odir}/{sample_id}.dup_read_numbers.txt"
            df.to_csv(dup_res, sep="\t")
        except:
            print(f"error{key}")


if __name__ == "__main__":
    description = (
        "This script is to count the duplicated reads of all reads of input fastq file."
    )
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-o", "--odir", default=None, help="output dir for results of duplication."
    )
    parser.add_argument(
        "-s", "--sample_id", default=None, help="sample id for the output file."
    )
    parser.add_argument("-f", "--fqgz", default=None, help="input fastq.gz file.")
    parser.add_argument(
            "-n",
            "--n_num",
            default=0,
            help="input int, 0 indicates do not filter sequences containing N, 1/2/3/... indicates filter to retain sequences containing < 1/2/3/... N.")
    args = parser.parse_args()
    dup(args.odir, args.sample_id, args.fqgz, args.n_num)
