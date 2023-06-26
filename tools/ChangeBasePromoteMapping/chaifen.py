import numpy as np
def chaifen():
    with open("R001C001noMap.txt","r") as f:
        reads_num = f.readlines()
    # length = len(reads)
    # step = 20
    # size = int(length/step)

    with open("R001C001correctACGT.txt","r") as f:
        reads = f.readlines()
        nomappedreads = []
        for ele in reads_num:
            number = int(ele.replace("\n",""))
            nomappedreads.append(reads[number])
            print(ele)

    with open("nomapReads_total.txt",'w') as f:
        f.writelines(nomappedreads)

import os

def split_file(input_file, output_dir, num_files):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    total_lines = len(lines)
    lines_per_file = total_lines // num_files
    remaining_lines = total_lines % num_files

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_files):
        start_idx = i * lines_per_file
        end_idx = start_idx + lines_per_file

        if i == num_files - 1:
            end_idx += remaining_lines

        output_file = os.path.join(output_dir, f'file_{i+1}.txt')

        with open(output_file, 'w') as f:
            f.writelines(lines[start_idx:end_idx])

    print(f'{num_files} files created in {output_dir}.')

input_file = 'input.txt'  # 输入文件路径
output_dir = 'output_files'  # 输出文件夹路径
num_files = 20  # 分割文件数量

split_file("nomapReads_total.txt", "result", num_files)
