import fastQ
with open('../fastq/fast21.fq','r') as f:
    lines = f.readlines()
    a = len(lines)
    print(len(a[-3]))
    print(a)
#     linesnew = lines[(219279)*4:]
# with open('../fastq/fast21.6.fq','w') as t:
#     for i in linesnew:
#         t.write(i)
