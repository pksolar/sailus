import glob
import fastQ
import numpy as np

dictacgt = {1:"A",2:"C",3:"G",4:"T",5:"N"}


labels = glob.glob(r"E:\code\python_PK\channelAttention-2-finetune\fastq\bingjilabel\Lane01\deepLearnDataUpdate\*\label\*.npy")
mask = np.load(r"E:\code\python_PK\channelAttention-2-finetune\fastq\bingjilabel\Lane01\deepLearnDataUpdate\R001C001_mask.npy")
mask_faltten = mask.flatten()
indice = np.where(mask_faltten !=0)

cycle = 1
size_h = 1360
size_w = 2560

pad_width_img = ((0, 0), (10, 10), (0, 0))
pad_width = ((10, 10), (0, 0))


listacgt  =[]
for labelp in labels:
    cycname = labelp.split("\\")[-3]
    print(cycname)
    label = np.load(labelp)

    label = np.pad(label, pad_width_img, mode='constant')
    msk = np.pad(msk, pad_width, mode='constant')
    h,w = label.shape
    rows = int(h / size_h)
    cols = int(w / size_w)
    label_total = None
    for i in range(rows):
        for j in range(cols):
            print("rows:", i, " cols:", j)
            # input_crop = inputs[:,:,i*size_h:(i+1)*size_h,j*size_w:(j+1)*size_w]
            label_crop = labels[ i * size_h:(i + 1) * size_h, j * size_w:(j + 1) * size_w]
            msk_crop = msk[ i * size_h:(i + 1) * size_h, j * size_w:(j + 1) * size_w]
            try:
                label_total = np.concatenate((label_total, label), axis=0)
            except:
                label_total = label



    for i,ele in enumerate(label_pick):
                if cycname == "Cyc001":  # 说明是第一个cycle，创建包含几个reads的列表 ，
                    listacgt.append(dictacgt[ele])
                else:  # 第二个cycle以后依次写入：3
                    listacgt[i] = listacgt[i] + dictacgt[ele]
    # if cycname == "Cyc010":
    #     break
fastQ.writeFq("fastq/labelfast_fast/"+'Lane01_fastq_notbing.fq', listacgt, 'R001C001')

