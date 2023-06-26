import torch
import glob
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.utils.data as data
from datasets_ import Dataset_npy, Dataset_npy_val
import os
from natsort import natsorted
import numpy as np
from mymodel import DNA_Sequencer_Atten
import fastQ
import time
import subprocess
import deepMappingUpdateDataset_forMatrix


def autoMap(ori_mappingdir):
    #run_version = fr"{machine_name}{key}"
    dir_path = fr'{ori_mappingdir}\\'
    fastq_file = fr'{ori_mappingdir}\\Lane01_fastq.fq'
    gz_file = fr'{ori_mappingdir}\\Lane01_fastq.fq.gz'
    sam_file = fr'{ori_mappingdir}\\Lane01_fastq.fq.sam'

    stater_file = fr'{ori_mappingdir}\\Lane01_fastq.fq_total.out'
    split_file = fr'{ori_mappingdir}\\Lane01_fastq.fq_split.out'


    index_file = r'E:\software\sailu\WinMappingScript\reference\Ecoli.fa_index'
    # 第一条命令
    cmd1 = fr'E:\software\sailu\WinMappingScript\software\7z.exe x {gz_file} -y -o{dir_path}'
    os.system(cmd1)

    # 第二条命令
    cmd2 = fr'E:\software\sailu\WinMappingScript\software\bowtie2-align-s.exe --very-fast -p 10 -q {fastq_file} -S {sam_file} -x {index_file}'
    os.system(cmd2)

    # 第三条命令
    cmd3 = fr'python E:\software\sailu\WinMappingScript\samStaterThree.py {sam_file} {stater_file} --splitFov {split_file}'

    os.system(cmd3)

    process = subprocess.Popen(cmd2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    output, error = process.communicate()

    try:
        result = error.split(" ")[-4][-6:-1]
        savetxtdir = ori_mappingdir.replace("Lane01","")
        with open(rf"{savetxtdir}/{result}.txt","w") as f:
            f.write("xxx")
        # newname = ori_mappingdir.replace("-",rf"-map{result}")
        # os.rename(rf"fastq/{ori_mappingdir}",rf"fastq/{newname}")
    except:
        print("wrong done")



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        # print("self.avg:",self.avg)


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir + filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))


def main():
    dataset_name = '08'

    dictacgt = {1: 'A', 2: "C", 3: "G", 4: "T"}
    rate = 1
    """
    输入的数据is npy : num x cycle x channel     25w x 100 x 4 
    把 25w 分成n个图, 1000 x  250  切成如此。
    #不会在外面设计数据。

    two direction


    """
    save_dir = 'savedir_pth/'
    best_acc = 0
    batchsize = 1
    height = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dir_name = r"E:\data\liangdujuzhen\img\\"

    val_path = dir_name + "08h_R001C001_withNoMap.npy"
    fov = "08h_R001C001"
    val_label_path = val_path.replace("img","label")
    val_data = np.load(val_path)
    val_label = np.load(val_label_path)

    num2, cycle_num2, channel2 = val_data.shape
    num_pic2 = np.floor(num2 / height).astype(int)

    #把val_data 弄成一个列表可能更加科学一点。
    val_data_list = []
    val_label_list = []
    #val_data = val_data[:num_pic2 * height].reshape(num_pic2, height, cycle_num2, channel2)
    for i in range(num_pic2):
        val_data_list.append(val_data[height*i:height*(i+1)])
        val_label_list.append(val_label[height*i:height*(i+1)])
    # 切块以后，还剩余的数据也要不足上去
    val_data_list.append(val_data[num_pic2*height:])
    val_label_list.append(val_label[num_pic2*height:])

    #val_label = val_label[:num_pic2 * height].reshape(num_pic2, height, cycle_num2)
    val_dataset = Dataset_npy_val(val_data_list, val_label_list)
    val_loader = data.DataLoader(val_dataset, batch_size=batchsize, num_workers=0, shuffle=False, drop_last=False)

    model = DNA_Sequencer_Atten().to(device)
    # 定义损失函数和优化器
    #best_model = torch.load("savedir_pth/acc98.731.pth.tar")['state_dict']
    best_model = torch.load("savedir_pth/acc99.659.pth.tar")['state_dict']

    model.load_state_dict(best_model)

    idx = 0
    accurate_show = AverageMeter()

    dictacgt = {1: "A", 2: "C", 3: "G", 4: "T"}
    predict_list = []
    s = time.time()
    with torch.no_grad():
        for inputs, labels in val_loader: #
            idx += 1
            model.eval()
            inputs = inputs.to(device)  # b,c,h,w   b 4 1000 100 ，因为是全卷积，即使不是1000也可以搞定吧。
            labels = labels.to(device)  # b,c,h,w
            outputs = model(inputs)

            _, pred = torch.max(outputs, 1)  # pred 取 0,1,2,3
            _, label = torch.max(labels, 1)  #
            #outputs_np = outputs.cpu().numpy().transpose([])
            c = (pred == label)  # 里面有多少true和false，
            right = torch.sum(c).item()
            accurate = 100 * right / (pred.shape[0] * pred.shape[1] * pred.shape[2])
            accurate_show.update(accurate)
            #print('acc:%.3f %%' % ( accurate))

            pred_np = pred.cpu().numpy().astype(int).squeeze() # 1,1000,100
            str_acgt = ""
            for i in range(pred_np.shape[0]):
                for j in range(pred_np.shape[1]):

                    str_acgt = str_acgt + dictacgt[pred_np[i,j]+1]
                predict_list.append(str_acgt)
                str_acgt=""


        print('total acc: %.3f %%' % (accurate_show.avg))

    timestr = time.strftime("%Y%m%d-%H%M%S")
    # fastQ.writeFq('fastq/{:.3f}-fast08_{}.fq'.format(accurate_show.avg,timestr), predict_list, 'ROO1C001')



    save_path = fr'fastq/{fov}_{accurate_show.avg:.3f}_{timestr}//'
    temp_save_path = fr'E:\code\python_PK\channelAttention-2-finetune\fastq/{fov}_{accurate_show.avg:.3f}_{timestr}/Lane01/sfile//'
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(temp_save_path, exist_ok=True)
    # 将坐标保存到txt文件
    fastQ.writeFq(save_path+"/Lane01/" + 'Lane01_fastq.fq', predict_list, 'R001C001')
    end = time.time()
    print("time:", end - s)
    deepMappingUpdateDataset_forMatrix.deepUpdateData(r"E:\code\python_PK\channelAttention-2-finetune\\",save_path,r"E:\data\liangdujuzhen\update\\")

    #autoMap(save_path)

if __name__ == "__main__":
    main()