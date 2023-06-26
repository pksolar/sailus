import torch
import glob
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.utils.data as data
import os
from natsort import natsorted

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
        #print("self.avg:",self.avg)

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
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
    seq_len is 3 
    two direction


    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_array = np.load("../08_R001C001_pure.npy")[:224496]
    label_array = np.load("../08_R001C001_label_pure.npy")[:224496]
    seq_len = 3
    train_dataset = Dataset_npy(data_array, label_array, seq_len)

    train_loader = data.DataLoader(train_dataset, batch_size=batchsize, num_workers=0, shuffle=False, drop_last=True)