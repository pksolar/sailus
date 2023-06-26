import glob
import shutil

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.utils.data as data
import os
from natsort import natsorted

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F



import torch
import torch.nn as nn

class GHMLoss(nn.Module):
    def __init__(self, bins=10, momentum=0.5):
        super(GHMLoss, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = torch.linspace(0, 1, bins+1)
        self.edges[-1] += 1e-6  # To include the rightmost value
        self.acc_sum = torch.zeros(bins+1)

    def forward(self, pred, target, sample_weight=None):
        edges = self.edges.to(pred.device)
        g = torch.abs(pred.sigmoid().detach() - target)  # Calculate the gradient errors
        valid_mask = torch.isfinite(g) & torch.isfinite(target)
        g = g[valid_mask]
        target = target[valid_mask]
        weights = torch.zeros_like(g)

        # Calculate weights based on gradient errors
        total_samples = g.numel()
        inds = torch.bucketize(g, edges)
        sample_in_bin = torch.bincount(inds, minlength=self.bins)
        self.acc_sum.to(pred.device)
        acc_sum = self.momentum * self.acc_sum + (1 - self.momentum) * sample_in_bin.float()
        total_positives = acc_sum.sum()
        total_negatives = total_samples - total_positives
        total_positives = torch.max(total_positives, torch.tensor([1.0]).to(pred.device))  # Avoid division by zero
        weights[target > pred.detach()] = total_negatives / total_positives
        weights[target <= pred.detach()] = 1.0

        if sample_weight is not None:
            weights *= sample_weight[valid_mask]

        loss = nn.functional.cross_entropy(pred, target, weight=weights, reduction='mean')
        return loss




class GHMC(nn.Module):
    def __init__(self, bins=10, momentum=0, use_sigmoid=True, loss_weight=1.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.edges[-1] += 1e-6
        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer('acc_sum', acc_sum)
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight=0.5, *args, **kwargs):
        # the target should be binary class label
        # if pred.dim() != target.dim():
        #     target, label_weight = _expand_onehot_labels(target, label_weight, pred.size(-1))
        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - target)

        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.cross_entropy(
            pred, target, weights, reduction='sum') / tot
        return loss * self.loss_weight




class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets,number):
        N, C,_,_ = inputs.size()
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            FL_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        else:
            FL_loss = (1 - pt) ** self.gamma * BCE_loss
        return FL_loss.sum()/number

class FocalLoss_4(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss_4, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.argmax(dim=1).view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()




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

def save_checkpoint(state, save_dir='pth', filename='checkpoint.pth.tar', max_model_num=12):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0]) #删除第一个
        model_lists = natsorted(glob.glob(save_dir + '*'))
def copy_checkpoint(ori_file, save_dir='pth', filename='checkpoint.pth.tar', max_model_num=12):
    dest_file = "pth/" + filename
    shutil.copy2(ori_file,dest_file)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0]) #删除第一个
        model_lists = natsorted(glob.glob(save_dir + '*'))