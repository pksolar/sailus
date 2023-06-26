from __future__ import division

import glob
import warnings

from Networks.HR_Net.seg_hrnet import get_seg_model

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import dataset
import math
from image import *
from utils import *

import logging
import nni
from nni.utils import merge_parameter
from config import return_args, args
import time

warnings.filterwarnings('ignore')
'''fixed random seed '''
setup_seed(args.seed)

logger = logging.getLogger('mnist_AutoML')

"""
 transforms.Normalize(mean=[0.02710745 ,0.03826159, 0.03559004 ,0.03568636],
                                                     std=[0.02371992, 0.02972373, 0.03165162, 0.02961854])

"""
def main(args):
    if args['dataset'] == 'ShanghaiA':
        train_file = './npydata/ShanghaiA_train.npy'
        test_file = './npydata/ShanghaiA_test.npy'
    elif args['dataset'] == 'base':
        train_file = './npydata/base_train.npy' #npy文件里全部都是路径。其实可以用glob来读取。
        test_file = './npydata/base_test.npy'
    elif args['dataset'] == 'ShanghaiB':
        train_file = './npydata/ShanghaiB_train.npy'
        test_file = './npydata/ShanghaiB_test.npy'
    elif args['dataset'] == 'UCF_QNRF':
        train_file = './npydata/qnrf_train.npy'
        test_file = './npydata/qnrf_test.npy'
    elif args['dataset'] == 'JHU':
        train_file = './npydata/jhu_train.npy'
        test_file = './npydata/jhu_val.npy'
    elif args['dataset'] == 'NWPU':
        train_file = './npydata/nwpu_train.npy'
        test_file = './npydata/nwpu_val.npy'

    with open(train_file, 'rb') as outfile:
        train_list = np.load(outfile).tolist()
    with open(test_file, 'rb') as outfile:
        test_list = np.load(outfile).tolist()

    train_img_list = glob.glob(r"E:\code\python_PK\tools\hrnet_label\img2base\imgdata\*.npy")
    test_list = glob.glob(r"E:\code\python_PK\tools\hrnet_label\val\imgdata\*.npy")
    #train_label_list = glob.glob(r"E:\code\python_PK\tools\hrnet_label\img2base\label\*.npy")

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_id']
    model = get_seg_model(train=True)
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()

    optimizer = torch.optim.Adam(
        [  #
            {'params': model.parameters(), 'lr': args['lr']},
        ], lr=args['lr'], weight_decay=args['weight_decay'])

    # criterion = nn.MSELoss(size_average=False).cuda()
    criterion = nn.MSELoss().cuda()

    print(args['pre'])

    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])

    if args['pre']:
        if os.path.isfile(args['pre']):
            print("=> loading checkpoint '{}'".format(args['pre']))
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            #args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
        else:
            print("=> no checkpoint found at '{}'".format(args['pre']))

    torch.set_num_threads(args['workers'])
    print(args['best_pred'], args['start_epoch'])

    if args['preload_data'] == True:
        train_data = pre_data(train_img_list, args, train=True)
        test_data = pre_data(test_list, args, train=False)
    else:
        train_data = train_list
        test_data = test_list


    for epoch in range(args['start_epoch'], args['epochs']):

        train(train_data, model, criterion, optimizer, epoch, args)

        '''inference '''
        if epoch % 30 == 0 and epoch >= 100:
            mse = validate(test_data, model, args)
            best_mse = min(mse, args['best_pred'])
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args['pre'],
                'state_dict': model.state_dict(),
                'best_prec1': best_mse,
                'optimizer': optimizer.state_dict(),
            },save_dir = "checkpoint/", filename="mse{:.5f}.pth".format(mse))

def pre_data(train_list, args, train):
    print("Pre_load dataset ......")
    data_keys = {}
    count = 0
    for j in range(len(train_list)):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        # print(fname)
        img, gt,msk = load_data_fidt(Img_path, args, train)

        blob = {}
        blob['img'] = img

        blob['gt'] = gt
        blob['fname'] = fname
        blob['msk'] = msk
        data_keys[count] = blob
        count += 1

    return data_keys


def train(Pre_data, model, criterion, optimizer, epoch, args):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),

                            ]),
                            train=True,
                            batch_size=args['batch_size'],
                            num_workers=args['workers'],
                            args=args),
        batch_size=args['batch_size'], drop_last=False)
    args['lr'] = optimizer.param_groups[0]['lr']
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args['lr']))

    model.train()
    end = time.time()


    for i, (fname, img, gt,msk) in enumerate(train_loader):

        data_time.update(time.time() - end)
        img = img.cuda()
        gt = gt.cuda()
        msk = msk.unsqueeze(1).cuda()

        # gt = gt.type(torch.FloatTensor)

        d6 = model(img)

        if d6.shape != gt.shape:
            print("the shape is wrong, please check. Both of prediction and GT should be [B, C, H, W].")
            exit()
        loss = criterion(d6*msk, gt*msk) #msk是点的msk。

        losses.update(loss.item(), img.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args['print_freq'] == 0:
            print('4_Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))



def validate(Pre_data, model, args):
    print('begin test')
    batch_size = 1
    mse_show = AverageMeter()
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=False,
                            args=args, train=False),
        batch_size=1)

    model.eval()
    criterion = torch.nn.MSELoss().cuda()

    if not os.path.exists('./local_eval/loc_file'):
        os.makedirs('./local_eval/loc_file')

    for i, (fname, img, gt,msk) in enumerate(test_loader):

        count = 0
        img = img.cuda()
        gt = gt.cuda()
        msk = msk.unsqueeze(1).cuda()


        with torch.no_grad():
            d6 = model(img)
            mse = criterion(d6*msk,gt*msk)
            print(f"val_mse:{mse.item()}")
        mse_show.update(mse.item())

    return mse_show.avg


def show_map(input):
    input[input < 0] = 0
    input = input[0][0]
    fidt_map1 = input
    fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
    fidt_map1 = fidt_map1.astype(np.uint8)
    fidt_map1 = cv2.applyColorMap(fidt_map1, 2)
    return fidt_map1


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


if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    params = vars(merge_parameter(return_args, tuner_params))
    print(params)

    main(params)
