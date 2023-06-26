import torch
from torch.utils.data import Dataset
import os
import random
from image import *
import numpy as np
import numbers
from torchvision import datasets, transforms

class listDataset_full(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1,
                 num_workers=4, args=None):
        if train:
            random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.args = args

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        if self.args['preload_data'] == True:
            fname = self.lines[index]['fname']
            img = self.lines[index]['img']
            gt = self.lines[index]['gt']
            msk = self.lines[index]['msk']

        else:
            img_path = self.lines[index]
            fname = os.path.basename(img_path)
            img, fidt_map, kpoint = load_data_fidt(img_path, self.args, self.train)

        '''data augmention'''
        if self.train == True:
            if random.random() > 0.5:
                gt = np.fliplr(gt)
                msk = np.fliplr(msk)
                img =np.fliplr(img)




        img = img.copy()
        # msk[msk<0] = 0


        img = torch.from_numpy(img).cuda().float()

        # if self.transform is not None:
        #     img = self.transform(img.transpose(2,1,0))

        '''crop size'''
        if self.train == True:
            gt = gt.copy()
            msk = msk.copy()
            gt = torch.from_numpy(gt).cuda().float()
            msk = torch.from_numpy(msk).cuda().float()
            width = self.args['crop_size']
            height = self.args['crop_size']
            # print(img.shape)
            crop_size_x = random.randint(0, img.shape[1] - width)
            crop_size_y = random.randint(0, img.shape[2] - height)
            img = img[:, crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height]
            gt = gt[:,crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height].float()
            msk =  msk[crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height].float()
            return fname, img, gt,msk
        return fname,img



class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1,
                 num_workers=4, args=None):
        if train:
            random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.args = args

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        if self.args['preload_data'] == True:
            fname = self.lines[index]['fname']
            img = self.lines[index]['img']
            gt = self.lines[index]['gt']
            msk = self.lines[index]['msk']

        else:
            img_path = self.lines[index]
            fname = os.path.basename(img_path)
            img, fidt_map, kpoint = load_data_fidt(img_path, self.args, self.train)

        '''data augmention'''
        if self.train == True:
            if random.random() > 0.5:
                gt = np.fliplr(gt)
                msk = np.fliplr(msk)
                img =np.fliplr(img)
        img = img.copy()
        # msk[msk<0] = 0
        img = torch.from_numpy(img).cuda().float()
        gt = gt.copy()
        msk = msk.copy()
        gt = torch.from_numpy(gt).cuda().float()
        msk = torch.from_numpy(msk).cuda().float()
        # if self.transform is not None:
        #     img = self.transform(img.transpose(2,1,0))

        '''crop size'''
        if self.train == True:
            msk[msk < 0] = 0
            width = self.args['crop_size']
            height = self.args['crop_size']
            crop_size_x = random.randint(0, img.shape[1] - width)
            crop_size_y = random.randint(0, img.shape[2] - height)
            img = img[:, crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height]
            gt = gt[:,crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height].float()
            msk =  msk[crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height].float()
        return fname, img, gt,msk

