import torch

from datasets import Dataset_3cycle,Dataset_3cycle_val
from callNet import  DNA_Sequencer,Max
from model_NestedUnet import NestedUNet as UNet
from utils import  *
import numpy as np
import json
import fastQ
import time
from datasets_read_img import Dataset_3cycle_test
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
from deepMappingUpdateDataset import deepUpdateData

rootdir = r"E:\code\python_PK\img2base_cnn_seg\fastq"
# 验证集会给出machine，fov，acc，time,不必我去读。
file_name = "08h_R001C001_98.404_0611152213"
save_root_dir = "C:\deepdata\image_update2"
os.makedirs(rootdir + rf"//{file_name}", exist_ok=True)
deepUpdateData(rootdir, file_name, save_root_dir)