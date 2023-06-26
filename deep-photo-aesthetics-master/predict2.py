#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models, transforms
from collections import OrderedDict
from tqdm import tqdm
import pprint
import cv2
import random
import torch
import sys
import os
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#get_ipython().run_line_magic('matplotlib', 'inline')
sys.path.append("..")


# In[12]:


use_cuda = torch.cuda.is_available()


# In[13]:


from modelmy.resnet_FT import ResNetGAPFeatures as Net
from utils.data import read_data, create_dataloader, AestheticsDataset


# In[14]:


from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# ## Create dataset

# In[15]:



test = read_data("data/test.csv", "images", is_test = True)


# In[16]:



test_loader = create_dataloader(test, batch_size=1, is_train=False)


# ## Create Model

# In[17]:


save_path = "checkpoint/001"
checkpoint = "epoch_12.loss_0.3802574505435238.pth"
resnet = models.resnet50(weights=True)
net = Net(resnet, n_features=12)
if use_cuda:
    resnet = resnet.cuda()
    net = net.cuda()
    net.load_state_dict(torch.load(f"{save_path}/{checkpoint}"))
else:
    net.load_state_dict(torch.load(f"{save_path}/{checkpoint}", map_location=lambda storage, loc: storage))


# ## Result 

# In[18]:


# In[19]:





# In[20]:




# In[21]:





# ## Visualization

# In[23]:


attr_keys = ['BalacingElements', 'ColorHarmony', 'Content', 'DoF',
             'Light', 'MotionBlur', 'Object', 'RuleOfThirds', 'VividColor']
non_neg_attr_keys = ['Repetition', 'Symmetry', 'score']
all_keys = attr_keys + non_neg_attr_keys
used_keys = ["ColorHarmony", "Content", "DoF", "Object", "VividColor", "score"]


# In[24]:


weights = {k: net.attribute_weights.weight[i, :] for i, k in enumerate(all_keys)} 


# In[25]:


def extract_pooled_features(inp, net):
    _ = net(inp)
    pooled_features = [features.feature_maps for features in net.all_features] 
    return pooled_features


# In[26]:


def downsample_pooled_features(features):
    dim_reduced_features = []
    for pooled_feature in pooled_features:
        if pooled_feature.size()[-1] == 75:
            dim_reduced_features.append(F.avg_pool2d(pooled_feature, kernel_size=(7, 7)))
        elif pooled_feature.size()[-1] == 38:
            dim_reduced_features.append(F.avg_pool2d(pooled_feature, kernel_size = (4, 4), padding=1))
        elif pooled_feature.size()[-1] == 19:
            dim_reduced_features.append(F.avg_pool2d(pooled_feature, kernel_size = (2, 2), padding=1))
        elif pooled_feature.size()[-1] == 10:
            dim_reduced_features.append(pooled_feature)
    dim_reduced_features = torch.cat(dim_reduced_features, dim=1).squeeze()
    return dim_reduced_features


# In[27]:


def scale(image, low=-1, high=1):
    im_max = np.max(image)
    im_min = np.min(image)
    return (high - low) * (image - np.min(image))/(im_max - im_min) + low 

def extract_heatmap(features, weights, w, h):
#     cam = np.ones((10, 10), dtype=np.float32) 
    
#     # Sum up the feature maps 
#     temp = weight.view(-1, 1, 1) * features
#     summed_temp = torch.sum(temp, dim=0).data.cpu().numpy()
#     cam = cam + summed_temp
#     cam = cv2.resize(cam, (w, h))
#     cam = np.maximum(cam, 0)
#     cam = np.uint8(255*(cam/np.max(cam)))
    cam = np.zeros((10, 10), dtype=np.float32) 
    temp = weights.view(-1, 1, 1) * downsampled_pooled_features
    summed_temp = torch.sum(temp, dim=0).data.cpu().numpy()
    cam = cam + summed_temp
    cam = cv2.resize(cam, (w, h))
    cam = scale(cam)
    return cam 


# In[28]:


def extract_prediction(inp, net):
    d = dict()
    net.eval()
    output = net(inp)
    g = output[:,1]

    for i, key in enumerate(all_keys):
        d[key] = output[:, i].data[0].item()
    return d


# In[29]:



test_dataset = AestheticsDataset(test, is_train=False)


# In[30]:


def sample_data(dataset, image_path=None):
    idx = random.sample(range(len(dataset)), 1)[0]
    return dataset[idx]


# In[31]:


# Get some test data to see how the heatmaps look
data = sample_data(test_dataset)
lista = []
listscore = []
for data in test_dataset:
    ax = data['score']
    lista.append(ax)
    image = data['image']
    image_path = data['image_path']
    image_default = mpimg.imread(image_path)
    img_shape = image_default.shape
    h, w = img_shape[0], img_shape[1]


# In[ ]:


    # plt.imshow(image_default)


    # In[ ]:


    inp = Variable(image).unsqueeze(0)
    if use_cuda:
        inp = inp.cuda()


    # In[ ]:


    predicted_values = extract_prediction(inp, net)
    listscore.append(predicted_values['score'])
    print(ax,",",predicted_values['score'])

#
# # for data in test_dataset:
# # print(test_dataset['BalancingElements'])
print(lista)
print(listscore)
plt.scatter(lista,listscore)
plt.table("here")
plt.show()










