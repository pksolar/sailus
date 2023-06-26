#!/usr/bin/env python
# coding: utf-8
import glob

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



# In[16]:





# ## Create Model

# In[17]:


save_path = "checkpoint/001"
checkpoint = "epoch_8.loss_0.49296483916827244.pth"
resnet = models.resnet50(weights=True)
net = Net(resnet, n_features=12)
if use_cuda:
    resnet = resnet.cuda()
    net = net.cuda()
    net.load_state_dict(torch.load(f"{save_path}/{checkpoint}"))
else:
    net.load_state_dict(torch.load(f"{save_path}/{checkpoint}", map_location=lambda storage, loc: storage))





attr_keys = ['BalacingElements', 'ColorHarmony', 'Content', 'DoF',
             'Light', 'MotionBlur', 'Object', 'RuleOfThirds', 'VividColor']
non_neg_attr_keys = ['Repetition', 'Symmetry', 'score']
all_keys = attr_keys + non_neg_attr_keys
used_keys = ["ColorHarmony", "Content", "DoF", "Object", "VividColor", "score"]


# In[24]:


weights = {k: net.attribute_weights.weight[i, :] for i, k in enumerate(all_keys)} 


# In[25]:




# In[26]:




# In[27]:


def scale(image, low=-1, high=1):
    im_max = np.max(image)
    im_min = np.min(image)
    return (high - low) * (image - np.min(image))/(im_max - im_min) + low 


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





# In[30]:


def sample_data(dataset, image_path=None):
    idx = random.sample(range(len(dataset)), 1)[0]
    return dataset[idx]


# In[31]:
path_images = glob.glob(r"imageD\*.jpg")

# Get some test data to see how the heatmaps look
for image in path_images:
data = sample_data(test_dataset)

image = data['image']
image_path = data['image_path']
image_default = mpimg.imread(image_path)
img_shape = image_default.shape
h, w = img_shape[0], img_shape[1]


# In[ ]:


plt.imshow(image_default)


# In[ ]:


inp = Variable(image).unsqueeze(0)
if use_cuda:
    inp = inp.cuda()


# In[ ]:


predicted_values = extract_prediction(inp, net)
print(predicted_values)




