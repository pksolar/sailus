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


train = read_data("data/train.csv", "images")
val = read_data("data/val.csv", "images")
test = read_data("data/test.csv", "images", is_test = True)


# In[16]:


train_loader = create_dataloader(train, batch_size=1)
val_loader = create_dataloader(val, batch_size=1, is_train=False)
test_loader = create_dataloader(test, batch_size=1, is_train=False)


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


# ## Result 

# In[18]:


train_results = pd.read_csv(f"{save_path}/train_results.csv").drop(["Unnamed: 0"], 1)
val_results = pd.read_csv(f"{save_path}/val_results.csv").drop(["Unnamed: 0"], 1)
train_corr_results = pd.read_csv(f"{save_path}/train_corr_results.csv").drop(["Unnamed: 0"], 1)
val_corr_results = pd.read_csv(f"{save_path}/val_corr_results.csv").drop(["Unnamed: 0"], 1)


# In[19]:


train_results.groupby("epoch").mean()


# In[20]:


val_results.groupby("epoch").mean()


# In[21]:


train_corr_results.groupby("epoch").mean()


# In[22]:


val_corr_results.groupby("epoch").mean()


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


train_dataset = AestheticsDataset(train, is_train=False)
val_dataset = AestheticsDataset(val, is_train=False)
test_dataset = AestheticsDataset(test, is_train=False)


# In[30]:


def sample_data(dataset, image_path=None):
    idx = random.sample(range(len(dataset)), 1)[0]
    return dataset[idx]


# In[31]:


# Get some test data to see how the heatmaps look
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
pooled_features = extract_pooled_features(inp, net)
downsampled_pooled_features = downsample_pooled_features(pooled_features)


# In[ ]:


fig, ax = plt.subplots(figsize=(20, 20))
y, x = np.mgrid[0:h, 0:w]
fig.subplots_adjust(right=1,top=1,hspace=0.5,wspace=0.5)
for i, k in enumerate(used_keys): 
    heatmap = extract_heatmap(downsampled_pooled_features, weights[k], w=w, h=h)
    ax = fig.add_subplot(2, 4, i+1)
    ax.imshow(image_default, cmap='gray')
    cb = ax.contourf(x, y, heatmap, cmap='jet', alpha=0.75)
    ax.set_title(f"Attribute: {k}\nScore: {data[k][0]}\nPredicted Score: {round(predicted_values[k], 2)}")
ax = fig.add_subplot(2, 4, 7)
ax.imshow(image_default) 
plt.colorbar(cb)
plt.tight_layout()


# #### Correlation for Training Set

# In[ ]:


from tqdm import tqdm
df_train_data = []
for train_data in tqdm(train_dataset):
    image = train_data['image']
    image_path = train_data['image_path']

    inp = Variable(image).unsqueeze(0)
    if use_cuda:
        inp = inp.cuda()
    output = net(inp).squeeze().data
    row_data = extract_prediction(inp, net) 
    row_data['img_path'] = image_path
    df_train_data.append(row_data)


# In[ ]:


predicted_train_df = pd.DataFrame(df_train_data).sort_values(['img_path'])[used_keys+['img_path']].reset_index(drop=True)
sorted_train_df = train.sort_values(['img_path'])[used_keys+['img_path']].reset_index(drop=True)


# In[ ]:


predicted_train_df.corrwith(sorted_train_df)


# #### Correlation for Validation Set

# In[ ]:


from tqdm import tqdm
df_val_data = []
for val_data in tqdm(val_dataset):
    image = val_data['image']
    image_path = val_data['image_path']

    inp = Variable(image).unsqueeze(0)
    if use_cuda:
        inp = inp.cuda()
    output = net(inp).squeeze().data
    row_data = extract_prediction(inp, net) 
    row_data['img_path'] = image_path
    df_val_data.append(row_data)
predicted_val_df = pd.DataFrame(df_val_data).sort_values(['img_path'])[used_keys+['img_path']].reset_index(drop=True)
sorted_val_df = val.sort_values(['img_path'])[used_keys+['img_path']].reset_index(drop=True)


# In[ ]:


predicted_val_df.corrwith(sorted_val_df)


# #### Correlation for Test Set

# In[ ]:


from tqdm import tqdm
df_test_data = []
for test_data in tqdm(test_dataset):
    image = test_data['image']
    image_path = test_data['image_path']

    inp = Variable(image).unsqueeze(0)
    if use_cuda:
        inp = inp.cuda()
    output = net(inp).squeeze().data
    row_data = extract_prediction(inp, net) 
    row_data['img_path'] = image_path
    df_test_data.append(row_data)
    


# In[ ]:


predicted_test_df = pd.DataFrame(df_test_data).sort_values(['img_path'])[used_keys+['img_path']].reset_index(drop=True)
sorted_test_df = test.sort_values(['img_path'])[used_keys+['img_path']].reset_index(drop=True)


# In[ ]:


predicted_test_df.corrwith(sorted_test_df)


# In[ ]:





# In[ ]:




