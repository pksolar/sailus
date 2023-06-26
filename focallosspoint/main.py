import torch
"""
 输入是配准后的4通道的图像的叠加：h,w,c  其中c=4.
 是否要输入前后cycle值得探究，如何将前后cycle的影响考虑到值得探究
 输入的数据最好是切块处理的，512x512
 采用的网络，应当是采用分类网络。采用hrnet来进行特征提取。
 分类损失函数采用focal loss.(正负样本极度不平衡。)
 如何进行：
        1、先看懂fidt论文，以及代码，它如何采用fidt map 或者如何采用hrnet来检测关键点
        2、将fidt模型以及权重迁移到自己的数据集上。其中的问题是：如果我要采用4通道的，如何使用。
        3、是否有可能将4通道的信息，压缩到3维？然后通过3维度信息也可将它分类。是否可以用PCA进行降维。
        
        所以先用PCA对图像4维数据进行降维度。再一次性输入3张图片。
"""