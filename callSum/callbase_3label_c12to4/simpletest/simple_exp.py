import torch
import numpy as np
import glob

def glob_test():
    a = glob.glob(r"E:\code\python_PK\callbase\**")
    print(len(a))


def accurate2():
    """
    计算一个batch，不同类别准确率
    :return:
    """
    classes = [i for i in range(0, 5)]
    names = ['N', 'A', 'G', 'C', 'T']
    class_correct = torch.zeros(5)
    class_total = torch.zeros(5)
    # print(class_correct)
    a = torch.randint(5, (5, 3, 4, 5))
    b = torch.randint(5, (5, 3, 4, 5))
    # print("a", a)
    # print("b", b)
    c = (a == b)
    c = c.flatten(0)
    b = b.flatten(0)
    print("c flatten shape:", c.shape[0])

    for j in range(300):
            b_ = b[j]
            class_correct[b_] += c[j].item()
            class_total[b_] += 1
    print(torch.sum(class_total))
    print(torch.sum(class_correct))
    for j in range(5):
            print('Accuracy of %5s:%.2f %%' % (names[j], 100 * class_correct[j] / class_total[j]))
    print("total accuray: %.2f %%"%  (100 * torch.sum(class_correct)/torch.sum(class_total)))
def accurate1():
    """
    分batch，分类别计算准确率

    :return:
    """
    classes = [i for i in range(0,5)]
    names = ['N','A','G','C','T']
    class_correct = torch.zeros(5,5)
    class_total = torch.zeros(5,5)
    # print(class_correct)
    a = torch.randint(5,(5,3,4,5))
    b = torch.randint(5,(5,3,4,5))
    print("a",a)
    print("b",b)
    c = (a==b)
    c = c.flatten(1)
    b = b.flatten(1)
    print("c flatten shape:",c.shape)
    for i in range(5): #labels:  b c h w
        for j in range(60):
            b_ = b[i][j]
            class_correct[i][b_] += c[i][j].item()
            class_total[i][b_] += 1
    print(torch.sum(class_total))
    print(torch.sum(class_correct))
    for i in range(5):
        for j in range(5):
            print('batch: %2d,Accuracy of %5s:%.2f %%'%(i,names[j],100  * class_correct[i][j]/class_total[i][j]))
glob_test()
# print(a)
# print(b)
# print("c_size:",c.shape)
# print(c)
# d = c.squeeze()
# print(d.shape)
# print(d)

# _,pred = torch.max(a,1)
# print(pred)
# print(pred.shape)

# a = np.array([[0,1,2,3],
#               [2,3,1,1]]).astype('int64')
# a_t = torch.from_numpy(a)
# print(a_t)
# print(torch.max(a_t,0)[0].tolist())
# preds = [1,3,2,4,2,1,3,4,3,2,1]
# base_list = ['A', 'C', 'G', 'T', 'N']
# preds = [base_list[pred] for pred in preds]
# print(preds)


#
# label = np.load('label.npy').astype('int')  # [0,37]
#
# print(label.shape)  # (530, 730)
#
# num_class = int(np.max(label) - np.min(label) + 1)
# print(num_class)  # 38
#
# # print(label[0])
# print(label[200])
# # see some class value
# print(label[200][0])
# print(label[200][48])
# print(label[200][240])
#
# # (530, 730) -> # (1, 530, 730)
# label = label[np.newaxis, :, :]  # add new dim in any dim
# print(label.shape)
#
# # (1, 530, 730) -> [1, 530, 730, 1]
# label = torch.LongTensor(label).unsqueeze(3)
# print(label.shape)
#
# # [1, 530, 730, 1] -> [1, 530, 730, 38]
# label = torch.zeros(label.shape[0], label.shape[1], label.shape[2], num_class).scatter_(3, label, 1).long()
# print(label.shape)
#
# # see the class value -> one-hot tensor
# print(label[0][200][0])
# print(label[0][200][48])
# print(label[0][200][240])
#
# # [1, 530, 730, 38] -> [1, 38, 530, 730]
# label = label.transpose(1, 3).transpose(2, 3)
#
# print(label.shape)
#
# # print(label[])