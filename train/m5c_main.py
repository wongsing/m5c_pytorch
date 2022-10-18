from cProfile import label
from re import X
import sys
import os
from turtle import forward
path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(path)

import torch
from torch import nn, tensor
import numpy as np
import pandas as pd
from transformers import FlaxT5Model

import matplotlib.pyplot as plt

from feature_extraction import Kmer
from feature_extraction import KNFC
from feature_extraction import PCP
from feature_extraction import PseDNC

from sklearn.preprocessing import scale

def data_concat(data1,data2,not_final=True):
    data_new = np.concatenate((data1,data2),axis=-1)
    if(not_final):       
        print('data_new.shape:',data_new.shape)
        data=data_new[:,0:]
        n1=np.shape(data)[-1] #n1是特征数
    else:
        #目前问题
        print('before reshape:data_new.shape:',data_new.shape)
        # data=data_new.reshape(-1,data_new.shape[-1])
        # print('after rshape:',data.shape)
        # n1=np.shape(data)[-1] #n1是特征数
    # shu=scale(data) #三维不能进行scale，得进行reshape,数据太多也不好scale
    # X1=shu
    # X=np.reshape(X1,(-1,1,n1))
    # print('X.shape:',X.shape)
    #将numpy转换成tensor
    input = torch.tensor(data_new).float()
    print('input.shape:',input.shape)
    return input

"""测试特征提取方法的调用"""
def feature_extraction1(path1,path2):
    print('**********************特征提取**********************')
    #Kmer
    a1 = Kmer.read_fasta(path1)
    kmer = Kmer.Kmer(a1)
    print('Kmer:',kmer,'shape:',kmer.shape)
    #KNFC
    a2 = KNFC.ObtainSequenceAndLabels(path1)
    knfc = KNFC.generate_fn_file(a2,4)
    print('KNFC:',knfc,'shape:',knfc.shape)
    #PCP
    prop_data_transformed,prop_key = PCP.dnc_key(path2)
    pcp = PCP.get_pcp(path1,prop_data_transformed,prop_key)
    print('PCP:',pcp,'shape:',pcp.shape)
    #PseDNC
    DNC_value_scale,DNC_key = PseDNC.dnc_key(path2)
    pse = PseDNC.get_PseNNC(path1,DNC_value_scale,DNC_key)
    print('PseDNC:',pse,'shape:',pse.shape)
    return kmer,knfc,pcp,pse

from fusion import feature_fusion
"""测试特征融合调用"""
def feature_fusion1(kmer,knfc,pcp,pse):     
    print('**********************特征融合**********************')
    #bilstm对PCP/PseDNC进行特征融合->input_a
    #attentin对Kmer/KNFC进行特征融合->input_b
    input_a = data_concat(pcp,pse)
    input_b = data_concat(kmer,knfc)

    attn = feature_fusion.attention(input_a,1)
    print('attn:',attn,'shape:',attn.shape)

    lstm = feature_fusion.bilstm(input_b,100,True)
    print('lstm:',lstm,'shape:',lstm.shape)
    #将特征合起来
    final_data = data_concat(attn,lstm,False)
    print(final_data)
    return final_data

#path
path1 = 'data/Athaliana/A.thaliana1000indep_pos.fasta'
path2 = 'feature_extraction/physical_chemical_properties_RNA.txt'

# kmer,knfc,pcp,pse = feature_extraction1(path1,path2)
# final_data = feature_fusion1(kmer,knfc,pcp,pse)

# pd.DataFrame(final_data.detach().numpy()).to_csv(r'A.thaliana1000indep_pos.csv', header=None, index=None)

"""将处理完的特征数据整合"""
def dataset(pos_file,neg_file):
    print('***************特征数据处理*******************')
    data_pos = pd.read_csv(pos_file)
    data_neg = pd.read_csv(neg_file)
    data_pos_len = data_pos.shape[0]
    data_neg_len = data_neg.shape[0]
    label1 = np.ones((data_pos_len,1))
    label2 = np.zeros((data_neg_len,1))
    data_pos = np.array(data_pos)
    data_neg = np.array(data_neg)
    print('data_pos.shape:',data_pos.shape)
    print('data_neg.shape:',data_neg.shape)
    data_total = np.concatenate([data_pos,data_neg],axis=0)
    label_total = np.concatenate([label1,label2],axis=0)
    data_all= np.concatenate([data_total,label_total],axis=1)
    print('data_all.shape:',data_all.shape)
    """这里需要考虑模型数据的随机性，导致性能评估不一致！"""
    # print('data_all before shuffle:',data_all)
    # np.random.shuffle(data_all)
    # print('data_all after shuffle:',data_all)
    data_all = torch.tensor(data_all).unsqueeze(-1)
    # print('before permute:',data_all.shape)
    data_all = data_all.permute(0,2,1).to(torch.float32)
    # print('after permute:',data_all.shape)
    return data_all[:,:,:-1],data_all[:,:,-1].to(torch.long)

#已经获得了打乱的训练数据集
X , y = dataset('A.thaliana5289_pos.csv','A.thaliana5289_neg.csv')
# print('X.shape:',X.shape,'y.shape:',y.shape)

"""将数据集分为batch"""
from data import data_iter
batch_size = 5
train_iter = data_iter.get_data_iter(X,y,batch_size)

"""conv1d channel"""
# Conv1d中的数据有三个维度，第一个维度N一般是batch_size，
# 第二个维度一般为in_channel，第三个维度为序列的时间维度，在NLP中为词向量大小；
# 输出维度基本相同，但是输出的第二个维度为out_channel。

# """训练"""
# input_size = 622
"""CNN model"""

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn,self).__init__()
        self.conv1 = nn.Conv1d(1,64,3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(64,32,3)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(32,16,3)
        self.relu3 = nn.ReLU()
        self.maxpol = nn.MaxPool1d(3)
        self.fc1 = nn.Linear(3280,1600)
        self.fc2 = nn.Linear(1600,800)
        self.fc3 = nn.Linear(800,2)

    def forward(self,x):
        out1 = self.relu1(self.conv1(x))
        # print('out1.shape:',out1.shape)
        out2 = self.relu2(self.conv2(out1))
        # print('out2.shape:',out1.shape)
        out3 = self.relu3(self.conv3(out2))
        # print('out3.shape:',out1.shape)
        X = self.maxpol(out3)
        # print('X.shape:',X.shape)
        return self.fc3(self.fc2(self.fc1(X.view(-1,3280))))



#检查模型
# net = nn.Sequential(
#             # nn.Conv1d(600,300,kernel_size=3),nn.ReLU(),
#             nn.Conv1d(300,100,kernel_size=3),nn.ReLU(),
#             nn.Conv1d(100,64,kernel_size=3),nn.ReLU(),
#             nn.Conv1d(64,32,kernel_size=3),nn.ReLU(),
#             nn.Conv1d(32,16,kernel_size=3),nn.ReLU(),
#             nn.MaxPool1d(kernel_size=3),
#             nn.Flatten(),
#             nn.Linear(16,2)
#         )
# test = torch.rand(size=(5,300,32),dtype=torch.float32)
# for layer in net:
#     x =layer(test)
#     print(layer.__class__.__name__,'output shape:\t',x.shape)

net = Cnn()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(net.parameters())

num_epochs = 10
train_loss = []
for epoch in range(num_epochs):
    for bacth_idx,data in enumerate(train_iter):
        # print('data:',data)
        x,y = data
        optimizer.zero_grad()
        out = net(x)
        # print('out.shape:',out.shape)
        # print('target.shape:',y.shape)
        # if out[0] > out[1]:
        #     pre = 0
        # else:
        #     pre = 1
        # print('pre:',pre)
        loss = loss_function(out,y.squeeze(dim=1))
        #清除梯度
        
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
#绘制损失曲线
plt.figure(figsize=(8,3))
plt.grid(True,linestyle='--',alpha=0.5)
plt.plot(train_loss,label='loss')
plt.legend(loc="best")
plt.show()





    

