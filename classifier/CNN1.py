import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import numpy as np


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        # input_size = input.shape[-1]
        self.net = nn.Sequential(
            nn.Conv1d(1,64,kernel_size=3),nn.ReLU(),
            nn.Conv1d(64,32,kernel_size=3),nn.ReLU(),
            nn.Conv1d(32,16,kernel_size=3),nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Flatten(),
            nn.Linear(16,2)
        )
# a = np.array(['ACDCFSADASDAS'])
# print(a.shape)
# net = CNN(a)
# print('net:',net)

#train
# loss = nn.CrossEntropyLoss()
# optimizer = torch.optim.RMSprop()

# #data
# data1 = pd.read_csv(r'attention.csv',header=None)
# data2 = pd.read_csv(r'bilstm_output.csv',header=None)
# print('data1:',data1.shape)
# print('data2:',data2.shape)
# data_a = np.array(data1)
# data_b = np.array(data2)
# print('data_a:',data_a.shape)
# print('data_b:',data_b.shape)
# data=data[:,0:]
# [m1,n1]=np.shape(data)
# label1=np.ones((int(m1/2),1))#Value can be changed
# label2=np.zeros((int(m1/2),1))
# #label1=np.ones((544,1))#Value can be changed
# #label2=np.zeros((407,1))
# label=np.append(label1,label2)
# shu=scale(data)
# X=shu
# y=label
# sepscores = []

# ytest=np.ones((1,2))*0.5
# yscore=np.ones((1,2))*0.5

# X = X.reshape(m1, n1,1)