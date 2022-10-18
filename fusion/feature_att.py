from pickletools import optimize
# from keras.models import Sequential
import pandas as pd
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc
# from keras.models import Model
# from keras.models import Sequential
# from keras.layers import Flatten,Bidirectional
# from keras.layers import LSTM
# #from keras.layers import Attention
# from attention import Attention

import torch
from torch import lstsq, nn
import torch.nn.functional as F
import math


#导入数据集
data_kmer=pd.read_csv(r"2-mer.csv",header=None)
data_knfc=pd.read_csv(r"knfc.csv",header=None)

data1 = np.array(data_knfc)
data2 = np.array(data_kmer)

print('data1.shape:',data1.shape)
print('data2.shape:',data2.shape)

data_new = np.concatenate((data2,data1),axis=1)
print('data_new.shape:',data_new.shape)

data=data_new[:,0:]
[m1,n1]=np.shape(data) #n1是特征数
shu=scale(data)
X1=shu

X=np.reshape(X1,(-1,1,n1))
print('X.shape:',X.shape)
#将numpy转换成tensor
input = torch.tensor(X).float()
print('input.shape:',input.shape)

embed_dim = input.shape[-1]
print('input:',input,'shape:',input.shape)

#这里用多头注意力机制
def attention
multihead_attn = nn.MultiheadAttention(embed_dim, num_heads=8)
attn_output, attn_output_weights = multihead_attn(input,input, input)
print('attn_output:',attn_output,'shape:',attn_output.shape) #(1000,1,272)
t = attn_output.tolist()

#遇到的问题：如何导出数据集？
#要么直接改写成类来进行调用

# print('list:',t)
# pd.DataFrame(t).to_csv(r'attention.csv', header=None, index=None)

# model = Sequential()
# model.add(Attention())
# model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics =['accuracy'])

# data_=pd.read_csv(r"input file",header=None)
# data=np.array(data_)
# data=data[:,0:]
# [m1,n1]=np.shape(data)
# shu=scale(data)
# X1=shu


# X=np.reshape(X1,(-1,1,n1))
# cv_clf = model

# #这是用来调试的
# # tf.config.experimental_run_functions_eagerly(True)

# feature=cv_clf.predict(X)

# data_csv = pd.DataFrame(data=feature)
# data_csv.to_csv(r'save file',header=None,index=False)
