import pandas as pd
import numpy as np
import torch
from torch import embedding, nn
import torch.nn.functional as F

"""通过bilstm,组成成分/物理化学成分特征融合"""
def bilstm(input,hidden_size,bidirectional):
    input_size = input.shape[-1]
    print('lstm.input_size:',input_size)
    rnn = torch.nn.LSTM(input_size,hidden_size,bidirectional=bidirectional)
    output,(hn,cn) = rnn(input)
    print('output.shape:',output.shape) #(1000,1,400)
    print('hn.shape:',hn.shape)
    print('cn.shape:',cn.shape)
    return output.detach().numpy()

"""通过attention,位置特征融合"""
def attention(input,num_heads):
    embed_dim = input.shape[-1]
    multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
    attn_output, attn_output_weights = multihead_attn(input,input, input)
    print('attn_output:',attn_output.shape,'weights:',attn_output_weights.shape) #(1000,1,272)
    return attn_output.detach().numpy()

    

