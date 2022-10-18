import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

def get_data_iter(src,trg,batch_size):
    print('src:',src.shape)
    print('trg:',trg.shape)
    data = TensorDataset(src, trg)
    print('len(data):',len(data))
    # print('data:',data)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    for _ in range(5):
        for inputs,labels in data_loader:
            # print('inputs:',inputs)
            # print('labels:',labels)
            break
    # for i_batch, batch_data in enumerate(data_loader):
    #     print(i_batch)  # 打印batch编号
    #     # print('batch.shape:',batch_data.shape)
    #     print(batch_data[0].shape)  # 打印该batch里面src
    #     print(batch_data[1].shape)  # 打印该batch里面trg
    return data_loader
