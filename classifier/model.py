import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self,input):
        super(CNN,self).__init__()
        input_size = input.shape[-1]
        self.net = nn.Sequential(
            nn.Conv1d(input_size,64,kernel_size=3),nn.ReLU(),
            nn.Conv1d(64,32,kernel_size=3),nn.ReLU(),
            nn.Conv1d(32,16,kernel_size=3),nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Flatten(),
            nn.Linear(16,2)
        )

class GRU(nn.Module):
    def __init__(self,input):
        super().__init__()
        input_size = input.shape[-1]
        self.net = nn.GRU(input_size,64)