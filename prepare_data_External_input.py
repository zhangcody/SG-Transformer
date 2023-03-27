import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


def sliding_window(train, sw_width=1, n_out=7, in_start=0):
    data = train
    X, Y = [], []
    for _ in range(len(data)):
        in_end = in_start + n_out  # 7
        out_end = in_end + n_out  # 8
        if out_end < len(data):
            train_seq = data[in_start:in_end]
            train_seq = train_seq.reshape((len(train_seq)), 1)
            X.append(train_seq)
            Y.append(data[in_end])
        in_start += sw_width
    return np.array(X), np.array(Y)

def sliding_window_n(train, sw_width=3, n_out=27, in_start=0):
    data = train
    X, Y = [], []
    for _ in range(len(data)):
        in_end = in_start + n_out  # 7
        out_end = in_end + n_out  # 8
        inEnd = in_end-3
        if out_end < len(data):
            train_seq = data[in_start:inEnd]
            train_seq = train_seq.reshape((len(train_seq)), 1)
            X.append(train_seq)
            test_seq = data[inEnd:in_end]
            Y.append(test_seq)
        in_start += sw_width
    return np.array(X), np.array(Y)

class MyData(Dataset):
    def __init__(self,dataset,n_out=7):
        x, y = sliding_window(dataset,n_out=n_out)
        self.x, self.y = torch.from_numpy(x).float(),torch.from_numpy(y).float()

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.y)

def get_dataloader(dataset,batch_size,n_out=7):
    data = MyData(dataset,n_out=n_out)
    return DataLoader(dataset=data,batch_size=batch_size,shuffle=False,drop_last=True)

def get_dataloader_shuffle(dataset,batch_size,n_out=7):
    data = MyData(dataset,n_out=n_out)
    return DataLoader(dataset=data,batch_size=batch_size,shuffle=True,drop_last=True)

def get_dataloader_shuffle_n(dataset,batch_size,n_out=27):
    data = MyData_n(dataset,n_out=n_out)
    return DataLoader(dataset=data,batch_size=batch_size,shuffle=True,drop_last=True)

class MyData_n(Dataset):
    def __init__(self,dataset,n_out=27):
        x, y = sliding_window_n(dataset,n_out=n_out)
        self.x, self.y = torch.from_numpy(x).float(),torch.from_numpy(y).float()

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.y)