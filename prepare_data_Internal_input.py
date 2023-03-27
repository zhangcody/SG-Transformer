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


class MyData(Dataset):
    def __init__(self,n_out=7):
        dataset = pd.read_csv('data/fuhe.csv', header=0, index_col=0, parse_dates=True)
        df = pd.DataFrame(dataset)  # 整体数据的全部字典类型
        do = df['溶解氧(mg/L)']  # 返回溶解氧那一列，用字典的方式
        DO = []
        for i in range(0, len(do)):
            DO.append([do[i]])
        scaler_DO = MinMaxScaler(feature_range=(0, 1))
        DO = scaler_DO.fit_transform(DO)
        x, y = sliding_window(DO,n_out=n_out)
        self.x, self.y = torch.from_numpy(x).float(),torch.from_numpy(y).float()

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.y)

def get_dataloader(batch_size,n_out=7):
    data = MyData(n_out=n_out)
    return DataLoader(dataset=data,batch_size=batch_size,shuffle=True,drop_last=True)