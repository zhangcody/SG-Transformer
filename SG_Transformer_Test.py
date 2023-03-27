from math import sqrt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from prepare_data_External_input import get_dataloader
from Network import *
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from utils import plot_curve
import scipy.signal as sg

#hyperparams
enc_seq_len = 6
dec_seq_len = 2
output_sequence_length = 1

dim_val = 10
dim_attn = 5
lr = 0.002
epochs = 1
n_heads = 1

n_decoder_layers = 3
n_encoder_layers = 3

batch_size = 15
scaler_DO = MinMaxScaler(feature_range=(0, 1))

#init network and optimizer
t = Transformer(dim_val, dim_attn, 1,dec_seq_len,  output_sequence_length, n_decoder_layers, n_encoder_layers, n_heads)
optimizer = torch.optim.Adam(t.parameters(), lr=lr)
xlf_lo = torch.load("./model/model_SG_transformer.pth")  # 读取模型
xlf_lo.eval()

# 数据准备
dataset = pd.read_csv('data/fuhe.csv', header=0, index_col=0, parse_dates=True)
df = pd.DataFrame(dataset)  # 整体数据的全部字典类型
do = df['溶解氧(mg/L)']  # 返回溶解氧那一列，用字典的方式
DO = []
for i in range(0, len(do)):
    DO.append([do[i]])
sg = sg.savgol_filter(do, 5, 2)
scaler_DO = MinMaxScaler(feature_range=(0, 1))
DO_sg = scaler_DO.fit_transform(np.array(sg).reshape(-1,1))
DO_test = DO_sg[9900:11000]
dataloader = get_dataloader(DO_test,batch_size,n_out=6)

#keep track of loss for graph
losses = []
out = []
rmse_out = []
rmses = []
Y_Pre_list = []
Y_Real_list = []

for X_, Y_ in dataloader:
    optimizer.zero_grad()

    net_out = xlf_lo(X_)

    Y = Y_.reshape(len(Y_),1)
    loss = torch.mean((net_out - Y) ** 2)
    Y_pre = net_out.detach().numpy()

    Y_preList = Y_pre.tolist()
    Y_realList = Y.tolist()

    Y_Pre_list.append(Y_preList)
    Y_Real_list.append(Y_realList)
    mse = mean_squared_error(Y,Y_pre)
    rmse = sqrt(mse)
    rmse_out.append(rmse)
    # backwards pass
    loss.backward()
    optimizer.step()

    out.append(loss.detach().numpy())

losses.append(sum(out)/len(out))
rmses.append(sum(rmse_out)/len(rmse_out))
Pre = [c for a in Y_Pre_list for b in a for c in b]
Real = [c for a in Y_Real_list for b in a for c in b]
y_pre = scaler_DO.inverse_transform(np.array(Pre).reshape(-1,1))
y_real = scaler_DO.inverse_transform(np.array(Real).reshape(-1,1))
print('predict', len(y_pre))
print('Y', len(y_real))
plot_curve(y_pre,y_real)
print('rmse',rmses[-1])
print('epoch',losses[-1])