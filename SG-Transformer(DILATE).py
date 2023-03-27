from math import sqrt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from prepare_data_External_input import get_dataloader,get_dataloader_shuffle
from Network import *
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from DILATE.loss.dilate_loss import dilate_loss
from utils import plot_curve,plot_singleLine
import scipy.signal as sg
import warnings; warnings.simplefilter('ignore')

#hyperparams
enc_seq_len = 6
dec_seq_len = 1
output_sequence_length = 1

dim_val = 10
dim_attn = 5
lr = 0.005
epochs = 11
n_heads = 5

n_decoder_layers = 3
n_encoder_layers = 3

batch_size = 15
scaler_DO = MinMaxScaler(feature_range=(0, 1))

#init network and optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
t = Transformer(dim_val, dim_attn, 1,dec_seq_len,  output_sequence_length, n_decoder_layers, n_encoder_layers, n_heads)
optimizer = torch.optim.Adam(t.parameters(), lr=lr)

dataset = pd.read_csv('data/fuhe.csv', header=0, index_col=0, parse_dates=True)
df = pd.DataFrame(dataset)  # 整体数据的全部字典类型
do = df['溶解氧(mg/L)']  # 返回溶解氧那一列，用字典的方式
DO = []
for i in range(0, len(do)):
    DO.append([do[i]])
sg = sg.savgol_filter(do, 5, 2)
scaler_DO = MinMaxScaler(feature_range=(0, 1))
DO_sg = scaler_DO.fit_transform(np.array(sg).reshape(-1,1))
DO_train = DO_sg[9900:11000]
dataloader = get_dataloader_shuffle(DO_train,batch_size,n_out=24)

def tensor_list(t):
    a = torch.arange(1, 16, 1).view(15, 1)
    x = torch.cat((a, t), 1)
    d = torch.tensor(x).view(15, 2, -1)
    d = d.tolist()
    out = torch.tensor(d)
    return out
#keep track of loss for graph
losses = []
rmses = []
Y_Pre_list = []
Y_Real_list = []
criterion = torch.nn.MSELoss()
for e in range(epochs):
    out = []
    rmse_out = []

    for i, data in enumerate(dataloader, 0):

        inputs, target= data
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        target = torch.tensor(target, dtype=torch.float32).to(device)
        outputs = t(inputs)

        net_out = outputs
        Y_ = target
        outputs = tensor_list(outputs)
        target = tensor_list(target)
        loss, loss_shape, loss_temporal = dilate_loss(outputs, target, 0.5, 0.001, device)
        Y = Y_.reshape(len(Y_),1)
        Y_pre = net_out.detach().numpy()

        Y_preList = Y_pre.tolist()
        Y_realList = Y.tolist()

        if e==10:
            Y_Pre_list.append(Y_preList)
            Y_Real_list.append(Y_realList)
        mse = mean_squared_error(Y,Y_pre)
        rmse = sqrt(mse)
        rmse_out.append(rmse)
        # backwards pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track losses and draw rgaph
        # out.append([net_out.detach().numpy(), Y])
        out.append(loss.detach().numpy())

    losses.append(sum(out)/len(out))
    rmses.append(sum(rmse_out)/len(rmse_out))
        # break

    if e==60:
        torch.save(t,"../model/SG-Transformer(DILATE).pth")
        Pre = [c for a in Y_Pre_list for b in a for c in b]
        Real = [c for a in Y_Real_list for b in a for c in b]
        y_pre = scaler_DO.inverse_transform(np.array(Pre).reshape(-1,1))
        y_real = scaler_DO.inverse_transform(np.array(Real).reshape(-1,1))
        print('predict', len(y_pre))
        print('Y', len(y_real))
        plot_curve(y_pre,y_real)
    print('rmse',e,rmses[-1])
    print('epoch',e,losses[-1])
    # break
print(losses)
print(rmses)
plot_singleLine(losses)
plot_singleLine(rmses)