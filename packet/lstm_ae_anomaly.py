from feature_extractor import pcap_to_dataframe, preprocess_dataframe

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from matplotlib import pyplot as plt


class LSTMchar(nn.Module):
    def __init__(self, start_dim):
        super(LSTMchar, self).__init__()
        self.start_dim = start_dim

        self.lstm_e1 = nn.LSTM(self.start_dim, 32)
        self.lstm_e2 = nn.LSTM(32, 12)

        self.lstm_d1 = nn.LSTM(12, 32)
        self.lstm_d2 = nn.LSTM(32, self.start_dim)

    def forward(self, x):
        out, (h, c) = self.lstm_e1(x)
        out, (h, c) = self.lstm_e2(out)

        x_d = h.repeat(10, 1, 1) # 10=maxlen

        out, (h, c) = self.lstm_d1(x_d)
        out, (h, c) = self.lstm_d2(out)

        return out


df = pcap_to_dataframe("iot-client-1_eth0_to_Switch2_Ethernet1.pcap")
df = preprocess_dataframe(df)
timestamps = df["timestamp"].values
df = df.drop(columns=["timestamp"])

train_valid_split = int(df.shape[0] * 0.70)
X_train = torch.from_numpy(df[:train_valid_split].to_numpy().astype(np.float32))
X_valid = torch.from_numpy(df[train_valid_split:].to_numpy().astype(np.float32))

maxlen = 10
train_x = []
for i in range(X_train.shape[0] - maxlen - 1):
    train_x.append(X_train[i:i+maxlen, :])
train_x = torch.stack(train_x, dim=0)

valid_x = []
for i in range(X_valid.shape[0] - maxlen - 1):
    valid_x.append(X_valid[i:i+maxlen, :])
valid_x = torch.stack(valid_x, dim=0)

bs = 32

train_dl = DataLoader(train_x, batch_size=bs, shuffle=False)

valid_dl = DataLoader(valid_x, batch_size=bs, shuffle=False)

model = LSTMchar(df.shape[1])
loss_func = F.mse_loss
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
num_epochs = 100


def fit(epochs, train_generator, valid_generator):
    for epoch in range(epochs):
        train_loss_acc = 0.0
        valid_loss_acc = 0.0
        model.train()
        for x_batch in train_generator:
            x_batch = x_batch.transpose(0, 1)
            
            preds = model(x_batch)
            preds = preds.transpose(0, 1)
            x_batch = x_batch.transpose(0, 1)

            loss = loss_func(preds, x_batch)
            train_loss_acc += loss.item()

            loss.backward()
            opt.step()
            opt.zero_grad()
        
        with torch.no_grad():
            model.eval()
            for x_batch in valid_generator:
                x_batch = x_batch.transpose(0, 1)

                preds = model(x_batch)
                preds = preds.transpose(0, 1)
                x_batch = x_batch.transpose(0, 1)
                valid_loss_acc += loss_func(preds, x_batch).item()
        
        print(f"epoch {epoch+1}/{epochs}: train loss {train_loss_acc/len(train_generator):.8f}; valid loss {valid_loss_acc/len(valid_generator):.8f}")


fit(num_epochs, train_dl, valid_dl)

with torch.no_grad():
    recerror=[]
    Xnp = df.to_numpy().astype(np.float32)
    X = torch.from_numpy(Xnp)
    for i in range(X.shape[0]-maxlen):
        xi = X[i:i+maxlen, :].view(maxlen, 1, -1)
        pred = model(xi)
        recerror.append(loss_func(pred.transpose(0,1), xi.transpose(0,1)).item())

plt.plot(recerror)
plt.show()
