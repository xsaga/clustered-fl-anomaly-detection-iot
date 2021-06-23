from feature_extractor import pcap_to_dataframe, preprocess_dataframe

from collections import OrderedDict
from typing import Dict, List, Tuple, Union
from pathlib import Path
import hashlib
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split


def split_train_valid_eval(df :pd.DataFrame, eval_split=None, train_size=0.8):
    if eval_split:
        df_train_valid, df_eval = train_test_split(df, shuffle=False, train_size=eval_split)
        df_train, df_valid = train_test_split(df_train_valid, shuffle=False, train_size=train_size)
        return df_train, df_valid, df_eval
    else:
        df_train, df_valid = train_test_split(df, shuffle=False, train_size=train_size)
        return df_train, df_valid, None


def state_dict_hash(state_dict: Union['OrderedDict[str, torch.Tensor]', Dict[str, torch.Tensor]]) -> str:
    h = hashlib.md5()
    for k, v in state_dict.items():
        h.update(k.encode("utf-8"))
        h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()


def make_sequences(data, sequence_len):
    data_seq_x = []
    data_seq_y = []
    for i in range(data.shape[0] - sequence_len):
        data_seq_x.append(data[i:i+sequence_len, :])
        data_seq_y.append(data[i+1:i+sequence_len+1, :])
    return torch.stack(data_seq_x, dim=0), torch.stack(data_seq_y, dim=0)


def load_data(pcap_filename):
    df = pcap_to_dataframe(pcap_filename)
    df = preprocess_dataframe(df)
    df = df.drop(columns=["timestamp"])
    df_train, df_valid, _ = split_train_valid_eval(df, train_size=0.8)
    X_train = torch.from_numpy(df_train.to_numpy(dtype=np.float32))
    X_valid = torch.from_numpy(df_valid.to_numpy(dtype=np.float32))
    seq_len = 10
    train_x, train_y = make_sequences(X_train, seq_len)
    valid_x, valid_y = make_sequences(X_valid, seq_len)
    bs = 32
    train_ds = TensorDataset(train_x, train_y)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=False)
    valid_ds = TensorDataset(valid_x, valid_y)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False)
    return train_dl, valid_dl


class LSTMchar(nn.Module):
    def __init__(self):
        super(LSTMchar, self).__init__()
        self.start_dim = 27

        self.lstm = nn.LSTM(self.start_dim, 128)
        self.linear = nn.Linear(128, self.start_dim)
        self.activ = nn.Softmax(dim=2)  # nn.ReLU()

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        out = self.activ(out)
        out = self.linear(out)
        return out


def fit(model, optimizer, loss_function, epochs, train_generator):
    model.train()
    for epoch in range(epochs):
        train_loss_acc = 0.0
        for x_batch, y_batch in train_generator:
            x_batch = x_batch.transpose(0, 1)

            preds = model(x_batch)
            preds = preds.transpose(0, 1)

            loss = loss_function(preds, y_batch)
            train_loss_acc += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print(f"epoch {epoch+1}/{epochs}: train loss {train_loss_acc/len(train_generator):.8f}")


model = LSTMchar()
torch.set_num_threads(1)
initial_random_model_path = "initial_random_model_clustering_lstm.pt"
model.load_state_dict(torch.load(initial_random_model_path), strict=True)
print(f"Loaded initial model {initial_random_model_path} with hash {state_dict_hash(model.state_dict())}")

pcap_filename = sys.argv[1]
print(f"Loading data {pcap_filename}... ")
train_dl, _ = load_data(pcap_filename)
print("Loaded")

num_epochs = int(sys.argv[2])
opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_func = F.mse_loss
print(f"Training for {num_epochs} epochs in {pcap_filename}")
fit(model, optimizer=opt, loss_function=loss_func, epochs=num_epochs, train_generator=train_dl)
print(f"Fitted model hash {state_dict_hash(model.state_dict())}")
model_savefile = f"{Path(pcap_filename).stem}_{num_epochs}epochs_lstm.pt"
torch.save(model.state_dict(), model_savefile)
print(f"Saved model in {model_savefile}")
