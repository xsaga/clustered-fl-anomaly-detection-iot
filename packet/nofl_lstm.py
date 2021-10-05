# testear arquitectura y params
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
from matplotlib import pyplot as plt
import seaborn as sns


def state_dict_hash(state_dict: Union['OrderedDict[str, torch.Tensor]', Dict[str, torch.Tensor]]) -> str:
    h = hashlib.md5()
    for k, v in state_dict.items():
        h.update(k.encode("utf-8"))
        h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()


def split_train_valid_eval(df :pd.DataFrame, eval_split=None, train_size=0.8):
    if eval_split:
        df_train_valid, df_eval = train_test_split(df, shuffle=False, train_size=eval_split)
        df_train, df_valid = train_test_split(df_train_valid, shuffle=False, train_size=train_size)
        return df_train, df_valid, df_eval
    else:
        df_train, df_valid = train_test_split(df, shuffle=False, train_size=train_size)
        return df_train, df_valid, None


def make_sequences(data, sequence_len):
    data_seq_x = []
    data_seq_y = []
    for i in range(data.shape[0] - sequence_len):
        data_seq_x.append(data[i:i+sequence_len, :])
        data_seq_y.append(data[i+1:i+sequence_len+1, :])
    return torch.stack(data_seq_x, dim=0), torch.stack(data_seq_y, dim=0)


def load_data(pcap_filename, seq_len=10):
    df = pcap_to_dataframe(pcap_filename)
    df = preprocess_dataframe(df)
    df = df.drop(columns=["timestamp"])
    df_train, df_valid, _ = split_train_valid_eval(df, train_size=0.8)
    X_train = torch.from_numpy(df_train.to_numpy(dtype=np.float32))
    X_valid = torch.from_numpy(df_valid.to_numpy(dtype=np.float32))
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

        self.lstm = nn.LSTM(self.start_dim, 64, 2)
        self.linear = nn.Linear(64, self.start_dim)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        out = self.linear(out)
        return out


def fit(model, optimizer, loss_function, epochs, train_generator, valid_generator):
    epoch_list = []
    loss_list = []
    valid_loss_list = []
    for epoch in range(epochs):
        model.train()
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
        epoch_list.append(epoch)
        loss_list.append(train_loss_acc/len(train_generator))
        valid_loss_list.append(test(model, loss_func, valid_generator))
    return epoch_list, loss_list, valid_loss_list


def test(model, loss_function, valid_generator):
    valid_loss_acc = 0.0
    with torch.no_grad():
        model.eval()
        for x_batch, y_batch in valid_generator:
            x_batch = x_batch.transpose(0, 1)
            preds = model(x_batch)
            preds = preds.transpose(0, 1)
            valid_loss_acc += loss_function(preds, y_batch).item()
    print(f"valid loss {valid_loss_acc/len(valid_generator):.8f}")
    return valid_loss_acc/len(valid_generator)


def prediction_error(model, loss_func, samples_x, samples_y):
    with torch.no_grad():
        model.eval()
        samples_x = samples_x.transpose(0, 1)
        samples_y = samples_y.transpose(0, 1)
        pred = model(samples_x)
        future = pred[-1, :, :]
        target = samples_y[-1, :, :]
        results = torch.mean(loss_func(future, target, reduction="none"), dim=1)
    return results



pcap_filename = "mqtt-device-t2_reducido.pcap"
print(f"Loading data {pcap_filename}...")
seq_len = 25
train_dl, valid_dl = load_data(pcap_filename, seq_len)

df = pcap_to_dataframe(pcap_filename)
df = preprocess_dataframe(df)
timestamps = df["timestamp"].values
df = df.drop(columns=["timestamp"])


torch.set_num_threads(2)

num_epochs = 50
model = LSTMchar()
opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-7)
loss_func = F.mse_loss

s = 0
for k,v in model.state_dict().items():
    s += torch.numel(v)*4
print(s)

# train
print(f"Training for {num_epochs} epochs in {pcap_filename}, number of samples {len(train_dl)}.")
epoch_l, loss_l, valid_loss_l = fit(model, optimizer=opt, loss_function=loss_func, epochs=num_epochs, train_generator=train_dl, valid_generator=valid_dl)
plt.plot(epoch_l, loss_l, marker="o", label="train loss")
plt.plot(epoch_l, valid_loss_l, marker="o", label="valid loss")
plt.legend()
plt.show()
# np.savez(f"results_nofl_epoch_losses_lstm_{Path(pcap_filename).stem}.npz", np.array(epoch_l), np.array(loss_l), np.array(valid_loss_l))


results = prediction_error(model, loss_func, *make_sequences(torch.from_numpy(df.to_numpy(dtype=np.float32)), seq_len))

results_df = pd.DataFrame({"ts":timestamps[seq_len:], "rec_err":results})
sns.scatterplot(data=results_df, x="ts", y="rec_err", linewidth=0, alpha=0.4)
plt.show()


results_eval = prediction_error(model, loss_func, *make_sequences(torch.from_numpy(df_eval.to_numpy(dtype=np.float32)), seq_len))
results_df = pd.DataFrame({"ts":timestamps_eval[seq_len:], "rec_err":results_eval})
sns.scatterplot(data=results_df, x="ts", y="rec_err", linewidth=0, alpha=0.4)
plt.show()
