import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler


def flowmeter_to_df(filename :str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df.sort_values(by="Timestamp", inplace=True)
    return df


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


def label_by_ip(df :pd.DataFrame, ip_list):
    labels = np.zeros(df.shape[0])
    labels[df["Src IP"].apply(lambda x: x in ip_list)] = 1
    labels[df["Dst IP"].apply(lambda x: x in ip_list)] = 1
    return labels


class LSTMchar(nn.Module):
    def __init__(self, start_dim):
        super(LSTMchar, self).__init__()
        self.start_dim = start_dim

        self.lstm = nn.LSTM(self.start_dim, 128)
        self.linear = nn.Linear(128, self.start_dim)
        self.activ = nn.Softmax(dim=2)  # nn.ReLU()

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        out = self.activ(out)
        out = self.linear(out)
        return out


def fit(model, optimizer, loss_func, epochs, train_generator, valid_generator):
    for epoch in range(epochs):
        train_loss_acc = 0.0
        valid_loss_acc = 0.0
        model.train()
        for x_batch, y_batch in train_generator:
            x_batch = x_batch.transpose(0, 1)
            
            preds = model(x_batch)
            preds = preds.transpose(0, 1)

            loss = loss_func(preds, y_batch)
            train_loss_acc += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        with torch.no_grad():
            model.eval()
            for x_batch, y_batch in valid_generator:
                x_batch = x_batch.transpose(0, 1)

                preds = model(x_batch)
                preds = preds.transpose(0, 1)
                valid_loss_acc += loss_func(preds, y_batch).item()
        
        print(f"epoch {epoch+1}/{epochs}: train loss {train_loss_acc/len(train_generator):.8f}; valid loss {valid_loss_acc/len(valid_generator):.8f}")


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


train_flowmeter_filename = "coap-iot-bot-1_normal.pcap_Flow.csv"
eval_flowmeter_filename = "coap-iot-bot-1_attack.pcap_Flow.csv"
attack_victim_ip = ("192.168.0.254", "192.168.0.90")

df = flowmeter_to_df(train_flowmeter_filename)
df_timestamp = df["Timestamp"]

df = df.drop(columns=["Flow ID",
                      "Src IP",
                      "Src Port",
                      "Dst IP",
                      "Dst Port",
                      "Protocol",
                      "Timestamp",
                      "Label"])

df_train, df_valid, _ = split_train_valid_eval(df, train_size=0.85)

scaler = MinMaxScaler()
np_train = scaler.fit_transform(df_train)
np_valid = scaler.transform(df_valid)

X_train = torch.from_numpy(np_train.astype(np.float32))
X_valid = torch.from_numpy(np_valid.astype(np.float32))

seq_len = 10
train_x, train_y = make_sequences(X_train, seq_len)
valid_x, valid_y = make_sequences(X_valid, seq_len)

bs = 32
train_ds = TensorDataset(train_x, train_y)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=False)

valid_ds = TensorDataset(valid_x, valid_y)
valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False)

model = LSTMchar(df.shape[1])
loss_func = F.mse_loss
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
num_epochs = 100

fit(model, opt, loss_func, num_epochs, train_dl, valid_dl)

results = prediction_error(model, loss_func, *make_sequences(torch.from_numpy(scaler.transform(df).astype(np.float32)), seq_len))

results_df = pd.DataFrame({"ts":df_timestamp[seq_len:], "rec_err":results})
sns.scatterplot(data=results_df, x="ts", y="rec_err", linewidth=0, alpha=0.4).set(title=f"LSTM. {train_flowmeter_filename}")
plt.show()

# ###

df_eval = flowmeter_to_df(eval_flowmeter_filename)
df_eval_timestamp = df_eval["Timestamp"]
df_eval_labels = label_by_ip(df_eval, attack_victim_ip)
df_eval = df_eval.drop(columns=["Flow ID",
                                "Src IP",
                                "Src Port",
                                "Dst IP",
                                "Dst Port",
                                "Protocol",
                                "Timestamp",
                                "Label"])

results_eval = prediction_error(model, loss_func, *make_sequences(torch.from_numpy(scaler.transform(df_eval).astype(np.float32)), seq_len))
results_eval_df = pd.DataFrame({"ts":df_eval_timestamp[seq_len:], "rec_err":results_eval, "label":df_eval_labels[seq_len:]})
sns.scatterplot(data=results_eval_df, x="ts", y="rec_err", hue="label", linewidth=0, alpha=0.4).set(yscale="log", title=f"LSTM. {eval_flowmeter_filename}")
plt.show()
