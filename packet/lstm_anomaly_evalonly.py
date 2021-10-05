from feature_extractor import pcap_to_dataframe, preprocess_dataframe

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc

from matplotlib import pyplot as plt
import seaborn as sns


def label_by_ip(df :pd.DataFrame, ip_list):
    labels = np.zeros(df.shape[0])
    labels[df["ip_src"].apply(lambda x: x in ip_list)] = 1
    labels[df["ip_dst"].apply(lambda x: x in ip_list)] = 1
    return labels


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


# trained model
model = LSTMchar()
ckpt = torch.load("22jun/clustered_captures/cluster3/fl_28jun_lstm_10rounds_5localepochs/models_global/round_10/global_model_round_10.tar")
model.load_state_dict(ckpt["state_dict"])
loss_func = F.mse_loss

# dataset eval
df = pcap_to_dataframe("06may_normal_ataque/iot-client-bot-1_normal.pcap")
attack_victim_ip = ("192.168.0.254", "192.168.0.50")

labels = label_by_ip(df, attack_victim_ip)

df = preprocess_dataframe(df)
timestamps = df["timestamp"].values
df = df.drop(columns=["timestamp"])
seq_len = 25
results = prediction_error(model, loss_func, *make_sequences(torch.from_numpy(df.to_numpy(dtype=np.float32)), seq_len))
results = results.numpy()

results_df = pd.DataFrame({"ts":timestamps[seq_len:], "rec_err":results, "label": labels[seq_len:]})
sns.scatterplot(data=results_df, x="ts", y="rec_err", hue="label", linewidth=0, alpha=0.4)
plt.show()

fpr, tpr, thresholds = roc_curve(labels[10:], results, pos_label=1)
print(auc(fpr, tpr))
plt.plot(fpr, tpr)
plt.show()