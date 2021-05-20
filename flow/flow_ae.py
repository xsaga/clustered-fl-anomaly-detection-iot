import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

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


def label_by_ip(df :pd.DataFrame, ip_list):
    labels = np.zeros(df.shape[0])
    labels[df["Src IP"].apply(lambda x: x in ip_list)] = 1
    labels[df["Dst IP"].apply(lambda x: x in ip_list)] = 1
    return labels


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(76, 32), # 26,12
            # nn.Dropout(0.05), # !
            nn.ReLU(), # nn.LeakyReLU(), nn.ReLU()
            nn.Linear(32, 8), # 12,4
            nn.ReLU() # nn.Sigmoid(), nn.Tanh(), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 32), # 4,12
            nn.ReLU(), # nn.LeakyReLU(), nn.ReLU()
            nn.Linear(32, 76), # 12,26
            # nn.Dropout(0.05), # !
            nn.ReLU() # nn.ReLU()
        )

    def forward(self, x):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return decoded


def fit(model, optimizer, loss_func, epochs, train_generator, valid_generator):
    for epoch in range(epochs):
        train_loss_acc = 0.0
        valid_loss_acc = 0.0
        model.train()
        for x_batch in train_generator:
            preds = model(x_batch)
            loss = loss_func(preds, x_batch)
            train_loss_acc += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        with torch.no_grad():
            model.eval()
            for x_batch in valid_generator:
                preds = model(x_batch)
                valid_loss_acc += loss_func(preds, x_batch).item()
        
        print(f"epoch {epoch+1}/{epochs}: train loss {train_loss_acc/len(train_generator):.8f}; valid loss {valid_loss_acc/len(valid_generator):.8f}")


def reconstruction_error(model, loss_func, samples):
    with torch.no_grad():
        model.eval()
        predictions = model(samples)
        rec_error = torch.mean(loss_func(samples, predictions, reduction="none"), dim=1)
    return rec_error


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

bs = 32
train_dl = DataLoader(X_train, batch_size=bs, shuffle=True)
valid_dl = DataLoader(X_valid, batch_size=bs, shuffle=False)

model = Autoencoder()
lr = 1e-3  # 1e-3
num_epochs = 100  # 100
loss_func = F.mse_loss
opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

fit(model, opt, loss_func, num_epochs, train_dl, valid_dl)

results = reconstruction_error(model, loss_func, torch.from_numpy(scaler.transform(df).astype(np.float32)))
results = results.numpy()

results_df = pd.DataFrame({"ts":df_timestamp, "rec_err":results})
sns.scatterplot(data=results_df, x="ts", y="rec_err", linewidth=0, alpha=0.4).set(title=f"AE. {train_flowmeter_filename}")
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

results_eval = reconstruction_error(model, loss_func, torch.from_numpy(scaler.transform(df_eval).astype(np.float32)))
results_eval_df = pd.DataFrame({"ts":df_eval_timestamp, "rec_err":results_eval, "label":df_eval_labels})
sns.scatterplot(data=results_eval_df, x="ts", y="rec_err", hue="label", linewidth=0, alpha=0.4).set(yscale="log", title=f"AE. {eval_flowmeter_filename}")
plt.show()
