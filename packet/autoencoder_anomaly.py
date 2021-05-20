from feature_extractor import pcap_to_dataframe, preprocess_dataframe

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(27, 12), # 26,12
            # nn.Dropout(0.05), # !
            nn.ReLU(), # nn.LeakyReLU(), nn.ReLU()
            nn.Linear(12, 4), # 12,4
            nn.ReLU()  # nn.Sigmoid() # nn.Tanh(), nn.Sigmoid(), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 12), # 4,12
            nn.ReLU(), # nn.LeakyReLU(), nn.ReLU()
            nn.Linear(12, 27), # 12,26
            # nn.Dropout(0.05), # !
            nn.ReLU() # nn.ReLU()
        )

    def forward(self, x):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return decoded


def fit(model, optimizer, loss_function, epochs, train_generator, valid_generator):
    for epoch in range(epochs):
        train_loss_acc = 0.0
        valid_loss_acc = 0.0
        model.train()
        for x_batch in train_generator:
            preds = model(x_batch)
            loss = loss_function(preds, x_batch)
            train_loss_acc += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        with torch.no_grad():
            model.eval()
            for x_batch in valid_generator:
                preds = model(x_batch)
                valid_loss_acc += loss_function(preds, x_batch).item()
        
        print(f"epoch {epoch+1}/{epochs}: train loss {train_loss_acc/len(train_generator):.8f}; valid loss {valid_loss_acc/len(valid_generator):.8f}")


def reconstruction_error(model, loss_function, samples):
    with torch.no_grad():
        model.eval()
        predictions = model(samples)
        rec_error = torch.mean(loss_function(samples, predictions, reduction="none"), dim=1)
    return rec_error


df = pcap_to_dataframe("iot-client-bot-1_normal.pcap")
attack_victim_ip = ("192.168.0.254", "192.168.0.50")

## labels para los ataques
labels = label_by_ip(df, attack_victim_ip)

df = preprocess_dataframe(df)
timestamps = df["timestamp"].values
df = df.drop(columns=["timestamp"])

df_train, df_valid, _ = split_train_valid_eval(df, train_size=0.7)
X_train = torch.from_numpy(df_train.to_numpy(dtype=np.float32))
X_valid = torch.from_numpy(df_valid.to_numpy(dtype=np.float32))

bs = 32  # 64
train_dl = DataLoader(X_train, batch_size=bs, shuffle=True)
valid_dl = DataLoader(X_valid, batch_size=bs, shuffle=False)

model = Autoencoder()

lr = 1e-3  # 1e-3
num_epochs = 100  # 100
loss_func = F.mse_loss
# opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# importante el weight_decay. Sin el decay (=0) los resultados
# varian mucho segun el entrenamiento. Con el decay cada entrenamiento
# es más consistente.
opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
# opt = optim.RMSprop(model.parameters(), lr=lr)

fit(model, opt, loss_func, num_epochs, train_dl, valid_dl)

results = reconstruction_error(model, loss_func, torch.from_numpy(df.to_numpy(dtype=np.float32)))
results = results.numpy()

#plt.plot(results)
#plt.vlines(train_valid_split, 0, max(results), color="black")
#plt.show()
results_df = pd.DataFrame({"ts":timestamps, "rec_err":results, "label": labels})
sns.scatterplot(data=results_df, x="ts", y="rec_err", hue="label", linewidth=0, alpha=0.4)
plt.show()

results=np.array(results)
th = np.max(results[:13100])
confusion_matrix(labels, (results>th)+0)