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


def load_data(pcap_filename):
    df = pcap_to_dataframe(pcap_filename)
    df = preprocess_dataframe(df)
    df = df.drop(columns=["timestamp"])
    df_train, df_valid, _ = split_train_valid_eval(df, train_size=0.8)
    X_train = torch.from_numpy(df_train.to_numpy(dtype=np.float32))
    X_valid = torch.from_numpy(df_valid.to_numpy(dtype=np.float32))
    bs = 32
    train_dl = DataLoader(X_train, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(X_valid, batch_size=bs, shuffle=False)
    return train_dl, valid_dl


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(34, 17), # 26,12
            # nn.Dropout(0.05), # !
            nn.ReLU(), # nn.LeakyReLU(), nn.ReLU()
            nn.Linear(17, 8), # 12,4
            nn.ReLU()  # nn.Sigmoid() # nn.Tanh(), nn.Sigmoid(), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 17), # 4,12
            nn.ReLU(), # nn.LeakyReLU(), nn.ReLU()
            nn.Linear(17, 34), # 12,26
            # nn.Dropout(0.05), # !
            nn.ReLU() # nn.ReLU()
        )

    def forward(self, x):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return decoded


def fit(model, optimizer, loss_function, epochs, train_generator, valid_generator):
    epoch_list = []
    loss_list = []
    valid_loss_list = []
    for epoch in range(epochs):
        model.train()
        train_loss_acc = 0.0
        for x_batch in train_generator:
            preds = model(x_batch)
            loss = loss_function(preds, x_batch)
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
        for x_batch in valid_generator:
            preds = model(x_batch)
            valid_loss_acc += loss_function(preds, x_batch).item()
    print(f"valid loss {valid_loss_acc/len(valid_generator):.8f}")
    return valid_loss_acc/len(valid_generator)


def reconstruction_error(model, loss_function, samples):
    with torch.no_grad():
        model.eval()
        predictions = model(samples)
        rec_error = torch.mean(loss_function(samples, predictions, reduction="none"), dim=1)
    return rec_error


# datasets
pcap_filename = "mqtt-device-t2-1_0-0_to_OpenvSwitch-13_1-0.pcap"
print(f"Loading data {pcap_filename}...")
train_dl, valid_dl = load_data(pcap_filename)

df = pcap_to_dataframe(pcap_filename)
df = preprocess_dataframe(df)
timestamps = df["timestamp"].values
df = df.drop(columns=["timestamp"])

eval_pcap_filename = "mqtt-device-t2-17_0-0_to_OpenvSwitch-14_2-0.pcap"
df_eval = pcap_to_dataframe(eval_pcap_filename)
df_eval = preprocess_dataframe(df_eval)
timestamps_eval = df_eval["timestamp"].values
df_eval = df_eval.drop(columns=["timestamp"])




num_epochs = 30
model = Autoencoder()
opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
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
#np.savez(f"results_nofl_epoch_losses_ae_{Path(pcap_filename).stem}.npz", np.array(epoch_l), np.array(loss_l), np.array(valid_loss_l))

# th
results = reconstruction_error(model, loss_func, torch.from_numpy(df.to_numpy(dtype=np.float32)))
results = results.numpy()

results_df = pd.DataFrame({"ts":timestamps, "rec_err":results})
sns.scatterplot(data=results_df, x="ts", y="rec_err", linewidth=0, alpha=0.4)
plt.show()

th_max = np.max(results)
th_99 = np.percentile(results, 99)

# eval
results_eval = reconstruction_error(model, loss_func, torch.from_numpy(df_eval.to_numpy(dtype=np.float32)))
results_eval = results_eval.numpy()

results_df = pd.DataFrame({"ts":timestamps_eval, "rec_err":results_eval})
sns.scatterplot(data=results_df, x="ts", y="rec_err", linewidth=0, alpha=0.4)
plt.hlines(th_max, timestamps_eval[0], timestamps_eval[-1], color="red")
plt.hlines(th_99, timestamps_eval[0], timestamps_eval[-1], color="red")
plt.show()

print(np.sum(results_eval > th_max)/results_eval.shape[0]*100, "%")
print(np.sum(results_eval > th_99)/results_eval.shape[0]*100, "%")
