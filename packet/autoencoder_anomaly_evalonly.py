from feature_extractor import pcap_to_dataframe, preprocess_dataframe

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, f1_score

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


def reconstruction_error(model, loss_function, samples):
    with torch.no_grad():
        model.eval()
        predictions = model(samples)
        rec_error = torch.mean(loss_function(samples, predictions, reduction="none"), dim=1)
    return rec_error

# trained model
model = Autoencoder()
ckpt = torch.load("global_model_round_30.tar")
model.load_state_dict(ckpt["state_dict"])
loss_func = F.mse_loss


# dataset estimate threshold
df = pcap_to_dataframe("coap-iot-bot-1_normal.pcap")
df = preprocess_dataframe(df)
timestamps = df["timestamp"].values
df = df.drop(columns=["timestamp"])
results = reconstruction_error(model, loss_func, torch.from_numpy(df.to_numpy(dtype=np.float32)))
results = results.numpy()
results_df = pd.DataFrame({"ts":timestamps, "rec_err":results})
sns.scatterplot(data=results_df, x="ts", y="rec_err", linewidth=0, alpha=0.4)
plt.show()

th = np.max(results[1:])
print(th)

# dataset eval
df = pcap_to_dataframe("coap-iot-bot-1_attack.pcap")
attack_victim_ip = ("192.168.0.254", "192.168.0.90")

## labels para los ataques
labels = label_by_ip(df, attack_victim_ip)

df = preprocess_dataframe(df)
timestamps = df["timestamp"].values
df = df.drop(columns=["timestamp"])



results = reconstruction_error(model, loss_func, torch.from_numpy(df.to_numpy(dtype=np.float32)))
results = results.numpy()

results_df = pd.DataFrame({"ts":timestamps, "rec_err":results, "label": labels})
sns.scatterplot(data=results_df, x="ts", y="rec_err", hue="label", linewidth=0, alpha=0.4)
plt.axhline(y=th, c="k")
plt.show()

labels_pred = (results>th)+0
confusion_matrix(labels, labels_pred, labels=[1,0])  # 0:normal, 1:attack; positive class is attack
tp, fn, fp, tn = confusion_matrix(labels, labels_pred, labels=[1,0]).ravel()
accuracy_score(labels, labels_pred)
f1_score(labels, labels_pred, pos_label=1)

fpr, tpr, thresholds = roc_curve(labels, results, pos_label=1)
print(auc(fpr, tpr))
plt.plot(fpr, tpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


##
sec_since = timestamps-timestamps[0]
plt.scatter(sec_since[labels==1], results[labels==1], marker = "^", c='#ff7f0e', alpha=0.4, label="Attack (ground truth)")
plt.scatter(sec_since[labels==0], results[labels==0], marker = "o", c='#1f77b4', alpha=0.4, label="Normal (ground truth)")
plt.axhline(y=th, linewidth=0.5, c="k", label="Threshold")
plt.xlabel("Seconds since beginning of capture")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.show()