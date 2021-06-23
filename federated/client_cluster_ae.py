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


def fit(model, optimizer, loss_function, epochs, train_generator):
    model.train()
    for epoch in range(epochs):
        train_loss_acc = 0.0
        for x_batch in train_generator:
            preds = model(x_batch)
            loss = loss_function(preds, x_batch)
            train_loss_acc += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print(f"epoch {epoch+1}/{epochs}: train loss {train_loss_acc/len(train_generator):.8f}")


model = Autoencoder()
torch.set_num_threads(1)
initial_random_model_path = "initial_random_model_clustering_ae.pt"
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
model_savefile = f"{Path(pcap_filename).stem}_{num_epochs}epochs_ae.pt"
torch.save(model.state_dict(), model_savefile)
print(f"Saved model in {model_savefile}")
