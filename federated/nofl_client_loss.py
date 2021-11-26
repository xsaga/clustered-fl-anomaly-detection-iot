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

# parallel --verbose --bar --jobs 20 python nofl_client_loss.py {1} ::: *.pcap


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


def load_data(pcap_filename, cache_tensors=True):
    cache_filename = Path(pcap_filename + "_cache_tensors.pt")
    if cache_tensors and cache_filename.is_file():
        print("loading data from cache: ", cache_filename)
        serialize_tensors = torch.load(cache_filename)
        X_train = serialize_tensors["X_train"]
        X_valid = serialize_tensors["X_valid"]
    else:
        df = pcap_to_dataframe(pcap_filename)
        df = preprocess_dataframe(df)
        df = df.drop(columns=["timestamp"])
        df_train, df_valid, _ = split_train_valid_eval(df, train_size=0.8)
        X_train = torch.from_numpy(df_train.to_numpy(dtype=np.float32))
        X_valid = torch.from_numpy(df_valid.to_numpy(dtype=np.float32))
        if cache_tensors:
            serialize_tensors = {"X_train": X_train, "X_valid": X_valid}
            torch.save(serialize_tensors, cache_filename)
    
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



model = Autoencoder()
torch.set_num_threads(1)

pcap_filename = sys.argv[1]
print(f"Loading data {pcap_filename}...")
train_dl, valid_dl = load_data(pcap_filename)

num_epochs = 50
opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_func = F.mse_loss

# train
print(f"Training for {num_epochs} epochs in {pcap_filename}, number of samples {len(train_dl)}.")
epoch_l, loss_l, valid_loss_l = fit(model, optimizer=opt, loss_function=loss_func, epochs=num_epochs, train_generator=train_dl, valid_generator=valid_dl)
np.savez(f"results_nofl_epoch_losses_ae_{Path(pcap_filename).stem}.npz", np.array(epoch_l), np.array(loss_l), np.array(valid_loss_l))

