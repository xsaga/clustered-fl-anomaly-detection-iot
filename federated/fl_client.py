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


def load_data(pcap_filename, cache_tensors=False):
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
    return train_loss_acc/len(train_generator)  # type: ignore


def test(model, loss_function, valid_generator):
    valid_loss_acc = 0.0
    with torch.no_grad():
        model.eval()
        for x_batch in valid_generator:
            preds = model(x_batch)
            valid_loss_acc += loss_function(preds, x_batch).item()
    print(f"valid loss {valid_loss_acc/len(valid_generator):.8f}")
    return valid_loss_acc/len(valid_generator)


local_models_dir_path = Path("./models_local")
global_models_dir_path = Path("./models_global")

global_round_dirs = [p for p in global_models_dir_path.iterdir() if p.is_dir() and p.match("round_*")]
global_round_dirs = sorted(global_round_dirs, key=lambda p: int(p.stem.rsplit("_", 1)[-1]))

if not global_round_dirs:
    print("No global models found.")
    sys.exit(0)

recent_global_round_dir = global_round_dirs[-1]
current_local_round_dir = local_models_dir_path / recent_global_round_dir.name
current_local_round_dir.mkdir(exist_ok=True)

current_round = int(current_local_round_dir.name.rsplit("_", 1)[-1])

model = Autoencoder()
torch.set_num_threads(1)

global_model_path = [p for p in recent_global_round_dir.iterdir() if p.is_file() and p.match("global_model_round_*.tar")]
assert len(global_model_path) == 1
global_model_path = global_model_path[0]

global_checkpoint = torch.load(global_model_path)
assert global_checkpoint["model_hash"] == state_dict_hash(global_checkpoint["state_dict"])
model.load_state_dict(global_checkpoint["state_dict"], strict=True)

print(f"Starting with global model {state_dict_hash(model.state_dict())} at {global_model_path}. FL round {current_round}.")

pcap_filename = sys.argv[1]
print(f"Loading data {pcap_filename}...")
train_dl, valid_dl = load_data(pcap_filename)

num_epochs = int(global_checkpoint["local_epochs"])
opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_func = F.mse_loss

# train
print(f"Training for {num_epochs} epochs in {pcap_filename}, number of samples {len(train_dl)}.")
train_loss = fit(model, optimizer=opt, loss_function=loss_func, epochs=num_epochs, train_generator=train_dl)

# eval
valid_loss = test(model, loss_func, valid_dl)

model_savefile = current_local_round_dir / f"{Path(pcap_filename).stem}_round{current_round}_epochs{num_epochs}.tar"
checkpoint = {"state_dict": model.state_dict(),
              "model_hash": state_dict_hash(model.state_dict()),
              "local_epochs": num_epochs,
              "loss": valid_loss,
              "train_loss": train_loss,
              "num_samples": len(train_dl)}
torch.save(checkpoint, model_savefile)
print(f"Saved model in {model_savefile} with hash {checkpoint['model_hash']}")
