"""Defines the Autoencoder model and basic functions."""
import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from feature_extractor import pcap_to_dataframe, preprocess_dataframe


def state_dict_hash(state_dict: Union['OrderedDict[str, torch.Tensor]', Dict[str, torch.Tensor]]) -> str:
    """Get a hash of the model's state_dict. For checks and debugging."""
    h = hashlib.md5()
    for k, v in state_dict.items():
        h.update(k.encode("utf-8"))
        h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()


def split_train_valid_eval(df: pd.DataFrame, eval_split: Optional[Union[float, int]]=None, train_size: Union[float, int]=0.8) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Split a Dataframe into two or three splits.

    If eval_split is set: first the DataFrame is divided into two
    splits (train_valid and eval) based on eval_split. Then,
    trin_valid is further divided into two splits (train and valid)
    based on train_size. If eval split is not set: the DataFrame is
    divided into two splits based on train_size.
    """
    if eval_split:
        df_train_valid, df_eval = train_test_split(df, shuffle=False, train_size=eval_split)
        df_train, df_valid = train_test_split(df_train_valid, shuffle=False, train_size=train_size)
        return df_train, df_valid, df_eval

    df_train, df_valid = train_test_split(df, shuffle=False, train_size=train_size)
    return df_train, df_valid, None


def load_data(pcap_filename: str, use_serialized_dataframe_if_available: bool=False, cache_tensors: bool=True, port_mapping: Optional[List[Tuple[Sequence[int], str]]]=None, sport_bins: Optional[List[int]]=None, dport_bins: Optional[List[int]]=None, train_data_fraction=0.8) -> Tuple[DataLoader, DataLoader]:
    """ Reads a pcap file and transforms it into a training and validation DataLoader."""
    cache_filename = Path(pcap_filename + "_cache_tensors.pt")
    serialized_df_filename = Path(pcap_filename).with_suffix(".pickle")
    if cache_tensors and cache_filename.is_file():
        print("loading data from cache: ", cache_filename)
        serialize_tensors: Dict[str, torch.Tensor] = torch.load(cache_filename)
        X_train = serialize_tensors["X_train"]
        X_valid = serialize_tensors["X_valid"]
    else:
        if use_serialized_dataframe_if_available and serialized_df_filename.is_file():
            print("loading serialized dataframe: ", serialized_df_filename)
            df = pd.read_pickle(serialized_df_filename)
        else:
            df = pcap_to_dataframe(pcap_filename)
        df = preprocess_dataframe(df, port_mapping, sport_bins, dport_bins)
        df = df.drop(columns=["timestamp"])
        df_train, df_valid, _ = split_train_valid_eval(df, train_size=train_data_fraction)
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
    def __init__(self, num_input: int):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_input, num_input // 2),
            nn.ReLU(),
            nn.Linear(num_input // 2, num_input // 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(num_input // 4, num_input // 2),
            nn.ReLU(),
            nn.Linear(num_input // 2, num_input),
            nn.ReLU()
        )

    def forward(self, x):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return decoded


def fit(model, optimizer, loss_function, epochs, train_generator) -> float:
    model.train()
    train_loss_acc = 0.0
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
    return train_loss_acc / len(train_generator)


def test(model, loss_function, valid_generator) -> float:
    valid_loss_acc = 0.0
    with torch.no_grad():
        model.eval()
        for x_batch in valid_generator:
            preds = model(x_batch)
            valid_loss_acc += loss_function(preds, x_batch).item()
    print(f"valid loss {valid_loss_acc/len(valid_generator):.8f}")
    return valid_loss_acc / len(valid_generator)
