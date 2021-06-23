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

import flwr as fl
from flwr.common import Scalar


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


def make_sequences(data, sequence_len):
    data_seq_x = []
    data_seq_y = []
    for i in range(data.shape[0] - sequence_len):
        data_seq_x.append(data[i:i+sequence_len, :])
        data_seq_y.append(data[i+1:i+sequence_len+1, :])
    return torch.stack(data_seq_x, dim=0), torch.stack(data_seq_y, dim=0)


def load_data(pcap_filename):
    df = pcap_to_dataframe(pcap_filename)
    df = preprocess_dataframe(df)
    df = df.drop(columns=["timestamp"])
    df_train, df_valid, _ = split_train_valid_eval(df, train_size=0.8)
    X_train = torch.from_numpy(df_train.to_numpy(dtype=np.float32))
    X_valid = torch.from_numpy(df_valid.to_numpy(dtype=np.float32))
    seq_len = 10
    train_x, train_y = make_sequences(X_train, seq_len)
    valid_x, valid_y = make_sequences(X_valid, seq_len)
    bs = 32
    train_ds = TensorDataset(train_x, train_y)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=False)
    valid_ds = TensorDataset(valid_x, valid_y)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False)
    return train_dl, valid_dl


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


def fit(model, optimizer, loss_function, epochs, train_generator):
    model.train()
    for epoch in range(epochs):
        train_loss_acc = 0.0
        for x_batch, y_batch in train_generator:
            x_batch = x_batch.transpose(0, 1)

            preds = model(x_batch)
            preds = preds.transpose(0, 1)

            loss = loss_function(preds, y_batch)
            train_loss_acc += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print(f"epoch {epoch+1}/{epochs}: train loss {train_loss_acc/len(train_generator):.8f}")


def test(model, loss_function, valid_generator):
    valid_loss_acc = 0.0
    with torch.no_grad():
        model.eval()
        for x_batch, y_batch in valid_generator:
            x_batch = x_batch.transpose(0, 1)
            preds = model(x_batch)
            preds = preds.transpose(0, 1)
            valid_loss_acc += loss_function(preds, y_batch).item()
    print(f"valid loss {valid_loss_acc/len(valid_generator):.8f}")
    return valid_loss_acc/len(valid_generator)


class AnomalyClient(fl.client.NumPyClient):
    def __init__(self, model: nn.Module, trainloader: DataLoader, testloader: DataLoader):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        params_list = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_list})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Scalar]) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        # update the params of the local model with the params form the server
        self.set_parameters(parameters)
        print("Fit received model hash: ", state_dict_hash(self.model.state_dict()))
        # train local model with local data
        opt = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        loss_func = F.mse_loss
        fit(self.model, optimizer=opt, loss_function=loss_func, epochs=1, train_generator=self.trainloader)
        print("Fitted model hash: ", state_dict_hash(self.model.state_dict()))
        torch.save(self.model.state_dict(), f"{Path(pcap_filename).stem}.pt")  # TODO name, poner a la clase el nombre del dataset
        # return trained weights and config
        return self.get_parameters(), len(self.trainloader), {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        # update the params of the local model with the params form the server
        self.set_parameters(parameters)
        print("Evaluating model hash: ", state_dict_hash(self.model.state_dict()))
        # evaluate the updated model on the local valid set
        loss_func = F.mse_loss
        loss = test(self.model, loss_func, self.testloader)
        # return metrics
        return float(loss), len(self.testloader), {}


model = LSTMchar()
pcap_filename = sys.argv[1]
print(f"Client loading data: {pcap_filename}")
train_dl, valid_dl = load_data(pcap_filename)
print("Loaded")
client = AnomalyClient(model, train_dl, valid_dl)
fl.client.start_numpy_client("127.0.0.1:8080", client=client)