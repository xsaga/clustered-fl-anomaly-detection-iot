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


def fedavg(model_weights: List[List[torch.Tensor]], num_training_samples: List[int]) -> List[torch.Tensor]:
    new_weights = []
    total_training_samples = sum(num_training_samples)
    for layers in zip(*model_weights):
        weighted_layers = torch.stack([torch.mul(l, w) for l, w in zip(layers, num_training_samples)])
        averaged_layers =  torch.div(torch.sum(weighted_layers, dim=0), total_training_samples)
        new_weights.append(averaged_layers)
    return new_weights


def average_weighted_loss(model_losses: List[float], num_training_samples: List[int]) -> float:
    weighted_losses = [l*n for l, n in zip(model_losses, num_training_samples)]
    return sum(weighted_losses)/sum(num_training_samples)


local_models_dir_path = Path("./models_local")
global_models_dir_path = Path("./models_global")

initial_random_model_path = global_models_dir_path / "round_0" / "global_model_round_0.tar"

local_models_dir_path.mkdir(exist_ok=True)
global_models_dir_path.mkdir(exist_ok=True)

local_round_dirs = [p for p in local_models_dir_path.iterdir() if p.is_dir() and p.match("round_*")]
local_round_dirs = sorted(local_round_dirs, key=lambda p: int(p.stem.rsplit("_", 1)[-1]))

LOCAL_EPOCHS = 5
global_model = LSTMchar()

if not local_round_dirs:
    if not initial_random_model_path.is_file():
        # prepare initial random model
        initial_random_model_path.parent.mkdir(exist_ok=True)
        checkpoint = {"state_dict": global_model.state_dict(),
                      "model_hash": state_dict_hash(global_model.state_dict()),
                      "local_epochs": LOCAL_EPOCHS,
                      "loss": 0.0,
                      "train_loss": 0.0,
                      "num_samples": 0}
        torch.save(checkpoint, initial_random_model_path)
        print(f"Created initial random model with hash: {checkpoint['model_hash']}")
        sys.exit(0)
    else:
        # run fl client
        checkpoint = torch.load(initial_random_model_path)
        assert checkpoint["model_hash"] == state_dict_hash(checkpoint["state_dict"])
        print(f"Initial model hash: {checkpoint['model_hash']}. Start FL clients.")
        sys.exit(0)

# local round directories found
recent_local_round_dir = local_round_dirs[-1]
current_fl_round = int(recent_local_round_dir.stem.rsplit("_", 1)[-1]) + 1
print(f"Recent local round: {recent_local_round_dir}. Current FL round: {current_fl_round}")

recent_local_round_models = [p for p in recent_local_round_dir.iterdir() if p.is_file() and p.match("*.tar")]
print(f"Found {len(recent_local_round_models)} local models.")
if not recent_local_round_models:
    sys.exit(0)

all_models = []
all_training_samples = []
all_loss = []
all_train_loss = []
for model_path in recent_local_round_models:
    chkpt = torch.load(model_path)
    assert chkpt["model_hash"] == state_dict_hash(chkpt["state_dict"])
    print(f"Loading {model_path}, hash: {chkpt['model_hash']}, training samples: {chkpt['num_samples']}")
    all_models.append(list(chkpt["state_dict"].values()))
    all_training_samples.append(int(chkpt["num_samples"]))
    all_loss.append(chkpt["loss"])
    all_train_loss.append(chkpt["train_loss"])

new_global_model = fedavg(all_models, all_training_samples)
avg_loss = average_weighted_loss(all_loss, all_training_samples)
global_model.load_state_dict(OrderedDict(zip(global_model.state_dict().keys(), new_global_model)))
new_checkpoint = {"state_dict": global_model.state_dict(),
                  "model_hash": state_dict_hash(global_model.state_dict()),
                  "local_epochs": LOCAL_EPOCHS,
                  "loss": avg_loss,
                  "train_loss": average_weighted_loss(all_train_loss, all_training_samples),
                  "num_samples": sum(all_training_samples)}
new_global_model_dir = global_models_dir_path / f"round_{current_fl_round}"
new_global_model_dir.mkdir(exist_ok=True)
new_global_model_path = new_global_model_dir / f"global_model_round_{current_fl_round}.tar"
torch.save(new_checkpoint, new_global_model_path)
np.savez(new_global_model_dir/f"losses_num_samples_round{current_fl_round}.npz", np.array(all_loss), np.array(all_train_loss), np.array(all_training_samples))
print(f"New global model {new_checkpoint['model_hash']} saved to {new_global_model_path}. Avg loss {avg_loss}. Avg train loss {new_checkpoint['train_loss']}.")
