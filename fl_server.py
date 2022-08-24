import argparse
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from feature_extractor import pcap_to_dataframe, port_basic_three_map, port_hierarchy_map, port_hierarchy_map_iot, preprocess_dataframe
from model_ae import Autoencoder, state_dict_hash, test


def fedavg(model_weights: List[List[torch.Tensor]], num_training_samples: List[int]) -> List[torch.Tensor]:
    assert len(model_weights) == len(num_training_samples)
    new_weights = []
    total_training_samples = sum(num_training_samples)
    for layers in zip(*model_weights):
        weighted_layers = torch.stack([torch.mul(l, w) for l, w in zip(layers, num_training_samples)])
        averaged_layers = torch.div(torch.sum(weighted_layers, dim=0), total_training_samples)
        new_weights.append(averaged_layers)
    return new_weights


# FedOpt
def delta_updates(model_weights: List[List[torch.Tensor]], num_training_samples: List[int], previous_model: List[torch.Tensor]) -> List[torch.Tensor]:
    # model_weights : list of models : list of list of tensors
    avg_model_weights = fedavg(model_weights, num_training_samples)
    delta = []
    for i in range(len(avg_model_weights)):
        delta.append(avg_model_weights[i] - previous_model[i])
    return delta


def serveropt(arguments, optimizer_state, model_weights: List[List[torch.Tensor]], num_training_samples: List[int], previous_model: List[torch.Tensor]) -> Tuple[List[torch.Tensor], Any]:
    prev_model = [t.detach().clone() for t in previous_model]
    pseudogradient = [torch.neg(t) for t in delta_updates(model_weights, num_training_samples, prev_model)]

    params = [t.requires_grad_(True) for t in prev_model]

    if arguments.serveropt == "adam":
        opt = optim.Adam(params, lr=arguments.lr, betas=(arguments.b1, arguments.b2), eps=arguments.eps, weight_decay=arguments.wdecay)
    else:  # args.serveropt == "sgd"
        opt = optim.SGD(params, lr=arguments.lr, momentum=arguments.momentum, weight_decay=arguments.wdecay)

    print("ServerOpt: ", opt)

    if optimizer_state:
        opt.load_state_dict(optimizer_state)
        print("Loaded serveropt optimizer state")

    for i, param in enumerate(params):
        param.grad = pseudogradient[i]
    opt.step()

    return [t.detach().clone() for t in params], opt.state_dict()


def average_weighted_loss(model_losses: List[float], num_training_samples: List[int]) -> float:
    weighted_losses = [l * n for l, n in zip(model_losses, num_training_samples)]
    return sum(weighted_losses) / sum(num_training_samples)


parser = argparse.ArgumentParser(description="FL server.")
parser.add_argument("-b", "--base", type=lambda p: Path(p).absolute(), default=Path("./FLbase").absolute())
parser.add_argument("-d", "--dimensions", type=int, default=27)
parser.add_argument("-e", "--epochs", type=int, default=5)
parser.add_argument("--serveropt", type=str, required=True, choices=("adam", "sgd"))
parser.add_argument("--lr", type=float, default=1.0)
parser.add_argument("--b1", type=float, default=0.9)
parser.add_argument("--b2", type=float, default=0.999)
parser.add_argument("--eps", type=float, default=1e-08)
parser.add_argument("--wdecay", type=float, default=0)
parser.add_argument("--momentum", type=float, default=0)
args = parser.parse_args()

# seed
torch.manual_seed(0)
np.random.seed(0)

local_models_dir_path = args.base / "./models_local"
global_models_dir_path = args.base / "./models_global"

initial_random_model_path = global_models_dir_path / "round_0" / "global_model_round_0.tar"

local_models_dir_path.mkdir(parents=True, exist_ok=True)
global_models_dir_path.mkdir(parents=True, exist_ok=True)

local_round_dirs = sorted([p for p in local_models_dir_path.iterdir() if p.is_dir() and p.match("round_*")],
                          key=lambda p: int(p.stem.rsplit("_", 1)[-1]))

global_model = Autoencoder(args.dimensions)

if not local_round_dirs:
    if not initial_random_model_path.is_file():
        # prepare initial random model
        initial_random_model_path.parent.mkdir(exist_ok=True)
        checkpoint = {"state_dict": global_model.state_dict(),
                      "model_hash": state_dict_hash(global_model.state_dict()),
                      "local_epochs": args.epochs,
                      "loss": 0.0,
                      "train_loss": 0.0,
                      "num_samples": 0}
        torch.save(checkpoint, initial_random_model_path)
        print(f"Created initial random model with hash: {checkpoint['model_hash']}. Start FL clients.")
        sys.exit(0)
    else:
        # run fl client
        checkpoint = torch.load(initial_random_model_path)
        assert checkpoint["model_hash"] == state_dict_hash(checkpoint["state_dict"])
        print(f"Initial model hash: {checkpoint['model_hash']}. Start FL clients.")
        sys.exit(0)

# local round directories found
last_local_round_dir = local_round_dirs[-1]
current_fl_round = int(last_local_round_dir.stem.rsplit("_", 1)[-1]) + 1
print(f"Recent local round: {last_local_round_dir}. Current FL round: {current_fl_round}")

last_local_round_models = [p for p in last_local_round_dir.iterdir() if p.is_file() and p.match("*.tar")]
print(f"Found {len(last_local_round_models)} local models.")
if not last_local_round_models:
    sys.exit(0)

# load eval dataset
eval_pcap_filename = "./eval/eval.pcap"
if Path(eval_pcap_filename).is_file():
    print(f"Loading evaluation data {eval_pcap_filename}...")

    try:
        X_eval = torch.load(eval_pcap_filename + ".pt")
    except FileNotFoundError:
        eval_df = pcap_to_dataframe(eval_pcap_filename)
        eval_df = preprocess_dataframe(eval_df, port_mapping=port_hierarchy_map_iot, sport_bins=None, dport_bins=None)
        eval_df = eval_df.drop(columns=["timestamp"])
        X_eval = torch.from_numpy(eval_df.to_numpy(dtype=np.float32))
        torch.save(X_eval, eval_pcap_filename + ".pt")

    eval_dl = DataLoader(X_eval, batch_size=32, shuffle=False)
else:
    eval_dl = None


all_models = []
all_training_samples = []
all_loss = []
all_train_loss = []
all_local_model_loss_eval_dataset = []
for model_path in last_local_round_models:
    chkpt = torch.load(model_path)
    assert chkpt["model_hash"] == state_dict_hash(chkpt["state_dict"])
    print(f"Loading {model_path}, hash: {chkpt['model_hash']}, training samples: {chkpt['num_samples']}")
    all_models.append(list(chkpt["state_dict"].values()))
    all_training_samples.append(int(chkpt["num_samples"]))
    all_loss.append(chkpt["loss"])
    all_train_loss.append(chkpt["train_loss"])

    # eval dataset loss
    local_model = Autoencoder(args.dimensions)
    local_model.load_state_dict(chkpt["state_dict"])
    if eval_dl:
        local_model_loss_valid_dataset = test(local_model, F.mse_loss, eval_dl)
        all_local_model_loss_eval_dataset.append(local_model_loss_valid_dataset)

# Server optimization (model aggregation)
new_global_model_fedavg = fedavg(all_models, all_training_samples)
# --- fedopt ---
# load previous step global model
prev_step_global_model_path = global_models_dir_path / f"round_{current_fl_round-1}" / f"global_model_round_{current_fl_round-1}.tar"
prev_step_chkpt = torch.load(prev_step_global_model_path)
print("PREVIOUS STEP GLOBAL MODEL HASH: ", prev_step_chkpt["model_hash"])

# load serveropt optim state
optim_state_path = global_models_dir_path / "optim_state.pt"
try:
    serveropt_optim_state = torch.load(optim_state_path)
    print("Deserializing serveropt optimizer state")
except FileNotFoundError:
    serveropt_optim_state = None

new_global_model, serveropt_optim_state = serveropt(args, serveropt_optim_state, all_models, all_training_samples, list(prev_step_chkpt["state_dict"].values()))

# save serveropt optim state
torch.save(serveropt_optim_state, optim_state_path)

# checks
for i in range(len(new_global_model)):
    print("SERVEROPT == FEDAVG ?", torch.allclose(new_global_model[i], new_global_model_fedavg[i]))
# --- ------ ---
avg_loss = average_weighted_loss(all_loss, all_training_samples)
global_model.load_state_dict(OrderedDict(zip(global_model.state_dict().keys(), new_global_model)))

# eval dataset loss
if eval_dl:
    global_model_loss_eval_dataset = test(global_model, F.mse_loss, eval_dl)

new_checkpoint = {"state_dict": global_model.state_dict(),
                  "model_hash": state_dict_hash(global_model.state_dict()),
                  "local_epochs": args.epochs,
                  "loss": avg_loss,
                  "train_loss": average_weighted_loss(all_train_loss, all_training_samples),
                  "num_samples": sum(all_training_samples)}
new_global_model_dir = global_models_dir_path / f"round_{current_fl_round}"
new_global_model_dir.mkdir(exist_ok=True)
new_global_model_path = new_global_model_dir / f"global_model_round_{current_fl_round}.tar"
torch.save(new_checkpoint, new_global_model_path)

np.savez(new_global_model_dir / f"losses_num_samples_round{current_fl_round}.npz", np.array(all_loss), np.array(all_train_loss), np.array(all_training_samples))
print(f"New global model {new_checkpoint['model_hash']} saved to {new_global_model_path}. Avg loss {avg_loss}. Avg train loss {new_checkpoint['train_loss']}.")
