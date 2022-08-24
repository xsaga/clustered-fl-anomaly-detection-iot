import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim

from feature_extractor import port_basic_three_map, port_hierarchy_map, port_hierarchy_map_iot
from model_ae import Autoencoder, fit, load_data, state_dict_hash, test


parser = argparse.ArgumentParser(description="FL client.")
parser.add_argument("-b", "--base", type=lambda p: Path(p).absolute(), default=Path("./FLbase").absolute())
parser.add_argument("-d", "--dimensions", type=int, default=27)
parser.add_argument("--clientopt", type=str, required=True, choices=("adam", "sgd"))
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--b1", type=float, default=0.9)
parser.add_argument("--b2", type=float, default=0.999)
parser.add_argument("--eps", type=float, default=1e-08)
parser.add_argument("--wdecay", type=float, default=1e-5)
parser.add_argument("--momentum", type=float, default=0.0)
parser.add_argument("pcap")
args = parser.parse_args()

local_models_dir_path = args.base / "./models_local"
global_models_dir_path = args.base / "./models_global"

global_round_dirs = sorted([p for p in global_models_dir_path.iterdir() if p.is_dir() and p.match("round_*")],
                           key=lambda p: int(p.stem.rsplit("_", 1)[-1]))

if not global_round_dirs:
    print("No global models found. Run FL server.")
    sys.exit(0)

last_global_round_dir = global_round_dirs[-1]
current_local_round_dir = local_models_dir_path / last_global_round_dir.name
current_local_round_dir.mkdir(parents=True, exist_ok=True)
current_round = int(current_local_round_dir.name.rsplit("_", 1)[-1])

model = Autoencoder(args.dimensions)
torch.set_num_threads(1)

global_model_path = [p for p in last_global_round_dir.iterdir() if p.is_file() and p.match("global_model_round_*.tar")]
assert len(global_model_path) == 1
global_model_path = global_model_path[0]

global_checkpoint = torch.load(global_model_path)
assert global_checkpoint["model_hash"] == state_dict_hash(global_checkpoint["state_dict"])
model.load_state_dict(global_checkpoint["state_dict"], strict=True)

print(f"Starting with global model {state_dict_hash(model.state_dict())}\nat {global_model_path}.\nFL round {current_round}.")

print(f"Loading data from {args.pcap}...")
train_dl, valid_dl = load_data(args.pcap, cache_tensors=True, port_mapping=port_hierarchy_map_iot, sport_bins=None, dport_bins=None)

num_epochs = int(global_checkpoint["local_epochs"])

if args.clientopt == "adam":
    opt = optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2), eps=args.eps, weight_decay=args.wdecay)
else:  # args.clientopt == "sgd"
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wdecay)

print("ClientOpt: ", opt)

loss_func = F.mse_loss

# train
print(f"Training for {num_epochs} epochs in {args.pcap}, number of samples {len(train_dl)}.")
train_loss = fit(model, optimizer=opt, loss_function=loss_func, epochs=num_epochs, train_generator=train_dl)

# eval
valid_loss = test(model, loss_func, valid_dl)

model_savefile = current_local_round_dir / f"{Path(args.pcap).stem}_round{current_round}_epochs{num_epochs}.tar"
checkpoint = {"state_dict": model.state_dict(),
              "model_hash": state_dict_hash(model.state_dict()),
              "local_epochs": num_epochs,
              "loss": valid_loss,
              "train_loss": train_loss,
              "num_samples": len(train_dl)}
torch.save(checkpoint, model_savefile)
print(f"Saved model in {model_savefile} with hash {checkpoint['model_hash']}")
