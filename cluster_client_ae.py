"""Partial training of the model for the device clustering. Represents a single client (one pcap file).

Not to be executed directly. This program is called by 'cluster_train.sh'
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim

from feature_extractor import pcap_to_dataframe, port_hierarchy_map, port_hierarchy_map_iot, preprocess_dataframe
from model_ae import Autoencoder, fit, load_data, state_dict_hash


parser = argparse.ArgumentParser(description="Train partial models for clustering.")
parser.add_argument("-d", "--dimensions", type=int, default=27)
parser.add_argument("-e", "--epochs", type=int, default=4)
parser.add_argument("pcap")
args = parser.parse_args()

model = Autoencoder(args.dimensions)
torch.set_num_threads(1)
initial_random_model_path = "initial_random_model_ae.pt"
model.load_state_dict(torch.load(initial_random_model_path), strict=True)
print(f"Loaded initial model {initial_random_model_path} with hash {state_dict_hash(model.state_dict())}")

print(f"Loading data from {args.pcap}... ")
train_dl, _ = load_data(args.pcap, use_serialized_dataframe_if_available=True, cache_tensors=False, port_mapping=port_hierarchy_map_iot, sport_bins=None, dport_bins=None)
print("Loaded")

opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_func = F.mse_loss
print(f"Training for {args.epochs} epochs in {args.pcap}")
fit(model, optimizer=opt, loss_function=loss_func, epochs=args.epochs, train_generator=train_dl)
print(f"Fitted model hash {state_dict_hash(model.state_dict())}")
model_savefile = f"{Path(args.pcap).stem}_{args.epochs}epochs_ae.pt"
torch.save(model.state_dict(), model_savefile)
print(f"Saved model in {model_savefile}")
