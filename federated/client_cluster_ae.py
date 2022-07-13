from feature_extractor import pcap_to_dataframe, preprocess_dataframe, port_hierarchy_map, port_hierarchy_map_iot
from model_ae import Autoencoder, state_dict_hash, load_data, fit
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch import optim


data_dimensions = 27
model = Autoencoder(data_dimensions)
torch.set_num_threads(1)
initial_random_model_path = "initial_random_model_ae.pt"
model.load_state_dict(torch.load(initial_random_model_path), strict=True)
print(f"Loaded initial model {initial_random_model_path} with hash {state_dict_hash(model.state_dict())}")

pcap_filename = sys.argv[1]
print(f"Loading data {pcap_filename}... ")
train_dl, _ = load_data(pcap_filename, cache_tensors=False, port_mapping = None, sport_bins = None, dport_bins = None)
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
