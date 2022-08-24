import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from feature_extractor import port_basic_three_map, port_hierarchy_map, port_hierarchy_map_iot
from model_ae import Autoencoder, load_data, test


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
        loss_list.append(train_loss_acc / len(train_generator))
        valid_loss_list.append(test(model, loss_func, valid_generator))
    return epoch_list, loss_list, valid_loss_list


parser = argparse.ArgumentParser(description="NoFL client.")
parser.add_argument("-b", "--base", type=lambda p: Path(p).absolute(), default=Path("./NoFLbase").absolute())
parser.add_argument("-d", "--dimensions", type=int, default=27)
parser.add_argument("-e", "--epochs", type=int, default=5)
parser.add_argument("--opt", type=str, required=True, choices=("adam", "sgd"))
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--b1", type=float, default=0.9)
parser.add_argument("--b2", type=float, default=0.999)
parser.add_argument("--eps", type=float, default=1e-08)
parser.add_argument("--wdecay", type=float, default=1e-5)
parser.add_argument("--momentum", type=float, default=0.0)
parser.add_argument("pcap")
args = parser.parse_args()


args.base.mkdir(parents=True, exist_ok=True)
model = Autoencoder(args.dimensions)
torch.set_num_threads(1)

print(f"Loading data {args.pcap}...")
train_dl, valid_dl = load_data(args.pcap, cache_tensors=True, port_mapping=port_hierarchy_map_iot, sport_bins=None, dport_bins=None)

num_epochs = args.epochs  # num_local_epochs x num_fl_rounds

if args.opt == "adam":
    opt = optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2), eps=args.eps, weight_decay=args.wdecay)
else:  # args.opt == "sgd"
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wdecay)
print("Opt: ", opt)

loss_func = F.mse_loss

# train
print(f"Training for {num_epochs} epochs in {args.pcap}, number of samples {len(train_dl)}.")
epoch_l, loss_l, valid_loss_l = fit(model, optimizer=opt, loss_function=loss_func, epochs=num_epochs, train_generator=train_dl, valid_generator=valid_dl)

out_fname = args.base / f"results_nofl_epoch_losses_ae_{Path(args.pcap).stem}.npz"
print(f"Serializing results to {out_fname}")
np.savez(out_fname, np.array(epoch_l), np.array(loss_l), np.array(valid_loss_l))
