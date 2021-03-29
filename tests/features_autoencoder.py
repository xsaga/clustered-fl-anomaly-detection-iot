# script que une 'features.py' y 'autoencoder_test.py'

import datetime
import math
import pickle
from collections import Counter

import numpy as np
import pandas as pd

from scapy.all import *

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

input_capture_name = "mqtt1.pcap"
output_model_name = "M1_1.pt"

cap = rdpcap(input_capture_name)

def dr(i):
    return(cap[i].time/1000 - cap[0].time/1000)


def dt(i):
    return (cap[i].time/1000 - cap[i-1].time/1000)


def inter_arrival_time(idx, capture):
    if idx == 0:
        return 0
    return capture[idx].time / 1000 - capture[idx-1].time / 1000


def port_ranges(port):  # !!! incrementar la clasificacion, jerarquias mas detalladas.
    if port in range(0, 1024):
        return "system"
    elif port in range(1024, 49152):
        return "user"
    else:
        return "dynamic"

PORT_CATEGORIES = ["system", "user", "dynamic"]

def entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum( count/lns * math.log(count/lns, 2) for count in p.values())

pkts_features = []
labels = []
times = []

IP_MF = 0b001
IP_DF = 0b010
IP_R =  0b100

TCP_FIN = 0b000000001
TCP_SYN = 0b000000010
TCP_RST = 0b000000100
TCP_PSH = 0b000001000
TCP_ACK = 0b000010000
TCP_URG = 0b000100000
TCP_ECE = 0b001000000
TCP_CWR = 0b010000000
TCP_NS =  0b100000000

for idx, pkt in enumerate(cap):

    features = {"packet_length": 0,
                "inter_arrival_time": 0,
                "h": 0,
                "ip_src": "127.0.0.1",
                "ip_dst": "127.0.0.1",
                "ip_tos": 0,
                "ip_flags": "",
                "ip_flag_mf": 0,
                "ip_flag_df": 0,
                "ip_flag_r": 0,
                "ip_ttl": 64,
                "ip_protocol": "",
                "ip_proto_tcp": 0,
                "ip_proto_udp": 0,
                "ip_proto_icmp": 0,
                "ip_proto_other": 0,
                "sport": 0,
                "dport": 0,
                "flags": "",
                "flag_fin": 0,
                "flag_syn": 0,
                "flag_rst": 0,
                "flag_psh": 0,
                "flag_ack": 0,
                "flag_urg": 0,
                "flag_other": 0,
                "window": 0}
    # filtrar !!
    if IPv6 in pkt:
        print(f"DISCARDED: {repr(pkt)}")
        continue
    # times
    times.append(float(pkt.time - cap[0].time))
    # labels
    if IP in pkt:
        lyr = pkt.getlayer(IP)
        if lyr.src == "192.168.0.100" or lyr.dst == "192.168.0.100":
            labels.append(1)
        elif lyr.src == "192.168.0.3" or lyr.dst == "192.168.0.3":
            labels.append(2)
        else:
            labels.append(0)
    else:
        labels.append(0)

    # total packet length
    features["packet_length"] = len(pkt)

    # inter arrival time
    if idx == 0:
        features["inter_arrival_time"] = 0.0
    else:
        features["inter_arrival_time"] = float(cap[idx].time - cap[idx-1].time)

    features["h"] = entropy(raw(pkt).hex())

    if pkt.haslayer(IP):
        lyr = pkt.getlayer(IP)
        features["ip_src"] = lyr.src
        features["ip_dst"] = lyr.dst
        features["ip_tos"] = lyr.tos
        features["ip_flags"] = str(lyr.flags) # decode? str(f) o f.value (numerico), ['MF', 'DF', 'evil']
        features["ip_flag_mf"] = 1 if lyr.flags.value & IP_MF else 0
        features["ip_flag_df"] = 1 if lyr.flags.value & IP_DF else 0
        features["ip_flag_r"] = 1 if lyr.flags.value & IP_R else 0
        features["ip_ttl"] = lyr.ttl

    if pkt.haslayer(TCP):
        lyr = pkt.getlayer(TCP)
        features["ip_protocol"] = "TCP"
        features["ip_proto_tcp"] = 1
        features["sport"] = lyr.sport
        features["dport"] = lyr.dport
        features["flags"] = str(lyr.flags) # decode? scapy.fields.FlagValue type. str(f) o f.value (numerico)
        features["flag_fin"] = 1 if lyr.flags.value & TCP_FIN else 0
        features["flag_syn"] = 1 if lyr.flags.value & TCP_SYN else 0
        features["flag_rst"] = 1 if lyr.flags.value & TCP_RST else 0
        features["flag_psh"] = 1 if lyr.flags.value & TCP_PSH else 0
        features["flag_ack"] = 1 if lyr.flags.value & TCP_ACK else 0
        features["flag_urg"] = 1 if lyr.flags.value & TCP_URG else 0
        features["flag_other"] = 1 if lyr.flags.value >= TCP_ECE else 0
        features["window"] = lyr.window
    elif pkt.haslayer(UDP):
        lyr = pkt.getlayer(UDP)
        features["ip_protocol"] = "UDP"
        features["ip_proto_udp"] = 1
        features["sport"] = lyr.sport
        features["dport"] = lyr.dport
    elif pkt.haslayer(ICMP):
        lyr = pkt.getlayer(ICMP)
        features["ip_protocol"] = "ICMP"
        features["ip_proto_icmp"] = 1
        # icmp_type = lyr.type
    else:
        features["ip_protocol"] = "OTHER"
        features["ip_proto_other"] = 1
    
    pkts_features.append(features)

df = pd.DataFrame(pkts_features)

port_dtype = pd.api.types.CategoricalDtype(categories=PORT_CATEGORIES)
df["sport"] = df["sport"].apply(port_ranges).astype(port_dtype)
df["dport"] = df["dport"].apply(port_ranges).astype(port_dtype)

sport_onehot = pd.get_dummies(df["sport"], prefix="sport")
dport_onehot = pd.get_dummies(df["dport"], prefix="dport")
df = df.drop(columns=["ip_src", "ip_dst", "ip_flags", "ip_protocol", "flags", "sport", "dport"])

df = df.join([sport_onehot, dport_onehot])
df["packet_length"] = np.log1p(df["packet_length"])
df["inter_arrival_time"] = np.log1p(df["inter_arrival_time"])
df["window"] = np.log1p(df["window"])
df["ip_ttl"] = np.log1p(df["ip_ttl"])
# !!! aplicar log a ip_tos también, (por eso sale el error reconsturccion 1400 en esos icmp)
df["ip_tos"] = np.log1p(df["ip_tos"])

#### autoencoder_test

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(26, 12), # 26,12
            # nn.Dropout(0.05), # !
            nn.ReLU(), # nn.LeakyReLU(), nn.ReLU()
            nn.Linear(12, 4), # 12,4
            nn.Sigmoid() # nn.Tanh(), nn.Sigmoid(), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 12), # 4,12
            nn.ReLU(), # nn.LeakyReLU(), nn.ReLU()
            nn.Linear(12, 26), # 12,26
            # nn.Dropout(0.05), # !
            nn.ReLU() # nn.ReLU()
        )

    def forward(self, x):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return decoded


# dataset

X = df.to_numpy()
X = X.astype(np.float32)

print(X.shape)

train_dataset = torch.from_numpy(X)

bs = 32 # 64
train_dl = DataLoader(train_dataset, batch_size=bs, shuffle=True)

model = Autoencoder()

lr = 1e-3  # 1e-3
num_epochs = 200  # 100
loss_func = F.mse_loss
# opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# importante el weight_decay. Sin el decay (=0) los resultados
# varian mucho segun el entrenamiento. Con el decay cada entrenamiento
# es más consistente.
opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
# opt = optim.RMSprop(model.parameters(), lr=lr)

def fit(epochs, train_dl):
    model.train()
    for epoch in range(epochs):
        loss_acc = 0
        print(f"--- epoch #{epoch} ---")
        for xb in train_dl:
            preds = model(xb)
            loss = loss_func(preds, xb)
            
            loss.backward()
            opt.step()
            opt.zero_grad()

            loss_acc += loss.item()
        print(f"loss = {loss_acc/len(train_dl)}")

fit(num_epochs, train_dl)
print("done")

with torch.no_grad():
    model.eval()
    paks = torch.from_numpy(X)
    preds = model(paks)
    loss = loss_func(preds, paks)
    results = [loss_func(x,y).item() for x,y in zip(preds, paks)]

print("plotting")

df = pd.DataFrame({"resultado":results, "time":times, "labels":labels})

snsplot = sns.scatterplot(data=df, x="time", y="resultado", hue="labels",
                          palette={0:"g", 1:"r", 2:"b"}, alpha=0.1)

plt.xlabel("time (s)")
plt.ylabel("error reconstruccion")
plt.show()

torch.save(model.state_dict(), output_model_name)
print(f"Model saved to: {output_model_name}")
