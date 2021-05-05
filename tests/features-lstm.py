import datetime
import math
import pickle
import sys

from pathlib import Path

import numpy as np
import pandas as pd

from scipy import stats
from scapy.all import *

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

input_capture_name = "iot-client-1_eth0_to_Switch2_Ethernet1.pcap"
if len(sys.argv) > 1:
    input_capture_name = sys.argv[1]
output_model_name = f"{Path(input_capture_name).stem}.pt"

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

def entropy(x):
    cnt = np.bincount(x, minlength=256)
    return stats.entropy(cnt, base=2)

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

    features["h"] = entropy(bytearray(raw(pkt)))

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
df["ip_tos"] = np.log1p(df["ip_tos"])

df["h"] = df["h"]/(-math.log2(1/256))  # entropy/8.0

class LSTMchar(nn.Module):
    def __init__(self, start_dim):
        super(LSTMchar, self).__init__()
        self.start_dim = start_dim

        self.lstm = nn.LSTM(self.start_dim, 128)
        self.linear = nn.Linear(128, self.start_dim)
        self.activ = nn.Softmax(dim=2)  # nn.ReLU()

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        out = self.activ(out)
        out = self.linear(out)
        return out

X = df.to_numpy()
X = X.astype(np.float32)

train_x = []
train_y = []
maxlen = 10
for i in range(X.shape[0] - maxlen - 1):
    train_x.append(X[i:i+maxlen, :])
    train_y.append(X[i+1:i+maxlen+1, :])

train_x = np.array(train_x)
train_y = np.array(train_y)

train_x = torch.tensor(train_x)
train_y = torch.tensor(train_y)

# train_x = train_x.to(device) # ! en vez de moverlo aqui, moverlo en el loop de training. Sino la memoria de la GPU se llena
# train_y = train_y.to(device) # !

train_ds = TensorDataset(train_x, train_y)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=False)

model = LSTMchar(X.shape[1])
loss_func = F.mse_loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs=100
for epoch in range(epochs):
    model = model.train()
    acc_loss = 0
    for xb, yb in train_dl:
        xb = xb.transpose(0, 1)
        # yb = yb.transpose(0, 1)

        preds = model(xb)
        preds = preds.transpose(0, 1)
        loss = loss_func(preds, yb)
        acc_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"epoch {epoch}. loss for model {output_model_name} = {acc_loss/len(train_dl)}")

recerror=[]
with torch.no_grad():
    model = model.eval()
    for i in range(X.shape[0] - maxlen - 1):
        xi = X[i:i+maxlen,:]
        t  = X[i+maxlen+1]
        xi = torch.tensor(xi)
        t = torch.tensor(t)
        xi=xi.view(maxlen,1,-1)
        p = model(xi)
        p = p[-1,0,:]
        err = loss_func(p, t).item()
        recerror.append(err)

plt.plot(recerror)
plt.show()

torch.save(model.state_dict(), output_model_name)
print(f"Model saved to: {output_model_name}")
