import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

###
# torch.set_default_dtype(torch.float32)

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

capture_filename = "capture.pickle" # capture.pickle; capture2021_03_16T08_59_17.pickle
metadata_filename = "capture_times_labels.pickle" # capture_times_labels.pickle; capture_times_labels2021_03_16T08_59_17.pickle 

train_pkt_max = 4700 # "capture.pickle"
# train_pkt_max = 13400 # "capture2021_03_16T08_59_17.pickle"

with open(capture_filename, "rb") as f:
    X = pickle.load(f)
X = X.astype(np.float32)

print(X.shape)

train_dataset = torch.from_numpy(X[:train_pkt_max, :])
# test_dataset = torch.from_numpy(X[train_pkt_max:, :])

bs = 32 # 64
train_dl = DataLoader(train_dataset, batch_size=bs, shuffle=True)
# test_dl = DataLoader(test_dataset, batch_size=bs)

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

def describe(x):
    m = x.min()
    M = x.max()
    mean = x.mean()
    p80 = np.percentile(x, 80)
    p90 = np.percentile(x, 90)
    p99 = np.percentile(x, 99)
    print(f"m={m:.2f}, M={M:.2f}, mean={mean:.2f}, p80,90,99 = {p80:.2f},{p90:.2f},{p99:.2f}")

# In [51]: describe(results_train)
# m=0.75, M=0.86, mean=0.80, p80,90,99 = 0.82,0.82,0.86
# In [52]: describe(results_otro)
# m=0.75, M=1.27, mean=0.89, p80,90,99 = 0.88,0.88,1.16

print("plotting")

with open(metadata_filename, "rb") as f:
    times, labels = pickle.load(f)

df = pd.DataFrame({"resultado":results, "time":times, "labels":labels})

snsplot = sns.scatterplot(data=df, x="time", y="resultado", hue="labels",
                          palette={0:"g", 1:"r", 2:"b"}, alpha=0.1)

plt.xlabel("time (s)")
plt.ylabel("error reconstruccion")
plt.show()



# class AutoencoderZ(nn.Module):
#     ...:     def __init__(self):
#     ...:         super(AutoencoderZ, self).__init__()
#     ...:         self.encoder = nn.Sequential(
#     ...:             nn.Linear(52, 20), # 26,12
#     ...:             # nn.Dropout(0.05), # !
#     ...:             nn.ReLU(), # nn.LeakyReLU(), nn.ReLU()
#     ...:             nn.Linear(20, 5), # 12,4
#     ...:             nn.Sigmoid() # nn.Tanh(), nn.Sigmoid(), nn.ReLU()
#     ...: 
#     ...:         )
#     ...:         self.decoder = nn.Sequential(
#     ...:             nn.Linear(5, 12), # 4,12
#     ...:             nn.ReLU(), # nn.LeakyReLU(), nn.ReLU()
#     ...:             nn.Linear(12, 26), # 12,26
#     ...:             # nn.Dropout(0.05), # !
#     ...:             nn.ReLU() # nn.ReLU()
#     ...:         )
#     ...: 
#     ...:     def forward(self, x):
#     ...:         latent = self.encoder(x)
#     ...:         decoded = self.decoder(latent)
#     ...:         return decoded

# def fit(epochs, train_dl):
#     ...:     model.train()
#     ...:     for epoch in range(epochs):
#     ...:         loss_acc = 0
#     ...:         print(f"--- epoch #{epoch} ---")
#     ...:         for xb in train_dl:
#     ...:             preds = model(xb)
#     ...:             loss = loss_func(preds, xb[:,26:])
#     ...: 
#     ...:             loss.backward()
#     ...:             opt.step()
#     ...:             opt.zero_grad()
#     ...: 
#     ...:             loss_acc += loss.item()
#     ...:         print(f"loss = {loss_acc/len(train_dl)}")

# def fit(epochs, train_dl):
#     ...:     model.train()
#     ...:     for epoch in range(epochs):
#     ...:         loss_acc = 0
#     ...:         print(f"--- epoch #{epoch} ---")
#     ...:         for xb in train_dl:
#     ...:             preds = model(xb)
#     ...:             loss = loss_func(preds, xb[:,26:])
#     ...: 
#     ...:             loss.backward()
#     ...:             opt.step()
#     ...:             opt.zero_grad()
#     ...: 
#     ...:             loss_acc += loss.item()
#     ...:         print(f"loss = {loss_acc/len(train_dl)}")
