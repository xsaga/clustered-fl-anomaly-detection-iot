import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset


###
# torch.set_default_dtype(torch.float32)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(26, 12),
            nn.ReLU(),
            nn.Linear(12, 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 12),
            nn.ReLU(),
            nn.Linear(12, 26),
            nn.ReLU()
        )

    def forward(self, x):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return decoded


# dataset

with open("capture.pickle", "rb") as f:
    X = pickle.load(f)
X = X.astype(np.float32)
train_dataset = torch.from_numpy(X[:4700, :])
# test_dataset = torch.from_numpy(X[4700:, :])

bs = 64
train_dl = DataLoader(train_dataset, batch_size=bs, shuffle=True)
# test_dl = DataLoader(test_dataset, batch_size=bs)

model = Autoencoder()

lr = 1e-3
loss_func = F.mse_loss
# opt = optim.SGD(model.parameters(), lr=lr)  # con SGC parece que va muuy lento
opt = optim.Adam(model.parameters(), lr=lr)   # optimiza rapido

def fit(epochs, train_dl):
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


#for (xb,_) in test_dl:
#    break

#plt.imshow(xb[6].numpy().reshape(28,28)); plt.savefig("original.png"); plt.close()

#with torch.no_grad():
#    plt.imshow(model(xb[6].view(-1,784)).numpy().reshape(28,28)); plt.savefig("recons.png"); plt.close()

# In [51]: describe(results_train)
# m=0.75, M=0.86, mean=0.80, p80,90,99 = 0.82,0.82,0.86
# In [52]: describe(results_otro)
# m=0.75, M=1.27, mean=0.89, p80,90,99 = 0.88,0.88,1.16