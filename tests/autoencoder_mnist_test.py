import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import torchvision

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 784),
            nn.ReLU()
        )

    def forward(self, x):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return decoded


# dataset
train_dataset = torchvision.datasets.MNIST(root="/home/osboxes/torch_datasets",
    train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root="/home/osboxes/torch_datasets",
    train=False, transform=torchvision.transforms.ToTensor(), download=True)

bs = 64
train_dl = DataLoader(train_dataset, batch_size=bs)
test_dl = DataLoader(test_dataset, batch_size=bs)

model = Autoencoder()

lr = 1e-3
loss_func = F.mse_loss
# opt = optim.SGD(model.parameters(), lr=lr)  # con SGC parece que va muuy lento
opt = optim.Adam(model.parameters(), lr=lr)   # optimiza rapido

def fit(epochs, train_dl):
    for epoch in range(epochs):
        loss_acc = 0
        print(f"--- epoch #{epoch} ---")
        for (xb,_) in train_dl:
            xb = xb.view(-1, 784)
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
