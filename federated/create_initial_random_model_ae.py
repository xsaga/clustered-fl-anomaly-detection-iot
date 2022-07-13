import sys
from model_ae import Autoencoder
import torch


data_dimensions = int(sys.argv[1])
model = Autoencoder(data_dimensions)
print(model)
torch.save(model.state_dict(), "initial_random_model_ae.pt")
