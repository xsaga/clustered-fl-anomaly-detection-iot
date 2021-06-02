from collections import OrderedDict
import hashlib
import os
from typing import List, Tuple, Optional, Dict, Union
import flwr as fl
from flwr.common import Parameters, Scalar, FitRes, parameters_to_weights, weights_to_parameters
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(27, 12), # 26,12
            # nn.Dropout(0.05), # !
            nn.ReLU(), # nn.LeakyReLU(), nn.ReLU()
            nn.Linear(12, 4), # 12,4
            nn.ReLU()  # nn.Sigmoid() # nn.Tanh(), nn.Sigmoid(), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 12), # 4,12
            nn.ReLU(), # nn.LeakyReLU(), nn.ReLU()
            nn.Linear(12, 27), # 12,26
            # nn.Dropout(0.05), # !
            nn.ReLU() # nn.ReLU()
        )

    def forward(self, x):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return decoded


def flwr_parameters_to_state_dict(parameters: Parameters, keys: List[str]) -> 'OrderedDict[str, torch.Tensor]':
    weights = parameters_to_weights(parameters)
    return OrderedDict({k:torch.Tensor(v) for k,v in zip(keys, weights)})


def state_dict_hash(state_dict: Union['OrderedDict[str, torch.Tensor]', Dict[str, torch.Tensor]]) -> str:
    h = hashlib.md5()
    for k, v in state_dict.items():
        h.update(k.encode("utf-8"))
        h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()


class CustomStrategy(FedAvg):
    def aggregate_fit(self, rnd: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_result = super().aggregate_fit(rnd, results, failures)
        
        if aggregated_result:
            params = aggregated_result[0]
            if params:
                print("Saving model...")
                state_dict = flwr_parameters_to_state_dict(params, list(Autoencoder().state_dict().keys()))  #todo pasar keys a la clase
                print("Aggregated model hash: ", state_dict_hash(state_dict))
                torch.save(state_dict, f"global_model_round{rnd}.pt")
        return aggregated_result


model = Autoencoder()

initial_random_model_path = "initial_random_model.pt"
if os.path.isfile(initial_random_model_path):
    # load initial random model
    model.load_state_dict(torch.load(initial_random_model_path), strict=True)
    print(f"Loaded initial model {initial_random_model_path}")
else:
    # save initial random model
    torch.save(model.state_dict(), initial_random_model_path)
    print(f"Saved initial model {initial_random_model_path}")

init_weights = [val.cpu().numpy() for val in model.state_dict().values()]

configuration = {"num_rounds": 1}

strategy = CustomStrategy(fraction_fit=0.1,
                          fraction_eval=0.1,
                          min_fit_clients=2,
                          min_eval_clients=2,
                          min_available_clients=2,
                          initial_parameters=weights_to_parameters(init_weights))

print("Initial model hash: ", state_dict_hash(model.state_dict()))
fl.server.start_server("127.0.0.1:8080", config=configuration, strategy=strategy)
