import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt

global_models_dir_path = Path("./models_global")

global_round_dirs = [p for p in global_models_dir_path.iterdir() if p.is_dir() and p.match("round_*")]
global_round_dirs = sorted(global_round_dirs, key=lambda p: int(p.stem.rsplit("_", 1)[-1]))

w_losses = []
w_train_losses = []
for d in global_round_dirs[1:]:
    loss_f = [f for f in d.iterdir() if f.is_file() and f.match("losses_num_samples_round*.npz")][0]
    losses, train_losses, num_samples = np.load(loss_f).values()
    w_losses.append(np.average(losses, weights=num_samples))
    w_train_losses.append(np.average(train_losses, weights=num_samples))

plt.scatter(list(range(1, len(global_round_dirs))), w_losses, color="blue")
plt.scatter(list(range(1, len(global_round_dirs))), w_train_losses, color="red")
plt.plot(list(range(1, len(global_round_dirs))), w_losses, color="blue")
plt.plot(list(range(1, len(global_round_dirs))), w_train_losses, color="red")
plt.show()

# poner el LOSS en escala Log

# res = [p for p in curr.iterdir() if p.is_file() and p.match("results_nofl_epoch_losses_ae*")]
# In [48]: losses = []
#     ...: for f in res:
#     ...:     epochs, loss, val_loss = np.load(f).values()
#     ...:     losses.append(loss)
# losses = np.array(losses)
# avg_loss = np.mean(losses, axis=0)