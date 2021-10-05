import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt

global_models_dir_path = Path("./models_global")

global_round_dirs = [p for p in global_models_dir_path.iterdir() if p.is_dir() and p.match("round_*")]
global_round_dirs = sorted(global_round_dirs, key=lambda p: int(p.stem.rsplit("_", 1)[-1]))

# FL

fl_weighted_losses = []
fl_losses_min = []
fl_losses_max = []
fl_weighted_train_losses = []
fl_losses_raw = []
for d in global_round_dirs[1:]:
    loss_f = [f for f in d.iterdir() if f.is_file() and f.match("losses_num_samples_round*.npz")][0]
    fl_losses, fl_train_losses, num_samples = np.load(loss_f).values()
    fl_losses_raw.append(fl_losses)
    fl_weighted_losses.append(np.average(fl_losses, weights=num_samples))
    fl_losses_min.append(np.min(fl_losses))
    fl_losses_max.append(np.max(fl_losses))
    fl_weighted_train_losses.append(np.average(fl_train_losses, weights=num_samples))

plt.scatter(list(range(1, len(global_round_dirs))), fl_weighted_losses, color="blue")
plt.scatter(list(range(1, len(global_round_dirs))), fl_weighted_train_losses, color="red")
plt.plot(list(range(1, len(global_round_dirs))), fl_weighted_losses, color="blue")
plt.plot(list(range(1, len(global_round_dirs))), fl_weighted_train_losses, color="red")
plt.yscale("log")
plt.show()

fl_losses_raw = np.array(fl_losses_raw)
fl_losses_raw=fl_losses_raw.transpose((1,0))
# no FL

nofl_dir = Path("./nofl/")
nofl_files = [p for p in nofl_dir.iterdir() if p.is_file() and p.match("results_nofl_epoch_losses_*")]
losses = []
for f in nofl_files:
    epochs, loss, val_loss = np.load(f).values()
    losses.append(val_loss)
losses = np.array(losses)
min_loss = np.min(losses, axis=0)
max_loss = np.max(losses, axis=0)
avg_loss = np.mean(losses, axis=0)


# poner doble eje X
## plt.scatter(list(range(1, len(global_round_dirs))), fl_weighted_losses, color="blue")
plt.plot(list(range(1, len(global_round_dirs))), fl_weighted_losses, color="blue", label="Avg FL loss")
## plt.fill_between(list(range(1, len(global_round_dirs))), fl_losses_max, fl_losses_min, alpha=0.3, color="blue", hatch="\\", label="Min to Max FL loss range")
plt.boxplot(fl_losses_raw, showmeans=True, patch_artist=True, boxprops=dict(facecolor="blue", edgecolor="blue", hatch="\\", alpha=0.2), capprops=dict(color="blue"), whiskerprops=dict(color="blue"), flierprops=dict(color="blue", markeredgecolor="blue", alpha=0.2), medianprops=dict(color="blue"), meanprops=dict(markerfacecolor="blue", markeredgecolor="blue"))
## plt.scatter(list(range(1, len(epochs[4::5])+1)), avg_loss[4::5], color="green")
plt.plot(list(range(1, len(epochs[4::5])+1)), avg_loss[4::5], color="green", label="Avg isolated loss every 5 rounds")
## plt.fill_between(list(range(1, len(epochs[4::5])+1)), max_loss[4::5], min_loss[4::5], alpha=0.3, color="green", hatch="/", label="Min to Max isolated loss range")
plt.boxplot(losses[:,4::5], showmeans=True, patch_artist=True, boxprops=dict(facecolor="green", edgecolor="green", hatch="/", alpha=0.2), capprops=dict(color="green"), whiskerprops=dict(color="green"), flierprops=dict(color="green", markeredgecolor="green", alpha=0.2), medianprops=dict(color="green"), meanprops=dict(markerfacecolor="green", markeredgecolor="green"))
plt.yscale("log")
plt.xlabel("Training rounds")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()


####
#for i in range(len(nofl_files)):
#    print(nofl_files[i])
#    e, l, vl = np.load(nofl_files[i]).values()
#    plt.plot(e, l, color="blue", label="train loss")
#    plt.plot(e, vl, color="red", label="validation loss")
#    plt.title(nofl_files[i])
#    plt.legend(); plt.show()
