from pathlib import Path
import itertools
import re
import torch
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib import pyplot as plt
import seaborn as sns


base_path = Path("./")
exp_pattern = re.compile("trial([0-9]+)")

experiment_files = [d for d in base_path.iterdir() if d.is_dir() and re.match(exp_pattern, d.name)]
experiment_files = sorted(experiment_files, key=lambda p: int(re.match(exp_pattern, p.name).group(1)))
print(f"{len(experiment_files)} experiment files found.")

fl_rounds = None

all_experiments = {}
for experiment in experiment_files:
    experiment_name = experiment.name
    experiment_models_global_path = experiment / "models_global"

    global_round_dirs = [p for p in experiment_models_global_path.iterdir() if p.is_dir() and p.match("round_*")]
    global_round_dirs = sorted(global_round_dirs, key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    fl_rounds = len(global_round_dirs)

    experiment_eval_losses = []
    experiment_train_losses = []

    for round in global_round_dirs[1:]:
        print(f"{experiment_name}: {round.name}")
        ckpt_file = [f for f in round.iterdir() if f.is_file() and f.match("global_model_round_*.tar")][0]
        ckpt = torch.load(ckpt_file)
        experiment_eval_losses.append(ckpt["loss"])
        experiment_train_losses.append(ckpt["train_loss"])

    all_experiments[experiment_name] = {"eval": experiment_eval_losses, "train": experiment_train_losses}

markers = itertools.cycle(("o", "v", "^", "<", ">", "s", "p", "P", "*", "h", "X", "D", "1", "2", "3", "4", "+", "x"))
for exp_name, losses in all_experiments.items():
    eval_loss = losses["eval"]
    plt.plot(range(1, len(eval_loss)+1), eval_loss, label=exp_name, marker=next(markers), linestyle="dashed")

plt.xlim(left=1)
axlabels = [str(l) if l%10==0 else "" for l in np.arange(fl_rounds)]
axlabels[0] = ""
axlabels[1] = "1"
plt.xticks(np.arange(fl_rounds), axlabels)
plt.legend(loc="upper right")
plt.xlabel("FL rounds")
plt.ylabel("evaluation loss")
plt.yscale("log")
plt.tight_layout()
plt.show()


res={}
for key, val in all_experiments.items():
    subkey1 = key.split("_")[1]
    subkey2 = key.split("_")[2]
    if subkey1 in res:
        res[subkey1][subkey2] = val["eval"][-1]
    else:
        res[subkey1] = {subkey2:val["eval"][-1]}

df = pd.DataFrame(res)
df.sort_index(axis="columns", inplace=True)
df.sort_index(axis="columns", inplace=True)
print(np.log10(df))

# https://stackoverflow.com/questions/27037241/changing-the-rotation-of-tick-labels-in-seaborn-heatmap
sns.heatmap(np.log10(df), annot=True, cmap=sns.color_palette("Blues", as_cmap=True).reversed())
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()