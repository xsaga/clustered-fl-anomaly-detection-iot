from pathlib import Path
import itertools
import re
import torch
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib import pyplot as plt
import seaborn as sns


def get_exp_key(d, k):
    return list(filter(lambda x: k in x, d.keys()))


rcParams["font.family"] = ["Times New Roman"]
rcParams["font.size"] = 8
rcParams["xtick.labelsize"] = 8
rcParams["ytick.labelsize"] = 8
rcParams["axes.labelsize"] = 8
rcParams["legend.fontsize"] = 8
# rcParams["lines.linewidth"] = 0.75
rcParams["lines.markersize"] = 4
plot_width = 3.487  # in
plot_height = 3.487

base_path = Path("./cluster_mqtt/")
exp_pattern = re.compile("lrgridsearchA([0-9]+)")

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
fig, ax = plt.subplots()
for exp_name, losses in all_experiments.items():
    label_name = re.match(exp_pattern, exp_name).group(1)
    eval_loss = losses["eval"]
    ax.plot(range(1, len(eval_loss) + 1), eval_loss, label=label_name, marker=next(markers), linestyle="dashed")

ax.set_xlim(left=1)
axlabels = [str(lbl) if lbl % 10 == 0 else "" for lbl in np.arange(fl_rounds)]
axlabels[0] = ""
axlabels[1] = "1"
ax.set_xticks(np.arange(fl_rounds), axlabels)
ax.legend(loc="upper right", title="Trial", ncol=2)
ax.set_xlabel("FL rounds")
ax.set_ylabel("evaluation loss")
ax.set_yscale("log")
fig.set_size_inches(plot_width, plot_height)
fig.tight_layout()
# fig.show()
fig.savefig("cluster_mqtt_clientopt_serveropt.pdf", format="pdf")


res = {}
for key, val in all_experiments.items():
    subkey1 = key.split("_")[1]
    subkey2 = key.split("_")[2]
    subkey1 = subkey1.split("lr")[-1].replace("p", ".")
    subkey2 = subkey2.split("lr")[-1].replace("p", ".")
    if subkey1 in res:
        res[subkey1][subkey2] = val["eval"][-1]
    else:
        res[subkey1] = {subkey2: val["eval"][-1]}

df = pd.DataFrame(res)
df.sort_index(axis="columns", inplace=True)
df.sort_index(axis="columns", inplace=True)
print(np.log10(df))

# https://stackoverflow.com/questions/27037241/changing-the-rotation-of-tick-labels-in-seaborn-heatmap
sns.heatmap(np.log10(df), annot=True, fmt=".3g", cmap=sns.color_palette("Blues", as_cmap=True).reversed())
plt.yticks(rotation=0)
plt.xlabel("client learning rate")
plt.ylabel("server learning rate")
fig = plt.gcf()
fig.set_size_inches(plot_width, plot_height)
plt.tight_layout()
# plt.show()
plt.savefig("cluster_mqtt_lrgs_adam1_sgd.pdf", format="pdf")


for exp_name in get_exp_key(all_experiments, "adam1lr0p001_sgdlr1p5"):
    eval_loss = all_experiments[exp_name]["eval"]
    plt.plot(range(1, len(eval_loss) + 1), eval_loss, label=exp_name, marker=next(markers), linestyle="dashed")
plt.legend()
plt.yscale("log")
plt.show()
