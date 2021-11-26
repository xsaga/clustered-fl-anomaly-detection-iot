from pathlib import Path
import itertools
import re
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


base_path = Path("./")
exp_pattern = re.compile("exp([0-9]+)")

experiment_files = [d for d in base_path.iterdir() if d.is_dir() and re.match(exp_pattern, d.name)]
experiment_files = sorted(experiment_files, key=lambda p: int(re.match(exp_pattern, p.name).group(1)))

all_experiments = {}

for experiment in experiment_files:
    experiment_name = experiment.name
    experiment_models_global_path = experiment / "models_global"

    global_round_dirs = [p for p in experiment_models_global_path.iterdir() if p.is_dir() and p.match("round_*")]
    global_round_dirs = sorted(global_round_dirs, key=lambda p: int(p.stem.rsplit("_", 1)[-1]))

    experiment_eval_losses = []
    experiment_train_losses = []

    for round in global_round_dirs[1:]:
        print(f"{experiment_name}:: {round.name}")
        ckpt_file = [f for f in round.iterdir() if f.is_file() and f.match("global_model_round_*.tar")][0]
        ckpt = torch.load(ckpt_file)
        experiment_eval_losses.append(ckpt["loss"])
        experiment_train_losses.append(ckpt["train_loss"])

    all_experiments[experiment_name] = {"eval": experiment_eval_losses, "train": experiment_train_losses}

# las keys del dic estan ordenadas?

filtrar_exp = ("exp8", "exp14", "exp7_mal", "exp2")

rename_map = {'exp1': "trial 1",
              'exp2': "exp2",
              'exp3': "trial 2",
              'exp4': "trial 3",
              'exp5': "trial 4",
              'exp6': "trial 5",
              'exp7_mal': "exp7_mal",
              'exp8': "exp8",
              'exp9': "trial 6",
              'exp10': "trial 7",
              'exp11': "trial 8",
              'exp12': "trial 9",
              'exp13': "trial 10",
              'exp14': "exp14",
              'exp15': "trial 11"}

# table
for exp_key, exp_eval_train in all_experiments.items():
    if exp_key in filtrar_exp:
        continue
    eval_loss = exp_eval_train["eval"]
    print(f"{exp_key} == {rename_map[exp_key]}, last round mean eval loss: {eval_loss[-1]}")

# plot
markers = itertools.cycle(("o", "v", "^", "<", ">", "s", "p", "P", "*", "h", "X", "D"))
for exp_key, exp_eval_train in all_experiments.items():
    if exp_key in filtrar_exp:
        continue
    eval_loss = exp_eval_train["eval"]
    plt.plot(list(range(1, len(eval_loss)+1)), eval_loss, label=rename_map[exp_key], marker=next(markers))

plt.legend(loc="upper right")
plt.xlabel("FL rounds")
plt.ylabel("Evaluation loss")
plt.tight_layout()
plt.show()


# heatmap
# col = lr_s, idx=lr_c
          # lr_c
lr_idx = ["0.1", "0.01", "0.001", "0.0001"]

           # lr_s
lr_data = {"0.1":[all_experiments["lr_exp16"]["eval"][-1], all_experiments["lr_exp7"]["eval"][-1], all_experiments["lr_exp8"]["eval"][-1], all_experiments["lr_exp9"]["eval"][-1]],
           "0.5":[all_experiments["lr_exp14"]["eval"][-1], all_experiments["lr_exp10"]["eval"][-1], all_experiments["lr_exp11"]["eval"][-1], all_experiments["lr_exp12"]["eval"][-1]],
           "1.0":[all_experiments["lr_exp15"]["eval"][-1], all_experiments["lr_exp4"]["eval"][-1], all_experiments["lr_exp5"]["eval"][-1], all_experiments["lr_exp6"]["eval"][-1]],
           "1.5":[all_experiments["lr_exp13"]["eval"][-1], all_experiments["lr_exp1"]["eval"][-1], all_experiments["lr_exp2"]["eval"][-1], all_experiments["lr_exp3"]["eval"][-1]]}

lr_df = pd.DataFrame(lr_data, index=lr_idx)
sns.heatmap(np.log10(lr_df), annot=True, cmap=sns.color_palette("YlOrBr", as_cmap=True).reversed())
plt.xlabel("Server learning rate")
plt.ylabel("Client learning rate")
plt.tight_layout()
plt.show()
