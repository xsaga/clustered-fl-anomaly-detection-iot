import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import rcParams
import seaborn as sns

rcParams["font.family"] = ["Times New Roman"]
rcParams["font.size"] = 8
rcParams["xtick.labelsize"] = 8
rcParams["ytick.labelsize"] = 8
rcParams["axes.labelsize"] = 8
rcParams["legend.fontsize"] = 8
# rcParams["lines.linewidth"] = 0.75
# rcParams["lines.markersize"] = 2
plot_height = 2.502  # 3.487  # in
plot_width = plot_height * 1.2

total_results: Dict[str, List[Dict[str, float]]] = {}
#              ^- trials for different number of compromised devices
#                        ^- different repetitions for each trial
#                           of compromised devices
#                             ^- data

BASE_DIR = Path("./clustering_compromised")
trial_dirs = [d for d in BASE_DIR.iterdir() if d.is_dir() and d.match("trial_*")]
trial_dirs = sorted(trial_dirs, key=lambda d: int(d.name.split("_")[1]))

# Load data
for trial in trial_dirs:
    trial_list = []

    repetitions_dirs = [d for d in trial.iterdir() if d.is_dir() and d.match("clus_4epochs_re*")]
    repetitions_dirs = sorted(repetitions_dirs, key=lambda d: int(d.name.split("_re")[-1]))

    for repetition in repetitions_dirs:
        with open(repetition / "results.pickle", "rb") as f:
            trial_list.append(pickle.load(f))

    total_results[trial.name] = trial_list

# Transform data
transformed_results: Dict[str, Dict[str, List[float]]] = {}

for key, val in total_results.items():
    # Columns: Cluster 0, Cluster 1, ..., Cluster K
    # Rows   : r repetitions
    df = pd.DataFrame(val)
    transformed_results[key] = df.to_dict(orient="list")

# Transformed Clusters first
clusters_first = {}
for k in range(8):
    xxx = {}
    for trial in transformed_results.keys():
        xxx[trial] = transformed_results[trial][f"Cluster {k}"]
    clusters_first[f"Cluster {k}"] = xxx

# Number of devices in each cluster
num_dev_clus_map = {0:11,
                    1:15,
                    2:11,
                    3:15,
                    4:6,
                    5:10,
                    6:5,
                    7:5}

# Plots
for k in range(8):
    cluster_df = pd.DataFrame(clusters_first[f"Cluster {k}"])
    cluster_df.rename(mapper=lambda x: x.split("_")[1], axis="columns", inplace=True)
    num_devices = num_dev_clus_map[k]

    fig, ax = plt.subplots()
    # sns.boxplot(data=cluster_df, ax=ax, flierprops={"marker": "o"})
    # ax.set_title(f"Cluster {k}")

    ax.boxplot(cluster_df, labels=[int(i) if int(i) <= num_devices else num_devices for i in cluster_df.columns])
    ax.axvline(x=num_devices+0.5, ls="dashed", color="red")
    ax.set_xlabel("Number of compromised devices in cluster")
    ax.set_ylabel("Average cluster diameter")
    fig.set_size_inches(plot_width, plot_height)
    fig.tight_layout()
    fig.savefig(BASE_DIR / f"cluster_{k}_compromised_diameter.pdf", format="pdf")
