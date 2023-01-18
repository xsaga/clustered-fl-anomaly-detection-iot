# TODO cambiar nombre
import argparse
import pickle
import re
from pathlib import Path
from typing import Dict, List, Pattern, Tuple

import numpy as np
import pandas as pd
import torch
import scipy
from scipy.spatial import ConvexHull, distance_matrix
from matplotlib import cm
from matplotlib import rcParams
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from model_ae import Autoencoder


def count_cluster_items(cluster_predicted: np.ndarray, item_labels: np.ndarray) -> Dict[int, List[Tuple[str, int]]]:
    """Show the device composition of each identified cluster.

    For each cluster, return a list with the unique labels and counts
    of the devices belonging to that cluster.
    """
    result = {}
    for i in range(cluster_predicted.max() + 1):
        items_in_cluster = item_labels[cluster_predicted == i]
        result[i] = list(zip(*np.unique(items_in_cluster, return_counts=True)))
    return result


parser = argparse.ArgumentParser(description="Evaluate clustering results in compromised state.")
parser.add_argument("--dir", required=True, type=lambda p: Path(p).absolute())
parser.add_argument("--dimensions", required=True, type=int)
parser.add_argument("--show", action="store_true")
args = parser.parse_args()  # parser.parse_args(["--dir", "clus_4epochs_port_hier_iot/", "--dimensions", "69", "--show"])

if not args.show:
    rcParams["font.family"] = ["Times New Roman"]
    rcParams["font.size"] = 8
    rcParams["xtick.labelsize"] = 8
    rcParams["ytick.labelsize"] = 8
    rcParams["axes.labelsize"] = 8
    rcParams["legend.fontsize"] = 8
    # rcParams["lines.linewidth"] = 0.75
    # rcParams["lines.markersize"] = 2
    plot_width = 3.487  # in
    plot_height = 3.487


np.random.seed(1)

model_paths = list(args.dir.glob("*.pt"))
print(f"Found {len(model_paths)} models in {args.dir}")

weights = []
names = []
for model_path in model_paths:
    m = Autoencoder(args.dimensions)
    m.load_state_dict(torch.load(model_path))
    model_weights = np.concatenate([v.detach().numpy().flatten() for v in m.state_dict().values()])
    weights.append(model_weights)
    names.append(Path(model_path).stem)

names = [n.split("_")[0].replace("iotsim-", "") for n in names]
tags = np.array([n.rsplit("-", 1)[0] for n in names])
weights = np.array(weights)

# cluster based on non-compromised state (K=8)
name_cluster_map = {re.compile(r"air-quality-[0-9]*"): 0,
                    re.compile(r"building-monitor-[0-9]*"): 0,
                    re.compile(r"city-power-[0-9]*"): 2,
                    re.compile(r"combined-cycle-[0-9]*"): 2,
                    re.compile(r"combined-cycle-tls-[0-9]*"): 7,
                    re.compile(r"cooler-motor-[0-9]*"): 3,
                    re.compile(r"domotic-monitor-[0-9]*"): 0,
                    re.compile(r"hydraulic-system-[0-9]*"): 1,
                    re.compile(r"ip-camera-museum-[0-9]*"): 4,
                    re.compile(r"ip-camera-street-[0-9]*"): 4,
                    re.compile(r"predictive-maintenance-([1-9]|10)"): 5,  # from 1 to 10
                    re.compile(r"predictive-maintenance-[1-9]+"): 6,  # from 11 to 15
                    re.compile(r"stream-consumer-[0-9]*"): 4}


def map_regexp(name: str, regexmap: Dict[Pattern[str], int]) -> int:
    for k, v in regexmap.items():
        if k.fullmatch(name):
            return v
    return -1


labels = pd.Series(names).map(lambda x: map_regexp(x, name_cluster_map)).values

print("Cluster composition:")
for cluster_idx, items in count_cluster_items(labels, tags).items():
    print(f"Cluster {cluster_idx}: {items}")


def average_diameter_distance(points: np.ndarray) -> float:
    n = points.shape[0]
    distances = distance_matrix(points, points, p=2)
    # 1 / (N (N-1) / 2) * 1 / 2 dist_sum
    return distances.sum() / (n * (n - 1))


# print(average_diameter_distance(weights))
results = {}
for i in range(labels.max() + 1):
    cluster_points = weights[labels == i]
    results[f"Cluster {i}"] = average_diameter_distance(cluster_points)
    print(f"Cluster {i} ({cluster_points.shape[0]} devices) avg diameter {average_diameter_distance(cluster_points)}")

with open(args.dir / "results.pickle", "wb") as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

# 2D Viz
pca = PCA(n_components=2)
pca.fit(weights)
reduced_2d = pca.transform(weights)

cmap = cm.get_cmap("viridis", labels.max() + 1)


def select_position(positions, rest):
    """Find an 'acceptable' position to place the text label in a plot.

    From a list of candidate positions, return the position that
    maximizes the distance to the closest point.
    """
    distances = scipy.spatial.distance.cdist(positions, rest)
    closest = distances.min(axis=1)
    selection = np.argmax(closest)
    return positions[selection]


fig, ax = plt.subplots()
ax.scatter(reduced_2d[:, 0], reduced_2d[:, 1], c=labels, alpha=0.4, linewidth=0.5)

for i in range(labels.max() + 1):
    red_cluster = reduced_2d[labels == i]
    if red_cluster.shape[0] < 3:
        continue  # not enough points for convex hull
    hull = ConvexHull(red_cluster)
    for simplex in hull.simplices:
        ax.plot(red_cluster[simplex, 0], red_cluster[simplex, 1], c=cmap(i), linestyle="solid", alpha=0.5)

for i in range(labels.max() + 1):
    red_clus = reduced_2d[labels == i]
    clus_mean = red_clus.mean(axis=0)
    clus_ymax = red_clus[:, 1].max()
    clus_text = f"Cluster {i}"
    # ax.scatter(clus_mean[0], clus_mean[1], c="green", marker="v")

    options = [[clus_mean[0], red_clus[:, 1].max() + 0.25],
               [clus_mean[0], red_clus[:, 1].min() - 0.25]]
    # [red_clus[:, 0].max() + 0.2, clus_mean[1]],
    # [red_clus[:, 0].min() - 0.2, clus_mean[1]]]
    # text_x, text_y = clus_mean[0], clus_ymax + 0.2
    text_x, text_y = select_position(options, reduced_2d[labels != i])
    ax.text(text_x, text_y, s=clus_text, c="black", bbox=dict(facecolor=cmap(i), edgecolor=cmap(i), alpha=0.25, boxstyle="round", pad=0.1), horizontalalignment="center")

if args.show:
    for i, n in enumerate(names):
        n = f"({labels[i]}) {n}"
        ax.annotate(n, (reduced_2d[i, 0], reduced_2d[i, 1])).set_alpha(0.1)

ax.set_xlabel("principal component 1")
ax.set_ylabel("principal component 2")
if args.show:
    plt.show()
else:
    fig.set_size_inches(plot_width, plot_height)
    fig.tight_layout()
    fig.savefig(args.dir / "2d_map.pdf", format="pdf")
