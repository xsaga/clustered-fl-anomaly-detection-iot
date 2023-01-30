"""Visualize and evaluate client clustering results.

Use after 'cluster_train.sh'
"""
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import scipy
from scipy.spatial import ConvexHull
from s_dbw import S_Dbw
from matplotlib import cm
from matplotlib import rcParams
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, completeness_score, homogeneity_completeness_v_measure, homogeneity_score, normalized_mutual_info_score, rand_score, v_measure_score

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


def create_clustering(experiment_dir: Path, data_dimensions) -> Dict[int, np.ndarray]:
    np.random.seed(1)

    model_paths = list(experiment_dir.glob("*.pt"))
    # print(f"Found {len(model_paths)} models in {experiment_dir}")

    weights = []
    names = []
    for model_path in model_paths:
        m = Autoencoder(data_dimensions)
        m.load_state_dict(torch.load(model_path))
        model_weights = np.concatenate([v.detach().numpy().flatten() for v in m.state_dict().values()])
        weights.append(model_weights)
        names.append(Path(model_path).stem)

    names = [n.split("_")[0].replace("iotsim-", "") for n in names]
    weights = np.array(weights)

    # cluster
    pca_cluster = PCA(n_components=0.9)  # explain 90% var
    reduced_weights = pca_cluster.fit_transform(weights)
    # print(reduced_weights.shape)

    # range n_cluster in cluster_numbers instead of selecting the specific value of interest
    # to keep the same random seed and make it comparable to cluster_client_evaluation.py
    cluster_numbers = list(range(2, min(41, weights.shape[0] - 1)))
    kmeans_labels: Dict[int, np.ndarray] = {}

    for n_clusters in cluster_numbers:
        # print(n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=50).fit(reduced_weights)
        kmeans_labels[n_clusters] = kmeans.labels_

    return kmeans_labels


def eval_clustering(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    return adjusted_rand_score(labels_true, labels_pred)


parser = argparse.ArgumentParser(description="Evaluate clustering results.")
parser.add_argument("--dir", required=True, type=lambda p: Path(p).absolute())
parser.add_argument("--dimensions", required=True, type=int)
# parser.add_argument("--clusters", required=False, type=int, default=0)
parser.add_argument("--show", action="store_true")
parser.add_argument("--image-format", required=False, type=str, choices=["pdf", "png"], default="pdf")
args = parser.parse_args()  # parse_args(["--dir", "clustering_results", "--dimensions", "27"])

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
    plot_height = 2.155  # 3.487

# Target clustering
reference_cluster = [('air-quality-1', 0),
                     ('building-monitor-1', 0),
                     ('building-monitor-2', 0),
                     ('building-monitor-3', 0),
                     ('building-monitor-4', 0),
                     ('building-monitor-5', 0),
                     ('city-power-1', 2),
                     ('combined-cycle-10', 2),
                     ('combined-cycle-1', 2),
                     ('combined-cycle-2', 2),
                     ('combined-cycle-3', 2),
                     ('combined-cycle-4', 2),
                     ('combined-cycle-5', 2),
                     ('combined-cycle-6', 2),
                     ('combined-cycle-7', 2),
                     ('combined-cycle-8', 2),
                     ('combined-cycle-9', 2),
                     ('combined-cycle-tls-1', 7),
                     ('combined-cycle-tls-2', 7),
                     ('combined-cycle-tls-3', 7),
                     ('combined-cycle-tls-4', 7),
                     ('combined-cycle-tls-5', 7),
                     ('cooler-motor-10', 3),
                     ('cooler-motor-11', 3),
                     ('cooler-motor-12', 3),
                     ('cooler-motor-13', 3),
                     ('cooler-motor-14', 3),
                     ('cooler-motor-15', 3),
                     ('cooler-motor-1', 3),
                     ('cooler-motor-2', 3),
                     ('cooler-motor-3', 3),
                     ('cooler-motor-4', 3),
                     ('cooler-motor-5', 3),
                     ('cooler-motor-6', 3),
                     ('cooler-motor-7', 3),
                     ('cooler-motor-8', 3),
                     ('cooler-motor-9', 3),
                     ('domotic-monitor-1', 0),
                     ('domotic-monitor-2', 0),
                     ('domotic-monitor-3', 0),
                     ('domotic-monitor-4', 0),
                     ('domotic-monitor-5', 0),
                     ('hydraulic-system-10', 1),
                     ('hydraulic-system-11', 1),
                     ('hydraulic-system-12', 1),
                     ('hydraulic-system-13', 1),
                     ('hydraulic-system-14', 1),
                     ('hydraulic-system-15', 1),
                     ('hydraulic-system-1', 1),
                     ('hydraulic-system-2', 1),
                     ('hydraulic-system-3', 1),
                     ('hydraulic-system-4', 1),
                     ('hydraulic-system-5', 1),
                     ('hydraulic-system-6', 1),
                     ('hydraulic-system-7', 1),
                     ('hydraulic-system-8', 1),
                     ('hydraulic-system-9', 1),
                     ('ip-camera-museum-1', 4),
                     ('ip-camera-museum-2', 4),
                     ('ip-camera-street-1', 4),
                     ('ip-camera-street-2', 4),
                     ('predictive-maintenance-10', 5),
                     ('predictive-maintenance-11', 6),
                     ('predictive-maintenance-12', 6),
                     ('predictive-maintenance-13', 6),
                     ('predictive-maintenance-14', 6),
                     ('predictive-maintenance-15', 6),
                     ('predictive-maintenance-1', 5),
                     ('predictive-maintenance-2', 5),
                     ('predictive-maintenance-3', 5),
                     ('predictive-maintenance-4', 5),
                     ('predictive-maintenance-5', 5),
                     ('predictive-maintenance-6', 5),
                     ('predictive-maintenance-7', 5),
                     ('predictive-maintenance-8', 5),
                     ('predictive-maintenance-9', 5),
                     ('stream-consumer-1', 4),
                     ('stream-consumer-2', 4)]

reference_labels = np.array(list(map(lambda x: x[1], reference_cluster)))
best_n_clusters = np.max(reference_labels) + 1

# loop
file_pattern = re.compile("clus_4epochs_port_hier_iot_rep([0-9]+)_([0-9.]+)frac")

results = []

for child in filter(lambda x: x.is_dir() and re.match(file_pattern, x.name), args.dir.iterdir()):
    child_match = re.match(file_pattern, child.name)
    repetition = int(child_match.group(1))
    fraction = float(child_match.group(2))
    all_clusters = create_clustering(child, args.dimensions)
    score = eval_clustering(reference_labels, all_clusters[best_n_clusters])
    print(f"-> rep {repetition}\tfrac {fraction}:\t{score}")
    results.append((fraction, repetition, score))

results_df = pd.DataFrame(results, columns=["Fraction", "Repetition", "Score"])
fig, ax = plt.subplots()
sns.boxplot(data=results_df, x="Fraction", y="Score",
            flierprops={"marker": "o"},
            ax=ax)
fig.show()

# new figure
results_df_pivoted = pd.pivot(results_df, index="Repetition", columns="Fraction")
fraction_labels = results_df_pivoted.columns.get_level_values(1).to_numpy() * 100

fig, ax = plt.subplots()
ax.boxplot(results_df_pivoted, labels=fraction_labels, capprops=dict(color="black", linewidth=3))
ax.hlines(y=1.0, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], colors="silver", linestyles="dotted")
ax.set_xlabel("training data fraction (%)")
ax.set_ylabel("adjusted rand score")
if args.show:
    fig.show()
else:
    fig.set_size_inches(plot_width, plot_height)
    fig.tight_layout()
    fig.savefig(args.dir / f"cluster_stability_varying_fraction.{args.image_format}", format=args.image_format)
