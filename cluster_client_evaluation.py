import argparse
import numpy as np

import torch
from model_ae import Autoencoder

from pathlib import Path

from matplotlib import cm
from matplotlib import rcParams
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

import scipy
from scipy.spatial import ConvexHull

from s_dbw import S_Dbw

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import rand_score, adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, homogeneity_completeness_v_measure


def count_cluster_items(cluster_predicted, item_labels):
    result = {}
    for i in range(cluster_predicted.max() + 1):
        items_in_cluster = item_labels[cluster_predicted==i]
        result[i] = list(zip(*np.unique(items_in_cluster, return_counts=True)))
    return result


parser = argparse.ArgumentParser(description="Evaluate clustering results.")
parser.add_argument("--dir", required=True, type=lambda p: Path(p).absolute())
parser.add_argument("--dimensions", required=True, type=int)
parser.add_argument("--clusters", required=False, type=int, default=0)
parser.add_argument("--show", action="store_true")
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
    plot_height = 3.487


np.random.seed(1)
data_dimensions = args.dimensions

model_paths = list(args.dir.glob("*.pt"))
print(f"Found {len(model_paths)} models in {args.dir}")

weights = []
names = []
for model_path in model_paths:
    m = Autoencoder(data_dimensions)
    m.load_state_dict(torch.load(model_path))
    model_weights = np.concatenate([v.detach().numpy().flatten() for v in m.state_dict().values()])
    weights.append(model_weights)
    names.append(Path(model_path).stem)

names = [n.split("_")[0].replace("iotsim-", "") for n in names]
tags = np.array([n.rsplit("-", 1)[0] for n in names])
weights = np.array(weights)

# cluster
pca_cluster = PCA(n_components=0.9)  # explain 90% var
reduced_weights = pca_cluster.fit_transform(weights)
print(reduced_weights.shape)

cluster_numbers = list(range(2, min(41, weights.shape[0]-1)))
kmeans_labels = dict()
score_ss = []
score_ch = []
score_db = []
score_sdbw = []
for n_clusters in cluster_numbers:
    print(n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=50).fit(reduced_weights)
    kmeans_labels[n_clusters] = kmeans.labels_

    score_ss.append(silhouette_score(reduced_weights, kmeans.labels_))
    score_ch.append(calinski_harabasz_score(reduced_weights, kmeans.labels_))
    score_db.append(davies_bouldin_score(reduced_weights, kmeans.labels_))
    score_sdbw.append(S_Dbw(reduced_weights, kmeans.labels_, centers_id=None, method="Tong", centr="mean", metric="euclidean"))

fig, ax1 = plt.subplots()
ax1.set_xlabel("number of clusters")
ax1.set_ylabel("scores")
ax1.plot(cluster_numbers, score_ss, label="Silhouette (max)", marker="o", color="#1f77b4")  # (MAX)
ax1.plot(cluster_numbers, score_db, label="Davies-Bouldin (min)", marker="D", color="#ff7f0e")  # (MIN)
ax1.plot(cluster_numbers, score_sdbw, label="S_Dbw (min)", marker="^", color="#2ca02c")  # (MIN)
# ax2 = ax1.twinx()
# ax2.set_ylabel("score calinski-harabasz")
# ax2.plot(cluster_numbers, score_ch, label="calinski-harabasz (max)", marker="P", color="#d62728")  # (MAX)
if args.clusters:
    ax1.axvline(x=args.clusters, linestyle=":", color="k")
ax1.xaxis.set_major_locator(MultipleLocator(2))
ax1.xaxis.set_minor_locator(MultipleLocator(1))
# fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), bbox_transform=ax1.transAxes)
ax1.legend(loc="best")
if args.show:
    ax1.set_title(f"Cluster analysis for {args.dir.name}")
    plt.show()
else:
    fig.set_size_inches(plot_width, plot_height)
    fig.tight_layout()
    fig.savefig(args.dir/"score_unsupervised.pdf", format="pdf")

device_name_map = {"air-quality": 0,
                   "building-monitor": 1,
                   "city-power": 2,
                   "combined-cycle": 3,
                   "combined-cycle-tls": 4,
                   "cooler-motor": 5,
                   "domotic-monitor": 6,
                   "hydraulic-system": 7,
                   "ip-camera-museum": 8,
                   "ip-camera-street": 9,
                   "predictive-maintenance": 10,
                   "stream-consumer": 11}

labels_true = np.array(list(map(lambda x: device_name_map[x], tags)))

ard_score = []
ami_score = []
vme_score = []
for n_clusters in cluster_numbers:
    ard_score.append(adjusted_rand_score(labels_true, kmeans_labels[n_clusters]))
    ami_score.append(adjusted_mutual_info_score(labels_true, kmeans_labels[n_clusters]))
    vme_score.append(v_measure_score(labels_true, kmeans_labels[n_clusters]))

fig, ax = plt.subplots()
ax.plot(cluster_numbers, ard_score, marker="o", label="Adjusted Rand")
ax.plot(cluster_numbers, ami_score, marker="D", label="Adjusted Mutual Information")
ax.plot(cluster_numbers, vme_score, marker="^", label="V Measure")
ax.set_xlabel("number of clusters")
ax.set_ylabel("scores")
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.legend(loc="best")
if args.show:
    plt.show()
else:
    fig.set_size_inches(plot_width, plot_height)
    fig.tight_layout()
    fig.savefig(args.dir/"score_groundtruth.pdf", format="pdf")

best_n_clusters = args.clusters if args.clusters > 0 else int(input("Selected number of clusters? "))
print("Selected ", best_n_clusters, " clusters")
print("Cluster composition:")
for cluster_idx, items in count_cluster_items(kmeans_labels[best_n_clusters], tags).items():
    print(f"Cluster {cluster_idx}: {items}")

# 2D Viz
pca = PCA(n_components=2)
pca.fit(weights)
reduced_2d = pca.transform(weights)

cmap = cm.get_cmap("viridis", best_n_clusters)

def select_position(positions, rest):
    # find a better position to place the text labels
    distances = scipy.spatial.distance.cdist(positions, rest)
    closest = distances.min(axis=1)
    selection = np.argmax(closest)
    return positions[selection]

fig, ax = plt.subplots()
ax.scatter(reduced_2d[:, 0], reduced_2d[:, 1], c=kmeans_labels[best_n_clusters], alpha=0.4, linewidth=0.5)

for i in range(best_n_clusters):
    red_cluster = reduced_2d[kmeans_labels[best_n_clusters] == i]
    if red_cluster.shape[0] < 3:
        continue  # not enough points for convex hull
    hull = ConvexHull(red_cluster)
    for simplex in hull.simplices:
        ax.plot(red_cluster[simplex, 0], red_cluster[simplex, 1], c=cmap(i), linestyle="solid", alpha=0.5)

for i in range(best_n_clusters):
    red_clus = reduced_2d[kmeans_labels[best_n_clusters] == i]
    clus_mean = red_clus.mean(axis=0)
    clus_ymax = red_clus[:, 1].max()
    clus_text = f"Cluster {i}"
    # ax.scatter(clus_mean[0], clus_mean[1], c="green", marker="v")

    options = [[clus_mean[0], red_clus[:, 1].max() + 0.25],
               [clus_mean[0], red_clus[:, 1].min() - 0.25]]
              # [red_clus[:, 0].max() + 0.2, clus_mean[1]],
              # [red_clus[:, 0].min() - 0.2, clus_mean[1]]]
    # text_x, text_y = clus_mean[0], clus_ymax + 0.2
    text_x, text_y = select_position(options, reduced_2d[kmeans_labels[best_n_clusters] != i])
    ax.text(text_x, text_y, s=clus_text, c="black", bbox=dict(facecolor=cmap(i), edgecolor=cmap(i), alpha=0.25, boxstyle="round", pad=0.1), horizontalalignment="center")

if args.show:
    for i, n in enumerate(names):
        n = f"({kmeans_labels[best_n_clusters][i]}) {n}"
        ax.annotate(n, (reduced_2d[i, 0], reduced_2d[i, 1])).set_alpha(0.1)

ax.set_xlabel("principal component 1")
ax.set_ylabel("principal component 2")
if args.show:
    plt.show()
else:
    fig.set_size_inches(plot_width, plot_height)
    fig.tight_layout()
    fig.savefig(args.dir/"2d_map.pdf", format="pdf")
