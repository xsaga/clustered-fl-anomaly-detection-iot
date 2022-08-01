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
parser.add_argument("--show", action="store_true")
args = parser.parse_args()  # parse_args(["--dir", "clustering_results", "--dimensions", "27"])

if not args.show:
    rcParams["font.family"] = ["Times New Roman"]
    rcParams["font.size"] = 4
    rcParams["xtick.labelsize"] = 6
    rcParams["ytick.labelsize"] = 6
    rcParams["axes.labelsize"] = 6
    rcParams["legend.fontsize"] = 4
    rcParams["lines.linewidth"] = 0.75
    rcParams["lines.markersize"] = 2
    plot_width = 1.714  # in
    plot_height = 1.285


np.random.seed(1)
data_dimensions = args.dimensions

model_paths = list(args.dir.glob("*.pt"))
print(f"Found {len(model_paths)} models in {args.dir}")

modelos = []
pesos = []
names = []
for model_path in model_paths:
    m = Autoencoder(data_dimensions)
    m.load_state_dict(torch.load(model_path))
    modelos.append(m)
    all_weights = np.concatenate([v.detach().numpy().flatten() for v in m.state_dict().values()])
    pesos.append(all_weights)
    # pesos.append(m.decoder[2].weight.flatten().detach().numpy())
    names.append(Path(model_path).stem)

names = [n.split("_")[0].replace("iotsim-", "") for n in names]
tags = np.array([n.rsplit("-", 1)[0] for n in names])
pesos = np.array(pesos)

# cluster
pca_cluster = PCA(n_components=0.9) # explain 90% var
pesos_reducido = pca_cluster.fit_transform(pesos)
print(pesos_reducido.shape)

cluster_numbers = list(range(2, min(41, pesos.shape[0]-1)))
kmeans_labels = dict()
score_ss = []
score_ch = []
score_db = []
score_sdbw = []
for n_clusters in cluster_numbers:
    print(n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=50).fit(pesos_reducido)
    kmeans_labels[n_clusters] = kmeans.labels_

    score_ss.append(silhouette_score(pesos_reducido, kmeans.labels_))
    score_ch.append(calinski_harabasz_score(pesos_reducido, kmeans.labels_))
    score_db.append(davies_bouldin_score(pesos_reducido, kmeans.labels_))
    score_sdbw.append(S_Dbw(pesos_reducido, kmeans.labels_, centers_id=None, method="Tong", centr="mean", metric="euclidean"))

fig, ax1 = plt.subplots()
ax1.set_xlabel("number of clusters")
ax1.set_ylabel("scores")
ax1.plot(cluster_numbers, score_ss, label="silhouette", marker="o", color="#1f77b4")  # (MAX)
ax1.plot(cluster_numbers, score_db, label="davies-bouldin", marker="D", color="#ff7f0e")  # (MIN)
ax1.plot(cluster_numbers, score_sdbw, label="S_Dbw", marker="P", color="#2ca02c")  # (MIN)
ax2 = ax1.twinx()
ax2.set_ylabel("score calinski-harabasz")
ax2.plot(cluster_numbers, score_ch, label="calinski-harabasz", marker="^", color="#d62728")  # (MAX)
ax1.xaxis.set_major_locator(MultipleLocator(2))
ax1.xaxis.set_minor_locator(MultipleLocator(1))
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), bbox_transform=ax1.transAxes)
fig.tight_layout()
ax1.set_title(f"Cluster analysis for {args.dir.name}")
if args.show:
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
ax.plot(cluster_numbers, ard_score, marker="o", label="ard")
ax.plot(cluster_numbers, ami_score, marker="o", label="ami")
ax.plot(cluster_numbers, vme_score, marker="o", label="vme")
ax.legend()
if args.show:
    plt.show()
else:
    fig.set_size_inches(plot_width, plot_height)
    fig.tight_layout()
    fig.savefig(args.dir/"score_groundtruth.pdf", format="pdf")

best_n_clusters = 8
print("Selected ", best_n_clusters, " clusters")
print("Cluster composition:")
for cluster_idx, items in count_cluster_items(kmeans_labels[best_n_clusters], tags).items():
    print(f"Cluster {cluster_idx}: {items}")

# 2D Viz
pca = PCA(n_components=2)
pca.fit(pesos)
reducido = pca.transform(pesos)

colm = cm.get_cmap("viridis", best_n_clusters)

def select_position(positions, rest):
    # poner label en el hueco mas alejado
    distances = scipy.spatial.distance.cdist(positions, rest)
    closest = distances.min(axis=1)
    selection = np.argmax(closest)
    return positions[selection]

fig, ax = plt.subplots()
ax.scatter(reducido[:, 0], reducido[:, 1], c=kmeans_labels[best_n_clusters], alpha=0.4, linewidth=0.5)

for i in range(best_n_clusters):
    red = reducido[kmeans_labels[best_n_clusters] == i]
    if red.shape[0] < 3:
        continue  # not enough points for convex hull
    hull = ConvexHull(red)
    for simplex in hull.simplices:
        ax.plot(red[simplex, 0], red[simplex, 1], c=colm(i), linestyle="solid", alpha=0.5)

for i in range(best_n_clusters):
    reducido_clus = reducido[kmeans_labels[best_n_clusters] == i]
    clus_mean = reducido_clus.mean(axis=0)
    clus_ymax = reducido_clus[:, 1].max()
    clus_text = f"Cluster {i}"
    # ax.scatter(clus_mean[0], clus_mean[1], c="green", marker="v")

    options = [[clus_mean[0], reducido_clus[:, 1].max() + 0.25],
               [clus_mean[0], reducido_clus[:, 1].min() - 0.25]]
              # [reducido_clus[:, 0].max() + 0.2, clus_mean[1]],
              # [reducido_clus[:, 0].min() - 0.2, clus_mean[1]]]
    # text_x, text_y = clus_mean[0], clus_ymax + 0.2
    text_x, text_y = select_position(options, reducido[kmeans_labels[best_n_clusters] != i])
    ax.text(text_x, text_y, s=clus_text, c="black", bbox=dict(facecolor=colm(i), edgecolor=colm(i), alpha=0.25, boxstyle="round", pad=0.1), horizontalalignment="center")

for i, n in enumerate(names):
    n = f"({kmeans_labels[best_n_clusters][i]}) {n}"
    ax.annotate(n, (reducido[i, 0], reducido[i, 1])).set_alpha(0.1)

ax.set_xlabel("principal component 1")
ax.set_ylabel("principal component 2")
if args.show:
    plt.show()
else:
    fig.set_size_inches(plot_width, plot_height)
    fig.tight_layout()
    fig.savefig(args.dir/"2d_map.pdf", format="pdf")