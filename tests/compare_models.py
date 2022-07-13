import glob
import os
import numpy as np

import torch
import torch.nn as nn

from pathlib import Path

from matplotlib import rcParams
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.lines as mlines
import seaborn as sns

import scipy

from s_dbw import S_Dbw

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import rand_score, adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, homogeneity_completeness_v_measure

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

# no no no
def select_best_cluster_idx(score: np.ndarray, significant=None, best=np.max, arg_best=np.argmax, worst=np.min, arg_worst=np.argmin) -> int:
    score = np.copy(score)
    if not significant:
        significant = 0.1*(best(score) - worst(score))
    
    gold = best(score)
    arg_gold = arg_best(score)

    score[arg_gold] = worst(score)

    silver = best(score)
    arg_silver = arg_best(score)

    print("gold ", gold, " at ", arg_gold)
    print("silver ", silver, " at ", arg_silver)
    if np.isclose(np.var(score), 0):
        print("0 var")
        return arg_gold 
    if gold - silver <= significant:
        print("not significant")
        return select_best_cluster_idx(score, significant, best, arg_best, worst, arg_worst)
    else:
        print("significant")
        return arg_gold


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(34, 17), # 26,12
            # nn.Dropout(0.05), # !
            nn.ReLU(), # nn.LeakyReLU(), nn.ReLU()
            nn.Linear(17, 8), # 12,4
            nn.ReLU()  # nn.Sigmoid() # nn.Tanh(), nn.Sigmoid(), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 17), # 4,12
            nn.ReLU(), # nn.LeakyReLU(), nn.ReLU()
            nn.Linear(17, 34), # 12,26
            # nn.Dropout(0.05), # !
            nn.ReLU() # nn.ReLU()
        )

    def forward(self, x):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return decoded

# load
EPSILON=2
paths_glob_pattern = f"./{EPSILON}epoch/*.pt"
model_paths = glob.glob(paths_glob_pattern)
print(f"found {len(model_paths)} models")

modelos = []
pesos = []
names = []
for model_path in model_paths:
    m = Autoencoder()
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

cluster_numbers = list(range(2, min(21, pesos.shape[0]-1)))
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
ax1.axvline(x=3, color="k")
ax1.plot(cluster_numbers, score_ss, label="silhouette", marker="o", color="#1f77b4")  # (MAX)
ax1.plot(cluster_numbers, score_db, label="davies-bouldin", marker="D", color="#ff7f0e")  # (MIN)
ax1.plot(cluster_numbers, score_sdbw, label="S_Dbw", marker="P", color="#2ca02c")  # (MIN)
ax2 = ax1.twinx()
ax2.set_ylabel("score calinski-harabasz")
ax2.plot(cluster_numbers, score_ch, label="calinski-harabasz", marker="^", color="#d62728")  # (MAX)
ax1.xaxis.set_major_locator(MultipleLocator(2))
ax1.xaxis.set_minor_locator(MultipleLocator(1))
fig.legend(loc="center right", bbox_to_anchor=(1, 0.5), bbox_transform=ax1.transAxes)
fig.set_size_inches(plot_width, plot_height)
fig.tight_layout()
# plt.title(f"Cluster analysis for {paths_glob_pattern}")
fig.savefig(os.path.expanduser(f"~/{EPSILON}epochs_scores_markers.pdf"), format="pdf")
# plt.show()

best_n_clusters = 3

# pca
pca = PCA(n_components=2)
pca.fit(pesos)

reducido = pca.transform(pesos)

fig, ax = plt.subplots()
# ax.scatter(reducido[:, 0], reducido[:, 1], c=kmeans_labels[best_n_clusters], alpha=0.3)
# for i, n in enumerate(names):
#     ax.annotate(n, (reducido[i, 0], reducido[i, 1])).set_alpha(0.3)
### dynamic colormap based on number of clusters
# colormin = np.min(kmeans_labels[best_n_clusters])
# colormax = np.max(kmeans_labels[best_n_clusters])
# ax.scatter(reducido[tags=="coap-t1", 0], reducido[tags=="coap-t1", 1], marker="o", s=42, cmap="tab10", c=kmeans_labels[best_n_clusters][tags=="coap-t1"], vmin=colormin, vmax=colormax, alpha=0.5, edgecolors="face", label="coap-t1")
# ax.scatter(reducido[tags=="mqtt-t1", 0], reducido[tags=="mqtt-t1", 1], marker="s", s=42, cmap="tab10", c=kmeans_labels[best_n_clusters][tags=="mqtt-t1"], vmin=colormin, vmax=colormax, alpha=0.5, edgecolors="face", label="mqtt-t1")
# ax.scatter(reducido[tags=="mqtt-t2", 0], reducido[tags=="mqtt-t2", 1], marker="^", s=42, cmap="tab10", c=kmeans_labels[best_n_clusters][tags=="mqtt-t2"], vmin=colormin, vmax=colormax, alpha=0.5, edgecolors="face", label="mqtt-t2")

### fixed colors, final plot
ax.scatter(reducido[tags=="coap-t1", 0], reducido[tags=="coap-t1", 1], marker="o", color="#1f77b4", alpha=0.4, linewidth=0.2, label="coap-t1")  # s=42
ax.scatter(reducido[tags=="mqtt-t1", 0], reducido[tags=="mqtt-t1", 1], marker="s", color="#ff7f0e", alpha=0.4, linewidth=0.2, label="mqtt-t1")
ax.scatter(reducido[tags=="mqtt-t2", 0], reducido[tags=="mqtt-t2", 1], marker="^", color="#2ca02c", alpha=0.4, linewidth=0.2, label="mqtt-t2")


# for i, n in enumerate(names):
    # ax.annotate(n, (reducido[i, 0], reducido[i, 1])).set_alpha(0.3)

# plt.title(f"Clusters for {paths_glob_pattern}")

coap1_line = mlines.Line2D([], [], color="#1f77b4", marker="o", ls="", label="coap-t1")  # markersize=10
mqtt1_line = mlines.Line2D([], [], color="#ff7f0e", marker="s", ls="", label="mqtt-t1")
mqtt2_line = mlines.Line2D([], [], color="#2ca02c", marker="^", ls="", label="mqtt-t2")

ax.set_xlabel("principal component 1")
ax.set_ylabel("principal component 2")

plt.legend(handles=[coap1_line, mqtt1_line, mqtt2_line])
fig.set_size_inches(plot_width, plot_height)
fig.tight_layout()
fig.savefig(os.path.expanduser(f"~/{EPSILON}epochs_pca.pdf"), format="pdf")

######

for i in range(best_n_clusters):
    devs = tags[kmeans_labels[best_n_clusters]==i]
    print(f"C{i}: ", end="")
    unq, cnt =np.unique(devs, return_counts=True)
    for u,c in zip(unq, cnt):
        print(u, c, end="; ")
    # print("\n")

# probar multiples z
# z = torch.ones(26)
# z = torch.ones(26) - 0.5
# z = torch.randn(26)
z = torch.zeros(26)

modelos_eval = np.array(list(map(lambda m: m(z).detach().numpy(), modelos)))

# Pearson product-moment correlation
corr = np.corrcoef(modelos_eval)

# cmap = sns.diverging_palette(240,20,as_cmap=True)
# cmap = "vlag"
cmap = sns.color_palette("Spectral", as_cmap=True)
cmap = cmap.reversed()

sns.heatmap(corr, cmap=cmap, xticklabels=names, yticklabels=names, square=True, annot=True)
plt.show()

sns.clustermap(corr, cmap=cmap, xticklabels=names, yticklabels=names)
plt.show()

# avg
corr_l = []
for i in range(100):
    z = torch.randn(26)
    modelos_eval = np.array(list(map(lambda m: m(z).detach().numpy(), modelos)))
    corr_l.append(np.corrcoef(modelos_eval))
corr_l = np.array(corr_l)
corr = corr_l.mean(axis=0)
sns.heatmap(corr, cmap=cmap, xticklabels=names, yticklabels=names, square=True, annot=True)
plt.show()

sns.clustermap(corr, cmap=cmap, xticklabels=names, yticklabels=names)
plt.show()
# ####

# select number of clusters
num_clusters = list(range(2, corr.shape[0]))
silhouette_scores = []
for n in num_clusters:
    clustering = AgglomerativeClustering(n_clusters=n).fit(corr)
    silhouette_scores.append(silhouette_score(corr, clustering.labels_))

plt.plot(num_clusters, silhouette_scores)
plt.show()

# Spearman
# spearman_corr = np.zeros(corr.shape)
# for i in range(modelos_eval.shape[0]):
#     for j in range(i, modelos_eval.shape[0]):
#         spearman_corr[i, j] = scipy.stats.spearmanr(modelos_eval[i, :], modelos_eval[j, :])[0]
#         spearman_corr[j, i] = spearman_corr[i, j]

spearman_corr, pvalues = scipy.stats.spearmanr(modelos_eval, axis=1)

sns.heatmap(spearman_corr, cmap=cmap, xticklabels=names, yticklabels=names, square=True, annot=True)
plt.show()

sns.clustermap(spearman_corr, cmap=cmap, xticklabels=names, yticklabels=names)
plt.show()

# select number of clusters
num_clusters = list(range(2, spearman_corr.shape[0]))
silhouette_scores = []
for n in num_clusters:
    clustering = AgglomerativeClustering(n_clusters=n).fit(spearman_corr)
    silhouette_scores.append(silhouette_score(spearman_corr, clustering.labels_))

plt.plot(num_clusters, silhouette_scores)
plt.show()

#  https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py

###

In [31]: cluster_numbers = list(range(2, min(21, pesos.shape[0]-1)))
    ...: agg_labels = dict()
    ...: score_ss = []
    ...: score_ch = []
    ...: score_db = []
    ...: score_sdbw = []
    ...: for n_clusters in cluster_numbers:
    ...:     print(n_clusters)
    ...:     agg = AgglomerativeClustering(n_clusters=n_clusters).fit(pesos_reducido)
    ...:     agg_labels[n_clusters] = agg.labels_
    ...: 
    ...:     score_ss.append(silhouette_score(pesos_reducido, agg.labels_))
    ...:     score_ch.append(calinski_harabasz_score(pesos_reducido, agg.labels_))
    ...:     score_db.append(davies_bouldin_score(pesos_reducido, agg.labels_))
    ...:     score_sdbw.append(S_Dbw(pesos_reducido, agg.labels_, centers_id=None, method="Tong", centr="mean", metric="euclidean"))
    ...: 


In [32]: fig, ax1 = plt.subplots()
    ...: ax1.set_xlabel("number of clusters")
    ...: ax1.set_ylabel("scores")
    ...: ax1.plot(cluster_numbers, score_ss, label="silhouette", marker="o", color="#1f77b4")  # (MAX)
    ...: ax1.plot(cluster_numbers, score_db, label="davies-bouldin", marker="D", color="#ff7f0e")  # (MIN)
    ...: ax1.plot(cluster_numbers, score_sdbw, label="S_Dbw", marker="P", color="#2ca02c")  # (MIN)
    ...: ax2 = ax1.twinx()
    ...: ax2.set_ylabel("score calinski-harabasz")
    ...: ax2.plot(cluster_numbers, score_ch, label="calinski-harabasz", marker="^", color="#d62728")  # (MAX)
    ...: ax1.xaxis.set_major_locator(MultipleLocator(2))
    ...: ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ...: fig.legend(loc="center right", bbox_to_anchor=(1, 0.5), bbox_transform=ax1.transAxes)
    ...: fig.show()


# supervised score

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

plt.plot(cluster_numbers, ard_score, label="ard")
plt.plot(cluster_numbers, ami_score, label="ami")
plt.plot(cluster_numbers, vme_score, label="vme")
plt.legend()
plt.show()

homogeneity_completeness_v_measure(labels_true, kmeans_labels[12])