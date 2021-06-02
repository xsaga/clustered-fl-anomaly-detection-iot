import glob

import numpy as np

import torch
import torch.nn as nn

from pathlib import Path

from matplotlib import pyplot as plt
import seaborn as sns

import scipy

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(26, 12), # 26,12
            # nn.Dropout(0.05), # !
            nn.ReLU(), # nn.LeakyReLU(), nn.ReLU()
            nn.Linear(12, 4), # 12,4
            nn.Sigmoid() # nn.Tanh(), nn.Sigmoid(), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 12), # 4,12
            nn.ReLU(), # nn.LeakyReLU(), nn.ReLU()
            nn.Linear(12, 26), # 12,26
            # nn.Dropout(0.05), # !
            nn.ReLU() # nn.ReLU()
        )

    def forward(self, x):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return decoded


model_paths = glob.glob("./*.pt")
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

pesos = np.array(pesos)


# cluster
cluster_numbers = list(range(2, min(10, pesos.shape[0]-1)))
score_ss = []
score_db = []
for n_clusters in cluster_numbers:
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++").fit(pesos)
    kmeans_labels = kmeans.labels_
    # davies_bouldin_score :: parece que sale mejor con davies_bouldin
    # silhouette_score
    score_ss.append(silhouette_score(pesos, kmeans_labels))
    score_db.append(davies_bouldin_score(pesos, kmeans_labels))

plt.plot(cluster_numbers, score_ss)
plt.plot(cluster_numbers, score_db)
plt.show()


# pca
pca = PCA(n_components=2)
pca.fit(pesos)

reducido = pca.transform(pesos)

fig, ax = plt.subplots()
ax.scatter(reducido[:, 0], reducido[:, 1])

for i, n in enumerate(names):
    ax.annotate(n, (reducido[i, 0], reducido[i, 1]))

plt.show()

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
