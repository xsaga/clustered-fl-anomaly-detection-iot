import glob

import numpy as np

import torch
import torch.nn as nn

from pathlib import Path

from matplotlib import pyplot as plt
import seaborn as sns

import scipy

from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

class LSTMchar(nn.Module):
    def __init__(self, start_dim):
        super(LSTMchar, self).__init__()
        self.start_dim = start_dim

        self.lstm = nn.LSTM(self.start_dim, 128)
        self.linear = nn.Linear(128, self.start_dim)
        self.activ = nn.Softmax(dim=2)  # nn.ReLU()

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        out = self.activ(out)
        out = self.linear(out)
        return out


model_paths = glob.glob("./*.pt")
print(f"found {len(model_paths)} models")

modelos = []
pesos = []
names = []
for model_path in model_paths:
    m = LSTMchar(26)
    m.load_state_dict(torch.load(model_path))
    modelos.append(m)
    pesos.append(m.linear.weight.flatten().detach().numpy())
    names.append(Path(model_path).stem)

pesos = np.array(pesos)

pca = PCA(n_components=2)
pca.fit(pesos)

reducido = pca.transform(pesos)

fig, ax = plt.subplots()
ax.scatter(reducido[:, 0], reducido[:, 1])

for i, n in enumerate(names):
    ax.annotate(n, (reducido[i, 0], reducido[i, 1]))

plt.show()

# probar multiples z
seq = 10
# z = torch.ones(seq, 1, 26)
# z = torch.ones(seq, 1, 26) - 0.5
# z = torch.randn(seq, 1, 26)
z = torch.zeros(seq, 1, 26)

modelos_eval = np.array(list(map(lambda m: m(z)[-1,:,:].squeeze().detach().numpy(), modelos)))

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
    z = torch.randn(seq, 1, 26)
    modelos_eval = np.array(list(map(lambda m: m(z)[-1,:,:].squeeze().detach().numpy(), modelos)))
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
    silhouette_scores.append(metrics.silhouette_score(corr, clustering.labels_))

plt.plot(num_clusters, silhouette_scores)
plt.show()

# Spearman
# spearman_corr = np.zeros(corr.shape)
# for i in range(modelos_eval.shape[0]):
#     for j in range(i, modelos_eval.shape[0]):
#         spearman_corr[i, j] = scipy.stats.spearmanr(modelos_eval[i, :], modelos_eval[j, :])[0]
#         spearman_corr[j, i] = spearman_corr[i, j]
z = torch.zeros(seq, 1, 26)
modelos_eval = np.array(list(map(lambda m: m(z)[-1,:,:].squeeze().detach().numpy(), modelos)))
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
    silhouette_scores.append(metrics.silhouette_score(spearman_corr, clustering.labels_))

plt.plot(num_clusters, silhouette_scores)
plt.show()
