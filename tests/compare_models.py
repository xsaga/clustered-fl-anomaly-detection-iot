import glob

import numpy as np

import torch
import torch.nn as nn

from pathlib import Path

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

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
    pesos.append(m.decoder[2].weight.flatten().detach().numpy())
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
# z = torch.ones(26)
# z = torch.ones(26) - 0.5
# z = torch.randn(26)
z = torch.zeros(26)

modelos_eval = np.array(list(map(lambda m: m(z).detach().numpy(), modelos)))
corr = np.corrcoef(modelos_eval)

# cmap = sns.diverging_palette(240,20,as_cmap=True)
# cmap = "vlag"
cmap = sns.color_palette("Spectral", as_cmap=True)
cmap = cmap.reversed()

sns.heatmap(corr, cmap=cmap, xticklabels=names, yticklabels=names, square=True, annot=True)
plt.show()

sns.clustermap(corr, cmap=cmap, xticklabels=names, yticklabels=names)
plt.show()

# select number of clusters
num_clusters = list(range(2, corr.shape[0]))
silhouette_scores = []
for n in num_clusters:
    clustering = AgglomerativeClustering(n_clusters=n).fit(corr)
    silhouette_scores.append(metrics.silhouette_score(corr, clustering.labels_))

plt.plot(num_clusters, silhouette_scores)
plt.show()
