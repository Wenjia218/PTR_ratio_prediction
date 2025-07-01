import pandas as pd

df = pd.read_csv("data/rinalmo/all_embeddings_with_gene_id.tsv", sep="\t")

r3_cols = [c for c in df.columns if c.startswith("r3")]
r5_cols = [c for c in df.columns if c.startswith("r5")]
emb_cols = r3_cols + r5_cols
X = df[emb_cols]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.decomposition import PCA

# First, fit with all components to see explained variance:
pca_full = PCA()
pca_full.fit(X_scaled)

cumvar = pca_full.explained_variance_ratio_.cumsum()
# You can inspect cumvar to pick a cutoff (e.g. the index where cumvar >= .90)

pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(cumvar)
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.grid(True)
plt.show()

