import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load your TSV
df = pd.read_csv("project01/group04/data/rinalmo/all_embeddings_complete_with_gene_id.tsv", sep="\t")

# 2. Identify columns
r5_cols = [c for c in df.columns if c.startswith("r5_")]
r3_cols = [c for c in df.columns if c.startswith("r3_")]

# 3. (Optional) scale
scaler5 = StandardScaler()
r5_scaled = scaler5.fit_transform(df[r5_cols])

scaler3 = StandardScaler()
r3_scaled = scaler3.fit_transform(df[r3_cols])

# 4. Fit PCA
n_components_5 = 256    # <-- choose how many dims you want for 5' UTR
n_components_3 = 256     # <-- choose how many dims you want for 3' UTR

pca5 = PCA(n_components=n_components_5)
r5_pcs = pca5.fit_transform(r5_scaled)

pca3 = PCA(n_components=n_components_3)
r3_pcs = pca3.fit_transform(r3_scaled)

# 5. Build column names and merge back
r5_pc_names = [f"r5_pc{i+1}" for i in range(n_components_5)]
r3_pc_names = [f"r3_pc{i+1}" for i in range(n_components_3)]

df_pca5 = pd.DataFrame(r5_pcs, columns=r5_pc_names, index=df.index)
df_pca3 = pd.DataFrame(r3_pcs, columns=r3_pc_names, index=df.index)

# Combine
df_reduced = pd.concat([ df.drop(columns=r5_cols + r3_cols),
                         df_pca5, df_pca3 ],
                       axis=1)


df_reduced.to_csv("project01/group04/data/rinalmo/embeddings.tsv", sep="\t", index=False)
output_file = f"project01/group04/data/rinalmo/embeddings_r5-{n_components_5}_r3-{n_components_3}.tsv"

# write out
df_reduced.to_csv(output_file, sep="\t", index=False)