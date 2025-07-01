import pandas as pd

# 1. Load the master table (Table_EV4.tsv)
master = pd.read_csv('C:/Users/aquar/PycharmProjects/PTR_project/PTR_ratio_prediction/data/paper/Table_EV4.tsv', sep='\t', usecols=['GeneName','EnsemblTranscriptID','EnsemblGeneID'])

# 2. Load your target CSV which has only GeneName & EnsemblTranscriptID
#    (replace 'your.csv' with your actual filename)
target = pd.read_csv('C:/Users/aquar/PycharmProjects/PTR_project/PTR_ratio_prediction/data/rinalmo/all_embeddings_complete.csv')

# 3. Merge to pull in the EnsemblGeneID
merged = target.merge(
    master,
    on=['EnsemblTranscriptID'],
    how='left'
)

# 4. (Optional) check for any missing IDs
missing = merged['EnsemblGeneID'].isna().sum()
print(f"{missing} rows did not find a matching EnsemblGeneID")

merged.rename(columns={'GeneName_x': 'GeneName'}, inplace=True)
if 'GeneName_y' in merged.columns:
    merged.drop(columns=['GeneName_y'], inplace=True)

# 2. Reorder columns so that EnsemblGeneID is the 3rd column
cols = merged.columns.tolist()
# ensure the key columns are in the order you want:
desired = ['GeneName', 'EnsemblTranscriptID', 'EnsemblGeneID']
# pull out any other columns after those three
others = [c for c in cols if c not in desired]
merged = merged[desired + others]


# 5. Save out the augmented CSV
merged.to_csv('C:/Users/aquar/PycharmProjects/PTR_project/PTR_ratio_prediction/data/rinalmo/all_embeddings_complete_with_gene_id.tsv',  sep='\t', index=False)

