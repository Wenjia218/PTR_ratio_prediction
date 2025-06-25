import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize
from sklearn.metrics import r2_score
import statsmodels.formula.api as smf
from tqdm import tqdm
import argparse


def data_prep_melt(feature_path,ptr_path):
    
    features = pd.read_csv(feature_path, sep='\t')

    if feature_path == 'data/Table_EV6.tsv' :
        features = features.iloc[:, 0:208]

    ptr = pd.read_csv(ptr_path, sep='\t')
    ptr = ptr[[col for col in ptr.columns if "mRNA" not in col]]
    ptr = ptr[[col for col in ptr.columns if "protein" not in col]]

    id_vars = ptr.columns[:4]
    value_vars = ptr.columns[4:] 
    ptr_long = ptr.melt(id_vars=id_vars, value_vars=value_vars, 
                var_name='tissue', value_name='PTR')
    ptr_long['tissue'] = ptr_long['tissue'].str.replace('_PTR', '', regex=False)
    ptr_long['PTR'] = pd.to_numeric(ptr_long['PTR'], errors='coerce')
    ptr_long = ptr_long.dropna(subset=['PTR'])

    cols_to_drop = [col for col in ['GeneName', 'EnsemblTranscriptID', 'EnsemblProteinID'] if col in features.columns]
    features = features.drop(columns=cols_to_drop)

    cols_to_drop_ptr = [col for col in ['GeneName', 'EnsemblTranscriptID', 'EnsemblProteinID'] if col in ptr_long.columns]
    ptr_long = ptr_long.drop(columns=cols_to_drop_ptr)
    
    return features, ptr_long



def mixed_model(features, ptr_long):

    merged = ptr_long.merge(features, on='EnsemblGeneID')
    merged['tissue'] = merged['tissue'].astype('category')
    merged.columns = merged.columns.str.replace('.', '_', regex=False)

    zero_cols = features.columns[(features == 0).all()]
    print(f"Dropping features with all zero values: {list(zero_cols)}")
    features = features.drop(columns=zero_cols)
    features = features.drop('EnsemblGeneID', axis=1)
    
    features = features.columns.str.replace('.', '_', regex=False)
    print(features)
    fixed_effects = 'C(tissue) + ' + ' + '.join(features) 

    formula = f'PTR ~ {fixed_effects}'
    print(formula)
    kf = KFold(n_splits=10, shuffle=True)

    predicted_ptr = np.zeros(len(merged))
    coefs_list = []


    for train_idx, test_idx in tqdm(kf.split(merged), total=kf.get_n_splits(), desc="Cross-validation folds"):
        
        merged_train = merged.iloc[train_idx]
        model = smf.mixedlm(formula, data=merged_train, groups=merged_train['EnsemblGeneID'])

        result = model.fit()

        predicted_ptr[test_idx] = result.predict(merged.iloc[test_idx])
        coefs_list.append(result.params)

    
    ptr_long['predicted_PTR'] = predicted_ptr

    return coefs_list, ptr_long


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run mixed model analysis")
    parser.add_argument('feature_path', type=str, help="Path to the feature file")
    parser.add_argument('ptr_path', type=str, help="Path to the PTR file")
    args = parser.parse_args()
    
    feature_filename = os.path.basename(args.feature_path).split('.')[0]
    ptr_filename = os.path.basename(args.ptr_path).split('.')[0]

    # Prepare data
    features, ptr_long = data_prep_melt(args.feature_path, args.ptr_path)

    # Run model
    coefs_list, ptr_pred = mixed_model(features, ptr_long)

    # Save outputs with descriptive filenames
    np.save(f'data/coefs_{feature_filename}.npy', coefs_list)
    ptr_pred.to_csv(f'data/ptr_pred_{feature_filename}.tsv', sep='\t', index=False)




