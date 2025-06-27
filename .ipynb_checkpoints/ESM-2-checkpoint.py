import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import KFold
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import argparse
torch.cuda.empty_cache()




def run_esm2(input, model, device, batch_size, seqs):
    logits = []
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(input["input_ids"]), batch_size)):
            batch_input = {k: v[i:i+batch_size].to(device) for k, v in input.items()}
            output = model(**batch_input)
            logits.append(output.logits.cpu())

    logits = torch.cat(logits, dim=0)
    logits_df = pd.DataFrame(logits)
    logits_df = logits_df.add_prefix("logit_")
    logits_df['EnsemblGeneID'] = seqs['EnsemblGeneID']
    return logits_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ESM-2 model on protein sequences.")
    parser.add_argument('model_path', type=str)
    parser.add_argument('num_labels', type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    seqs = pd.read_csv('data/Table_EV4.tsv', sep='\t')
    seqs = seqs[['EnsemblGeneID','ProteinSequence']]
    if args.test:
        seqs = seqs.iloc[:10]
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    #with max_length=5000 only 0.14% of seqs get cut
    inputs = tokenizer(list(seqs['ProteinSequence']), return_tensors="pt", padding=True, truncation=True, max_length=5000)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=args.num_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    logits_df = run_esm2(inputs, model, device, args.batch_size, seqs)
    
    model_name = os.path.basename(os.path.normpath(args.model_path))
    output_file = f"data/{model_name}_{args.num_labels}_embed.tsv"

    logits_df.to_csv(output_file, sep='\t', index=False)


