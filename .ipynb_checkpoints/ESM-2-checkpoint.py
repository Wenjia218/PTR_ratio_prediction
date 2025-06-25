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

torch.cuda.empty_cache()

seqs = pd.read_csv('data/Table_EV4.tsv', sep='\t')
seqs = seqs[['EnsemblGeneID','ProteinSequence']]

tokenizer = AutoTokenizer.from_pretrained("../esm2_t12_35M_UR50D")
input=tokenizer(list(seqs['ProteinSequence']),return_tensors="pt",padding=True,truncation=True,max_length=5000)

model = AutoModelForSequenceClassification.from_pretrained("../esm2_t12_35M_UR50D",num_labels=100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval();


def run_esm2(input, batch_size):
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
   
logits_df = run_esm2(input,4)

logits_df.to_tsv("data/esm2_t12_35M_UR50D_100_embed.tsv", index=False)