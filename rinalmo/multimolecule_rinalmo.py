#!/usr/bin/env python3
import os, csv, gc, sys
import pandas as pd
import numpy as np
import torch
import traceback
from multimolecule import RnaTokenizer, RiNALMoModel

# 1) Load & filter your EV4 table
df = pd.read_csv(
    "/s/project/ml4rg_students/2025/project01/data/Table_EV4/Table_EV4.tsv", sep="\t"
)
mask = df["UTR5_Sequence"].notna() & df["UTR3_Sequence"].notna()
df = df[mask].reset_index(drop=True)
df = df[
    (df["UTR5_Sequence"].str.strip().str.len() > 0)
    & (df["UTR3_Sequence"].str.strip().str.len() > 0)
].reset_index(drop=True)

total = len(df)
print(f"→ Loaded {total} transcripts from EV4")

# 2) Device setup: GPU if you have one
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"→ Using device: {device}")

# 3) Load tokenizer & model **once**, move model to device
tokenizer = RnaTokenizer.from_pretrained("multimolecule/rinalmo")
model = RiNALMoModel.from_pretrained("multimolecule/rinalmo").to(device)
model.eval()


# 4) Helper to embed a single sequence
def embed_seq(seq: str) -> np.ndarray:
    # tokenize, move inputs to device
    # 0.005 sequences got cut using the max length of 10000
    inputs = tokenizer(
        seq, return_tensors="pt", padding=True, truncation=True, max_length=10000
    ).to(device)
    with torch.no_grad():
        out = model(**inputs).last_hidden_state  # [1, L, D]
        mask = inputs.attention_mask.unsqueeze(-1)  # [1, L, 1]
        summed = (out * mask).sum(dim=1)  # [1, D]
        lengths = mask.sum(dim=1)  # [1, 1]
        vec = (summed / lengths).cpu().numpy().reshape(-1)  # [D]
    # clean up GPU tensors
    del inputs, out, mask, summed, lengths
    torch.cuda.empty_cache()
    return vec


# 5) Open CSV for streaming
hidden_dim = model.config.hidden_size
out_path = "all_embeddings.csv"
with open(out_path, "w", newline="") as fout:
    writer = csv.writer(fout)
    writer.writerow(
        ["GeneName", "EnsemblTranscriptID"]
        + [f"r5_{i}" for i in range(hidden_dim)]
        + [f"r3_{i}" for i in range(hidden_dim)]
    )

    # 6) Main loop: one transcript at a time
    for idx, row in df.iterrows():
        seq5 = row["UTR5_Sequence"]
        seq3 = row["UTR3_Sequence"]
        print(
            f"\n→ Row {idx+1}/{total}: length5={len(seq5)}, length3={len(seq3)}",
            end="\n",
        )

        try:
            v5 = embed_seq(seq5)
            v3 = embed_seq(seq3)
        except Exception as e:
            # Print the full traceback to see the real error
            print(
                f"⚠️ Error on row {idx+1} (ID={row['EnsemblTranscriptID']}):",
                file=sys.stderr,
            )
            traceback.print_exc()
            continue

        writer.writerow([row["GeneName"], row["EnsemblTranscriptID"], *v5, *v3])
        print(f"✔️   OK: processed {idx+1}/{total}", end="\r")
        del v5, v3
        gc.collect()


print("\n✅ Finished—all embeddings saved to", out_path)


"""
sequence = "UAGCUUAUCAGACUGAUGUUG"
inputs = tokenizer(sequence, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

embeddings = outputs.last_hidden_state  # shape: [1, seq_len, hidden_dim]
print("Shape:", embeddings.shape)

# Convert to numpy array to inspect actual values (optional)
embedding_array = embeddings.squeeze(0).numpy()  # shape: [seq_len, hidden_dim]

# Print values for each position
for i, token_emb in enumerate(embedding_array):
    print(f"Token {i}: {token_emb[:5]}...")
"""
