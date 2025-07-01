#!/bin/bash
#SBATCH --job-name=PTR_prediction
#SBATCH --partition=student_project
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# -------- INPUT FILES --------
EMBED_FILE="data/rinalmo/all_embeddings_with_gene_id.tsv"
TABLE_FILE="data/paper/Table_EV3.tsv"

# -------- LOG FILE NAME CONSTRUCTION --------
EMBED_BASENAME=$(basename "$EMBED_FILE" .tsv)
TABLE_BASENAME=$(basename "$TABLE_FILE" .tsv)

LOG_DIR="sh_logs"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/PTR_${EMBED_BASENAME}__${TABLE_BASENAME}_%j.out"
ERR_FILE="${LOG_DIR}/PTR_${EMBED_BASENAME}__${TABLE_BASENAME}_%j.err"

#SBATCH --output=/dev/null  # Disable default SLURM log output

# -------- REDIRECT OUTPUT MANUALLY --------
exec > >(tee "${LOG_FILE//%j/${SLURM_JOB_ID}}") 2> >(tee "${ERR_FILE//%j/${SLURM_JOB_ID}}" >&2)

# -------- RUN PYTHON SCRIPT --------
python PTR_prediction.py "$EMBED_FILE" "$TABLE_FILE"
