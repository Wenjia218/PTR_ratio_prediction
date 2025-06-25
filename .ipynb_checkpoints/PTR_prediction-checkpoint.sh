#!/bin/bash
#SBATCH --job-name=PTR_prediciton
#SBATCH --partition=student_project
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00  # Set appropriate time limit
#SBATCH --output=sh_logs/PTR_prediciton_%j.out  # Output file (%j = job ID)

# Run your Python script
python PTR_prediciton.py data/Table_EV6.tsv data/Table_EV3.tsv