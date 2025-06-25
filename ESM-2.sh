#!/bin/bash
# SBATCH --job-name=ESM-2
# SBATCH --partition=student_project
# SBATCH --cpus-per-task=16
# SBATCH --mem=64G
# SBATCH --gres=gpu:1
# SBATCH --time=08:00:00  # Set appropriate time limit
# SBATCH --output=sh_logs/esm2_%j.out  # Output file (%j = job ID)

# Run your Python script
python ESM-2.py
