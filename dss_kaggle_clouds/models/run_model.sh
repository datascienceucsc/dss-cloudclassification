#!/bin/bash
#SBATCH --job-name experiment01
#SBATCH --mail-user=apoirel@ucsc.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --partition=96x24gpu4
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=apoirel_%j.out
#SBATCH --error=apoirel_%j.out

module load python/3.6.2
python train_model.py
