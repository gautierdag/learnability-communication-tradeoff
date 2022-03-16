#!/bin/bash
#
#SBATCH --job-name=gp-grid
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-23:00:00
#SBATCH --mem-per-cpu=2G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gp

python som.py --average_k 16 --workers 16
