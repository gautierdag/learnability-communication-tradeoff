#!/bin/bash
#SBATCH --mem=2000
#SBATCH --cpus-per-task=110
#SBATCH --time=2:00:00

set -e # fail fast

dt=$(date '+%d_%m_%y_%H_%M');
echo "I am job ${SLURM_JOB_ID}"
echo "I'm running on ${SLURM_JOB_NODELIST}"
echo "Job started at ${dt}"

# ====================
# Activate Anaconda environment
# ====================
source ~/.bashrc
conda activate gp

python -m learnability_frontier.py

echo "Job ${SLURM_JOB_ID} is done!"