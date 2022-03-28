#!/bin/bash
#SBATCH --job-name=GP
#SBATCH --mem=2G
#SBATCH --cpus-per-task=55
#SBATCH --time=3-12:00:00

set -e # fail fast

dt=$(date '+%d_%m_%y_%H_%M');
echo "I am job ${SLURM_JOB_ID}"
echo "I'm running on ${SLURM_JOB_NODELIST}"
echo "Job started at ${dt}"

# ====================
# Activate Anaconda environment
# ====================
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gp

python -u learnability_frontier.py

echo "Job ${SLURM_JOB_ID} is done!"
