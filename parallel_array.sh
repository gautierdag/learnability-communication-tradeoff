#!/bin/bash
#
#SBATCH --job-name=grid-search
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --mem-per-cpu=2G
#
#SBATCH --array=0-63
#SBATCH --requeue

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gp

# myscript.py will be called multiple times (in parallel)
# in SBATCH --array=0-9 we specify a range (10 times in total)
# and myscript.py will be called with the respective input.
# You can use that value to index a python array with your hyperparameters.
python grid_search_som.py $SLURM_ARRAY_TASK_ID
