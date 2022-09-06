#!/bin/bash
#
#SBATCH --job-name=proc-ce
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --mem-per-cpu=2G
#
#SBATCH --array=0-31
#SBATCH --requeue

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gp

# myscript.py will be called multiple times (in parallel)
# in SBATCH --array=0-9 we specify a range (10 times in total)
# and myscript.py will be called with the respective input.
# You can use that value to index a python array with your hyperparameters.

# python grid_search_som.py $SLURM_ARRAY_TASK_ID
# python -u communication.py --seed 42 --average_k 50 --optimal --i $SLURM_ARRAY_TASK_ID
python -u process_results.py $SLURM_ARRAY_TASK_ID
