#!/bin/bash
#SBATCH --job-name=r2t_train
#SBATCH --output=logs_%j.out   # %j adds the job ID to the filename
#SBATCH --error=logs_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=shard:30
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# --- 1. Load System Modules ---
# This makes the 'conda' command available on the compute node
module load anaconda3 || module load miniconda3

# --- 2. Activate Conda ---
# Batch scripts need to manually 'source' the conda profile to use 'conda activate'
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate your_env_name

# --- 3. Run the actual training ---
# Now that the environment is ready, run the python command
# (Copy the python line from your original .sh file here)
python 