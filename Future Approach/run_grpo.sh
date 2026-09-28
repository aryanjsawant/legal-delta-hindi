#!/bin/bash
#SBATCH --job-name=yea_train
#SBATCH --output=logs_%j.out
#SBATCH --error=logs_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=shard:40          
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00

# ==============================
# Move to project directory
# ==============================
cd /home/tanmoyhazra/RLnew

echo "=========================================="
echo "Starting Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working Dir: $(pwd)"
echo "=========================================="

# ==============================
# Load modules
# ==============================
module load anaconda3-2024.2
module load cuda-12.8

# ==============================
# Initialize conda - use base environment first
# ==============================
eval "$(/apps/compilers/anaconda3-24.2/bin/conda shell.bash hook)"

# ==============================
# Activate environment
# ==============================
conda activate grpo

echo "=========================================="
echo "Environment Info"
echo "Python: $(which python)"
python --version
echo "Conda Env: $CONDA_DEFAULT_ENV"
python -c "import torch; print(f'torch: {torch.__version__}'); print(f'torch.int1 available: {hasattr(torch, \"int1\")}')"
echo "=========================================="

# ==============================
# GPU Debug Info
# ==============================
echo "GPU Info:"
nvidia-smi

: "${HF_TOKEN:?HF_TOKEN must be set in the environment}"
export HF_HUB_ENABLE_HF_TRANSFER=1

# Run the script
python src/train_il_tur.py
