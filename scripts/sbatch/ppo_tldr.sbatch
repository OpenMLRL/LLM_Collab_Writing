#!/bin/bash
#SBATCH --account=bfzm-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100g
#SBATCH --time=24:00:00
#SBATCH --job-name=ppo_tldr
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.out

cd $SLURM_SUBMIT_DIR    # ensure we start where sbatch was called

# Create logs directory if it doesn't exist
mkdir -p logs

# load env
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate comlrl

# Set environment variables
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export HF_HOME="/work/hdd/bfzm/$USER/hf_cache"

# run script
python3 -u train_ac.py \
    --config configs/ac_tldr_config.yaml \
    --override wandb.entity="ryanamiri05-northeastern-university" \
               wandb.project="mlrl" \
               wandb.name="ppo_tldr"
