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
#SBATCH --output=/projects/bfzm/%u/src/LLM_Collab_Writing/logs/%x.%j.out
#SBATCH --error=/projects/bfzm/%u/src/LLM_Collab_Writing/logs/%x.%j.out

cd /projects/bfzm/$USER/src    # Run from parent directory to access CoMLRL

# Create logs directory if it doesn't exist
mkdir -p LLM_Collab_Writing/logs

# load env
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate comlrl

# Set environment variables
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH}:$(pwd)/CoMLRL:$(pwd)/LLM_Collab_Writing"
export HF_HOME="/work/hdd/bfzm/$USER/hf_cache"

# run script
cd LLM_Collab_Writing
python3 -u train_ac.py \
    --config configs/ac_tldr_config.yaml \
    --override wandb.entity="ryanamiri05-northeastern-university" \
               wandb.project="mlrl" \
               wandb.name="ppo_tldr"
