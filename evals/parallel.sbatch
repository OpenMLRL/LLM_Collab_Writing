#!/bin/bash
#SBATCH --job-name=eval_parallel_tldr
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH --cpus-per-task=8
#SBATCH --output=evals/logs/parallel.%j.out

# ==============================================================================
# Parallel Evaluation Script for TLDR
# ==============================================================================
# This script runs the parallel (naive concatenation) evaluation where both
# agents generate summaries simultaneously with NO communication.
# Agent 1: Concise summary
# Agent 2: Detailed elaboration (2-3x longer)
# ==============================================================================

# Print job information
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "=============================================="

# Load required modules (adjust for your cluster)
module load anaconda 2>/dev/null || true
module load cuda/12.1 2>/dev/null || true

# Activate conda environment
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate comlrl 2>/dev/null || conda activate base
fi

# Configuration
CONFIG_FILE="${CONFIG_FILE:-evals/configs/parallel_config.yaml}"

echo ""
echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo ""

# Create output directories
mkdir -p "evals/results"
mkdir -p "evals/logs"

# Run evaluation
python evals/eval_parallel.py \
    --config "$CONFIG_FILE" \
    --verbose

# Capture exit status
EXIT_STATUS=$?

echo ""
echo "=============================================="
echo "Job completed at: $(date)"
echo "Exit status: $EXIT_STATUS"
echo "Results saved to: evals/results"
echo "=============================================="

exit $EXIT_STATUS
