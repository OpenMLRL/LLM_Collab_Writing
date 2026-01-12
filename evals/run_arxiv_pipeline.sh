#!/bin/bash
# Run arXiv Pipeline Evaluation
#
# Pipeline: Agent 1 generates first, Agent 2 receives Agent 1's output in prompt
# Key difference from Parallel: Agent 2 sees Agent 1's background section and builds upon it
#
# Usage:
#   cd ~/projects/LLM_Collab_Writing
#   bash evals/run_arxiv_pipeline.sh
#
# To run in background with tmux:
#   tmux new -s arxiv_pipeline
#   bash evals/run_arxiv_pipeline.sh
#   # Press Ctrl+b then d to detach

# Print job information
echo "=============================================="
echo "arXiv Pipeline Evaluation"
echo "Started at: $(date)"
echo "Working directory: $(pwd)"
echo "=============================================="

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate comlrl
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'comlrl'."
    echo "Please ensure the environment exists: conda create -n comlrl python=3.10"
    exit 1
fi
echo "✓ Conda environment activated: $CONDA_DEFAULT_ENV"

# Configuration
CONFIG_FILE="evals/configs/arxiv_pipeline_config.yaml"

echo ""
echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Mode: Pipeline (Agent 2 sees Agent 1's output)"
echo ""

# Create output directories
mkdir -p "evals/results"
mkdir -p "evals/logs"

# Run evaluation
python evals/eval_arxiv_pipeline.py \
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
