#!/bin/bash
# ==============================================================================
# Parallel Evaluation Script for arXiv
# ==============================================================================
# This script runs the parallel (naive concatenation) evaluation where both
# agents generate introduction sections simultaneously with NO communication.
# Agent 1: Background and motivation
# Agent 2: Methodology and implications
# ==============================================================================

set -e  # Exit on error

# Print script information
echo "=============================================="
echo "arXiv Parallel Evaluation"
echo "Started at: $(date)"
echo "Working directory: $(pwd)"
echo "=============================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please activate your conda environment first."
    exit 1
fi

# Activate conda environment
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate comlrl
    echo "✓ Conda environment activated: comlrl"
else
    echo "Warning: Could not find conda.sh. Make sure conda is in your PATH."
    echo "Attempting to activate comlrl environment..."
    conda activate comlrl || {
        echo "Error: Failed to activate comlrl environment"
        exit 1
    }
fi

# Configuration
CONFIG_FILE="${CONFIG_FILE:-evals/configs/arxiv_parallel_config.yaml}"

echo ""
echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo ""

# Create output directories
mkdir -p "evals/results"
mkdir -p "evals/logs"

# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."  # Go to LLM_Collab_Writing root

# Run evaluation
echo "Starting arXiv parallel evaluation..."
python evals/eval_arxiv_parallel.py \
    --config "$CONFIG_FILE" \
    --verbose

EXIT_STATUS=$?

echo ""
echo "=============================================="
echo "arXiv evaluation completed at: $(date)"
echo "Exit status: $EXIT_STATUS"
echo "Results saved to: evals/results"
echo "=============================================="

exit $EXIT_STATUS
