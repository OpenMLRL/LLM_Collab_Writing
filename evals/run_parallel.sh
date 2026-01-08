#!/bin/bash
# ==============================================================================
# Parallel Evaluation Script for TLDR
# ==============================================================================
# This script runs the parallel (naive concatenation) evaluation where both
# agents generate summaries simultaneously with NO communication.
# Agent 1: Concise summary
# Agent 2: Detailed elaboration (2-3x longer)
# ==============================================================================

set -e  # Exit on error

# Print script information
echo "=============================================="
echo "TLDR Parallel Evaluation"
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
CONFIG_FILE="${CONFIG_FILE:-evals/configs/parallel_config.yaml}"

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
echo "Starting TLDR parallel evaluation..."
python evals/eval_parallel.py \
    --config "$CONFIG_FILE" \
    --verbose

EXIT_STATUS=$?

echo ""
echo "=============================================="
echo "TLDR evaluation completed at: $(date)"
echo "Exit status: $EXIT_STATUS"
echo "Results saved to: evals/results"
echo "=============================================="

exit $EXIT_STATUS
