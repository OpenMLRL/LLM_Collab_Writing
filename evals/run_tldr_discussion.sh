#!/bin/bash
#
# TLDR Discussion (2-turn) Evaluation Runner
# 
# This script runs the Discussion configuration evaluation for TLDR.
# Discussion uses 2 turns:
#   - Turn 1: Both agents generate independently (Parallel)
#   - Turn 2: Both agents see each other's outputs and refine
#
# Usage:
#   ./run_discussion.sh                     # Use default config
#   ./run_discussion.sh --verbose           # With detailed output
#   ./run_discussion.sh --eval-split "test[:100]"  # Custom split
#
# To run in tmux (survives SSH disconnection):
#   tmux new-session -d -s tldr_discussion
#   tmux send-keys -t tldr_discussion 'cd /path/to/LLM_Collab_Writing && ./evals/run_discussion.sh' Enter
#   tmux attach -t tldr_discussion
#

set -e

# Print header
echo "=============================================="
echo "TLDR Discussion (2-turn) Evaluation"
echo "Started at: $(date)"
echo "Working directory: $(pwd)"
echo "=============================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project directory
cd "$PROJECT_DIR"

# Activate conda environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate comlrl
    echo "✓ Conda environment activated: comlrl"
else
    echo "Warning: conda not found, assuming environment is already active"
fi

# Configuration
CONFIG_FILE="evals/configs/tldr_discussion_config.yaml"
echo ""
echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Num turns: 2 (Parallel → Refinement)"

# Run the evaluation
echo ""
echo "Starting TLDR discussion (2-turn) evaluation..."
python evals/eval_tldr_discussion.py --config "$CONFIG_FILE" "$@"

# Print footer
echo ""
echo "=============================================="
echo "TLDR Discussion Evaluation Complete"
echo "Finished at: $(date)"
echo "=============================================="
