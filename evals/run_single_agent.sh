#!/bin/bash
#
# TLDR Single-Agent Evaluation Runner
# 
# This script runs the Single-Agent configuration evaluation for TLDR.
# Single-agent uses:
#   - One larger model (vs 2x smaller in multi-agent)
#   - 260 max_new_tokens (matching baseline, not 512)
#   - Generates both paragraphs with [PARAGRAPH_SPLIT] delimiter
#
# Usage:
#   ./run_single_agent.sh                     # Use default config
#   ./run_single_agent.sh --verbose           # With detailed output
#   ./run_single_agent.sh --eval-split "test[:100]"  # Custom split
#
# To run in tmux (survives SSH disconnection):
#   tmux new-session -d -s tldr_single
#   tmux send-keys -t tldr_single 'cd /path/to/LLM_Collab_Writing && ./evals/run_single_agent.sh' Enter
#   tmux attach -t tldr_single
#

set -e

# Print header
echo "=============================================="
echo "TLDR Single-Agent Evaluation"
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
CONFIG_FILE="evals/configs/single_agent_config.yaml"
echo ""
echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Max tokens: 260 (matching baseline)"

# Run the evaluation
echo ""
echo "Starting TLDR single-agent evaluation..."
python evals/eval_single_agent.py --config "$CONFIG_FILE" "$@"

# Print footer
echo ""
echo "=============================================="
echo "TLDR Single-Agent Evaluation Complete"
echo "Finished at: $(date)"
echo "=============================================="
