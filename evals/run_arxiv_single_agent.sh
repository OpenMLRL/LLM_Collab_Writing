#!/bin/bash
#
# arXiv Single-Agent Evaluation Runner
# 
# This script runs the Single-Agent configuration evaluation for arXiv.
# Single-agent uses:
#   - One larger model (4B vs 2x 1.7B in multi-agent)
#   - 512 max_new_tokens (both paragraphs need 128-256 tokens each)
#   - Generates both paragraphs with [PARAGRAPH_SPLIT] delimiter
#   - Equal-length paragraphs (unlike TLDR's 2-3x ratio)
#
# Usage:
#   ./run_arxiv_single_agent.sh                     # Use default config
#   ./run_arxiv_single_agent.sh --verbose           # With detailed output
#   ./run_arxiv_single_agent.sh --eval-split "val[:100]"  # Custom split
#
# To run in tmux (survives SSH disconnection):
#   tmux new-session -d -s arxiv_single
#   tmux send-keys -t arxiv_single 'cd /path/to/LLM_Collab_Writing && ./evals/run_arxiv_single_agent.sh' Enter
#   tmux attach -t arxiv_single
#

set -e

# Print header
echo "=============================================="
echo "arXiv Single-Agent Evaluation"
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
CONFIG_FILE="evals/configs/arxiv_single_agent_config.yaml"
echo ""
echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Max tokens: 512 (both paragraphs need 128-256 each)"

# Run the evaluation
echo ""
echo "Starting arXiv single-agent evaluation..."
python evals/eval_arxiv_single_agent.py --config "$CONFIG_FILE" "$@"

# Print footer
echo ""
echo "=============================================="
echo "arXiv Single-Agent Evaluation Complete"
echo "Finished at: $(date)"
echo "=============================================="
