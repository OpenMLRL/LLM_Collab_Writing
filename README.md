# LLM Collaboration – Writing

This repo provides the extended environments for [**CoMLRL**](https://github.com/OpenMLRL/CoMLRL).

This repository contains the writing-task experiments in [**[AAAI26]** **_LLM Collaboration with Multi‑Agent Reinforcement Learning_**](https://arxiv.org/abs/2508.04652). 

<img src="./demo_aw.gif" alt="Writing demo" width="600px">

## Installation

Install [CoMLRL](https://github.com/OpenMLRL/CoMLRL):

```bash
pip install comlrl
# Install PyTorch compatible with your device
```

Or via conda-forge:

```bash
conda install -c conda-forge comlrl
# Install PyTorch compatible with your device
```

## Benchmarks

- ArXiv Abstract Expansion: `LovelyBuggies/arXiv_abstract` (train[:1000], val[:1000])
- TLDR Summarization: `trl-lib/tldr` (train[:1000], test[:1000])

## Training Scripts

```bash
python LLM_Collab_Writing/train_grpo.py \
  --config LLM_Collab_Writing/configs/grpo_arxiv_config.yaml

python LLM_Collab_Writing/train_magrpo.py \
  --config LLM_Collab_Writing/configs/magrpo_tldr_config.yaml
```

Override any configuration value inline with `--override`:

```bash
python LLM_Collab_Writing/train_magrpo.py \
  --config LLM_Collab_Writing/configs/magrpo_arxiv_config.yaml \
  --override model.name='Qwen/Qwen3-7B' magrpo.learning_rate=3e-6
```

## Settings

### Single Turn

Writing runs are strictly single-turn. Both training entrypoints enforce `num_turns=1`; configs that specify other values will raise an error.

### Formatters

- **ArXiv**: Agent 1 writes background/motivation; Agent 2 writes methodology/implications.
- **TLDR**: Agent 1 produces a concise summary; Agent 2 expands with additional details and vocabulary diversity.
- **GRPO mode**: A single agent emits both paragraphs separated by `[PARAGRAPH_SPLIT]`, which the reward splits internally.

### Reward Structure

Rewards reuse the level-based metrics from the paper:

1. Structural token limits.
2. Relative length coordination.
3. Vocabulary diversity (unique word ratios).
4. Style mix (transition-word coverage + Jaccard overlap).

The same functions back evaluation loggers for the baselines.

### Logging

Evaluation wrappers adapt the original logging utilities to the unified `MAGRPOTrainer` API, yielding aggregated metrics such as token ratios, transition coverage, and gated vs. ungated rewards. Weights & Biases configs mirror the code-generation project; set `wandb.project`, `wandb.entity`, and `wandb.name` in YAML or via overrides.
