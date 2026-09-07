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

- ArXiv Abstract Expansion: `OpenMLRL/arXiv_abstract` (train[:1000], val[:1000])
- TLDR Summarization: `trl-lib/tldr` (train[:1000], test[:1000])

## Training Scripts

```bash
python LLM_Collab_Writing/train_grpo.py \
  --config LLM_Collab_Writing/configs/grpo_arxiv_config.yaml

python LLM_Collab_Writing/train_magrpo.py \
  --config LLM_Collab_Writing/configs/magrpo_tldr_config.yaml

python LLM_Collab_Writing/train_madpo.py \
  --config LLM_Collab_Writing/configs/madpo_tldr_config.yaml

python LLM_Collab_Writing/train_marlhf_iter.py \
  --config LLM_Collab_Writing/configs/marlhf_iter_tldr_config.yaml
```

Override any configuration value inline with `--override`:

```bash
python LLM_Collab_Writing/train_magrpo.py \
  --config LLM_Collab_Writing/configs/magrpo_arxiv_config.yaml \
  --override agent_model.name='Qwen/Qwen3-7B' magrpo.agent_learning_rate=3e-6
```

## Settings

### Single Turn

Writing runs are strictly single-turn. All training entrypoints enforce
`num_turns=1`; configs that specify other values raise an error.

### Preference Training

MADPO and MARLHF are available through `train_madpo.py` and `train_marlhf.py`.
Their iterative variants use `train_madpo_iter.py` and `train_marlhf_iter.py`.
The iterative configs support current, history-checkpoint, local-model, and API
comparators independently from the current, nearest-k, all-history, and
lambda-decay replay modes.

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

### Centralized MAGRPO

`train_magrpo.py --config configs/magrpo_tldr_config.yaml --override magrpo.collaboration_mode=centralized magrpo.num_turns=1`
trains one joint-input/joint-output actor using the task reward, without a
comparator, preference dataset, or learned reward model. The existing writing
adapter splits role outputs for rewards and evaluation. `max_new_tokens` is the
total joint-response budget. The default remains decentralized; arXiv uses the
same switch with its own configuration and adapter.

### Centralized Preference Collaboration

MADPO, MARLHF, and their iterative variants can train one model to generate both
writing roles. The default stays decentralized. Enable it with:

```bash
python train_madpo_iter.py --config configs/madpo_iter_tldr_config.yaml --override madpo_iter.collaboration_mode=centralized
python train_marlhf_iter.py --config configs/marlhf_iter_tldr_config.yaml --override marlhf_iter.collaboration_mode=centralized
```

The same switch works for Arxiv configs and for `madpo` / `marlhf` non-iterative
scripts. Keep `num_agents=2` for the task roles; `agent_model` is loaded once.
Explicit `agents` lists and actor device lists must each describe one model.
The existing writing adapter combines both original prompts and requests
`<agent_0>` / `<agent_1>` prose sections. Only task rewards and evaluation split
the sections; policy training and learned reward scoring use the entire joint
response. Iterative comparators automatically use the same centralized prompt
and parser, regardless of comparator source.

`max_new_tokens` is a joint, not per-role, budget; consider doubling the previous
per-role value. For MARLHF, `reward_max_length` must accommodate the joint prompt
and both responses. Reward/comparator device overrides remain available. Existing
decentralized behavior, dataset defaults, and environment-step counting do not
change.

### Logging

Evaluation wrappers adapt the original logging utilities to the unified `MAGRPOTrainer` API, yielding aggregated metrics such as token ratios, transition coverage, and gated vs. ungated rewards. Weights & Biases configs mirror the code-generation project; set `wandb.project`, `wandb.entity`, and `wandb.name` in YAML or via overrides.

## Slurm runtime cache

Training launchers isolate unset CUDA JIT caches per job on node-local storage.
Update CoMLRL alongside this checkout; see the [runtime cache guide](docs/runtime_cache.md).
