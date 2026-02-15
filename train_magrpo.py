"""
MAGRPO training entrypoint for collaborative writing tasks (arXiv abstracts and TLDR).

This script mirrors the layout of the code-generation project but specializes the
formatters, rewards, and evaluation logging for writing-focused datasets.
"""

import argparse
import os
from typing import Any, Callable, Dict, List, Optional

from config import Config, add_config_args, parse_overrides
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from loggers.arxiv_logger import (
    aggregate_arxiv_metrics_for_logging,
    arxiv_combined_reward_logger,
)
from loggers.tldr_logger import (
    aggregate_tldr_metrics_for_logging,
    tldr_combined_reward_logger,
)
from rewards.arxiv_rewards import arxiv_combined_reward
from rewards.tldr_rewards import tldr_combined_reward
from comlrl.utils.reward_processor import RewardProcessors
from comlrl.trainers.reinforce import MAGRPOConfig, MAGRPOTrainer


def background_agent_formatter(example: Dict[str, Any]) -> str:
    """Formatter for the background agent (Agent 1) for the arXiv dataset."""
    abstract = example.get("abstract_text", "")

    if not abstract:
        return "Error: No abstract provided."

    prompt_text = f"""Based on the following scientific abstract, expand content for an introduction section.

Abstract:
{abstract}

IMPORTANT INSTRUCTIONS:
- There is another agent that will provide methodology and implications
- You just need to focus on background and motivation
- Avoid repeating methodology and implications content
"""

    return prompt_text


def complementary_agent_formatter(example: Dict[str, Any]) -> str:
    """Formatter for the complementary agent (Agent 2) for the arXiv dataset."""
    abstract = example.get("abstract_text", "")

    if not abstract:
        return "Error: No abstract provided."

    prompt_text = f"""Based on the following scientific abstract, expand content for an introduction section.

Abstract:
{abstract}

IMPORTANT INSTRUCTIONS:
- There is another agent that will provide the background and motivation
- You just need to focus on methodology and implications
- Avoid repeating background and motivation content
"""

    return prompt_text


def summary_agent_formatter(example: Dict[str, Any]) -> str:
    """Formatter for the summary agent (Agent 1) for the TLDR dataset."""
    prompt = example.get("prompt", "")

    if not prompt:
        return "Error: No prompt provided."

    prompt_text = f"""Create a concise summary response to this post.

Query:
{prompt}

IMPORTANT INSTRUCTIONS:
- Provide a brief, focused summary in one sentence or a few sentences
- Be factual and informative
"""

    return prompt_text


def elaboration_agent_formatter(example: Dict[str, Any]) -> str:
    """Formatter for the elaboration agent (Agent 2) for the TLDR dataset."""
    prompt = example.get("prompt", "")

    if not prompt:
        return "Error: No prompt provided."

    prompt_text = f"""Create a detailed summary response to this post.

Original Query:
{prompt}

IMPORTANT INSTRUCTIONS:
- Use more unique words
- Use some transition words to improve flow
"""

    return prompt_text


def get_formatters(dataset_type: str) -> List[Callable[[Dict[str, Any]], str]]:
    """Return per-agent formatter functions for the selected dataset."""
    if dataset_type is None:
        raise ValueError(
            "dataset.type not specified in config. Please add 'type: arxiv/tldr' to the dataset section."
        )

    formatters_map = {
        "arxiv": [background_agent_formatter, complementary_agent_formatter],
        "tldr": [summary_agent_formatter, elaboration_agent_formatter],
    }

    dataset_key = dataset_type.lower()
    if dataset_key not in formatters_map:
        raise ValueError(f"Unsupported dataset type '{dataset_type}' for writing tasks.")

    return formatters_map[dataset_key]


def _adapt_eval_logger(
    base_logger: Callable[[List[str], List[str]], List[Dict[str, Any]]]
) -> Callable[..., List[Dict[str, Any]]]:
    """Adapt two-agent loggers to the MAGRPO interface."""

    def logger(*, agent_completions_turns, **_: Any) -> List[Dict[str, Any]]:
        if not agent_completions_turns or len(agent_completions_turns) < 2:
            return []

        num_samples = len(agent_completions_turns[0])
        completions1: List[str] = []
        completions2: List[str] = []

        for idx in range(num_samples):
            turns_agent1 = agent_completions_turns[0][idx]
            turns_agent2 = agent_completions_turns[1][idx]

            completion1 = turns_agent1[-1] if turns_agent1 else ""
            completion2 = turns_agent2[-1] if turns_agent2 else ""

            completions1.append(completion1)
            completions2.append(completion2)

        return base_logger(completions1, completions2)

    return logger


def _adapt_eval_aggregator(
    base_aggregator: Callable[[List[Dict[str, Any]]], Dict[str, float]]
) -> Callable[..., Dict[str, float]]:
    """Wrap aggregators so they match the MAGRPO signature (accepting num_turns)."""

    def aggregator(metrics: List[Dict[str, Any]], num_turns: int = 1) -> Dict[str, float]:
        return base_aggregator(metrics)

    return aggregator


def get_eval_logging(dataset_type: str) -> Dict[str, Callable]:
    """Return evaluation logger/aggregator wrappers when available."""
    dataset_key = dataset_type.lower()
    if dataset_key == "arxiv":
        return {
            "eval_logger": _adapt_eval_logger(arxiv_combined_reward_logger),
            "eval_aggregator": _adapt_eval_aggregator(
                aggregate_arxiv_metrics_for_logging
            ),
        }
    if dataset_key == "tldr":
        return {
            "eval_logger": _adapt_eval_logger(tldr_combined_reward_logger),
            "eval_aggregator": _adapt_eval_aggregator(
                aggregate_tldr_metrics_for_logging
            ),
        }
    return {}


def make_reward_function(
    dataset_type: str,
) -> Callable[..., List[float]]:
    """Create a MAGRPO-compatible reward function for the dataset."""
    dataset_key = dataset_type.lower()

    if dataset_key == "arxiv":
        base_reward = arxiv_combined_reward
    elif dataset_key == "tldr":
        base_reward = tldr_combined_reward
    else:
        raise ValueError(f"Unsupported dataset type '{dataset_type}'.")

    def reward_fn(*agent_completions, batch_items=None, prompts=None):
        if len(agent_completions) < 2:
            raise ValueError(
                "Writing tasks expect two agent completions for reward calculation."
            )

        completions1 = agent_completions[0]
        completions2 = agent_completions[1]
        return base_reward(completions1, completions2)

    return reward_fn


def infer_dataset_type(dataset_name: str, explicit_type: Optional[str]) -> str:
    """Infer dataset type from name when not explicitly provided."""
    if explicit_type:
        return explicit_type.lower()

    name = dataset_name.lower()
    if "arxiv" in name:
        return "arxiv"
    if "tldr" in name:
        return "tldr"

    raise ValueError(
        f"Could not infer dataset type from dataset name '{dataset_name}'. "
        "Please specify dataset.type in the config (arxiv or tldr)."
    )


def main():
    """Configure and launch MAGRPO training for writing datasets."""
    parser = argparse.ArgumentParser(
        description="Train MAGRPO for collaborative writing tasks."
    )
    add_config_args(parser)
    args = parser.parse_args()

    if not args.config:
        raise ValueError("Please provide a configuration file via --config.")

    config = Config(args.config)
    if args.override:
        overrides = parse_overrides(args.override)
        config.update(overrides)
    model_config = config.get_agent_model_config()
    model_name = model_config.name

    dataset_name = config.get("dataset.name")
    dataset_type = infer_dataset_type(dataset_name, config.get("dataset.type"))

    output_base_dir = config.get("output.base_dir", "./output")
    output_verbose = bool(config.get("output.verbose", False))
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "no_job_id")
    output_dir = os.path.join(output_base_dir, f"job_{slurm_job_id}")
    os.makedirs(output_dir, exist_ok=True)

    train_split = config.get("dataset.train_split")
    eval_split = config.get("dataset.eval_split")

    train_dataset = load_dataset(dataset_name, split=train_split)
    eval_dataset = load_dataset(dataset_name, split=eval_split)

    model_kwargs: Dict[str, Any] = {}
    if model_config.torch_dtype is not None:
        model_kwargs["torch_dtype"] = model_config.torch_dtype

    agents_field = config.get("agents")
    agent_names = None
    if isinstance(agents_field, (list, tuple)):
        if not all(isinstance(x, str) for x in agents_field):
            raise ValueError("agents must be a list of model names.")
        agent_names = [str(x) for x in agents_field]
        agents_config = {"num_agents": len(agent_names)}
    elif isinstance(agents_field, dict):
        agents_config = agents_field
    elif agents_field is None:
        agents_config = {}
    else:
        raise ValueError("agents must be a list of model names.")
    num_agents = agents_config.get("num_agents", 2)
    if num_agents != 2:
        raise ValueError(
            f"Writing experiments expect exactly 2 agents; received num_agents={num_agents}."
        )
    tokenizer_source = agent_names[0] if agent_names else model_name
    if not tokenizer_source:
        raise ValueError("agent_model.name or agents must be provided.")
    if agent_names:
        tokenizers = [AutoTokenizer.from_pretrained(name) for name in agent_names]
    else:
        tokenizers = [AutoTokenizer.from_pretrained(tokenizer_source)]
    for tok in tokenizers:
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        padding_side = config.get("tokenizer.padding_side")
        if padding_side:
            tok.padding_side = padding_side
        if model_config.special_tokens:
            tok.add_special_tokens(model_config.special_tokens)
    tokenizer = tokenizers[0]

    if agent_names:
        agents = [
            AutoModelForCausalLM.from_pretrained(
                name,
                **model_kwargs,
            )
            for name in agent_names
        ]
    else:
        agents = [
            AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs,
            )
            for _ in range(num_agents)
        ]
    magrpo_cfg = config.get_section("magrpo")
    num_turns_cfg = magrpo_cfg.get("num_turns")
    if num_turns_cfg is not None and int(num_turns_cfg) != 1:
        raise ValueError(
            "Writing collaboration experiments are single-turn. "
            "Please set magrpo.num_turns=1 (or remove the field) in the config."
        )

    temperature = model_config.temperature
    top_p = model_config.top_p
    top_k = model_config.top_k

    magrpo_args = MAGRPOConfig(
        num_turns=1,
        num_train_epochs=magrpo_cfg.get("num_train_epochs", 1),
        agent_learning_rate=magrpo_cfg.get("agent_learning_rate", 5e-6),
        logging_steps=magrpo_cfg.get("logging_steps", 10),
        num_generations=magrpo_cfg.get("num_generations", 4),
        max_new_tokens=magrpo_cfg.get("max_new_tokens", 256),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_agents=num_agents,
        parallel_training=str(magrpo_cfg.get("parallel_training", "auto")).strip().lower(),
        agent_devices=magrpo_cfg.get("agent_devices", None),
        early_termination_threshold=magrpo_cfg.get(
            "early_termination_threshold", -0.2
        ),
        rollout_buffer_size=magrpo_cfg.get("rollout_buffer_size", 2),
        train_batch_size=magrpo_cfg.get("train_batch_size"),
        advantage_normalization=magrpo_cfg.get("advantage_normalization", True),
        eval_interval=magrpo_cfg.get("eval_interval", 4),
        eval_num_samples=magrpo_cfg.get("eval_num_samples", 4),
        eval_batch_size=magrpo_cfg.get("eval_batch_size", 1),
    )

    import rewards.arxiv_rewards as arxiv_rewards
    arxiv_rewards.VERBOSE = bool(output_verbose)
    import rewards.tldr_rewards as tldr_rewards
    tldr_rewards.VERBOSE = bool(output_verbose)
    formatters = get_formatters(dataset_type)
    reward_func = make_reward_function(dataset_type)

    wandb_section = config.get_section("wandb")
    if "name" in wandb_section:
        wandb_name = wandb_section["name"]
    elif "run_name" in wandb_section:
        wandb_name = wandb_section["run_name"]
    else:
        wandb_name = f"{dataset_type}-magrpo"

    output_section = dict(config.get_section("output") or {})
    if "verbose" not in output_section:
        output_section["verbose"] = False
        config.update({"output": {"verbose": False}})

    wandb_config = {
        "project": wandb_section.get("project", "mlrl"),
        "entity": wandb_section.get("entity", "OpenMLRL"),
        "name": wandb_name,
        "dir": wandb_section.get("dir", "./wandb"),
        "tags": wandb_section.get("tags", ["magrpo", dataset_type]),
        "config_sections": {
            "dataset": config.get_section("dataset"),
            "agent_model": config.get_section("agent_model"),
            "output": output_section,
            "trainer": magrpo_cfg,
        },
    }

    logging_wrappers = get_eval_logging(dataset_type)

    reward_processor = None
    if config.get("reward_processor.enabled", True):
        scale_factor = config.get("reward_processor.scale_factor", 1.0)
        reward_processor = RewardProcessors.scale(factor=scale_factor)
        shift_val = config.get("reward_processor.shift", None)
        if shift_val is not None:
            try:
                shift_val_f = float(shift_val)
            except (TypeError, ValueError):
                shift_val_f = None
            if shift_val_f is not None:
                shift_proc = RewardProcessors.shift(value=shift_val_f)
                prev = reward_processor
                reward_processor = (lambda p=prev, s=shift_proc: (lambda x: s(p(x))))()

    trainer_kwargs: Dict[str, Any] = {
        "agent_model": model_name or None,
        "agents": agents,
        "num_agents": num_agents,
        "reward_func": reward_func,
        "formatters": formatters,
        "args": magrpo_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "tokenizer": tokenizers if agent_names else tokenizer,
        "wandb_config": wandb_config,
        "dataset_type": dataset_type,
    }
    trainer_kwargs.update(logging_wrappers)

    if reward_processor is not None:
        trainer_kwargs["reward_processor"] = reward_processor

    trainer = MAGRPOTrainer(**trainer_kwargs)
    trainer.verbose = bool(output_verbose)
    trainer.train()

    if config.get("output.save_final_model", True):
        save_path = config.get("output.save_path", os.path.join(output_dir, "final_model"))
        trainer.save_model(save_path)
        if output_verbose:
            print(f"Model saved to: {save_path}")

    if hasattr(config, "save"):
        config_save_path = os.path.join(output_dir, "config.yaml")
        config.save(config_save_path)
        if output_verbose:
            print(f"Configuration saved to: {config_save_path}")


if __name__ == "__main__":
    main()
