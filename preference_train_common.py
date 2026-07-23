"""Shared setup for preference-based collaborative writing trainers."""

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import random
from typing import Any, Dict, Type

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from comlrl.utils.reward_processor import RewardProcessors
from config import Config
from loggers.ac_writing_metrics import build_ac_writing_metrics_callback
from train_magrpo import (
    get_eval_logging,
    get_formatters,
    infer_dataset_type,
    make_reward_function,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_reward_processor(config: Config):
    if not config.get("reward_processor.enabled", True):
        return None

    reward_processor = RewardProcessors.scale(
        factor=float(config.get("reward_processor.scale_factor", 1.0))
    )
    shift_value = config.get("reward_processor.shift")
    if shift_value is None:
        return reward_processor

    shift_processor = RewardProcessors.shift(value=float(shift_value))
    return lambda value: shift_processor(reward_processor(value))


def run_preference_training(
    *,
    config: Config,
    section_name: str,
    args_cls: Type[Any],
    trainer_cls: Type[Any],
    algorithm_name: str,
) -> None:
    model_config = config.get_agent_model_config()
    model_name = model_config.name
    dataset_name = config.get("dataset.name")
    dataset_type = infer_dataset_type(dataset_name, config.get("dataset.type"))
    trainer_config: Dict[str, Any] = config.get_section(section_name)

    num_agents = int(trainer_config.get("num_agents", 2))
    if num_agents != 2:
        raise ValueError(
            f"Writing preference training requires exactly 2 agents; got {num_agents}."
        )
    num_turns = int(trainer_config.get("num_turns", 1))
    if num_turns != 1:
        raise ValueError(
            f"Writing preference training is single-turn; got num_turns={num_turns}."
        )

    set_seed(int(config.get("seed", trainer_config.get("seed", 42))))

    agent_names = config.get("agents")
    if agent_names is not None:
        if not isinstance(agent_names, (list, tuple)) or not all(
            isinstance(name, str) for name in agent_names
        ):
            raise ValueError("agents must be a list of model names.")
        if len(agent_names) != num_agents:
            raise ValueError(f"agents must contain exactly {num_agents} model names.")
        agent_names = [str(name) for name in agent_names]

    output_base_dir = str(config.get("output.base_dir", f"output_{section_name}"))
    job_id = os.environ.get("SLURM_JOB_ID", "no_job_id")
    output_dir = os.path.join(output_base_dir, f"job_{job_id}")
    os.makedirs(output_dir, exist_ok=True)
    config.save(os.path.join(output_dir, "config.yaml"))

    train_dataset = load_dataset(
        dataset_name,
        split=config.get("dataset.train_split"),
    )
    eval_dataset = load_dataset(
        dataset_name,
        split=config.get("dataset.eval_split"),
    )

    tokenizer_sources = agent_names or [model_name]
    if not tokenizer_sources[0]:
        raise ValueError("agent_model.name or agents must be provided.")
    tokenizers = [AutoTokenizer.from_pretrained(source) for source in tokenizer_sources]
    for tokenizer in tokenizers:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        padding_side = config.get("tokenizer.padding_side")
        if padding_side:
            tokenizer.padding_side = str(padding_side)
        if model_config.special_tokens:
            tokenizer.add_special_tokens(model_config.special_tokens)

    args_kwargs = dict(trainer_config)
    args_kwargs.setdefault("temperature", model_config.temperature)
    args_kwargs.setdefault("top_p", model_config.top_p)
    args_kwargs.setdefault("top_k", model_config.top_k)
    args_kwargs.setdefault("num_agents", num_agents)
    args_kwargs.setdefault("num_turns", num_turns)
    trainer_args = args_cls(**args_kwargs)

    formatters = get_formatters(dataset_type)
    reward_func = make_reward_function(dataset_type)
    eval_logging = get_eval_logging(dataset_type)

    output_verbose = bool(config.get("output.verbose", False))
    import rewards.arxiv_rewards as arxiv_rewards
    import rewards.tldr_rewards as tldr_rewards

    arxiv_rewards.VERBOSE = output_verbose
    tldr_rewards.VERBOSE = output_verbose

    wandb_section = config.get_section("wandb")
    wandb_name = (
        wandb_section.get("name")
        or wandb_section.get("run_name")
        or f"{dataset_type}-{algorithm_name.lower()}"
    )
    tags = wandb_section.get(
        "tags",
        [
            algorithm_name.lower(),
            dataset_type,
            "multi-agent",
            "turns_1",
        ],
    )
    wandb_config = {
        "project": wandb_section.get("project", "comlrl"),
        "entity": wandb_section.get("entity", "OpenMLRL"),
        "name": wandb_name,
        "dir": wandb_section.get("dir", output_base_dir),
        "output_dir": output_dir,
        "tags": list(tags),
        "config_sections": {
            "dataset": config.get_section("dataset"),
            "agent_model": config.get_section("agent_model"),
            "output": config.get_section("output"),
            "trainer": trainer_config,
        },
    }

    trainer_kwargs: Dict[str, Any] = {
        "agent_model": model_name if agent_names is None else None,
        "agents": agent_names,
        "num_agents": num_agents,
        "tokenizer": tokenizers if agent_names else tokenizers[0],
        "model_config": {
            "torch_dtype": model_config.torch_dtype,
            "special_tokens": model_config.special_tokens,
        },
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "dataset_type": dataset_type,
        "reward_func": reward_func,
        "formatters": formatters,
        "wandb_config": wandb_config,
        "args": trainer_args,
        **eval_logging,
    }

    if algorithm_name.lower() in {"marlhf", "marlhf_iter"}:
        trainer_kwargs["metrics_callback"] = build_ac_writing_metrics_callback(
            dataset_type,
            num_agents,
        )

    reward_processor = build_reward_processor(config)
    if reward_processor is not None:
        trainer_kwargs["reward_processor"] = reward_processor

    trainer = trainer_cls(**trainer_kwargs)
    trainer.verbose = output_verbose
    trainer.train()

    if config.get("output.save_final_model", False):
        save_path = config.get(
            "output.save_path",
            os.path.join(output_dir, "final_model"),
        )
        trainer.save_model(save_path)
