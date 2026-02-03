"""
MAAC training entrypoint for collaborative writing tasks (arXiv abstracts and TLDR).

This mirrors the code-generation MAAC script while reusing the writing-specific
formatters and reward functions.
"""

import argparse
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer

from config import Config, add_config_args, parse_overrides
from comlrl.trainers.maac import MAACConfig, MAACTrainer
from comlrl.utils.reward_processor import RewardProcessors
from rewards.arxiv_rewards import arxiv_combined_reward
from rewards.tldr_rewards import tldr_combined_reward


# -----------------------------------------------------------------------------
# Prompt formatters
# -----------------------------------------------------------------------------

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


def make_reward_function(dataset_type: str) -> Callable[..., List[float]]:
    """Create a MAAC-compatible reward function for the dataset."""
    dataset_key = dataset_type.lower()

    if dataset_key == "arxiv":
        base_reward = arxiv_combined_reward
    elif dataset_key == "tldr":
        base_reward = tldr_combined_reward
    else:
        raise ValueError(f"Unsupported dataset type '{dataset_type}'.")

    def reward_fn(prompts, completions1, completions2):
        _ = prompts  # prompt text not needed for reward
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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MAAC for collaborative writing tasks."
    )
    add_config_args(parser)
    args = parser.parse_args()

    if args.config:
        config = Config(args.config)
    else:
        default_config_path = Path(__file__).parent / "configs" / "maac_arxiv_config.yaml"
        if default_config_path.exists():
            config = Config(str(default_config_path))
        else:
            raise ValueError("Please provide a configuration file using --config.")

    if args.override:
        overrides = parse_overrides(args.override)
        config.update(overrides)

    model_config = config.get_model_config()
    model_name = model_config.name

    dataset_name = config.get("dataset.name")
    dataset_type = infer_dataset_type(dataset_name, config.get("dataset.type"))

    output_base_dir = config.get("output.base_dir", "./output")
    output_verbose = bool(config.get("output.verbose", False))
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "no_job_id")
    output_dir = os.path.join(output_base_dir, f"maac_job_{slurm_job_id}")
    os.makedirs(output_dir, exist_ok=True)

    train_split = config.get("dataset.train_split")
    eval_split = config.get("dataset.eval_split")

    maac_cfg = config.get_section("maac")
    seed_value = int(config.get("seed", maac_cfg.get("seed", 42)))
    _set_seed(seed_value)

    train_dataset = load_dataset(dataset_name, split=train_split)
    eval_dataset = load_dataset(dataset_name, split=eval_split)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, **model_config.tokenizer_kwargs
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    padding_side = config.get("tokenizer.padding_side")
    if padding_side:
        tokenizer.padding_side = padding_side

    if model_config.special_tokens:
        tokenizer.add_special_tokens(model_config.special_tokens)

    num_turns = int(maac_cfg.get("num_turns", 1))
    if num_turns != 1:
        raise ValueError(
            "Writing collaboration experiments are single-turn. "
            "Please set maac.num_turns=1 (or remove the field) in the config."
        )

    num_agents = int(maac_cfg.get("num_agents", 2))
    if num_agents != 2:
        raise ValueError(
            f"Writing experiments expect exactly 2 agents; received num_agents={num_agents}."
        )

    if "do_sample" in maac_cfg:
        use_sampling = bool(maac_cfg.get("do_sample"))
    else:
        use_sampling = bool(
            "temperature" in maac_cfg or "top_p" in maac_cfg or "top_k" in maac_cfg
        )
    temperature = maac_cfg.get("temperature", model_config.temperature)
    top_p = maac_cfg.get("top_p", model_config.top_p)
    top_k = maac_cfg.get("top_k")
    critic_model = (
        maac_cfg.get("critic_model")
        or maac_cfg.get("critic_model_name_or_path")
        or model_name
    )

    # Propagate verbosity to reward modules
    import rewards.arxiv_rewards as arxiv_rewards
    arxiv_rewards.VERBOSE = bool(output_verbose)
    import rewards.tldr_rewards as tldr_rewards
    tldr_rewards.VERBOSE = bool(output_verbose)
    formatters = get_formatters(dataset_type)
    reward_func = make_reward_function(dataset_type)

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

    trainer = MAACTrainer(
        model=model_name,
        tokenizer=tokenizer,
        reward_func=reward_func,
        reward_processor=reward_processor,
        formatters=formatters,
        metrics_callback=None,
        external_transition=None,
        args=MAACConfig(
            num_turns=1,
            num_train_epochs=maac_cfg.get("num_train_epochs", 1),
            actor_learning_rate=maac_cfg.get("actor_learning_rate", 5e-6),
            critic_learning_rate=maac_cfg.get("critic_learning_rate", 5e-6),
            value_loss_coef=maac_cfg.get("value_loss_coef", 0.6),
            rollout_buffer_size=maac_cfg.get("rollout_buffer_size", 4),
            max_new_tokens=maac_cfg.get("max_new_tokens", 256),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=use_sampling,
            num_agents=num_agents,
            num_generations=maac_cfg.get("num_generations", 1),
            critic_model_name_or_path=critic_model,
            discount=maac_cfg.get("discount", 0.9),
            critic_type=maac_cfg.get("critic_type", "v"),
            eval_interval=maac_cfg.get("eval_interval", 4),
            eval_num_samples=maac_cfg.get("eval_num_samples", 4),
            eval_batch_size=maac_cfg.get("eval_batch_size", 1),
            logging_steps=maac_cfg.get("logging_steps", 1),
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model_config={
            "tokenizer_kwargs": model_config.tokenizer_kwargs,
            "model_kwargs": model_config.model_kwargs,
            "critic_model_kwargs": maac_cfg.get(
                "critic_model_kwargs", model_config.model_kwargs
            ),
        },
        wandb_config=_build_wandb_config(config, model_name, dataset_type),
    )
    trainer.verbose = bool(output_verbose)
    trainer.train()

    if config.get("output.save_final_model", True):
        save_path = config.get(
            "output.save_path", os.path.join(output_dir, "final_model")
        )
        trainer.save_model(save_path)
        if output_verbose:
            print(f"Model saved to: {save_path}")

    if hasattr(config, "save"):
        config_save_path = os.path.join(output_dir, "config.yaml")
        config.save(config_save_path)
        if output_verbose:
            print(f"Configuration saved to: {config_save_path}")

    if wandb.run is not None:
        wandb.finish()


def _build_wandb_config(
    config: Config, model_name: str, dataset_type: str
) -> Dict[str, Any]:
    wandb_section = config.get_section("wandb")
    maac_section = config.get_section("maac")
    output_section = dict(config.get_section("output") or {})
    model_short_name = model_name.split("/")[-1].lower()

    if "name" in wandb_section:
        wandb_name = wandb_section["name"]
    elif "run_name" in wandb_section:
        wandb_name = wandb_section["run_name"]
    else:
        wandb_name = f"maac_{dataset_type}_{model_short_name}"

    tags = wandb_section.get("tags", ["maac", dataset_type, "multi-agent", "turns_1"])

    return {
        "project": wandb_section.get("project", "comlrl"),
        "entity": wandb_section.get("entity", "OpenMLRL"),
        "name": wandb_name,
        "dir": wandb_section.get("dir", "./wandb"),
        "tags": tags,
        "config_sections": {
            "dataset": config.get_section("dataset"),
            "model": config.get_section("model"),
            "output": output_section,
            "trainer": maac_section,
        },
    }


if __name__ == "__main__":
    main()
