"""
Single-agent Actor-Critic training entrypoint for writing tasks.

Uses IACTrainer with num_agents=1 and the GRPO-style formatter/reward split.
"""

import argparse
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer

from config import Config, add_config_args, parse_overrides
from comlrl.trainers.actor_critic import IACConfig, IACTrainer
from comlrl.utils.reward_processor import RewardProcessors
from rewards.arxiv_rewards import arxiv_combined_reward
from rewards.tldr_rewards import tldr_combined_reward


# Prompt formatters and helpers

def arxiv_single_formatter(example: Dict[str, Any]) -> str:
    """Prompt the single agent to produce two coordinated paragraphs for arXiv."""
    abstract = example.get("abstract_text", "")
    if not abstract:
        return "Error: No abstract provided."

    return f"""Please provide an expanded introduction of this abstract text in exactly two paragraphs with SAME LENGTH.

{abstract}

Instructions:
- First paragraph: Provide the background and motivation for this research, include as many categories of transition words as possible to improve flow.
- Second paragraph: Provide the framework, method, contribution, and the implications of this research, using same number of vocabulary words as the first paragraph, include as many categories of transition words as possible to improve flow, maintaining a consistent style

IMPORTANT: Separate the two paragraphs with exactly this delimiter: [PARAGRAPH_SPLIT]

FORMAT:
Paragraph 1: ...
[PARAGRAPH_SPLIT]
Paragraph 2: ...
"""


def tldr_single_formatter(example: Dict[str, Any]) -> str:
    """Prompt the single agent to produce complementary TLDR paragraphs."""
    prompt = example.get("prompt", "")
    if not prompt:
        return "Error: No prompt provided."

    return f"""Please provide a summary of this Reddit post in exactly two paragraphs:

{prompt}

Instructions:
- First paragraph: Provide a concise summary of the main points
- Second paragraph: Expand on the summary with more details, using more unique vocabulary words, include as many categories of transition words as possible to improve flow, and make it 2-3 times longer than the first paragraph in terms of character count, while maintaining a consistent style

IMPORTANT REQUIREMENTS - FOLLOW EXACTLY:
- No paragraph should be less than 10 tokens or more than 200 tokens
- Use EXACTLY this delimiter between paragraphs: [PARAGRAPH_SPLIT]

FORMAT:
Paragraph 1: ...
[PARAGRAPH_SPLIT]
Paragraph 2: ...
"""


def split_response_into_paragraphs(response: str) -> Tuple[str, str]:
    """Split a model response into two paragraphs with sensible fallbacks."""
    import re

    response = response.strip()
    delimiter = "[PARAGRAPH_SPLIT]"

    if delimiter in response:
        first, second = response.split(delimiter, 1)
        para1 = first.strip()
        para2 = second.strip()
    else:
        para2_pattern = re.compile(r"Paragraph 2:\s*", re.IGNORECASE)
        match = para2_pattern.search(response)

        if match:
            para1 = response[: match.start()].strip()
            para2 = response[match.end() :].strip()
        else:
            midpoint = len(response) // 3 or 1
            para1 = response[:midpoint].strip()
            para2 = response[midpoint:].strip()

    para1 = re.sub(r"^Paragraph 1:\s*", "", para1, flags=re.IGNORECASE)
    para2 = re.sub(r"^Paragraph 2:\s*", "", para2, flags=re.IGNORECASE)
    return para1, para2


def make_reward_function(dataset_type: str) -> Callable[[List[str]], List[float]]:
    """Create a reward wrapper that converts single-agent output to paired reward."""
    dataset_key = dataset_type.lower()
    if dataset_key == "arxiv":
        base_reward = arxiv_combined_reward
    elif dataset_key == "tldr":
        base_reward = tldr_combined_reward
    else:
        raise ValueError(f"Unsupported dataset type '{dataset_type}'.")

    def reward_fn(responses: List[str]) -> List[float]:
        completions1: List[str] = []
        completions2: List[str] = []

        for response in responses:
            para1, para2 = split_response_into_paragraphs(response)
            completions1.append(para1)
            completions2.append(para2)

        return base_reward(completions1, completions2)

    return reward_fn


def infer_dataset_type(dataset_name: str, explicit_type: Optional[str]) -> str:
    """Infer dataset type from name when not explicitly provided in the config."""
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


def get_formatter(dataset_type: str) -> Callable[[Dict[str, Any]], str]:
    dataset_key = dataset_type.lower()
    if dataset_key == "arxiv":
        return arxiv_single_formatter
    if dataset_key == "tldr":
        return tldr_single_formatter
    raise ValueError(f"Unsupported dataset type '{dataset_type}'.")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train single-agent Actor-Critic for writing tasks."
    )
    add_config_args(parser)
    args = parser.parse_args()

    if args.config:
        config = Config(args.config)
    else:
        default_config_path = Path(__file__).parent / "configs" / "ac_arxiv_config.yaml"
        if default_config_path.exists():
            config = Config(str(default_config_path))
        else:
            raise ValueError("Please provide a configuration file using --config.")

    if args.override:
        overrides = parse_overrides(args.override)
        config.update(overrides)

    model_config = config.get_model_config()
    critic_config = None
    model_name = model_config.name

    dataset_name = config.get("dataset.name")
    dataset_type = infer_dataset_type(dataset_name, config.get("dataset.type"))

    output_base_dir = config.get("output.base_dir", "./output")
    output_verbose = bool(config.get("output.verbose", False))
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "no_job_id")
    output_dir = os.path.join(output_base_dir, f"ac_job_{slurm_job_id}")
    os.makedirs(output_dir, exist_ok=True)

    train_split = config.get("dataset.train_split")
    eval_split = config.get("dataset.eval_split")

    ac_cfg = config.get_section("ac")
    seed_value = int(config.get("seed", ac_cfg.get("seed", 42)))
    _set_seed(seed_value)

    train_dataset = load_dataset(dataset_name, split=train_split)
    eval_dataset = load_dataset(dataset_name, split=eval_split)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    padding_side = config.get("tokenizer.padding_side")
    if padding_side:
        tokenizer.padding_side = padding_side

    if model_config.special_tokens:
        tokenizer.add_special_tokens(model_config.special_tokens)

    num_turns = int(ac_cfg.get("num_turns", 1))
    if num_turns != 1:
        raise ValueError(
            "Writing collaboration experiments are single-turn. "
            "Please set ac.num_turns=1 (or remove the field) in the config."
        )

    num_agents = int(ac_cfg.get("num_agents", 1))
    if num_agents != 1:
        raise ValueError(
            f"Single-agent AC expects num_agents=1; received num_agents={num_agents}."
        )

    temperature = ac_cfg.get("temperature", model_config.temperature)
    top_p = ac_cfg.get("top_p", model_config.top_p)
    top_k = ac_cfg.get("top_k")
    use_separate_critic = bool(ac_cfg.get("use_separate_critic", True))
    model_kwargs: Dict[str, Any] = {}
    if model_config.torch_dtype is not None:
        model_kwargs["torch_dtype"] = model_config.torch_dtype
    critic_config = None
    critics = None
    if use_separate_critic:
        critic_config = config.get_critic_config()
        critic_name = critic_config.name
        if not critic_name:
            raise ValueError("critic.name must be provided when use_separate_critic is true")
        critics = [critic_name]
        critic_model_kwargs: Dict[str, Any] = {}
        if critic_config.torch_dtype is not None:
            critic_model_kwargs["torch_dtype"] = critic_config.torch_dtype
    else:
        critic_model_kwargs = model_kwargs

    # Propagate verbosity to reward modules
    import rewards.arxiv_rewards as arxiv_rewards
    arxiv_rewards.VERBOSE = bool(output_verbose)
    import rewards.tldr_rewards as tldr_rewards
    tldr_rewards.VERBOSE = bool(output_verbose)
    formatter = get_formatter(dataset_type)
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

    trainer = IACTrainer(
        model=model_name,
        tokenizer=tokenizer,
        reward_func=reward_func,
        reward_processor=reward_processor,
        formatters=formatter,
        metrics_callback=None,
        external_transition=None,
        args=IACConfig(
            num_turns=1,
            num_train_epochs=ac_cfg.get("num_train_epochs", 1),
            agent_learning_rate=ac_cfg.get("agent_learning_rate", 5e-6),
            critic_learning_rate=ac_cfg.get("critic_learning_rate", 5e-6),
            value_loss_coef=ac_cfg.get("value_loss_coef", 0.6),
            advantage_normalization=ac_cfg.get("advantage_normalization", True),
            value_clip_range=ac_cfg.get("value_clip_range", 0.2),
            rollout_buffer_size=ac_cfg.get("rollout_buffer_size", 4),
            max_new_tokens=ac_cfg.get("max_new_tokens", 256),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_agents=1,
            num_generations=ac_cfg.get("num_generations", 1),
            use_separate_critic=use_separate_critic,
            critic_value_head_hidden_dim=ac_cfg.get("critic_value_head_hidden_dim"),
            value_head_hidden_dim=ac_cfg.get("value_head_hidden_dim"),
            discount=ac_cfg.get("discount", 0.9),
            eval_interval=ac_cfg.get("eval_interval", 4),
            eval_num_samples=ac_cfg.get("eval_num_samples", 4),
            eval_batch_size=ac_cfg.get("eval_batch_size", 1),
            logging_steps=ac_cfg.get("logging_steps", 1),
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model_config={
            "model_kwargs": model_kwargs,
            "critic_model_kwargs": (
                critic_model_kwargs
                if critic_config is not None
                else model_kwargs
            ),
        },
        wandb_config=_build_wandb_config(config, model_name, dataset_type),
        critics=critics,
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
    ac_section = config.get_section("ac")
    output_section = dict(config.get_section("output") or {})
    if "name" in wandb_section:
        wandb_name = wandb_section["name"]
    elif "run_name" in wandb_section:
        wandb_name = wandb_section["run_name"]
    else:
        wandb_name = f"{dataset_type}-ac"

    tags = wandb_section.get("tags", ["ac", dataset_type, "single-agent", "turns_1"])

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
            "trainer": ac_section,
        },
    }


if __name__ == "__main__":
    main()
