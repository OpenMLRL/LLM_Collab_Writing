"""
GRPO-style single-agent training for collaborative writing tasks.

We reuse the MAGRPO trainer with a single agent, aligning with the structure used
in the code-generation repository while adapting rewards/formatters to the
writing datasets (arXiv abstracts and TLDR summaries).
"""

import argparse
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import Config, add_config_args, parse_overrides
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from rewards.arxiv_rewards import arxiv_combined_reward
from rewards.tldr_rewards import tldr_combined_reward
from comlrl.utils.reward_processor import RewardProcessors
from comlrl.trainers.reinforce import MAGRPOConfig, MAGRPOTrainer
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


def make_reward_function(dataset_type: str) -> Callable[..., List[float]]:
    """Create a reward wrapper that converts single-agent output to paired reward."""
    dataset_key = dataset_type.lower()
    if dataset_key == "arxiv":
        base_reward = arxiv_combined_reward
    elif dataset_key == "tldr":
        base_reward = tldr_combined_reward
    else:
        raise ValueError(f"Unsupported dataset type '{dataset_type}'.")

    def reward_fn(*agent_completions, batch_items=None, prompts=None):
        if not agent_completions:
            return []

        responses = agent_completions[0]
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


def main():
    """Run single-agent writing training using MAGRPO in GRPO mode."""
    parser = argparse.ArgumentParser(
        description="Train GRPO-style single agent for collaborative writing tasks."
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

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    padding_side = config.get("tokenizer.padding_side")
    if padding_side:
        tokenizer.padding_side = padding_side

    if model_config.special_tokens:
        tokenizer.add_special_tokens(model_config.special_tokens)

    model_kwargs: Dict[str, Any] = {}
    if model_config.torch_dtype is not None:
        model_kwargs["torch_dtype"] = model_config.torch_dtype

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )

    grpo_cfg = config.get_section("grpo")
    num_turns_cfg = grpo_cfg.get("num_turns")
    if num_turns_cfg is not None and int(num_turns_cfg) != 1:
        raise ValueError(
            "Writing collaboration experiments are single-turn. "
            "Please set grpo.num_turns=1 (or remove the field) in the config."
        )

    temperature = model_config.temperature
    top_p = model_config.top_p
    top_k = model_config.top_k

    grpo_args = MAGRPOConfig(
        num_turns=1,
        num_train_epochs=grpo_cfg.get("num_train_epochs", 1),
        agent_learning_rate=grpo_cfg.get("agent_learning_rate", 5e-6),
        logging_steps=grpo_cfg.get("logging_steps", 10),
        num_generations=grpo_cfg.get("num_generations", 4),
        max_new_tokens=grpo_cfg.get("max_new_tokens", 512),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_agents=1,
        parallel_training=str(grpo_cfg.get("parallel_training", "auto")).strip().lower(),
        agent_devices=grpo_cfg.get("agent_devices", None),
        early_termination_threshold=grpo_cfg.get(
            "early_termination_threshold", -0.2
        ),
        eval_interval=grpo_cfg.get("eval_interval", 4),
        eval_num_samples=grpo_cfg.get("eval_num_samples", 4),
        eval_batch_size=grpo_cfg.get("eval_batch_size", 1),
        train_batch_size=grpo_cfg.get("train_batch_size"),
        advantage_normalization=grpo_cfg.get("advantage_normalization", True),
    )

    import rewards.arxiv_rewards as arxiv_rewards
    arxiv_rewards.VERBOSE = bool(output_verbose)
    import rewards.tldr_rewards as tldr_rewards
    tldr_rewards.VERBOSE = bool(output_verbose)
    formatter = get_formatter(dataset_type)
    reward_func = make_reward_function(dataset_type)

    wandb_section = config.get_section("wandb")
    if "name" in wandb_section:
        wandb_name = wandb_section["name"]
    elif "run_name" in wandb_section:
        wandb_name = wandb_section["run_name"]
    else:
        wandb_name = f"{dataset_type}-grpo"

    output_section = dict(config.get_section("output") or {})
    if "verbose" not in output_section:
        output_section["verbose"] = False
        config.update({"output": {"verbose": False}})

    wandb_config = {
        "project": wandb_section.get("project", "mlrl"),
        "entity": wandb_section.get("entity", "OpenMLRL"),
        "name": wandb_name,
        "dir": wandb_section.get("dir", "./wandb"),
        "tags": wandb_section.get("tags", ["grpo", dataset_type, "single-agent"]),
        "config_sections": {
            "dataset": config.get_section("dataset"),
            "agent_model": config.get_section("agent_model"),
            "output": output_section,
            "trainer": grpo_cfg,
        },
    }

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
        "agents": [model],
        "num_agents": 1,
        "reward_func": reward_func,
        "formatters": formatter,
        "args": grpo_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "tokenizer": tokenizer,
        "wandb_config": wandb_config,
        "dataset_type": dataset_type,
    }

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
