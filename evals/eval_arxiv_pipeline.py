#!/usr/bin/env python3
"""
Pipeline Evaluation Script for arXiv (Abstract Expansion)

This script runs inference-only evaluation for the Pipeline (sequential)
configuration where Agent 2 receives Agent 1's output in its prompt.

Key Difference from Parallel:
- Parallel: Agent 2 creates complementary content independently
- Pipeline: Agent 2 sees Agent 1's background section and builds upon it

Agent 1 (Background): Creates background and motivation content
Agent 2 (Complementary): Creates methodology/implications that builds on Agent 1's output

Key differences from TLDR:
- Token range: 128-256 per completion (vs 8-256 for TLDR)
- Length ratio: ~1.0x (equal) instead of 2-3x
- Vocabulary ratio: ~1.0x (similar) instead of 2x+

Metrics tracked per agent:
- Time: Wall-clock generation time (seconds)
- Cost: Number of tokens produced
- Score: Reward from arxiv_combined_reward

Output:
- CSV file with per-sample metrics
- JSON file with aggregated statistics
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rewards.arxiv_rewards as arxiv_rewards_module
from loggers.arxiv_logger import arxiv_combined_reward_logger
from rewards.arxiv_rewards import arxiv_combined_reward


def background_agent_formatter(example: Dict[str, Any]) -> str:
    """Formatter for the background agent (Agent 1) for the arXiv dataset.
    
    Same as parallel - Agent 1 prompt is unchanged.
    """
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


def complementary_pipeline_formatter(example: Dict[str, Any], agent1_output: str) -> str:
    """Formatter for the complementary agent (Agent 2) in PIPELINE mode.
    
    Key difference from parallel: Agent 2 sees Agent 1's background section.
    """
    abstract = example.get("abstract_text", "")

    if not abstract:
        return "Error: No abstract provided."

    prompt_text = f"""Based on the following scientific abstract, expand content for an introduction section.

Abstract:
{abstract}

Background and motivation section (from another agent):
{agent1_output}

IMPORTANT INSTRUCTIONS:
- Focus on methodology and implications
- Build upon the background section provided above
- Avoid repeating what was already written in the background
"""

    return prompt_text


def generate_completion(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_return_sequences: int = 1,
) -> Tuple[List[str], float, List[int]]:
    """
    Generate completions and return (texts, time_seconds, token_counts).
    """
    device = model.device

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    prompt_length = inputs.input_ids.shape[1]

    # Time the generation
    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
            num_beams=1,
        )

    end_time = time.perf_counter()
    generation_time = end_time - start_time

    # Decode completions (exclude prompt)
    completions = []
    token_counts = []

    for i in range(outputs.shape[0]):
        completion_tokens = outputs[i, prompt_length:]
        token_counts.append(len(completion_tokens))
        completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
        completions.append(completion_text)

    return completions, generation_time, token_counts


def evaluate_pipeline(
    model_name: str,
    dataset_name: str,
    eval_split: str,
    output_dir: str,
    num_attempts: int = 1,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run pipeline evaluation on arXiv dataset.

    Pipeline: Agent 1 generates first, Agent 2 receives Agent 1's output in prompt.

    Args:
        model_name: HuggingFace model name
        dataset_name: Dataset name
        eval_split: Dataset split for evaluation
        output_dir: Directory to save results
        num_attempts: Number of attempts per problem
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling
        verbose: Print detailed output

    Returns:
        Dictionary with aggregated results
    """
    print(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load two separate model instances for agents
    print("Loading Agent 1 (background) model...")
    model_agent1 = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model_agent1.eval()

    print("Loading Agent 2 (complementary) model...")
    model_agent2 = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model_agent2.eval()

    # Load dataset
    print(f"Loading dataset: {dataset_name} split={eval_split}")
    dataset = load_dataset(dataset_name, split=eval_split)
    print(f"Evaluation dataset size: {len(dataset)}")

    # Storage for results
    all_results = []

    # Suppress verbose output from reward function
    arxiv_rewards_module.VERBOSE = False

    # Process each problem
    for prob_idx, item in enumerate(dataset):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Problem {prob_idx + 1}/{len(dataset)}")

        # Format prompt for Agent 1 (same as parallel)
        prompt_agent1 = background_agent_formatter(item)

        # Generate num_attempts attempts for this problem
        for attempt_idx in range(num_attempts):
            if verbose:
                print(f"  Attempt {attempt_idx + 1}/{num_attempts}")

            # STEP 1: Agent 1 generates background/motivation FIRST
            completions1, time1, tokens1 = generate_completion(
                model_agent1,
                tokenizer,
                prompt_agent1,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=1,
            )
            background_completion = completions1[0]
            background_tokens = tokens1[0]

            # STEP 2: Agent 2 receives Agent 1's output in its prompt (PIPELINE)
            # This is the key difference from parallel!
            prompt_agent2 = complementary_pipeline_formatter(item, background_completion)
            
            completions2, time2, tokens2 = generate_completion(
                model_agent2,
                tokenizer,
                prompt_agent2,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=1,
            )
            complementary_completion = completions2[0]
            complementary_tokens = tokens2[0]

            # Compute reward
            rewards = arxiv_combined_reward(
                [background_completion],
                [complementary_completion],
            )
            score = rewards[0] if rewards else 0.0

            # Get detailed metrics using logger
            metrics = arxiv_combined_reward_logger(
                [background_completion],
                [complementary_completion],
            )
            metric = metrics[0] if metrics else {}

            # Store attempt result
            attempt_result = {
                "problem_id": prob_idx,
                "attempt_id": attempt_idx,
                "agent1_time_s": round(time1, 4),
                "agent1_tokens": background_tokens,
                "agent2_time_s": round(time2, 4),
                "agent2_tokens": complementary_tokens,
                "total_time_s": round(time1 + time2, 4),
                "total_tokens": background_tokens + complementary_tokens,
                "score": round(score, 4),
                "level1_reward": round(metric.get("level1_reward", 0.0), 4),
                "level2_reward": round(metric.get("level2_reward", 0.0), 4),
                "level3_reward": round(metric.get("level3_reward", 0.0), 4),
                "level4_reward": round(metric.get("level4_reward", 0.0), 4),
                "length_ratio": round(metric.get("length_ratio", 0.0), 4),
                "unique_words_ratio": round(metric.get("unique_words_ratio", 0.0), 4),
            }
            all_results.append(attempt_result)

            if verbose:
                print(f"    Time: {time1:.2f}s + {time2:.2f}s = {time1+time2:.2f}s")
                print(f"    Tokens: {background_tokens} + {complementary_tokens} = {background_tokens+complementary_tokens}")
                print(f"    Score: {score:.2f}")

    # Compute aggregated statistics
    num_problems = len(dataset)

    aggregated = {
        "avg_time_agent1": sum(r["agent1_time_s"] for r in all_results) / len(all_results),
        "avg_time_agent2": sum(r["agent2_time_s"] for r in all_results) / len(all_results),
        "avg_time_total": sum(r["total_time_s"] for r in all_results) / len(all_results),
        "avg_tokens_agent1": sum(r["agent1_tokens"] for r in all_results) / len(all_results),
        "avg_tokens_agent2": sum(r["agent2_tokens"] for r in all_results) / len(all_results),
        "avg_tokens_total": sum(r["total_tokens"] for r in all_results) / len(all_results),
        "avg_score": sum(r["score"] for r in all_results) / len(all_results),
        "avg_level1_reward": sum(r["level1_reward"] for r in all_results) / len(all_results),
        "avg_level2_reward": sum(r["level2_reward"] for r in all_results) / len(all_results),
        "avg_level3_reward": sum(r["level3_reward"] for r in all_results) / len(all_results),
        "avg_level4_reward": sum(r["level4_reward"] for r in all_results) / len(all_results),
        "avg_length_ratio": sum(r["length_ratio"] for r in all_results) / len(all_results),
        "avg_unique_words_ratio": sum(r["unique_words_ratio"] for r in all_results) / len(all_results),
    }

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save CSV
    csv_path = os.path.join(output_dir, f"arxiv_pipeline_results_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nCSV results saved to: {csv_path}")

    # Save JSON summary
    summary = {
        "config": "pipeline",
        "domain": "arxiv",
        "model": model_name,
        "dataset": dataset_name,
        "eval_split": eval_split,
        "num_problems": num_problems,
        "num_attempts": num_attempts,
        "hyperparameters": {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
        },
        "timestamp": timestamp,
        "aggregated": {k: round(v, 4) for k, v in aggregated.items()},
    }

    json_path = os.path.join(output_dir, f"arxiv_pipeline_summary_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary saved to: {json_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("PIPELINE EVALUATION SUMMARY (arXiv)")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Problems evaluated: {num_problems}")
    print(f"Attempts per problem: {num_attempts}")
    print(f"\nAggregated Metrics:")
    print(f"  Avg Time (Agent 1): {aggregated['avg_time_agent1']:.2f}s")
    print(f"  Avg Time (Agent 2): {aggregated['avg_time_agent2']:.2f}s")
    print(f"  Avg Time (Total):   {aggregated['avg_time_total']:.2f}s")
    print(f"  Avg Tokens (Agent 1): {aggregated['avg_tokens_agent1']:.1f}")
    print(f"  Avg Tokens (Agent 2): {aggregated['avg_tokens_agent2']:.1f}")
    print(f"  Avg Tokens (Total):   {aggregated['avg_tokens_total']:.1f}")
    print(f"  Avg Score: {aggregated['avg_score']:.2f} / 3.0")
    print(f"\nReward Breakdown:")
    print(f"  Level 1 (Structural): {aggregated['avg_level1_reward']:.2f}")
    print(f"  Level 2 (Coordination): {aggregated['avg_level2_reward']:.2f}")
    print(f"  Level 3 (Vocabulary): {aggregated['avg_level3_reward']:.2f}")
    print(f"  Level 4 (Style): {aggregated['avg_level4_reward']:.2f}")

    return summary


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline evaluation for arXiv (Abstract Expansion)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="evals/configs/arxiv_pipeline_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="HuggingFace model name (overrides config)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name (overrides config)",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default=None,
        help="Dataset split for evaluation (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (overrides config)",
    )
    parser.add_argument(
        "--num-attempts",
        type=int,
        default=None,
        help="Number of attempts per problem (overrides config)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate (overrides config)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (overrides config)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p sampling (overrides config)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=None,
        help="Print detailed output (overrides config)",
    )
    
    args = parser.parse_args()
    
    # Load config file if it exists
    config = {}
    if os.path.exists(args.config):
        config = load_config(args.config)
    
    # Merge config with command-line args (args take precedence)
    final_config = {
        "model_name": args.model_name or config.get("model", {}).get("name", "Qwen/Qwen3-1.7B"),
        "dataset_name": args.dataset_name or config.get("dataset", {}).get("name", "OpenMLRL/arXiv_abstract"),
        "eval_split": args.eval_split or config.get("dataset", {}).get("eval_split", "val[:1100]"),
        "output_dir": args.output_dir or config.get("output", {}).get("base_dir", "evals/results"),
        "num_attempts": args.num_attempts or config.get("evaluation", {}).get("num_attempts", 1),
        "max_new_tokens": args.max_new_tokens or config.get("evaluation", {}).get("max_new_tokens", 256),
        "temperature": args.temperature or config.get("evaluation", {}).get("temperature", 0.7),
        "top_p": args.top_p or config.get("evaluation", {}).get("top_p", 0.9),
        "verbose": args.verbose if args.verbose is not None else config.get("output", {}).get("verbose", False),
    }
    
    return argparse.Namespace(**final_config)


if __name__ == "__main__":
    print("Starting arXiv pipeline evaluation...")
    args = parse_args()

    evaluate_pipeline(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        eval_split=args.eval_split,
        output_dir=args.output_dir,
        num_attempts=args.num_attempts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        verbose=args.verbose,
    )
