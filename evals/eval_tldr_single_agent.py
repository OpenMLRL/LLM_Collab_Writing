#!/usr/bin/env python3
"""
Single-Agent Evaluation Script for TLDR (Summarization)

This script runs inference-only evaluation for the Single-Agent configuration
matching the baseline approach that achieved 36.7% Return.

Key features (matching baseline):
- Single 4B model (Qwen3-4B)
- 260 max_new_tokens (matching baseline, not 512)
- Explicit [PARAGRAPH_SPLIT] delimiter requirement in prompt
- Detailed instructions about paragraph length (10-200 tokens) and transition words
- Fallback splitting with 2.0-3.0x ratio if delimiter missing
- 260 tokens ensures both paragraphs stay within reward function's 8-256 token range

Metrics tracked:
- Time: Wall-clock generation time (seconds)
- Cost: Number of tokens produced
- Score: Reward from tldr_combined_reward (split output)

Output:
- CSV file with per-sample metrics
- JSON file with aggregated statistics
"""

import argparse
import csv
import json
import os
import random
import re
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

import rewards.tldr_rewards as tldr_rewards_module
from loggers.tldr_logger import tldr_combined_reward_logger
from rewards.tldr_rewards import tldr_combined_reward


def single_agent_formatter(example: Dict[str, Any]) -> str:
    """
    Formatter for single-agent TLDR generation.
    Combines both agent tasks into one prompt.
    Prompt style matches multi-agent prompts as closely as possible.
    Uses natural paragraph breaks (blank line) instead of special delimiters.
    """
    prompt = example.get("prompt", "")

    if not prompt:
        return "Error: No prompt provided."

    prompt_text = f"""Please provide a summary of this Reddit post in exactly two paragraphs:

{prompt}

Instructions:
- First paragraph: Provide a concise summary of the main points
- Second paragraph: Expand on the summary with more details, using more unique vocabulary words, include as many categories of transition words as possible to improve flow, and make it 2-3 times longer than the first paragraph in terms of character count, while maintaining a consistent style

IMPORTANT REQUIREMENTS - FOLLOW EXACTLY:
- No paragraph should be less than 10 tokens or more than 200 tokens
- Use EXACTLY this delimiter between paragraphs: [PARAGRAPH_SPLIT]

Summary:
Paragraph 1:"""

    return prompt_text


def split_response_into_paragraphs(response: str) -> Tuple[str, str]:
    """
    Split the response into two paragraphs using the special delimiter.
    Matches baseline approach: prioritize delimiter, fallback to ratio-based split.
    """
    # Clean up the response
    response = response.strip()

    # Look for the special delimiter (primary method)
    delimiter = "[PARAGRAPH_SPLIT]"
    if delimiter in response:
        # Split on the delimiter
        paragraphs = response.split(delimiter, 1)  # Split only on first occurrence
        para1 = paragraphs[0].strip()
        para2 = paragraphs[1].strip() if len(paragraphs) > 1 else ""
        return _clean_paragraph_prefixes(para1, para2)
    else:
        # Fallback: split to make second paragraph 2.0-3.0x longer than first (matching baseline)
        ratio = random.uniform(2.0, 3.0)
        split_point = int(len(response) / (1 + ratio))
        # Find nearest space to avoid breaking words
        while split_point < len(response) and response[split_point] not in ' \n':
            split_point += 1
        para1 = response[:split_point].strip()
        para2 = response[split_point:].strip()
        return _clean_paragraph_prefixes(para1, para2)


def _clean_paragraph_prefixes(para1: str, para2: str) -> Tuple[str, str]:
    """
    Clean up common prefixes from paragraphs.
    """
    # Common prefixes to remove
    prefixes = [
        r"^(?:First paragraph|Paragraph 1|Part 1):\s*",
        r"^(?:Second paragraph|Paragraph 2|Part 2):\s*",
        r"^1\.\s+",
        r"^2\.\s+",
    ]
    
    for pattern in prefixes:
        para1 = re.sub(pattern, "", para1, flags=re.IGNORECASE)
        para2 = re.sub(pattern, "", para2, flags=re.IGNORECASE)
    
    return para1.strip(), para2.strip()


def generate_completion(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 260,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_return_sequences: int = 1,
) -> Tuple[List[str], float, List[int]]:
    """
    Generate completions and return (texts, time_seconds, token_counts).
    """
    device = model.device

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
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


def evaluate_single_agent(
    model_name: str,
    dataset_name: str,
    eval_split: str,
    output_dir: str,
    num_attempts: int = 1,
    max_new_tokens: int = 260,
    temperature: float = 0.7,
    top_p: float = 0.9,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run single-agent evaluation on TLDR dataset.
    
    Matches baseline approach that achieved 36.7% Return:
    - Uses explicit [PARAGRAPH_SPLIT] delimiter in prompt
    - 260 max_new_tokens (matching baseline, not 512)
    - Explicit instructions about paragraph length (10-200 tokens) and transition words
    - 260 tokens ensures both paragraphs stay within reward function's 8-256 token range

    Args:
        model_name: HuggingFace model name (should be 4B model for TLDR)
        dataset_name: Dataset name
        eval_split: Dataset split for evaluation
        output_dir: Directory to save results
        num_attempts: Number of attempts per problem
        max_new_tokens: Maximum tokens to generate (260 matching baseline)
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

    # Load single 7B model
    print("Loading single-agent model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Load dataset
    print(f"Loading dataset: {dataset_name} split={eval_split}")
    dataset = load_dataset(dataset_name, split=eval_split)
    print(f"Evaluation dataset size: {len(dataset)}")

    # Storage for results
    all_results = []

    # Suppress verbose output from reward function
    tldr_rewards_module.VERBOSE = False

    # Process each problem
    for prob_idx, item in enumerate(dataset):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Problem {prob_idx + 1}/{len(dataset)}")

        # Format prompt for single agent
        prompt = single_agent_formatter(item)

        # Generate num_attempts attempts for this problem
        for attempt_idx in range(num_attempts):
            if verbose:
                print(f"  Attempt {attempt_idx + 1}/{num_attempts}")

            # Generate completion
            completions, gen_time, token_counts = generate_completion(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=1,
            )
            completion = completions[0]
            tokens = token_counts[0]

            # Split into two paragraphs
            para1, para2 = split_response_into_paragraphs(completion)

            # Compute reward using same function as multi-agent
            rewards = tldr_combined_reward(
                [para1],
                [para2],
            )
            score = rewards[0] if rewards else 0.0

            # Get detailed metrics using logger
            metrics = tldr_combined_reward_logger(
                [para1],
                [para2],
            )
            metric = metrics[0] if metrics else {}

            # Store attempt result
            attempt_result = {
                "problem_id": prob_idx,
                "attempt_id": attempt_idx,
                "time_s": round(gen_time, 4),
                "tokens": tokens,
                "score": round(score, 4),
                "level1_reward": round(metric.get("level1_reward", 0.0), 4),
                "level2_reward": round(metric.get("level2_reward", 0.0), 4),
                "level3_reward": round(metric.get("level3_reward", 0.0), 4),
                "level4_reward": round(metric.get("level4_reward", 0.0), 4),
                "length_ratio": round(metric.get("length_ratio", 0.0), 4),
                "unique_words_ratio": round(metric.get("unique_words_ratio", 0.0), 4),
                "para1_len": len(para1),
                "para2_len": len(para2),
            }
            all_results.append(attempt_result)

            # Print progress
            print(f"{'='*60}")
            print(f"Problem {prob_idx + 1}/{len(dataset)}")
            print(f"Attempt {attempt_idx + 1}/{num_attempts}")
            print(f"  Time: {gen_time:.2f}s")
            print(f"  Tokens: {tokens}")
            print(f"  Score: {score:.2f}")
            print(f"  Length ratio: {metric.get('length_ratio', 0):.2f}")
            print(f"{'='*60}")

    # Compute aggregated statistics
    num_problems = len(dataset)

    aggregated = {
        "avg_time": sum(r["time_s"] for r in all_results) / len(all_results),
        "avg_tokens": sum(r["tokens"] for r in all_results) / len(all_results),
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
    csv_path = os.path.join(output_dir, f"single_agent_results_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nCSV results saved to: {csv_path}")

    # Save JSON summary
    summary = {
        "config": "single_agent",
        "domain": "tldr",
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

    json_path = os.path.join(output_dir, f"single_agent_summary_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary saved to: {json_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SINGLE-AGENT EVALUATION SUMMARY (TLDR)")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Problems evaluated: {num_problems}")
    print(f"Attempts per problem: {num_attempts}")
    print(f"\nAggregated Metrics:")
    print(f"  Avg Time: {aggregated['avg_time']:.2f}s")
    print(f"  Avg Tokens: {aggregated['avg_tokens']:.1f}")
    print(f"  Avg Score: {aggregated['avg_score']:.2f}")
    print(f"  Avg Length Ratio: {aggregated['avg_length_ratio']:.2f}")
    print(f"  Avg Unique Words Ratio: {aggregated['avg_unique_words_ratio']:.2f}")
    print(f"{'='*60}")

    return summary


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-agent evaluation for TLDR (Summarization)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="evals/configs/tldr_single_agent_config.yaml",
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
        "model_name": args.model_name or config.get("model", {}).get("name", "Qwen/Qwen3-4B"),
        "dataset_name": args.dataset_name or config.get("dataset", {}).get("name", "trl-lib/tldr"),
        "eval_split": args.eval_split or config.get("dataset", {}).get("eval_split", "test[:1100]"),
        "output_dir": args.output_dir or config.get("output", {}).get("base_dir", "evals/results"),
        "num_attempts": args.num_attempts or config.get("evaluation", {}).get("num_attempts", 1),
        "max_new_tokens": args.max_new_tokens or config.get("evaluation", {}).get("max_new_tokens", 260),
        "temperature": args.temperature or config.get("evaluation", {}).get("temperature", 0.7),
        "top_p": args.top_p or config.get("evaluation", {}).get("top_p", 0.9),
        "verbose": args.verbose if args.verbose is not None else config.get("output", {}).get("verbose", False),
    }
    
    return argparse.Namespace(**final_config)


if __name__ == "__main__":
    print("Starting TLDR single-agent evaluation...")
    args = parse_args()

    evaluate_single_agent(
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
