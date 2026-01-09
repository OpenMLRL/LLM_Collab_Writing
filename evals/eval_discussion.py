#!/usr/bin/env python3
"""
Discussion Evaluation Script for TLDR (Summarization)

This script runs inference-only evaluation for the Discussion (2-turn)
configuration where both agents generate in Turn 1, then refine in Turn 2
after seeing each other's outputs.

Turn 1: Parallel (no communication)
- Agent 1 generates concise summary
- Agent 2 generates detailed summary (independent)

Turn 2: Bidirectional refinement
- Agent 1 sees: prompt + summary(T1) + elaboration(T1) → generates refined summary
- Agent 2 sees: prompt + summary(T1) + elaboration(T1) → generates refined elaboration

Final evaluation uses Turn 2 outputs.

Metrics tracked per agent:
- Time: Wall-clock generation time (seconds) for both turns
- Cost: Number of tokens produced (both turns)
- Score: Reward from tldr_combined_reward (Turn 2 outputs)

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

import rewards.tldr_rewards as tldr_rewards_module
from loggers.tldr_logger import tldr_combined_reward_logger
from rewards.tldr_rewards import tldr_combined_reward


# =============================================================================
# TURN 1 FORMATTERS (same as Parallel)
# =============================================================================

def summary_agent_formatter(example: Dict[str, Any]) -> str:
    """Formatter for the summary agent (Agent 1) - Turn 1."""
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
    """Formatter for the elaboration agent (Agent 2) - Turn 1."""
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


# =============================================================================
# TURN 2 FORMATTERS (see each other's outputs)
# =============================================================================

def summary_turn2_formatter(example: Dict[str, Any], summary_turn1: str, elaboration_turn1: str) -> str:
    """Formatter for Agent 1 Turn 2 - sees both summary and elaboration from Turn 1."""
    prompt = example.get("prompt", "")

    prompt_text = f"""Refine your concise summary based on the detailed version.

Original Query:
{prompt}

Your previous concise summary:
{summary_turn1}

The detailed summary from another agent:
{elaboration_turn1}

IMPORTANT INSTRUCTIONS:
- Output ONLY the refined concise summary
- Ensure it provides a good foundation for the detailed version
- Keep it brief and focused
"""

    return prompt_text


def elaboration_turn2_formatter(example: Dict[str, Any], summary_turn1: str, elaboration_turn1: str) -> str:
    """Formatter for Agent 2 Turn 2 - sees both summary and elaboration from Turn 1."""
    prompt = example.get("prompt", "")

    prompt_text = f"""Refine your detailed summary now that you can see the concise version.

Original Query:
{prompt}

The concise summary:
{summary_turn1}

Your previous detailed summary:
{elaboration_turn1}

IMPORTANT INSTRUCTIONS:
- Build upon and expand the concise summary
- Use more unique words than the concise summary
- Use transition words to improve flow
- Make your response 2-3x longer than the concise summary
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


def evaluate_discussion(
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
    Run discussion (2-turn) evaluation on TLDR dataset.

    Turn 1: Parallel (both generate independently)
    Turn 2: Both see each other's outputs and refine

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

    # Load two separate model instances for agents
    print("Loading Agent 1 (summary) model...")
    model_agent1 = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model_agent1.eval()

    print("Loading Agent 2 (elaboration) model...")
    model_agent2 = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
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
    tldr_rewards_module.VERBOSE = False

    # Process each problem
    for prob_idx, item in enumerate(dataset):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Problem {prob_idx + 1}/{len(dataset)}")

        # Format Turn 1 prompts (same as Parallel)
        prompt_agent1_t1 = summary_agent_formatter(item)
        prompt_agent2_t1 = elaboration_agent_formatter(item)

        # Generate num_attempts attempts for this problem
        for attempt_idx in range(num_attempts):
            if verbose:
                print(f"  Attempt {attempt_idx + 1}/{num_attempts}")

            # =================================================================
            # TURN 1: Parallel (no communication)
            # =================================================================
            if verbose:
                print(f"    Turn 1: Parallel generation...")

            # Agent 1: Generate concise summary
            completions1_t1, time1_t1, tokens1_t1 = generate_completion(
                model_agent1,
                tokenizer,
                prompt_agent1_t1,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=1,
            )
            summary_turn1 = completions1_t1[0]
            summary_tokens_t1 = tokens1_t1[0]

            # Agent 2: Generate detailed summary (independent)
            completions2_t1, time2_t1, tokens2_t1 = generate_completion(
                model_agent2,
                tokenizer,
                prompt_agent2_t1,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=1,
            )
            elaboration_turn1 = completions2_t1[0]
            elaboration_tokens_t1 = tokens2_t1[0]

            # =================================================================
            # TURN 2: Both see each other's outputs
            # =================================================================
            if verbose:
                print(f"    Turn 2: Refinement with cross-visibility...")

            # Agent 1 Turn 2: Sees summary_turn1 + elaboration_turn1
            prompt_agent1_t2 = summary_turn2_formatter(item, summary_turn1, elaboration_turn1)
            completions1_t2, time1_t2, tokens1_t2 = generate_completion(
                model_agent1,
                tokenizer,
                prompt_agent1_t2,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=1,
            )
            summary_turn2 = completions1_t2[0]
            summary_tokens_t2 = tokens1_t2[0]

            # Agent 2 Turn 2: Sees summary_turn1 + elaboration_turn1
            prompt_agent2_t2 = elaboration_turn2_formatter(item, summary_turn1, elaboration_turn1)
            completions2_t2, time2_t2, tokens2_t2 = generate_completion(
                model_agent2,
                tokenizer,
                prompt_agent2_t2,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=1,
            )
            elaboration_turn2 = completions2_t2[0]
            elaboration_tokens_t2 = tokens2_t2[0]

            # =================================================================
            # Evaluate Turn 2 outputs
            # =================================================================
            rewards = tldr_combined_reward(
                [summary_turn2],
                [elaboration_turn2],
            )
            score = rewards[0] if rewards else 0.0

            # Get detailed metrics using logger
            metrics = tldr_combined_reward_logger(
                [summary_turn2],
                [elaboration_turn2],
            )
            metric = metrics[0] if metrics else {}

            # Calculate totals (both turns)
            total_time_agent1 = time1_t1 + time1_t2
            total_time_agent2 = time2_t1 + time2_t2
            total_tokens_agent1 = summary_tokens_t1 + summary_tokens_t2
            total_tokens_agent2 = elaboration_tokens_t1 + elaboration_tokens_t2

            # Store attempt result
            attempt_result = {
                "problem_id": prob_idx,
                "attempt_id": attempt_idx,
                # Turn 1 metrics
                "agent1_time_t1_s": round(time1_t1, 4),
                "agent1_tokens_t1": summary_tokens_t1,
                "agent2_time_t1_s": round(time2_t1, 4),
                "agent2_tokens_t1": elaboration_tokens_t1,
                # Turn 2 metrics
                "agent1_time_t2_s": round(time1_t2, 4),
                "agent1_tokens_t2": summary_tokens_t2,
                "agent2_time_t2_s": round(time2_t2, 4),
                "agent2_tokens_t2": elaboration_tokens_t2,
                # Totals
                "agent1_time_total_s": round(total_time_agent1, 4),
                "agent1_tokens_total": total_tokens_agent1,
                "agent2_time_total_s": round(total_time_agent2, 4),
                "agent2_tokens_total": total_tokens_agent2,
                "total_time_s": round(total_time_agent1 + total_time_agent2, 4),
                "total_tokens": total_tokens_agent1 + total_tokens_agent2,
                # Score (from Turn 2)
                "score": round(score, 4),
                "level1_reward": round(metric.get("level1_reward", 0.0), 4),
                "level2_reward": round(metric.get("level2_reward", 0.0), 4),
                "level3_reward": round(metric.get("level3_reward", 0.0), 4),
                "level4_reward": round(metric.get("level4_reward", 0.0), 4),
                "length_ratio": round(metric.get("length_ratio", 0.0), 4),
                "unique_words_ratio": round(metric.get("unique_words_ratio", 0.0), 4),
            }
            all_results.append(attempt_result)

            # Print progress
            print(f"{'='*60}")
            print(f"Problem {prob_idx + 1}/{len(dataset)}")
            print(f"Attempt {attempt_idx + 1}/{num_attempts}")
            print(f"  T1 Time: {time1_t1:.2f}s + {time2_t1:.2f}s")
            print(f"  T2 Time: {time1_t2:.2f}s + {time2_t2:.2f}s")
            print(f"  Total Time: {total_time_agent1 + total_time_agent2:.2f}s")
            print(f"  Total Tokens: {total_tokens_agent1} + {total_tokens_agent2} = {total_tokens_agent1 + total_tokens_agent2}")
            print(f"  Score: {score:.2f}")
            print(f"{'='*60}")

    # Compute aggregated statistics
    num_problems = len(dataset)

    aggregated = {
        # Turn 1 averages
        "avg_time_agent1_t1": sum(r["agent1_time_t1_s"] for r in all_results) / len(all_results),
        "avg_time_agent2_t1": sum(r["agent2_time_t1_s"] for r in all_results) / len(all_results),
        "avg_tokens_agent1_t1": sum(r["agent1_tokens_t1"] for r in all_results) / len(all_results),
        "avg_tokens_agent2_t1": sum(r["agent2_tokens_t1"] for r in all_results) / len(all_results),
        # Turn 2 averages
        "avg_time_agent1_t2": sum(r["agent1_time_t2_s"] for r in all_results) / len(all_results),
        "avg_time_agent2_t2": sum(r["agent2_time_t2_s"] for r in all_results) / len(all_results),
        "avg_tokens_agent1_t2": sum(r["agent1_tokens_t2"] for r in all_results) / len(all_results),
        "avg_tokens_agent2_t2": sum(r["agent2_tokens_t2"] for r in all_results) / len(all_results),
        # Totals
        "avg_time_agent1_total": sum(r["agent1_time_total_s"] for r in all_results) / len(all_results),
        "avg_time_agent2_total": sum(r["agent2_time_total_s"] for r in all_results) / len(all_results),
        "avg_time_total": sum(r["total_time_s"] for r in all_results) / len(all_results),
        "avg_tokens_agent1_total": sum(r["agent1_tokens_total"] for r in all_results) / len(all_results),
        "avg_tokens_agent2_total": sum(r["agent2_tokens_total"] for r in all_results) / len(all_results),
        "avg_tokens_total": sum(r["total_tokens"] for r in all_results) / len(all_results),
        # Score
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
    csv_path = os.path.join(output_dir, f"discussion_results_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nCSV results saved to: {csv_path}")

    # Save JSON summary
    summary = {
        "config": "discussion",
        "domain": "tldr",
        "model": model_name,
        "dataset": dataset_name,
        "eval_split": eval_split,
        "num_problems": num_problems,
        "num_attempts": num_attempts,
        "num_turns": 2,
        "hyperparameters": {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
        },
        "timestamp": timestamp,
        "aggregated": {k: round(v, 4) for k, v in aggregated.items()},
    }

    json_path = os.path.join(output_dir, f"discussion_summary_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary saved to: {json_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("DISCUSSION EVALUATION SUMMARY (TLDR)")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Problems evaluated: {num_problems}")
    print(f"Attempts per problem: {num_attempts}")
    print(f"Turns: 2 (Parallel → Refinement)")
    print(f"\nTurn 1 (Parallel) Metrics:")
    print(f"  Avg Time (Agent 1): {aggregated['avg_time_agent1_t1']:.2f}s")
    print(f"  Avg Time (Agent 2): {aggregated['avg_time_agent2_t1']:.2f}s")
    print(f"  Avg Tokens (Agent 1): {aggregated['avg_tokens_agent1_t1']:.1f}")
    print(f"  Avg Tokens (Agent 2): {aggregated['avg_tokens_agent2_t1']:.1f}")
    print(f"\nTurn 2 (Refinement) Metrics:")
    print(f"  Avg Time (Agent 1): {aggregated['avg_time_agent1_t2']:.2f}s")
    print(f"  Avg Time (Agent 2): {aggregated['avg_time_agent2_t2']:.2f}s")
    print(f"  Avg Tokens (Agent 1): {aggregated['avg_tokens_agent1_t2']:.1f}")
    print(f"  Avg Tokens (Agent 2): {aggregated['avg_tokens_agent2_t2']:.1f}")
    print(f"\nTotal Metrics:")
    print(f"  Avg Time (Total):   {aggregated['avg_time_total']:.2f}s")
    print(f"  Avg Tokens (Total): {aggregated['avg_tokens_total']:.1f}")
    print(f"  Avg Score: {aggregated['avg_score']:.2f}")
    print(f"  Avg Length Ratio: {aggregated['avg_length_ratio']:.2f}")
    print(f"  Avg Unique Words Ratio: {aggregated['avg_unique_words_ratio']:.2f}")

    return summary


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Discussion (2-turn) evaluation for TLDR (Summarization)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="evals/configs/discussion_config.yaml",
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
        "dataset_name": args.dataset_name or config.get("dataset", {}).get("name", "trl-lib/tldr"),
        "eval_split": args.eval_split or config.get("dataset", {}).get("eval_split", "test[:1100]"),
        "output_dir": args.output_dir or config.get("output", {}).get("base_dir", "evals/results"),
        "num_attempts": args.num_attempts or config.get("evaluation", {}).get("num_attempts", 1),
        "max_new_tokens": args.max_new_tokens or config.get("evaluation", {}).get("max_new_tokens", 256),
        "temperature": args.temperature or config.get("evaluation", {}).get("temperature", 0.7),
        "top_p": args.top_p or config.get("evaluation", {}).get("top_p", 0.9),
        "verbose": args.verbose if args.verbose is not None else config.get("output", {}).get("verbose", False),
    }
    
    return argparse.Namespace(**final_config)


if __name__ == "__main__":
    print("Starting TLDR discussion (2-turn) evaluation...")
    args = parse_args()

    evaluate_discussion(
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
