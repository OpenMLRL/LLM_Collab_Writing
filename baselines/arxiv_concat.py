import argparse
import json
import time
from typing import Any, Dict

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from loggers.arxiv_logger import (
    aggregate_arxiv_metrics_for_logging,
    arxiv_combined_reward_logger,
)


class TwoAgentQwenarXivBaseline:
    def __init__(self, model_name="Qwen/Qwen3-1.7B", device="auto"):
        """
        Initialize two Qwen models for arXiv baseline evaluation.

        Args:
            model_name: HuggingFace model name
            device: Device to run on ("auto", "cuda", "cpu")
        """
        self.model_name = model_name
        self.device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        print(f"Loading two instances of model {model_name} on {self.device}...")

        # Load tokenizer (shared between both agents)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load Agent 1 model (Summary Agent)
        self.agent1_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        # Load Agent 2 model (Elaboration Agent)
        self.agent2_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        if self.device == "cpu":
            self.agent1_model = self.agent1_model.to(self.device)
            self.agent2_model = self.agent2_model.to(self.device)

        self.agent1_model.eval()
        self.agent2_model.eval()

        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def background_agent_formatter(self, example: Dict[str, Any]) -> str:
        """
        Formatter for the background agent (Agent 1).
        Focuses on describing research background and motivation from the abstract.
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

    def complementary_agent_formatter(self, example: Dict[str, Any]) -> str:
        """
        Formatter for the complementary agent (Agent 2).
        Provides complementary content to expand the introduction further.
        """
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

    def generate_response(self, model, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate response for a given prompt using specified model."""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Extract only the generated part
        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()

    def evaluate_arxiv_baseline(
        self, num_samples: int = 1000, save_results: bool = True
    ):
        """
        Evaluate arXiv baseline with two agents on test set.

        Args:
            num_samples: Number of test samples to evaluate
            save_results: Whether to save detailed results to JSON
        """
        print(f"Loading arXiv dataset...")
        dataset = load_dataset("LovelyBuggies/arXiv_abstract")
        test_data = dataset["test"]

        # Take first num_samples
        test_samples = test_data.select(range(min(num_samples, len(test_data))))

        print(f"Evaluating on {len(test_samples)} samples...")

        agent1_completions = []  # Agent 1 completions (background)
        agent2_completions = []  # Agent 2 completions (complementary)
        response_times = []
        failed_samples = 0

        for i, sample in enumerate(tqdm(test_samples, desc="Generating responses")):
            try:
                # Generate Agent 1 response (Summary Agent)
                agent1_prompt_text = self.background_agent_formatter(sample)
                agent1_response = self.generate_response(
                    self.agent1_model, agent1_prompt_text, max_new_tokens=256
                )

                start_time = time.time()
                # Generate Agent 2 response (Elaboration Agent)
                agent2_prompt_text = self.complementary_agent_formatter(sample)
                agent2_response = self.generate_response(
                    self.agent2_model, agent2_prompt_text, max_new_tokens=256
                )
                end_time = time.time()

                agent1_completions.append(agent1_response)
                agent2_completions.append(agent2_response)
                response_times.append(end_time - start_time)

            except Exception as e:
                print(f"Failed on sample {i}: {e}")
                failed_samples += 1
                agent1_completions.append("")
                agent2_completions.append("")
                response_times.append(0.0)

        print(f"Generation complete. Failed samples: {failed_samples}")
        print(f"Average response time: {np.mean(response_times):.2f} seconds")

        # Calculate metrics using your reward function
        print("Calculating metrics...")
        metrics_list = arxiv_combined_reward_logger(
            agent1_completions, agent2_completions
        )
        aggregated_metrics = aggregate_arxiv_metrics_for_logging(metrics_list)

        # Add timing metrics
        aggregated_metrics["avg_response_time"] = np.mean(response_times)
        aggregated_metrics["total_time"] = np.sum(response_times)
        aggregated_metrics["failed_samples"] = failed_samples
        aggregated_metrics["success_rate"] = (len(test_samples) - failed_samples) / len(
            test_samples
        )

        # Print results
        print("\n" + "=" * 50)
        print("TWO-AGENT BASELINE EVALUATION RESULTS")
        print("=" * 50)
        print(f"Model: {self.model_name} (2 instances)")
        print(f"Agent 1: Background Agent")
        print(f"Agent 2: Complementary Agent")
        print(f"Samples evaluated: {len(test_samples)}")
        print(f"Success rate: {aggregated_metrics['success_rate']:.1%}")
        print(f"Average response time: {aggregated_metrics['avg_response_time']:.2f}s")
        print(f"Total time: {aggregated_metrics['total_time']:.1f}s")
        print()

        print("REWARD METRICS:")
        print(f"Level 1 (Structure): {aggregated_metrics.get('level1_reward', 0):.3f}")
        print(
            f"Level 2 (Coordination): {aggregated_metrics.get('level2_reward', 0):.3f}"
        )
        print(f"Level 3 (Vocabulary): {aggregated_metrics.get('level3_reward', 0):.3f}")
        print(
            f"Level 4 (Style): {aggregated_metrics.get('jaccard_reward', 0) + aggregated_metrics.get('transition_reward', 0):.3f}"
        )
        print(
            f"Gated Total Reward: {aggregated_metrics.get('gated_total_reward', 0):.3f}"
        )
        print(
            f"Ungated Total Reward: {aggregated_metrics.get('ungated_total_reward', 0):.3f}"
        )
        print()

        print("DETAILED METRICS:")
        for key, value in aggregated_metrics.items():
            if key not in [
                "avg_response_time",
                "total_time",
                "failed_samples",
                "success_rate",
            ]:
                print(f"  {key}: {value:.3f}")

        # Save detailed results
        if save_results:
            results = {
                "model_name": self.model_name,
                "num_samples": len(test_samples),
                "baseline_type": "two_agent_qwen",
                "agent1_role": "background_agent",
                "agent2_role": "complementary_agent",
                "aggregated_metrics": aggregated_metrics,
                "individual_metrics": metrics_list,
                "agent1_completions": agent1_completions,  # Summary Agent
                "agent2_completions": agent2_completions,  # Elaboration Agent
                "response_times": response_times,
            }

            filename = f"arxiv_two_agent_baseline_results_{int(time.time())}.json"
            with open(filename, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nDetailed results saved to: {filename}")

        return aggregated_metrics, metrics_list


def main():
    parser = argparse.ArgumentParser(description="arXiv Two-Agent Baseline Evaluation")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B", help="Model name")
    parser.add_argument(
        "--samples", type=int, default=100, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save detailed results"
    )

    args = parser.parse_args()

    # Initialize baseline evaluator
    evaluator = TwoAgentQwenarXivBaseline(model_name=args.model, device=args.device)

    # Run evaluation
    aggregated_metrics, individual_metrics = evaluator.evaluate_arxiv_baseline(
        num_samples=args.samples, save_results=not args.no_save
    )


if __name__ == "__main__":
    main()
