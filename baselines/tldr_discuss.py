import argparse
import json
import time
from typing import Any, Dict

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from loggers.tldr_logger import (
    aggregate_tldr_metrics_for_logging,
    tldr_combined_reward_logger,
)


class TwoRoundTwoAgentQwenTLDRBaseline:
    def __init__(self, model_name="Qwen/Qwen3-1.7B", device="auto"):
        """
        Initialize two Qwen models for TLDR baseline evaluation with two-round generation.

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

    def summary_agent_formatter_round1(self, example: Dict[str, Any]) -> str:
        """
        Formatter for the summary agent (Agent 1) - Round 1.
        Creates concise, topic-focused summaries that set up key concepts.
        """
        prompt = example.get("prompt", "")

        if not prompt:
            return "Error: No prompt provided."

        prompt_text = f"""Create a concise summary response to this post.

Query:
{prompt}

IMPORTANT INSTRUCTIONS:
- Provide a brief, focused summary in one sentence or a few sentences
- Be factual and informative

Summary:"""

        return prompt_text

    def elaboration_agent_formatter_round1(self, example: Dict[str, Any]) -> str:
        """
        Formatter for the elaboration agent (Agent 2) - Round 1.
        Creates detailed responses that build on Agent 1's summary with proper flow.
        """
        prompt = example.get("prompt", "")

        if not prompt:
            return "Error: No prompt provided."

        prompt_text = f"""Create a detailed summary response to this post.

Original Query:
{prompt}

Create a concise summary response to this query:

IMPORTANT INSTRUCTIONS:
- Use more unique words
- Use some transition words to improve flow

Summary:"""

        return prompt_text

    def summary_agent_formatter_round2(
        self, example: Dict[str, Any], other_agent_completion: str
    ) -> str:
        """
        Formatter for the summary agent (Agent 1) - Round 2.
        Now includes the other agent's completion for reference.
        """
        prompt = example.get("prompt", "")

        if not prompt:
            return "Error: No prompt provided."

        prompt_text = f"""Create a concise summary response to this post.

Query:
{prompt}

Other Agent's Response:
{other_agent_completion}

IMPORTANT INSTRUCTIONS:
- Provide a brief, focused summary in one sentence or a few sentences
- Your response should be about 2-3 times shorter than the other agent's response
- You can refer to the other agent's response to improve your own response

Summary:"""

        return prompt_text

    def elaboration_agent_formatter_round2(
        self, example: Dict[str, Any], other_agent_completion: str
    ) -> str:
        """
        Formatter for the elaboration agent (Agent 2) - Round 2.
        Now includes the other agent's completion for reference.
        """
        prompt = example.get("prompt", "")

        if not prompt:
            return "Error: No prompt provided."

        prompt_text = f"""Create a detailed summary response to this post.

Original Query:
{prompt}

Other Agent's Response:
{other_agent_completion}

Create a concise summary response to this query:

IMPORTANT INSTRUCTIONS:
- Use more unique words
- Use some transition words to improve flow
- You can refer to the other agent's response to improve your own response
- Your response should be about 2-3 times longer than the other agent's response

Summary:"""

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

    def evaluate_tldr_two_round(
        self, num_samples: int = 1000, save_results: bool = True
    ):
        """
        Evaluate TLDR with two-round generation where agents can see each other's completions in round 2.

        Args:
            num_samples: Number of test samples to evaluate
            save_results: Whether to save detailed results to JSON
        """
        print(f"Loading TLDR dataset...")
        dataset = load_dataset("trl-lib/tldr")
        test_data = dataset["test"]

        # Take first num_samples
        test_samples = test_data.select(range(min(num_samples, len(test_data))))

        print(f"Evaluating on {len(test_samples)} samples with two-round generation...")

        # Round 1 completions
        agent1_round1_completions = []
        agent2_round1_completions = []

        # Round 2 completions (final ones to evaluate)
        agent1_round2_completions = []
        agent2_round2_completions = []

        response_times = []
        failed_samples = 0

        for i, sample in enumerate(tqdm(test_samples, desc="Two-round generation")):
            try:

                # ROUND 1: Generate initial responses in parallel
                # Agent 1 Round 1
                agent1_round1_prompt_text = self.summary_agent_formatter_round1(sample)
                agent1_round1_response = self.generate_response(
                    self.agent1_model, agent1_round1_prompt_text, max_new_tokens=256
                )

                # Agent 2 Round 1
                start_time = time.time()
                agent2_round1_prompt_text = self.elaboration_agent_formatter_round1(
                    sample
                )
                agent2_round1_response = self.generate_response(
                    self.agent2_model, agent2_round1_prompt_text, max_new_tokens=256
                )
                end_time = time.time()
                round1_time = end_time - start_time

                # ROUND 2: Generate final responses with knowledge of other agent's completion
                # Agent 1 Round 2 (knows Agent 2's Round 1 response)
                agent1_round2_prompt_text = self.summary_agent_formatter_round2(
                    sample, agent2_round1_response
                )
                agent1_round2_response = self.generate_response(
                    self.agent1_model, agent1_round2_prompt_text, max_new_tokens=256
                )

                start_time = time.time()
                # Agent 2 Round 2 (knows Agent 1's Round 1 response)
                agent2_round2_prompt_text = self.elaboration_agent_formatter_round2(
                    sample, agent1_round1_response
                )
                agent2_round2_response = self.generate_response(
                    self.agent2_model, agent2_round2_prompt_text, max_new_tokens=256
                )
                end_time = time.time()
                round2_time = end_time - start_time

                # Store all completions
                agent1_round1_completions.append(agent1_round1_response)
                agent2_round1_completions.append(agent2_round1_response)
                agent1_round2_completions.append(agent1_round2_response)
                agent2_round2_completions.append(agent2_round2_response)
                response_times.append(round1_time + round2_time)

            except Exception as e:
                print(f"Failed on sample {i}: {e}")
                failed_samples += 1
                agent1_round1_completions.append("")
                agent2_round1_completions.append("")
                agent1_round2_completions.append("")
                agent2_round2_completions.append("")
                response_times.append(0.0)

        print(f"Generation complete. Failed samples: {failed_samples}")
        print(f"Average response time: {np.mean(response_times):.2f} seconds")

        # Calculate metrics using ONLY the Round 2 completions
        print("Calculating metrics for Round 2 completions...")
        metrics_list = tldr_combined_reward_logger(
            agent1_round2_completions, agent2_round2_completions
        )
        aggregated_metrics = aggregate_tldr_metrics_for_logging(metrics_list)

        # Add timing metrics
        aggregated_metrics["avg_response_time"] = np.mean(response_times)
        aggregated_metrics["total_time"] = np.sum(response_times)
        aggregated_metrics["failed_samples"] = failed_samples
        aggregated_metrics["success_rate"] = (len(test_samples) - failed_samples) / len(
            test_samples
        )

        # Print results
        print("\n" + "=" * 50)
        print("TWO-ROUND TWO-AGENT EVALUATION RESULTS")
        print("=" * 50)
        print(f"Model: {self.model_name} (2 instances)")
        print(f"Agent 1: Summary Agent (concise summaries)")
        print(f"Agent 2: Elaboration Agent (detailed summaries)")
        print(
            f"Generation: Two rounds (Round 2 agents see each other's Round 1 outputs)"
        )
        print(f"Evaluation: Based on Round 2 completions only")
        print(f"Samples evaluated: {len(test_samples)}")
        print(f"Success rate: {aggregated_metrics['success_rate']:.1%}")
        print(f"Average response time: {aggregated_metrics['avg_response_time']:.2f}s")
        print(f"Total time: {aggregated_metrics['total_time']:.1f}s")
        print()

        print("REWARD METRICS (Round 2 only):")
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
                "baseline_type": "two_round_two_agent_qwen",
                "agent1_role": "summary_agent",
                "agent2_role": "elaboration_agent",
                "evaluation_round": "round_2_only",
                "aggregated_metrics": aggregated_metrics,
                "individual_metrics": metrics_list,
                "agent1_round1_completions": agent1_round1_completions,
                "agent2_round1_completions": agent2_round1_completions,
                "agent1_round2_completions": agent1_round2_completions,  # These are evaluated
                "agent2_round2_completions": agent2_round2_completions,  # These are evaluated
                "response_times": response_times,
            }

            filename = f"tldr_two_round_two_agent_results_{int(time.time())}.json"
            with open(filename, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nDetailed results saved to: {filename}")

        return aggregated_metrics, metrics_list


def main():
    parser = argparse.ArgumentParser(description="TLDR Two-Round Two-Agent Evaluation")
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
    evaluator = TwoRoundTwoAgentQwenTLDRBaseline(
        model_name=args.model, device=args.device
    )

    # Run evaluation
    aggregated_metrics, individual_metrics = evaluator.evaluate_tldr_two_round(
        num_samples=args.samples, save_results=not args.no_save
    )


if __name__ == "__main__":
    main()
