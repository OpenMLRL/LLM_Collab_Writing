import argparse
import json
import re
import time

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from loggers.tldr_logger import (
    aggregate_tldr_metrics_for_logging,
    tldr_combined_reward_logger,
)


class QwenTLDRTwoParagraphBaseline:
    def __init__(self, model_name="Qwen/Qwen3-4B", device="auto"):
        """
        Initialize the Qwen model for TLDR two-paragraph baseline evaluation.

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

        print(f"Loading model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()

        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def create_prompt(self, post_text: str) -> str:
        """
        Create prompt for TLDR task that instructs the model to generate two paragraphs.

        Args:
            post_text: The reddit post text to summarize
        """
        prompt = f"""Please provide a summary of this Reddit post in exactly two paragraphs:

{post_text}

Instructions:
- First paragraph: Provide a concise summary of the main points
- Second paragraph: Expand on the summary with more details, using more unique vocabulary words, include as many categories of transition words as possible to improve flow, and make it 2-3 times longer than the first paragraph in terms of character count, while maintaining a consistent style

IMPORTANT REQUIREMENTS - FOLLOW EXACTLY:
- No paragraph should be less than 10 tokens or more than 200 tokens
- Use EXACTLY this delimiter between paragraphs: [PARAGRAPH_SPLIT]

Summary:
Paragraph 1:"""
        return prompt

    def generate_response(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate response for a given prompt."""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
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

    def split_paragraphs(self, response: str) -> tuple[str, str]:
        """
        Split the response into two paragraphs using the special delimiter.
        If delimiter not found, split from the middle.

        Args:
            response: The generated response containing two paragraphs

        Returns:
            tuple: (paragraph1, paragraph2)
        """
        # Clean up the response
        response = response.strip()

        # Look for the special delimiter
        delimiter = "[PARAGRAPH_SPLIT]"
        if delimiter in response:
            # Split on the delimiter
            paragraphs = response.split(delimiter, 1)  # Split only on first occurrence
            para1 = paragraphs[0].strip()
            para2 = paragraphs[1].strip()
        else:
            # Print warning and split to make second paragraph 1.0-1.3x longer than first
            print(
                f"Delimiter '{delimiter}' not found in response, splitting with random ratio"
            )
            import random

            ratio = random.uniform(2.0, 3.0)
            split_point = int(len(response) / (1 + ratio))
            para1 = response[:split_point].strip()
            para2 = response[split_point:].strip()

        # Remove any remaining "Paragraph 1:" or similar prefixes
        para1 = re.sub(r"^Paragraph 1:\s*", "", para1, flags=re.IGNORECASE)
        para2 = re.sub(r"^Paragraph 2:\s*", "", para2, flags=re.IGNORECASE)

        return para1, para2

    def evaluate_tldr_baseline(
        self, num_samples: int = 1000, save_results: bool = True
    ):
        """
        Evaluate TLDR two-paragraph baseline on test set.

        Args:
            num_samples: Number of test samples to evaluate
            save_results: Whether to save detailed results to JSON
        """
        print(f"Loading TLDR dataset...")
        dataset = load_dataset("trl-lib/tldr")
        test_data = dataset["test"]

        # Take first num_samples
        test_samples = test_data.select(range(min(num_samples, len(test_data))))

        print(f"Evaluating on {len(test_samples)} samples...")

        paragraph1_completions = []  # Agent 1 completions (first paragraph)
        paragraph2_completions = []  # Agent 2 completions (second paragraph)
        full_responses = []  # Store full responses for analysis
        response_times = []
        failed_samples = 0
        split_failures = 0

        for i, sample in enumerate(tqdm(test_samples, desc="Generating responses")):
            post_text = sample["prompt"]

            try:
                # Generate two-paragraph summary
                prompt = self.create_prompt(post_text)
                start_time = time.time()
                full_response = self.generate_response(prompt, max_new_tokens=260)
                end_time = time.time()

                # Split into two paragraphs
                try:
                    para1, para2 = self.split_paragraphs(full_response)
                    paragraph1_completions.append(para1)
                    paragraph2_completions.append(para2)
                    full_responses.append(full_response)
                    response_times.append(end_time - start_time)
                except Exception as e:
                    print(f"Failed to split paragraphs for sample {i}: {e}")
                    split_failures += 1
                    # Use fallback - treat entire response as paragraph 2, empty paragraph 1
                    paragraph1_completions.append("")
                    paragraph2_completions.append(full_response)
                    full_responses.append(full_response)
                    response_times.append(end_time - start_time)

            except Exception as e:
                print(f"Failed on sample {i}: {e}")
                failed_samples += 1
                paragraph1_completions.append("")
                paragraph2_completions.append("")
                full_responses.append("")
                response_times.append(0.0)

        print(f"Generation complete. Failed samples: {failed_samples}")
        print(f"Paragraph split failures: {split_failures}")
        print(f"Average response time: {np.mean(response_times):.2f} seconds")

        # Calculate metrics using your reward function
        # completions1 = first paragraphs, completions2 = second paragraphs
        print("Calculating metrics...")
        metrics_list = tldr_combined_reward_logger(
            paragraph1_completions, paragraph2_completions
        )
        aggregated_metrics = aggregate_tldr_metrics_for_logging(metrics_list)

        # Add timing and analysis metrics
        aggregated_metrics["avg_response_time"] = np.mean(response_times)
        aggregated_metrics["total_time"] = np.sum(response_times)
        aggregated_metrics["failed_samples"] = failed_samples
        aggregated_metrics["split_failures"] = split_failures
        aggregated_metrics["success_rate"] = (len(test_samples) - failed_samples) / len(
            test_samples
        )

        # Calculate paragraph length statistics
        para1_lengths = [len(p.split()) for p in paragraph1_completions if p]
        para2_lengths = [len(p.split()) for p in paragraph2_completions if p]

        if para1_lengths and para2_lengths:
            aggregated_metrics["avg_para1_length"] = np.mean(para1_lengths)
            aggregated_metrics["avg_para2_length"] = np.mean(para2_lengths)
            aggregated_metrics["length_ratio_para2_to_para1"] = np.mean(
                para2_lengths
            ) / np.mean(para1_lengths)

        # Print results
        print("\n" + "=" * 50)
        print("TWO-PARAGRAPH BASELINE EVALUATION RESULTS")
        print("=" * 50)
        print(f"Model: {self.model_name}")
        print(f"Samples evaluated: {len(test_samples)}")
        print(f"Success rate: {aggregated_metrics['success_rate']:.1%}")
        print(f"Split failures: {split_failures}")
        print(f"Average response time: {aggregated_metrics['avg_response_time']:.2f}s")
        print(f"Total time: {aggregated_metrics['total_time']:.1f}s")
        print()

        if para1_lengths and para2_lengths:
            print("PARAGRAPH LENGTH ANALYSIS:")
            print(
                f"Average Paragraph 1 length: {aggregated_metrics['avg_para1_length']:.1f} words"
            )
            print(
                f"Average Paragraph 2 length: {aggregated_metrics['avg_para2_length']:.1f} words"
            )
            print(
                f"Length ratio (Para2/Para1): {aggregated_metrics['length_ratio_para2_to_para1']:.2f}"
            )
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
                "split_failures",
                "avg_para1_length",
                "avg_para2_length",
                "length_ratio_para2_to_para1",
            ]:
                print(f"  {key}: {value:.3f}")

        # Save detailed results
        if save_results:
            results = {
                "model_name": self.model_name,
                "num_samples": len(test_samples),
                "baseline_type": "two_paragraph_single_agent",
                "aggregated_metrics": aggregated_metrics,
                "individual_metrics": metrics_list,
                "paragraph1_completions": paragraph1_completions,
                "paragraph2_completions": paragraph2_completions,
                "full_responses": full_responses,
                "response_times": response_times,
            }

            filename = f"tldr_two_paragraph_baseline_results_{int(time.time())}.json"
            with open(filename, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nDetailed results saved to: {filename}")

        return aggregated_metrics, metrics_list


def main():
    parser = argparse.ArgumentParser(
        description="TLDR Two-Paragraph Baseline Evaluation"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="Model name")
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
    evaluator = QwenTLDRTwoParagraphBaseline(model_name=args.model, device=args.device)

    # Run evaluation
    aggregated_metrics, individual_metrics = evaluator.evaluate_tldr_baseline(
        num_samples=args.samples, save_results=not args.no_save
    )


if __name__ == "__main__":
    main()
