"""Fixed, target-free inputs and budgets for direct RLHF / DPO controls."""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path

from .domain import DOMAIN, METRICS, messages, score, dataset_defaults, normalize_rows

ALGORITHMS = ("rlhf", "dpo")


@dataclass
class Settings:
    domain: str = DOMAIN
    algorithm: str = "rlhf"
    seed: int = 42
    model: str = "Qwen/Qwen3-8B"
    model_revision: str = "b968826d9c46dd6066d109eabc6255188de91218"
    dataset: str = dataset_defaults["dataset"]
    dataset_revision: str = dataset_defaults["revision"]
    train_samples: int = dataset_defaults["train_samples"]
    eval_samples: int = dataset_defaults["eval_samples"]
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    generation_batch_size: int = 20
    prompt_batch_size: int = 4
    preference_candidates: int = 80
    preference_pairs: int = 16
    preference_batch_size: int = 4
    grpo_generations: int = 4
    online_epochs: int = dataset_defaults["online_epochs"]
    dpo_epochs: int = dataset_defaults["dpo_epochs"]
    reward_epochs: int = 1
    actor_lr: float = 2e-5
    reward_lr: float = 1e-5
    dpo_beta: float = 0.1
    ratio_clip: float = 0.2
    max_grad_norm: float = 1.0
    eval_every_responses: int = dataset_defaults["eval_every_responses"]
    periodic_eval_samples: int = dataset_defaults["periodic_eval_samples"]
    eval_seed: int = 1729
    actor_device: str = "cuda:0"
    reward_device: str = "cuda:1"
    output_dir: str = "output_direct"
    wandb_enabled: bool = True
    wandb_entity: str = "OpenMLRL"
    wandb_project: str = dataset_defaults["project"]
    group: str = DOMAIN + "-direct-qwen3-8b"
    save_final_model: bool = True

    @property
    def dpo_response_cap(self):
        return 2 * self.train_samples * self.preference_pairs * self.dpo_epochs

    def validate(self):
        if self.algorithm not in ALGORITHMS or self.domain != DOMAIN:
            raise ValueError("Wrong domain or algorithm for this entrypoint")
        for name in ("train_samples", "eval_samples", "max_new_tokens", "generation_batch_size",
                     "prompt_batch_size", "preference_candidates", "preference_pairs",
                     "preference_batch_size", "grpo_generations", "online_epochs", "dpo_epochs",
                     "reward_epochs", "eval_every_responses", "periodic_eval_samples", "top_k"):
            if getattr(self, name) < 1:
                raise ValueError(name + " must be positive")
        if self.preference_candidates < 2 or self.grpo_generations < 2:
            raise ValueError("At least two candidates per group")
        if not 0 < self.temperature or not 0 < self.top_p <= 1 or not 0 < self.ratio_clip < 1:
            raise ValueError("Invalid sampling or clipping settings")
        if self.periodic_eval_samples > self.eval_samples:
            raise ValueError("Anchor eval must fit full eval split")
        for name in ("actor_lr", "reward_lr", "dpo_beta", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be positive and finite")

    def budget(self):
        return {"online_responses": self.train_samples * self.online_epochs * self.grpo_generations
                if self.algorithm == "rlhf" else 0,
                "preference_responses": self.train_samples * self.preference_candidates,
                "preference_pair_cap": self.train_samples * self.preference_pairs,
                "dpo_response_presentations_cap": self.dpo_response_cap if self.algorithm == "dpo" else 0,
                "note": "DPO presentations reuse offline data. Strict ties can reduce the pair count."}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def eval_key(key):
    return "eval/turn_1/" + ("reward_mean" if key == "reward" else key)


def aggregate_metrics(records):
    if not records:
        raise ValueError("Empty evaluation")
    return {eval_key(k): sum(r[k] for r in records) / len(records) for k in METRICS}


def format_prompt(tokenizer, row):
    return tokenizer.apply_chat_template(messages(row), tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def select_pairs(rewards, limit):
    if not all(math.isfinite(r) for r in rewards):
        raise ValueError("Non-finite preference labels")
    pairs = [(rewards[i] - rewards[j], i, j) for i in range(len(rewards))
             for j in range(len(rewards)) if rewards[i] > rewards[j]]
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
    return [(i, j) for _, i, j in pairs[:limit]]


def load_data(cfg):
    if Path(cfg.dataset).is_dir():
        raw = {split: [json.loads(line) for line in (Path(cfg.dataset) / (split + ".jsonl")).read_text().splitlines()]
               for split in ("train", "eval")}
        train, evaluation = raw["train"], raw["eval"]
    else:
        from datasets import load_dataset
        raw = {split: list(load_dataset(cfg.dataset, revision=cfg.dataset_revision, split=spec))
               for split, spec in dataset_defaults["splits"].items()}
        train, evaluation = normalize_rows(raw["train"], "train"), normalize_rows(raw["eval"], "eval")
    train, evaluation = train[:cfg.train_samples], evaluation[:cfg.eval_samples]
    if len(train) != cfg.train_samples or len(evaluation) != cfg.eval_samples:
        raise ValueError("Unexpected selected dataset sizes")
    ids = [r["id"] for r in train + evaluation]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate IDs / train-eval overlap")
    train_prompts = {r["prompt"] for r in train}
    if train_prompts.intersection(r["prompt"] for r in evaluation):
        raise ValueError("Train/eval prompt overlap")
    return train, evaluation, {"train_ids": [r["id"] for r in train], "eval_ids": [r["id"] for r in evaluation],
                               "train_sha256": fingerprint(train), "eval_sha256": fingerprint(evaluation)}


def protocol_metadata(cfg):
    return {"settings": asdict(cfg), "protocol": DOMAIN + "_direct_v1", "num_actors": 1,
            "enable_thinking": False, "reward_range": [0, 1], "budget": cfg.budget(),
            "preference_source": "Synthetic labels from the domain task evaluator, not human annotations",
            "reward_note": dataset_defaults["reward_note"], "replay": "single offline collection; non-iter"}
