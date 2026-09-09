"""Independent direct single-agent oracle-preference RLHF and standard DPO."""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
from pathlib import Path
import random
import sys
import time

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi

from baseline.direct_agent.protocol import (
    ALGORITHMS, METRICS, Settings, eval_key, fingerprint, format_prompt, protocol_metadata, score, select_pairs, aggregate_metrics, load_data,
)
from baseline.direct_agent.objectives import (
    ScalarModel, dpo_delta_gradient, dpo_loss, grpo_loss, normalized, token_logps,
)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(cfg: Settings, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, revision=cfg.model_revision, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    configured_dropout = any(float(getattr(model.config, key, 0.0) or 0.0) > 0
                             for key in ("attention_dropout", "hidden_dropout", "hidden_dropout_prob"))
    if configured_dropout or any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in model.modules()):
        raise ValueError("This protocol requires dropout=0 for exact cached-reference/streaming gradients")
    return model


from .generation import generate


class Experiment:
    def __init__(self, cfg: Settings, train, evaluation, manifest, tokenizer, model):
        self.cfg, self.train, self.evaluation = cfg, train, evaluation
        self.tokenizer, self.model, self.manifest = tokenizer, model, manifest
        self.out = Path(cfg.output_dir)
        self.online_responses = self.preference_responses = self.dpo_presentations = self.updates = 0
        self.last_eval = 0
        self.started = time.monotonic()
        self.wandb_run = None
        self.prompts = {r["id"]: torch.tensor(tokenizer.encode(format_prompt(tokenizer, r), add_special_tokens=False))
                        for r in [*train, *evaluation]}
        limit = int(getattr(model.config, "max_position_embeddings", 32768))
        if max(p.numel() for p in self.prompts.values()) + cfg.max_new_tokens > limit:
            raise ValueError("Prompt + completion budget exceeds context; no silent truncation is allowed")
        metadata = protocol_metadata(cfg)
        metadata.update(manifest)
        metadata["max_prompt_tokens"] = max(p.numel() for p in self.prompts.values())
        write_json(self.out / "protocol.json", metadata)
        if cfg.wandb_enabled:
            import wandb
            job = os.environ.get("SLURM_JOB_ID", "local")
            self.wandb_run = wandb.init(
                entity=cfg.wandb_entity, project=cfg.wandb_project, group=cfg.group,
                name=f"{cfg.domain}-direct-{cfg.algorithm}-qwen3-8b-seed{cfg.seed}-{job}",
                dir=str(self.out), config=metadata,
                tags=["single-agent", "direct-answer", "qwen3-8b", cfg.algorithm, "non-thinking"],
            )
            wandb.define_metric("progress/responses_seen", hidden=True)
            for key in ("online_responses", "preference_responses", "dpo_presentations", "updates", "wall_seconds"):
                wandb.define_metric(f"progress/{key}", hidden=True)
            for prefix in ("eval/*", "turn_1/*", "optimization/*"):
                wandb.define_metric(prefix, step_metric="progress/responses_seen")
            # Offline collection/fitting occur before the actor step counter
            # moves. Give these curves their actual progress axes, not x=0.
            wandb.define_metric("preference/*", step_metric="progress/preference_responses")
            wandb.define_metric("optimization/reward_model_pairs", hidden=True)
            wandb.define_metric("optimization/reward_model_loss", step_metric="optimization/reward_model_pairs")

    @property
    def responses_seen(self):
        return self.dpo_presentations if self.cfg.algorithm == "dpo" else self.online_responses

    def log(self, metrics):
        row = dict(metrics, **{
            "progress/responses_seen": self.responses_seen,
            "progress/online_responses": self.online_responses,
            "progress/preference_responses": self.preference_responses,
            "progress/dpo_presentations": self.dpo_presentations,
            "progress/updates": self.updates,
            "progress/wall_seconds": time.monotonic() - self.started,
        })
        with (self.out / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        if self.wandb_run is not None:
            self.wandb_run.log(row)
        print(json.dumps(row), flush=True)

    def evaluate(self, phase: str, count: int):
        """Same decoding policy/RNG across methods and checkpoints; eval does not perturb training RNG."""
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        records = []
        python_state = random.getstate()
        try:
            with torch.random.fork_rng(devices=devices):
                seed_all(self.cfg.eval_seed + self.cfg.seed)
                for row in self.evaluation[:count]:
                    _, text, _mask = generate(self.model, self.tokenizer, self.prompts[row["id"]], 1, self.cfg, row=row)[0]
                    records.append({"id": row["id"], "completion": text, **score(text, row)})
        finally:
            random.setstate(python_state)
        metrics = aggregate_metrics(records)
        result = {"phase": phase, "algorithm": self.cfg.algorithm, "seed": self.cfg.seed,
                  "count": len(records), "metrics": metrics, "records": records,
                  "eval_sha256": self.manifest["eval_sha256"],
                  "protocol_sha256": self.comparison_fingerprint()}
        write_json(self.out / f"eval_{phase}.json", result)
        # Charts always show the same anchors, including initial/final.
        # Full-split metrics remain in eval_final.json and run summary.
        self.log(aggregate_metrics(records[:self.cfg.periodic_eval_samples]))
        self.last_eval = self.responses_seen
        return result

    def comparison_fingerprint(self):
        return fingerprint({"model": self.cfg.model, "revision": self.cfg.model_revision,
                            "protocol": self.cfg.domain + "_direct_v1", "data": self.manifest,
                            "generation": [self.cfg.max_new_tokens, self.cfg.temperature,
                                           self.cfg.top_p, self.cfg.top_k, self.cfg.eval_seed,
                                           self.cfg.periodic_eval_samples]})

    def maybe_eval(self):
        if self.responses_seen - self.last_eval >= self.cfg.eval_every_responses:
            self.evaluate(f"periodic_{self.responses_seen}", self.cfg.periodic_eval_samples)

    def preferences(self):
        """One offline collection. DPO and RLHF at a seed share this sampling protocol."""
        pairs = []
        # Keep sampling independent of incidental actor/value-head initialization.
        seed_all(self.cfg.seed + 5000)
        for i, row in enumerate(self.train):
            prompt = self.prompts[row["id"]]
            outputs = generate(self.model, self.tokenizer, prompt, self.cfg.preference_candidates, self.cfg, row=row)
            rewards = [score(text, row)["reward"] for _, text, _ in outputs]
            indices = select_pairs(rewards, self.cfg.preference_pairs)
            # A fixed initial actor is the DPO reference; exact logps are cached before any update.
            reference = {}
            if self.cfg.algorithm == "dpo":
                with torch.no_grad():
                    for index in sorted({k for pair in indices for k in pair}):
                        reference[index] = float(token_logps(self.model, prompt, outputs[index][0], outputs[index][2]).sum())
            for winner, loser in indices:
                pairs.append({"id": row["id"], "prompt": prompt,
                              "winner": outputs[winner][0], "loser": outputs[loser][0],
                              "winner_mask": outputs[winner][2], "loser_mask": outputs[loser][2],
                              "winner_text": outputs[winner][1], "loser_text": outputs[loser][1],
                              "winner_reward": rewards[winner], "loser_reward": rewards[loser],
                              "reference_delta": reference[winner] - reference[loser] if reference else None})
            self.preference_responses += len(outputs)
            self.log({"preference/questions": i + 1, "preference/pairs": len(pairs),
                      "preference/candidate_reward_mean": sum(rewards) / len(rewards),
                      "preference/candidate_reward_std": float(torch.tensor(rewards).std(unbiased=False)),
                      "preference/selected_pairs_this_question": len(indices)})
        if not pairs:
            raise RuntimeError("No strict preferences: no training occurred")
        # Local artifact only, no W&B sample-table panel. Never unpickle an external file.
        torch.save(pairs, self.out / "preferences.pt")
        return pairs

    def fit_reward_model(self, pairs):
        self.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        backbone = load_model(self.cfg, self.cfg.reward_device)
        reward_model = ScalarModel(backbone).train()
        optimizer = torch.optim.AdamW(reward_model.parameters(), lr=self.cfg.reward_lr)
        for epoch in range(self.cfg.reward_epochs):
            random.shuffle(pairs)
            loss_sum = 0.0
            for i, pair in enumerate(pairs):
                optimizer.zero_grad(set_to_none=True)
                # Exact pairwise logistic gradient, retaining only one long
                # itinerary graph at a time. Dropout=0 is enforced at load.
                with torch.no_grad():
                    delta = (reward_model(torch.cat([pair["prompt"], pair["winner"]]))
                             - reward_model(torch.cat([pair["prompt"], pair["loser"]])))
                    loss = -torch.nn.functional.logsigmoid(delta)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite reward-model loss")
                coefficient = -torch.sigmoid(-delta)
                for side, sign in (("winner", 1.), ("loser", -1.)):
                    (sign * coefficient * reward_model(torch.cat([pair["prompt"], pair[side]]))).backward()
                torch.nn.utils.clip_grad_norm_(reward_model.parameters(), self.cfg.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                loss_sum += float(loss.detach())
                if (i + 1) % 32 == 0 or i + 1 == len(pairs):
                    self.log({"optimization/reward_model_loss": loss_sum / (i + 1),
                              "optimization/reward_model_pairs": (epoch * len(pairs)) + i + 1})
        reward_model.zero_grad(set_to_none=True)
        del optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        reward_model.requires_grad_(False).eval()
        self.model.to(self.cfg.actor_device)
        return reward_model

    def train_online(self, reward_model=None):
        cfg = self.cfg
        is_iac = False  # This entrypoint exposes RLHF and DPO only.
        scalar = ScalarModel(self.model) if is_iac else None
        parameters = list(scalar.parameters() if scalar is not None else self.model.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=cfg.iac_lr if is_iac else cfg.actor_lr)
        epochs = cfg.iac_epochs if is_iac else cfg.online_epochs
        generations = 1 if is_iac else cfg.grpo_generations
        for epoch in range(epochs):
            items = list(self.train)
            random.shuffle(items)
            for start in range(0, len(items), cfg.prompt_batch_size):
                samples, task_rewards, training_rewards = [], [], []
                self.model.train()
                batch_rows = items[start:start + cfg.prompt_batch_size]
                batched = None
                for row_index, row in enumerate(batch_rows):
                    prompt = self.prompts[row["id"]]
                    outputs = ([batched[row_index]] if batched is not None else
                               generate(self.model, self.tokenizer, prompt, generations, cfg, row=row))
                    values = []
                    for tokens, text, mask in outputs:
                        task_reward = score(text, row)["reward"]
                        with torch.no_grad():
                            reward = float(reward_model(torch.cat([prompt, tokens]))) if reward_model else task_reward
                            old = token_logps(self.model, prompt, tokens, mask).detach().cpu()
                            value = float(scalar(prompt)) if scalar is not None else 0.0
                        values.append(reward)
                        samples.append({"prompt": prompt, "tokens": tokens, "mask": mask, "old": old,
                                        "reward": reward, "value": value})
                        task_rewards.append(task_reward)
                        training_rewards.append(reward)
                    if not is_iac:
                        advantages = normalized(torch.tensor(values)).tolist()
                        for sample, advantage in zip(samples[-generations:], advantages):
                            sample["advantage"] = advantage
                if is_iac:
                    advantages = normalized(torch.tensor([s["reward"] - s["value"] for s in samples])).tolist()
                    for sample, advantage in zip(samples, advantages):
                        sample["advantage"] = advantage
                optimizer.zero_grad(set_to_none=True)
                policy_losses, value_losses = [], []
                # Accumulate the exact batch objective while retaining one completion graph at a time.
                for sample in samples:
                    new = token_logps(self.model, sample["prompt"], sample["tokens"], sample["mask"])
                    loss = (-new.sum() * sample["advantage"] if is_iac else
                            grpo_loss(new, sample["old"], sample["advantage"], cfg.ratio_clip))
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite actor loss")
                    (loss / len(samples)).backward()
                    policy_losses.append(float(loss.detach()))
                    if scalar is not None:
                        value_loss = (scalar(sample["prompt"]) - sample["reward"]).square()
                        (cfg.value_loss_coef * value_loss / len(samples)).backward()
                        value_losses.append(float(value_loss.detach()))
                grad = torch.nn.utils.clip_grad_norm_(parameters, cfg.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                self.updates += 1
                self.online_responses += len(samples)
                self.log({"turn_1/reward_mean": sum(task_rewards) / len(samples),
                          "turn_1/training_reward_mean": sum(training_rewards) / len(samples),
                          "optimization/policy_loss": sum(policy_losses) / len(samples),
                          "optimization/value_loss": sum(value_losses) / max(1, len(value_losses)),
                          "optimization/grad_norm": float(grad), "optimization/epoch": epoch + 1})
                self.maybe_eval()
        if scalar is not None:
            torch.save(scalar.head.state_dict(), self.out / "value_head.pt")
        self.model.zero_grad(set_to_none=True)
        del optimizer

    def train_dpo(self, pairs):
        cfg = self.cfg
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.actor_lr)
        for epoch in range(cfg.dpo_epochs):
            random.shuffle(pairs)
            for start in range(0, len(pairs), cfg.preference_batch_size):
                remaining = (cfg.dpo_response_cap - self.dpo_presentations) // 2
                if remaining < 1:
                    break
                batch = pairs[start:start + min(cfg.preference_batch_size, remaining)]
                optimizer.zero_grad(set_to_none=True)
                losses = []
                self.model.train()
                for pair in batch:
                    with torch.no_grad():
                        delta = float(token_logps(self.model, pair["prompt"], pair["winner"], pair["winner_mask"]).sum()
                                      - token_logps(self.model, pair["prompt"], pair["loser"], pair["loser_mask"]).sum())
                    coefficient = dpo_delta_gradient(delta, pair["reference_delta"], cfg.dpo_beta) / len(batch)
                    for side, sign in (("winner", 1.0), ("loser", -1.0)):
                        (sign * coefficient * token_logps(self.model, pair["prompt"], pair[side], pair[f"{side}_mask"]).sum()).backward()
                    losses.append(float(dpo_loss(torch.tensor(delta), pair["reference_delta"], cfg.dpo_beta)))
                grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                self.updates += 1
                self.dpo_presentations += 2 * len(batch)
                self.log({"optimization/dpo_loss": sum(losses) / len(losses),
                          "optimization/grad_norm": float(grad), "optimization/epoch": epoch + 1})
                self.maybe_eval()
        self.model.zero_grad(set_to_none=True)
        del optimizer

    def run(self):
        initial = self.evaluate("initial", len(self.evaluation))
        if self.cfg.algorithm == "raw":
            final = dict(initial, phase="final")
            write_json(self.out / "eval_final.json", final)
        else:
            if self.cfg.algorithm in ("dpo", "rlhf"):
                pairs = self.preferences()
                if self.cfg.algorithm == "dpo":
                    self.train_dpo(pairs)
                else:
                    reward_model = self.fit_reward_model(pairs)
                    self.train_online(reward_model)
                    if self.cfg.save_final_model:
                        reward_model.backbone.save_pretrained(self.out / "reward_model")
                        torch.save(reward_model.head.state_dict(), self.out / "reward_model" / "reward_head.pt")
            else:
                self.train_online()
            final = self.evaluate("final", len(self.evaluation))
            if self.cfg.save_final_model:
                self.model.save_pretrained(self.out / "actor")
                self.tokenizer.save_pretrained(self.out / "actor")
        summary = {"status": "finished", "algorithm": self.cfg.algorithm, "seed": self.cfg.seed,
                   "initial_metrics": initial["metrics"], "final_metrics": final["metrics"],
                   "online_responses": self.online_responses, "preference_responses": self.preference_responses,
                   "dpo_presentations": self.dpo_presentations, "updates": self.updates,
                   "wall_seconds": time.monotonic() - self.started}
        write_json(self.out / "completed.json", summary)
        if self.wandb_run is not None:
            self.wandb_run.summary.update(summary)
            self.wandb_run.finish()
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--algorithm", choices=ALGORITHMS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true", help="Validate settings and dataset; never load model weights")
    args = parser.parse_args(argv)
    values = json.loads(args.config.read_text()) if args.config else {}
    for key in ("algorithm", "seed", "output_dir"):
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)
    cfg = Settings(**values)
    cfg.validate()
    if cfg.model_revision is None and not Path(cfg.model).exists():
        cfg.model_revision = HfApi().model_info(cfg.model).sha
    train, evaluation, manifest = load_data(cfg)
    if args.dry_run:
        print(json.dumps(dict(protocol_metadata(cfg), **manifest), indent=2))
        return
    # A private cache before the FIRST torch CUDA query; no shared cache mutation.
    if "SLURM_JOB_ID" in os.environ and not os.environ.get("CUDA_CACHE_PATH"):
        import tempfile
        os.environ["CUDA_CACHE_PATH"] = tempfile.mkdtemp(prefix=f"{cfg.domain}-direct-{os.environ['SLURM_JOB_ID']}-")
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", dataclasses.asdict(cfg))
    seed_all(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.model_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(cfg, cfg.actor_device)
    experiment = Experiment(cfg, train, evaluation, manifest, tokenizer, model)
    try:
        experiment.run()
    except BaseException as exc:
        write_json(out / "failed.json", {"error": repr(exc), "algorithm": cfg.algorithm, "seed": cfg.seed})
        if experiment.wandb_run is not None:
            experiment.wandb_run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
