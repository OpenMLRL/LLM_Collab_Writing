"""Opt-in update-budgeted GRPO/IAC/DPO with exact optimizer/RNG resume.

The short-run entrypoint and all domain prompts, scores and sampling remain unchanged.
An epoch is repeated until target_updates; response caps do not terminate this entrypoint.
Only clean checkpoint-boundary pauses return 85 (for Slurm requeue); errors never do.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import os
from pathlib import Path
import random
import signal
import tempfile
import time
import uuid

import torch
from . import run as base

COUNTERS = ("online_responses", "preference_responses", "dpo_presentations", "updates", "last_eval")
BFCL = base.__package__.startswith("native_parallel.")
SUPPORTED = tuple(a for a in ("grpo", "iac", "dpo", "rlhf") if a in base.ALGORITHMS)


def atomic_save(value, path):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)  # Only this call's exact, newly created temporary file.


def rng_state():
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same CUDA device count")
        torch.cuda.set_rng_state_all(state["cuda"])


class SegmentComplete(Exception):
    """Checkpoint committed; safe for the launcher to requeue this same job."""


class LongExperiment(base.Experiment):
    # Keep the pause exception owned by the active entrypoint (also under -m).
    segment_complete = SegmentComplete

    def __init__(self, cfg, train, evaluation, manifest, tokenizer, model, *,
                 target_updates=3200, segment_seconds=79200, checkpoint_seconds=3600,
                 stop_after_updates=0, run_name=None):
        if cfg.algorithm not in SUPPORTED or target_updates < 1:
            raise ValueError("Unsupported update-budgeted algorithm or invalid target")
        enabled = cfg.wandb_enabled
        cfg.wandb_enabled = False
        try:
            super().__init__(cfg, train, evaluation, manifest, tokenizer, model)
        finally:
            cfg.wandb_enabled = enabled
        self.target_updates = target_updates
        self.deadline = time.monotonic() + segment_seconds
        self.checkpoint_seconds = checkpoint_seconds
        self.checkpoint_at = time.monotonic()
        self.stop_after_updates = stop_after_updates
        self.stop_requested = False
        self.epoch, self.position, self.order = 0, 0, []
        self.initial = None
        self.identity = base.fingerprint({"config": dataclasses.asdict(cfg), "manifest": manifest,
                                          "target_updates": target_updates, "long_protocol": 1})
        self.checkpoint_file = self.out / "checkpoint.pt"
        self.prepared_file = self.out / "prepared.pt"
        self.resume_state = None
        if self.checkpoint_file.exists():
            self.resume_state = torch.load(self.checkpoint_file, map_location="cpu", weights_only=True)
            if self.resume_state["identity"] != self.identity:
                raise ValueError("Checkpoint protocol/config/data mismatch")
            for key, value in self.resume_state["counters"].items():
                setattr(self, key, value)
            self.epoch, self.position, self.order = [self.resume_state[k] for k in ("epoch", "position", "order")]
            self.initial = self.resume_state["initial"]
            self.started -= self.resume_state["elapsed"]
        identity_file = self.out / "run_identity.json"
        if identity_file.exists():
            identity = json.loads(identity_file.read_text())
            if identity["protocol_identity"] != self.identity:
                raise ValueError("Existing output directory belongs to a different experiment")
        else:
            domain = "bfcl" if BFCL else getattr(cfg, "domain", "travel")
            job = os.environ.get("SLURM_JOB_ID", "local")
            identity = {"protocol_identity": self.identity, "wandb_id": uuid.uuid4().hex[:12],
                        "run_name": run_name or f"{domain}-direct-{cfg.algorithm}-qwen3-8b-u{target_updates}-seed{cfg.seed}-{job}"}
            base.write_json(identity_file, identity)
        metadata = base.protocol_metadata(cfg)
        metadata.update(manifest)
        metadata["budget"] = {"actor_optimizer_updates": target_updates,
                              "mode": "repeat_epochs_until_actual_update_target",
                              "legacy_epoch_and_response_caps_overridden": True}
        metadata["max_prompt_tokens"] = max(p.numel() for p in self.prompts.values())
        metadata["actor_update_rule"] = {"grpo": "GRPO", "iac": "policy_gradient_with_value_baseline",
                                         "dpo": "DPO", "rlhf": "GRPO_with_frozen_learned_reward"}[cfg.algorithm]
        base.write_json(self.out / "long_protocol.json", metadata)
        if enabled:
            import wandb
            self.wandb_run = wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project,
                group=cfg.group + f"-u{target_updates}", id=identity["wandb_id"], resume="allow",
                name=identity["run_name"], dir=str(self.out), config=metadata,
                tags=["single-agent", "direct-answer", "qwen3-8b", cfg.algorithm, f"updates-{target_updates}"])
            for key in ("updates", "responses_seen", "online_responses", "dpo_presentations",
                        "preference_responses", "wall_seconds"):
                wandb.define_metric(f"progress/{key}", hidden=True)
            for prefix in ("eval/*", "turn_1/*", "optimization/*"):
                wandb.define_metric(prefix, step_metric="progress/updates")
            wandb.define_metric("preference/*", step_metric="progress/preference_responses")
            self.wandb_run.summary["status"] = "training"
            self.wandb_run.summary["target_updates"] = target_updates

    def lp(self, prompt, tokens, mask=None):
        args = (self.model, prompt, tokens) if BFCL else (self.model, prompt, tokens, mask)
        return base.token_logps(*args)

    def outputs(self, row, count):
        args = (self.model, self.tokenizer, self.prompts[row["id"]], count, self.cfg)
        results = base.generate(*args) if BFCL else base.generate(*args, row=row)
        return [(v[0], v[1], None if len(v) == 2 else v[2]) for v in results]

    def maybe_eval(self):
        if self.responses_seen - self.last_eval >= self.cfg.eval_every_responses:
            self.evaluate(f"periodic_update_{self.updates}", self.cfg.periodic_eval_samples)

    def save_checkpoint(self, trainable, optimizer):
        state = {"identity": self.identity, "model": trainable.state_dict(), "optimizer": optimizer.state_dict(),
                 "counters": {k: getattr(self, k) for k in COUNTERS},
                 "epoch": self.epoch, "position": self.position, "order": self.order,
                 "rng": rng_state(), "initial": self.initial, "elapsed": time.monotonic() - self.started}
        atomic_save(state, self.checkpoint_file)
        base.write_json(self.out / "checkpoint_status.json", {
            "status": "checkpointed", "updates": self.updates, "target_updates": self.target_updates,
            "epoch": self.epoch, "position": self.position, "identity": self.identity,
            "phase": "actor", "work_progress": [3, self.updates], "artifact": "checkpoint.pt"})
        self.checkpoint_at = time.monotonic()

    def boundary(self, trainable, optimizer, force=False):
        pause = (self.stop_requested or time.monotonic() >= self.deadline or
                 (self.stop_after_updates > 0 and self.updates >= self.stop_after_updates))
        if force or pause or time.monotonic() - self.checkpoint_at >= self.checkpoint_seconds:
            self.save_checkpoint(trainable, optimizer)
        if pause and self.updates < self.target_updates:
            if self.wandb_run is not None:
                self.wandb_run.summary.update({"status": "paused_for_requeue", "updates": self.updates})
                self.wandb_run.mark_preempting()
                self.wandb_run.finish(exit_code=0)
            raise SegmentComplete()

    def next_batch(self, size, batch_size, *, retain_order):
        if not self.order or self.position >= len(self.order):
            if not retain_order or not self.order:
                self.order = list(range(size))
            random.shuffle(self.order)
            self.position = 0
            self.epoch += 1
        return self.order[self.position:self.position + batch_size]

    def prepare(self):
        if self.cfg.algorithm == "rlhf":
            from .rlhf_stages import prepare_reward
            return prepare_reward(self)
        if self.prepared_file.exists():
            state = torch.load(self.prepared_file, map_location="cpu", weights_only=True)
            if state["identity"] != self.identity:
                raise ValueError("Prepared data protocol mismatch")
            self.initial = state["initial"]
            if self.resume_state is None:
                for key, value in state["counters"].items():
                    setattr(self, key, value)
                restore_rng(state["rng"])
        else:
            if self.resume_state is not None:
                raise ValueError("Checkpoint is missing its prepared data")
            self.initial = self.evaluate("initial", len(self.evaluation))
            if self.cfg.algorithm == "dpo":
                self.preferences()  # Original C/P, strict ties, scores and cached reference.
            state = {"identity": self.identity, "initial": self.initial, "rng": rng_state(),
                     "counters": {k: getattr(self, k) for k in COUNTERS}}
            atomic_save(state, self.prepared_file)
        pairs = None
        if self.cfg.algorithm == "dpo":
            pairs = torch.load(self.out / "preferences.pt", map_location="cpu", weights_only=True)
            if not pairs:
                raise ValueError("No strict preferences; cannot manufacture 3200 useful updates")
        return pairs

    def train_budget(self, pairs, reward_model=None):
        cfg = self.cfg
        if cfg.algorithm == "rlhf" and reward_model is None:
            raise ValueError("RLHF must use a fitted, frozen reward model")
        iac = cfg.algorithm == "iac"
        scalar = base.ScalarModel(self.model) if iac else None
        trainable = scalar if scalar is not None else self.model
        optimizer = torch.optim.AdamW(trainable.parameters(), lr=cfg.iac_lr if iac else cfg.actor_lr)
        if self.resume_state is not None:
            state = self.resume_state
            trainable.load_state_dict(state.pop("model"))
            optimizer.load_state_dict(state.pop("optimizer"))
            restore_rng(state["rng"])
            self.resume_state = None
            del state
            gc.collect()
        size = len(pairs) if pairs is not None else len(self.train)
        batch_size = cfg.preference_batch_size if pairs is not None else cfg.prompt_batch_size
        effective = math.ceil(self.target_updates / math.ceil(size / batch_size))
        budget = {"target_updates": self.target_updates, "effective_epochs": effective,
                  "items_per_epoch": size, "batch_size": batch_size,
                  "strict_pair_questions": len({p["id"] for p in pairs}) if pairs is not None else None,
                  "unique_chosen_answers": len({(p["id"], p["winner_text"]) for p in pairs}) if pairs is not None else None}
        base.write_json(self.out / "actual_budget.json", budget)
        if self.wandb_run is not None:
            self.wandb_run.config.update({"actual_update_budget": budget}, allow_val_change=True)
        self.boundary(trainable, optimizer)  # Setup can consume much of the first allocation.
        while self.updates < self.target_updates:
            indices = self.next_batch(size, batch_size, retain_order=pairs is not None)
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            if pairs is not None:
                losses = []
                for index in indices:
                    pair = pairs[index]
                    with torch.no_grad():
                        delta = float(self.lp(pair["prompt"], pair["winner"], pair.get("winner_mask")).sum()
                                      - self.lp(pair["prompt"], pair["loser"], pair.get("loser_mask")).sum())
                    coefficient = base.dpo_delta_gradient(delta, pair["reference_delta"], cfg.dpo_beta) / len(indices)
                    for side, sign in (("winner", 1.), ("loser", -1.)):
                        (sign * coefficient * self.lp(pair["prompt"], pair[side], pair.get(side + "_mask")).sum()).backward()
                    losses.append(float(base.dpo_loss(torch.tensor(delta), pair["reference_delta"], cfg.dpo_beta)))
                metrics = {"optimization/dpo_loss": sum(losses) / len(losses)}
                self.dpo_presentations += 2 * len(indices)
            else:
                rows = [self.train[i] for i in indices]
                batched = (base.generate_prompt_batch(self.model, self.tokenizer,
                    [self.prompts[r["id"]] for r in rows], cfg, rows=rows)
                    if iac and hasattr(base, "generate_prompt_batch") else None)
                samples, rewards, task_rewards = [], [], []
                for row_index, row in enumerate(rows):
                    prompt = self.prompts[row["id"]]
                    outputs = [batched[row_index]] if batched is not None else self.outputs(row, 1 if iac else cfg.grpo_generations)
                    values = []
                    for tokens, text, mask in outputs:
                        task_reward = base.score(text, row)["reward"]
                        with torch.no_grad():
                            reward = float(reward_model(torch.cat([prompt, tokens]))) if reward_model is not None else task_reward
                            old = self.lp(prompt, tokens, mask).detach().cpu()
                            value = float(scalar(prompt)) if scalar is not None else 0.
                        samples.append(dict(prompt=prompt, tokens=tokens, mask=mask, old=old,
                                            reward=reward, value=value))
                        values.append(reward)
                        rewards.append(reward)
                        task_rewards.append(task_reward)
                    if not iac:
                        for sample, advantage in zip(samples[-len(outputs):], base.normalized(torch.tensor(values)).tolist()):
                            sample["advantage"] = advantage
                if iac:
                    for sample, advantage in zip(samples, base.normalized(torch.tensor(
                            [s["reward"] - s["value"] for s in samples])).tolist()):
                        sample["advantage"] = advantage
                policy_losses, value_losses = [], []
                for sample in samples:
                    new = self.lp(sample["prompt"], sample["tokens"], sample["mask"])
                    loss = (-new.sum() * sample["advantage"] if iac else
                            base.grpo_loss(new, sample["old"], sample["advantage"], cfg.ratio_clip))
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite actor loss")
                    (loss / len(samples)).backward()
                    policy_losses.append(float(loss.detach()))
                    if scalar is not None:
                        value_loss = (scalar(sample["prompt"]) - sample["reward"]).square()
                        (cfg.value_loss_coef * value_loss / len(samples)).backward()
                        value_losses.append(float(value_loss.detach()))
                self.online_responses += len(samples)
                metrics = {"turn_1/reward_mean": sum(task_rewards) / len(task_rewards),
                           "turn_1/training_reward_mean": sum(rewards) / len(rewards),
                           "optimization/policy_loss": sum(policy_losses) / len(policy_losses),
                           "optimization/value_loss": sum(value_losses) / max(1, len(value_losses))}
            grad = torch.nn.utils.clip_grad_norm_(trainable.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            self.updates += 1
            self.position += len(indices)
            self.log(dict(metrics, **{"optimization/grad_norm": float(grad), "optimization/epoch": self.epoch}))
            self.maybe_eval()
            self.boundary(trainable, optimizer)
        self.save_checkpoint(trainable, optimizer)
        trainable.zero_grad(set_to_none=True)
        if scalar is not None:
            torch.save(scalar.head.state_dict(), self.out / "value_head.pt")
        del optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self):
        pairs = self.prepare()
        if self.cfg.algorithm == "rlhf":
            self.train_budget(None, reward_model=pairs)
        else:
            self.train_budget(pairs)
        final = self.evaluate("final", len(self.evaluation))
        if self.cfg.save_final_model:
            self.model.save_pretrained(self.out / "actor")
            self.tokenizer.save_pretrained(self.out / "actor")
        result = {"status": "finished", "algorithm": self.cfg.algorithm, "seed": self.cfg.seed,
                  "target_updates": self.target_updates, "updates": self.updates,
                  "online_responses": self.online_responses, "preference_responses": self.preference_responses,
                  "dpo_presentations": self.dpo_presentations, "initial_metrics": self.initial["metrics"],
                  "final_metrics": final["metrics"], "wall_seconds": time.monotonic() - self.started}
        base.write_json(self.out / "completed.json", result)
        if self.wandb_run is not None:
            self.wandb_run.summary.update(result)
            self.wandb_run.finish()
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--algorithm", choices=SUPPORTED, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-updates", type=int, default=3200)
    parser.add_argument("--segment-seconds", type=int, default=79200)
    parser.add_argument("--checkpoint-seconds", type=int, default=3600)
    parser.add_argument("--run-name")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if min(args.target_updates, args.segment_seconds, args.checkpoint_seconds) < 1:
        parser.error("All budgets must be positive")
    values = json.loads(args.config.read_text())
    values.update(algorithm=args.algorithm, seed=args.seed, output_dir=args.output_dir)
    cfg = base.Settings(**values)
    cfg.validate()
    train, evaluation, manifest = base.load_data(cfg)
    if args.dry_run:
        print(json.dumps({"config": dataclasses.asdict(cfg), "target_updates": args.target_updates,
                          "legacy_epoch_caps_overridden": True, **manifest}, indent=2))
        return 0
    out = Path(cfg.output_dir)
    if (out / "completed.json").exists():
        done = json.loads((out / "completed.json").read_text())
        if args.resume and done["updates"] == args.target_updates:
            print("Already completed; no duplicate training")
            return 0
        raise ValueError("Output already completed")
    out.mkdir(parents=True, exist_ok=args.resume)
    config_file = out / "config.json"
    if config_file.exists() and json.loads(config_file.read_text()) != dataclasses.asdict(cfg):
        raise ValueError("Refusing to overwrite a different run configuration")
    base.write_json(config_file, dataclasses.asdict(cfg))
    base.seed_all(cfg.seed)
    tokenizer = base.AutoTokenizer.from_pretrained(cfg.model, revision=cfg.model_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = base.load_model(cfg, cfg.actor_device)
    experiment = LongExperiment(cfg, train, evaluation, manifest, tokenizer, model,
        target_updates=args.target_updates, segment_seconds=args.segment_seconds,
        checkpoint_seconds=args.checkpoint_seconds, run_name=args.run_name)
    signal.signal(signal.SIGUSR1, lambda *_: setattr(experiment, "stop_requested", True))
    try:
        experiment.run()
    except SegmentComplete:
        return 85
    except BaseException as exc:
        base.write_json(out / "failed.json", {"error": repr(exc), "updates": experiment.updates})
        if experiment.wandb_run is not None:
            experiment.wandb_run.finish(exit_code=1)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
