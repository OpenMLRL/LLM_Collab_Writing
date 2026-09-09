"""Resumable offline preference collection and scalar reward-model fitting for RLHF.

No actor/comparator protocol changes: original strict C/P selection, pairwise
logistic RM objective and GRPO on frozen learned rewards. Setup and actor have
separate atomic artifacts; a ready RM is reused verbatim on every actor segment.
"""
from __future__ import annotations

import gc
import time
import random
import torch

from . import run as base
from .long_run import BFCL, COUNTERS, atomic_save, rng_state, restore_rng


def pause_requested(ex):
    return ex.stop_requested or time.monotonic() >= ex.deadline


def save_setup(ex, state):
    state.update(identity=ex.identity, initial=ex.initial, rng=rng_state(),
                 counters={k: getattr(ex, k) for k in COUNTERS},
                 elapsed=time.monotonic() - ex.started)
    atomic_save(state, ex.out / "rlhf_setup.pt")
    rank = {"preferences": 1, "reward_model": 2, "ready": 3}[state["phase"]]
    progress = state.get("questions", 0) if rank == 1 else state.get("rm_updates", 0)
    if rank == 3:
        progress = 0
    base.write_json(ex.out / "checkpoint_status.json", {
        "status": "checkpointed", "identity": ex.identity,
        "phase": state["phase"], "work_progress": [rank, progress],
        "updates": ex.updates, "target_updates": ex.target_updates,
        "artifact": "rlhf_setup.pt"})
    ex.checkpoint_at = time.monotonic()


def pause(ex, state):
    save_setup(ex, state)
    if ex.wandb_run is not None:
        ex.wandb_run.summary.update(status="paused_for_requeue", phase=state["phase"],
                                    updates=ex.updates)
        ex.wandb_run.mark_preempting()
        ex.wandb_run.finish(exit_code=0)
    raise ex.segment_complete()


def prepare_reward(ex):
    cfg = ex.cfg
    path = ex.out / "rlhf_setup.pt"
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["identity"] != ex.identity:
            raise ValueError("RLHF setup protocol/config/data mismatch")
        ex.initial = state["initial"]
        if ex.resume_state is None:
            for key, value in state["counters"].items():
                setattr(ex, key, value)
            ex.started -= state["elapsed"]
            restore_rng(state["rng"])
        elif state["phase"] != "ready":
            raise ValueError("Actor checkpoint requires its completed reward model")
    else:
        if ex.resume_state is not None:
            raise ValueError("Actor checkpoint is missing its reward model")
        ex.initial = ex.evaluate("initial", len(ex.evaluation))
        base.seed_all(cfg.seed + 5000)
        state = {"phase": "preferences", "questions": 0, "pairs": []}
        save_setup(ex, state)

    if state["phase"] == "preferences":
        if pause_requested(ex):
            pause(ex, state)
        pairs = state["pairs"]
        for i in range(state["questions"], len(ex.train)):
            row = ex.train[i]
            prompt = ex.prompts[row["id"]]
            outputs = ex.outputs(row, cfg.preference_candidates)
            rewards = [base.score(text, row)["reward"] for _, text, _ in outputs]
            indices = base.select_pairs(rewards, cfg.preference_pairs)
            for winner, loser in indices:
                pair = {"id": row["id"], "prompt": prompt,
                        "winner": outputs[winner][0], "loser": outputs[loser][0],
                        "winner_text": outputs[winner][1], "loser_text": outputs[loser][1],
                        "winner_reward": rewards[winner], "loser_reward": rewards[loser],
                        "reference_delta": None}
                if not BFCL:
                    pair.update(winner_mask=outputs[winner][2], loser_mask=outputs[loser][2])
                pairs.append(pair)
            ex.preference_responses += len(outputs)
            ex.log({"preference/questions": i + 1, "preference/pairs": len(pairs),
                    "preference/candidate_reward_mean": sum(rewards) / len(rewards),
                    "preference/candidate_reward_std": float(torch.tensor(rewards).std(unbiased=False)),
                    "preference/selected_pairs_this_question": len(indices)})
            state["questions"] = i + 1
            # Small CPU artifact, committed after every question (no repeated sampling).
            save_setup(ex, state)
            if pause_requested(ex):
                pause(ex, state)
        if not pairs:
            raise RuntimeError("No strict preferences: no training occurred")
        atomic_save(pairs, ex.out / "preferences.pt")
    elif state["phase"] not in ("reward_model", "ready"):
        raise ValueError("Unknown RLHF setup phase")

    # Actor is never trained until RM is complete. Offload it during RM fitting.
    ex.model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    saved_rng = state["rng"] if state["phase"] != "preferences" else None
    reward_model = base.ScalarModel(base.load_model(cfg, cfg.reward_device))
    if not BFCL:
        reward_model.train()  # Match the original domain's RM mode exactly.
    if state["phase"] in ("reward_model", "ready"):
        reward_model.load_state_dict(state.pop("model"))
    if state["phase"] == "ready":
        # The constructor consumes RNG; restore the post-RM state only before the
        # first actor segment. Later segments restore their actor checkpoint RNG.
        if ex.resume_state is None:
            restore_rng(saved_rng)
        reward_model.requires_grad_(False).eval()
        del state
        gc.collect()
        ex.model.to(cfg.actor_device)
        return reward_model

    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=cfg.reward_lr)
    if state["phase"] == "reward_model":
        optimizer.load_state_dict(state.pop("optimizer"))
        restore_rng(saved_rng)
    else:
        state.update(phase="reward_model", rm_epoch=0, rm_position=0,
                     rm_order=list(range(len(state["pairs"]))), rm_updates=0,
                     rm_loss_sum=0., rm_epoch_started=False)
    gc.collect()

    def checkpoint(force=False):
        stop = pause_requested(ex)
        if force or stop or time.monotonic() - ex.checkpoint_at >= ex.checkpoint_seconds:
            state.update(model=reward_model.state_dict(), optimizer=optimizer.state_dict())
            if stop:
                pause(ex, state)
            save_setup(ex, state)
            # Do not retain optimizer tensor references beyond checkpoint writing.
            state.pop("model")
            state.pop("optimizer")

    checkpoint()
    while state["rm_epoch"] < cfg.reward_epochs:
        if not state["rm_epoch_started"]:
            random.shuffle(state["rm_order"])  # Equivalent to in-place shuffle(pairs).
            state.update(rm_position=0, rm_loss_sum=0., rm_epoch_started=True)
        while state["rm_position"] < len(state["pairs"]):
            pair = state["pairs"][state["rm_order"][state["rm_position"]]]
            optimizer.zero_grad(set_to_none=True)
            if BFCL:
                winner = reward_model(torch.cat([pair["prompt"], pair["winner"]]))
                loser = reward_model(torch.cat([pair["prompt"], pair["loser"]]))
                loss = -torch.nn.functional.logsigmoid(winner - loser)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite reward-model loss")
                loss.backward()
            else:
                # Same exact serial pairwise gradient as the original long-text RM.
                with torch.no_grad():
                    delta = (reward_model(torch.cat([pair["prompt"], pair["winner"]]))
                             - reward_model(torch.cat([pair["prompt"], pair["loser"]])))
                    loss = -torch.nn.functional.logsigmoid(delta)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite reward-model loss")
                coefficient = -torch.sigmoid(-delta)
                for side, sign in (("winner", 1.), ("loser", -1.)):
                    (sign * coefficient * reward_model(torch.cat([pair["prompt"], pair[side]]))).backward()
            torch.nn.utils.clip_grad_norm_(reward_model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            state["rm_position"] += 1
            state["rm_updates"] += 1
            state["rm_loss_sum"] += float(loss.detach())
            if state["rm_position"] % 32 == 0 or state["rm_position"] == len(state["pairs"]):
                ex.log({"optimization/reward_model_loss": state["rm_loss_sum"] / state["rm_position"],
                        "optimization/reward_model_pairs": state["rm_updates"]})
            checkpoint()
        state["rm_epoch"] += 1
        state["rm_epoch_started"] = False

    reward_model.zero_grad(set_to_none=True)
    del optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reward_model.requires_grad_(False).eval()
    ready = {"phase": "ready", "model": reward_model.state_dict(), "rm_updates": state["rm_updates"],
             "strict_pairs": len(state["pairs"]), "questions": state["questions"]}
    save_setup(ex, ready)
    if ex.wandb_run is not None:
        ex.wandb_run.summary.update(phase="actor", reward_model_updates=ready["rm_updates"],
                                    strict_preference_pairs=ready["strict_pairs"])
    ex.model.to(cfg.actor_device)
    return reward_model
