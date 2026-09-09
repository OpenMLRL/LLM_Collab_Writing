"""Exact continuation and short-run trajectory regression tests (tiny CPU Qwen)."""
import contextlib
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from baseline.direct_agent.long_run import LongExperiment, SegmentComplete, SUPPORTED, BFCL, atomic_save
from baseline.direct_agent import run as base


class Tokenizer:
    pad_token_id, eos_token_id = 0, 2
    def encode(self, *args, **kwargs):
        return [1, 3, 4]


def tiny():
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=128, eos_token_id=2, pad_token_id=0))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def generated(model, tokenizer, prompt, count, cfg, **kwargs):
    answer = []
    for i in range(count):
        # Use both Python and torch RNG to detect incorrect continuation.
        random.random()
        tail = int(torch.randint(10, 14, ()).item())
        tokens = torch.tensor([7 + i % 2, tail, 2])
        text = str(i % 2 if count > 1 else tail % 2)
        answer.append((tokens, text) if BFCL else (tokens, text, torch.ones(3, dtype=torch.bool)))
    return answer


def evaluation(experiment, phase, count):
    # Evaluations consume no training RNG and log the same stable anchors.
    result = {"phase": phase, "count": count, "metrics": {"eval/turn_1/reward_mean": .5}}
    experiment.last_eval = experiment.responses_seen
    base.write_json(experiment.out / ("eval_" + phase + ".json"), result)
    return result


class LongTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, directory, algorithm):
        cfg = base.Settings(algorithm=algorithm, train_samples=3, eval_samples=1,
            max_new_tokens=4, actor_device="cpu", reward_device="cpu", output_dir=directory,
            prompt_batch_size=2, preference_batch_size=2, preference_candidates=4,
            preference_pairs=2, periodic_eval_samples=1, eval_every_responses=100000,
            wandb_enabled=False, save_final_model=False)
        cfg.online_epochs = 3
        cfg.dpo_epochs = 2
        if hasattr(cfg, "iac_epochs"):
            cfg.iac_epochs = 3
        return cfg

    @contextlib.contextmanager
    def mocks(self):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(base, "format_prompt", return_value="safe"))
            stack.enter_context(patch.object(base, "generate", side_effect=generated))
            stack.enter_context(patch.object(base, "score", side_effect=lambda text, row: {"reward": float(text)}))
            stack.enter_context(patch.object(base, "load_model", side_effect=lambda *args: tiny()))
            if hasattr(base, "generate_prompt_batch"):
                stack.enter_context(patch.object(base, "generate_prompt_batch",
                    side_effect=lambda model, tok, prompts, cfg, rows: [
                        generated(model, tok, p, 1, cfg)[0] for p in prompts]))
            stack.enter_context(patch.object(LongExperiment, "evaluate", evaluation))
            yield

    def experiment(self, directory, algorithm, stop=0):
        cfg = self.config(directory, algorithm)
        train = [{"id": str(i)} for i in range(3)]
        manifest = {"train_sha256": "tiny-train", "eval_sha256": "tiny-eval"}
        return LongExperiment(cfg, train, [{"id": "eval"}], manifest, Tokenizer(), tiny(),
                              target_updates=6, segment_seconds=100000,
                              checkpoint_seconds=100000, stop_after_updates=stop)

    def test_resume_is_identical_to_uninterrupted(self):
        for algorithm in SUPPORTED:
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as full_dir, \
                    tempfile.TemporaryDirectory() as split_dir, self.mocks():
                base.seed_all(9)
                uninterrupted = self.experiment(full_dir, algorithm)
                result = uninterrupted.run()
                self.assertEqual(result["updates"], 6)
                full = torch.load(Path(full_dir) / "checkpoint.pt", weights_only=True)

                base.seed_all(9)
                partial = self.experiment(split_dir, algorithm, stop=2)
                with self.assertRaises(SegmentComplete):
                    partial.run()
                self.assertFalse((Path(split_dir) / "completed.json").exists())
                self.assertEqual(partial.updates, 2)

                base.seed_all(123456)  # Model/head initialization must not perturb continuation.
                continued = self.experiment(split_dir, algorithm)
                with patch.object(continued, "preferences", side_effect=AssertionError("must reuse pairs")):
                    result = continued.run()
                self.assertEqual(result["updates"], 6)
                split = torch.load(Path(split_dir) / "checkpoint.pt", weights_only=True)
                for key in full["model"]:
                    torch.testing.assert_close(full["model"][key], split["model"][key], rtol=0, atol=0)
                self.assertEqual(full["order"], split["order"])
                self.assertEqual(full["position"], split["position"])
                self.assertEqual(full["counters"], split["counters"])
                self.assertEqual(full["rng"]["python"], split["rng"]["python"])
                self.assertTrue(torch.equal(full["rng"]["torch"], split["rng"]["torch"]))
                for key, state in full["optimizer"]["state"].items():
                    for name, value in state.items():
                        torch.testing.assert_close(value, split["optimizer"]["state"][key][name], rtol=0, atol=0)

    def test_original_update_objective_trajectory(self):
        for algorithm in SUPPORTED:
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as short_dir, \
                    tempfile.TemporaryDirectory() as long_dir, self.mocks(), \
                    patch.object(base.Experiment, "evaluate", evaluation):
                base.seed_all(9)
                short = base.Experiment(self.config(short_dir, algorithm),
                    [{"id": str(i)} for i in range(3)], [{"id": "eval"}],
                    {"train_sha256": "tiny-train", "eval_sha256": "tiny-eval"}, Tokenizer(), tiny())
                if algorithm == "dpo":
                    pairs = short.preferences()
                    short.train_dpo(pairs)
                elif algorithm == "rlhf":
                    pairs = short.preferences()
                    rm = short.fit_reward_model(pairs)
                    short.train_online(rm)
                else:
                    short.train_online()
                self.assertEqual(short.updates, 6)
                base.seed_all(9)
                long = self.experiment(long_dir, algorithm)
                long.run()
                for key, value in short.model.state_dict().items():
                    torch.testing.assert_close(value, long.model.state_dict()[key], rtol=0, atol=0)

    def test_rlhf_setup_phase_resume(self):
        for phase in ("preferences", "reward_model"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as full_dir, \
                    tempfile.TemporaryDirectory() as split_dir, self.mocks():
                base.seed_all(9)
                full = self.experiment(full_dir, "rlhf")
                full.run()
                expected = torch.load(Path(full_dir) / "checkpoint.pt", weights_only=True)
                expected_rm = torch.load(Path(full_dir) / "rlhf_setup.pt", weights_only=True)

                base.seed_all(9)
                partial = self.experiment(split_dir, "rlhf")
                original_log = partial.log
                original_step = torch.optim.AdamW.step
                steps = [0]
                def log(metrics):
                    original_log(metrics)
                    if phase == "preferences" and metrics.get("preference/questions") == 1:
                        partial.stop_requested = True
                def step(optimizer, *args, **kwargs):
                    result = original_step(optimizer, *args, **kwargs)
                    steps[0] += 1
                    if phase == "reward_model" and steps[0] == 2:
                        partial.stop_requested = True
                    return result
                with patch.object(partial, "log", log), patch.object(torch.optim.AdamW, "step", step):
                    with self.assertRaises(SegmentComplete):
                        partial.run()
                saved = torch.load(Path(split_dir) / "rlhf_setup.pt", weights_only=True)
                self.assertEqual(saved["phase"], phase)
                self.assertEqual(partial.updates, 0)

                # The pretrained actor always starts with identical pinned weights.
                base.seed_all(9)
                resumed = self.experiment(split_dir, "rlhf")
                resumed.run()
                actual = torch.load(Path(split_dir) / "checkpoint.pt", weights_only=True)
                actual_rm = torch.load(Path(split_dir) / "rlhf_setup.pt", weights_only=True)
                for key in expected["model"]:
                    torch.testing.assert_close(expected["model"][key], actual["model"][key], rtol=0, atol=0)
                for key in expected_rm["model"]:
                    torch.testing.assert_close(expected_rm["model"][key], actual_rm["model"][key], rtol=0, atol=0)
                self.assertEqual(expected["counters"], actual["counters"])
                self.assertEqual(expected["order"], actual["order"])
                self.assertEqual(expected["rng"]["python"], actual["rng"]["python"])
                self.assertTrue(torch.equal(expected["rng"]["torch"], actual["rng"]["torch"]))
                self.assertEqual(actual_rm["rm_updates"], 6)
                self.assertEqual(actual["counters"]["preference_responses"], 12)

    def test_setup_pause_from_separate_entrypoint_namespace(self):
        import runpy
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            entry = runpy.run_module(LongExperiment.__module__, run_name="_rlhf_test_entrypoint")
        with tempfile.TemporaryDirectory() as directory, self.mocks():
            cfg = self.config(directory, "rlhf")
            ex = entry["LongExperiment"](cfg, [{"id": "0"}], [{"id": "eval"}],
                    {"train_sha256": "tiny-train", "eval_sha256": "tiny-eval"},
                    Tokenizer(), tiny(), target_updates=6, segment_seconds=-1)
            with patch.object(ex, "evaluate", side_effect=lambda phase, count: evaluation(ex, phase, count)):
                with self.assertRaises(entry["SegmentComplete"]):
                    ex.run()

    def test_rlhf_cannot_fall_back_to_task_reward(self):
        with tempfile.TemporaryDirectory() as directory, self.mocks():
            ex = self.experiment(directory, "rlhf")
            with self.assertRaisesRegex(ValueError, "fitted, frozen"):
                ex.train_budget(None)

    def test_budget_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory, self.mocks():
            ex = self.experiment(directory, SUPPORTED[0], stop=2)
            with self.assertRaises(SegmentComplete):
                ex.run()
            cfg = self.config(directory, SUPPORTED[0])
            with self.assertRaisesRegex(ValueError, "mismatch"):
                LongExperiment(cfg, [{"id": str(i)} for i in range(3)], [{"id": "eval"}],
                    {"train_sha256": "tiny-train", "eval_sha256": "tiny-eval"}, Tokenizer(), tiny(),
                    target_updates=7)

    def test_atomic_save_failure_preserves_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            atomic_save({"good": 1}, path)
            with patch("torch.save", side_effect=OSError("test disk failure")):
                with self.assertRaises(OSError):
                    atomic_save({"good": 2}, path)
            self.assertEqual(torch.load(path, weights_only=True), {"good": 1})
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_zero_pairs_fail_instead_of_false_completion(self):
        with tempfile.TemporaryDirectory() as directory, self.mocks(), \
                patch.object(base, "score", return_value={"reward": 1.}):
            ex = self.experiment(directory, "dpo")
            with self.assertRaisesRegex(RuntimeError, "No strict preferences"):
                ex.run()
            self.assertFalse((Path(directory) / "completed.json").exists())


if __name__ == "__main__":
    unittest.main()
