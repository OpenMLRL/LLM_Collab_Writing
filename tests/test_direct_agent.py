"""Direct RLHF/DPO contracts and tiny-Qwen end-to-end optimizer tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from baseline.direct_agent.protocol import Settings, messages, select_pairs, score, DOMAIN, load_data
from baseline.direct_agent.objectives import token_logps, dpo_loss, dpo_delta_gradient, ScalarModel
from baseline.direct_agent.run import Experiment


def tiny():
    torch.manual_seed(10)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=24, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=64, bos_token_id=1, eos_token_id=2, pad_token_id=0))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return json.dumps(messages)
    def encode(self, *args, **kwargs):
        return [1, 3, 4]


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_targets_excluded(self):
        row = dict(prompt="public request", test="SECRET", completion="SECRET", entry_point="f")
        self.assertNotIn("SECRET", json.dumps(messages(row)))

    def test_ties_and_budget(self):
        self.assertEqual(select_pairs([0, 0, 0], 16), [])
        self.assertEqual(select_pairs([.2, .2, .8], 16), [(2, 0), (2, 1)])
        for algo in ("rlhf", "dpo"):
            cfg = Settings(algorithm=algo)
            cfg.validate()
            self.assertEqual(cfg.budget()["preference_responses"], cfg.train_samples * 80)

    def test_logps_include_last_token(self):
        model = tiny().eval()
        p, c = torch.tensor([1, 3]), torch.tensor([4, 5, 2])
        ids = torch.cat([p, c[:-1]])[None, :]
        logits = model(ids).logits[0, -3:].float()
        expected = -torch.nn.functional.cross_entropy(logits, c, reduction="none")
        torch.testing.assert_close(token_logps(model, p, c), expected)

    def test_dpo_stream_gradient(self):
        a = tiny().train(); b = copy.deepcopy(a)
        p, w, l = torch.tensor([1, 3]), torch.tensor([4, 5, 2]), torch.tensor([6, 7, 2])
        delta = token_logps(a, p, w).sum() - token_logps(a, p, l).sum()
        dpo_loss(delta, 0.3, .1).backward()
        coeff = dpo_delta_gradient(float(delta.detach()), .3, .1)
        (coeff * token_logps(b, p, w).sum()).backward()
        (-coeff * token_logps(b, p, l).sum()).backward()
        for x, y in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(x.grad, y.grad, rtol=2e-4, atol=2e-6)

    def test_rm_stream_gradient(self):
        a = ScalarModel(tiny()).train(); b = copy.deepcopy(a)
        w, l = torch.tensor([1, 3, 4]), torch.tensor([1, 3, 5])
        delta = a(w) - a(l)
        (-torch.nn.functional.logsigmoid(delta)).backward()
        coeff = -torch.sigmoid(-delta.detach())
        (coeff * b(w)).backward(); (-coeff * b(l)).backward()
        for x, y in zip(a.parameters(), b.parameters()):
            if x.grad is not None:
                torch.testing.assert_close(x.grad, y.grad, rtol=2e-4, atol=2e-6)

    def test_both_complete_paths_and_eval_anchors(self):
        rows = [dict(id=str(i), prompt="public " + str(i)) for i in range(4)]
        def generated(model, tokenizer, prompt, count, cfg, **kwargs):
            return [(torch.tensor([5 + i % 2, 2]), str(i % 2), None) for i in range(count)]
        def scored(text, row):
            from baseline.direct_agent.protocol import METRICS
            return dict.fromkeys(METRICS, float(text))
        for algo in ("rlhf", "dpo"):
            with tempfile.TemporaryDirectory() as directory:
                cfg = Settings(algorithm=algo, output_dir=directory, train_samples=2, eval_samples=2,
                    periodic_eval_samples=1, eval_every_responses=4, max_new_tokens=4, online_epochs=1,
                    dpo_epochs=1, preference_candidates=4, preference_pairs=2, grpo_generations=2,
                    wandb_enabled=False, save_final_model=False, actor_device="cpu", reward_device="cpu")
                cfg.validate()
                with patch("baseline.direct_agent.run.generate", generated), \
                     patch("baseline.direct_agent.run.score", scored), \
                     patch("baseline.direct_agent.run.load_model", side_effect=lambda *_: tiny()):
                    ex = Experiment(cfg, rows[:2], rows[2:], {"eval_sha256": "fixed"}, Tokenizer(), tiny())
                    result = ex.run()
                self.assertEqual(result["status"], "finished")
                self.assertGreater(result["updates"], 0)
                self.assertEqual(json.loads((Path(directory) / "eval_final.json").read_text())["count"], 2)

    def test_domain_scoring(self):
        if DOMAIN == "tldr":
            from rewards import tldr_rewards
            a = "This community discussed gardening during winter and shared practical planting advice."
            b = a + " Furthermore several experienced gardeners explained greenhouse temperatures, watering schedules and suitable vegetables. In addition, participants compared costs and recommended testing different approaches."
            tldr_rewards.VERBOSE = False
            r = score(a + "[PARAGRAPH_SPLIT]" + b, {})
            self.assertAlmostEqual(r["reward"], tldr_rewards.tldr_combined_reward([a], [b])[0] / 3)
            self.assertEqual(score(a + "\n\n" + b, {}), r)
            self.assertEqual(score("unstructured", {})["reward"], 0)
            self.assertEqual(score(a + "\n\n" + b + "\n\nthird", {})["parse_success"], 0)
        else:
            row = dict(prompt="def f(x):\n    '''increment'''", entry_point="f", test="SECRET")
            with patch("baseline.direct_agent.domain.evaluate_code", return_value=dict(passed=1, total=2, all_passed=False)) as sandbox:
                r = score("def f(x):\n    return x", row)
                self.assertAlmostEqual(r["reward"], .6)
                self.assertEqual(r["all_tests_pass"], 0)
                sandbox.assert_called_once()
            self.assertEqual(score("nonsense!", row)["reward"], 0)

    def test_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for split in ("train", "eval"):
                (Path(directory) / (split + ".jsonl")).write_text(json.dumps(dict(id=split, prompt="duplicate")) + "\n")
            with self.assertRaises(ValueError):
                load_data(Settings(dataset=directory, train_samples=1, eval_samples=1))


if __name__ == "__main__":
    unittest.main()
