from pathlib import Path
from types import SimpleNamespace

import pytest

import preference_train_common as setup
from comlrl.trainers import preference
from comlrl.trainers.preference.collaboration import CentralizedCollaboration

ALGORITHMS = [
    ("madpo", preference.MADPOConfig),
    ("marlhf", preference.MARLHFConfig),
    ("madpo_iter", preference.MADPOIterConfig),
    ("marlhf_iter", preference.MARLHFIterConfig),
]


@pytest.mark.parametrize("section,config_cls", ALGORITHMS)
@pytest.mark.parametrize("dataset", ["tldr", "arxiv"])
@pytest.mark.parametrize("mode", ["decentralized", "centralized"])
def test_preference_setup_passes_collaboration_adapter(
    section, config_cls, dataset, mode, monkeypatch, tmp_path
):
    root = Path(__file__).resolve().parents[1]
    config = setup.Config(str(root / "configs" / f"{section}_{dataset}_config.yaml"))
    assert config.get(f"{section}.collaboration_mode") == "decentralized"
    config.update(
        {
            section: {"collaboration_mode": mode, "agent_devices": "cpu"},
            "output": {"base_dir": str(tmp_path), "save_final_model": False},
        }
    )
    item = {
        "prompt": "A task to solve",
        "abstract_text": "A scientific abstract",
        "entry_point": "solve",
    }
    monkeypatch.setattr(setup, "load_dataset", lambda *a, **kw: [item])
    tokenizer = SimpleNamespace(pad_token="<pad>", eos_token="<eos>")
    monkeypatch.setattr(
        setup.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer
    )
    observed = {}

    def create_trainer(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(train=lambda: None)

    setup.run_preference_training(
        config=config,
        section_name=section,
        args_cls=config_cls,
        trainer_cls=create_trainer,
        algorithm_name=section,
    )
    assert observed["num_agents"] == 2
    assert observed["args"].collaboration_mode == mode
    if mode == "centralized":
        runtime = CentralizedCollaboration(
            observed["centralized_comparator_adapter"],
            observed["formatters"],
            lambda left, right: [float(bool(left[0]) and bool(right[0]))],
            2,
        )
        prompt = runtime.build_prompt(item)
        assert "<agent_0>" in prompt
        assert runtime(
            ["<agent_0>short response</agent_0><agent_1>long response</agent_1>"],
            batch_items=[item],
        ) == [1.0]
        if section.endswith("_iter"):
            assert observed["args"].comparator_generation_mode == "centralized"
    elif not section.endswith("_iter"):
        assert "centralized_comparator_adapter" not in observed


@pytest.mark.parametrize("mode", ["centralized", "decentralized"])
def test_magrpo_entrypoint_selects_joint_actor(mode, monkeypatch, tmp_path):
    import sys
    import train_magrpo as entrypoint

    observed = {}
    def capture(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(train=lambda: None)

    monkeypatch.setattr(entrypoint, "MAGRPOTrainer", capture)
    monkeypatch.setattr(entrypoint, "CentralizedMAGRPOTrainer", capture)
    monkeypatch.setattr(entrypoint, "load_dataset", lambda *a, **kw: [{"prompt": "Task"}])
    monkeypatch.setattr(entrypoint.AutoTokenizer, "from_pretrained", lambda *a, **kw:
                        SimpleNamespace(pad_token="pad", eos_token="eos"))
    config = Path(__file__).resolve().parents[1] / "configs/magrpo_tldr_config.yaml"
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config), "--override",
                        f"magrpo.collaboration_mode={mode}", f"output.base_dir={tmp_path}",
                        "output.save_final_model=false", "wandb.enabled=false"])
    entrypoint.main()
    assert observed["num_agents"] == 2
    assert ("centralized_adapter" in observed) == (mode == "centralized")
    assert getattr(observed["args"], "collaboration_mode", "decentralized") == mode
