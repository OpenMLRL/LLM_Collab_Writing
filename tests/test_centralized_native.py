from pathlib import Path
from types import SimpleNamespace

import pytest

import preference_train_common as setup
from comlrl.trainers.preference.collaboration import CentralizedCollaboration

ITEM = {
    "prompt": "My landlord kept the deposit after I moved out. What can I do?",
    "abstract_text": "A study of lunar dust",
    "completion": "SECRET_GOLD",
}
LEFT = "Background evidence motivates a careful study of lunar dust."
RIGHT = (
    "Building on this background, our method measures the composition of lunar "
    "dust. Therefore these results explain the observations."
)
SOURCE = {"tldr": ITEM["prompt"], "arxiv": ITEM["abstract_text"]}


def _runtime(adapter, dataset):
    return CentralizedCollaboration(
        adapter,
        setup.get_formatters(dataset),
        setup.make_reward_function(dataset),
        2,
    )


@pytest.mark.parametrize("dataset", ["tldr", "arxiv"])
def test_native_prompt_states_the_source_once(dataset):
    from centralized_comparator import get_writing_centralized_comparator_adapter
    from centralized_native import WritingNativeCentralizedAdapter

    prompt = _runtime(WritingNativeCentralizedAdapter(dataset), dataset).build_prompt(
        ITEM
    )
    assert prompt.count(SOURCE[dataset]) == 1
    assert "SECRET_GOLD" not in prompt
    for absent in ("original prompt", "another agent", "decentralized"):
        assert absent not in prompt
    assert "<agent_0>" in prompt and "<agent_1>" in prompt
    concatenated = _runtime(
        get_writing_centralized_comparator_adapter(dataset), dataset
    ).build_prompt(ITEM)
    assert concatenated.count(SOURCE[dataset]) == 2
    assert len(prompt) < len(concatenated)


@pytest.mark.parametrize("dataset", ["tldr", "arxiv"])
def test_native_prompt_keeps_the_role_formatters_error_rows(dataset):
    from centralized_native import build_native_centralized_prompt

    empty = {"prompt": "", "abstract_text": ""}
    expected = setup.get_formatters(dataset)[0](empty)
    assert build_native_centralized_prompt(empty, dataset) == expected


@pytest.mark.parametrize("dataset", ["tldr", "arxiv"])
def test_native_split_gives_the_unchanged_reward_and_logs(dataset):
    from centralized_native import WritingNativeCentralizedAdapter

    reward = setup.make_reward_function(dataset)
    runtime = _runtime(WritingNativeCentralizedAdapter(dataset), dataset)
    joint = f"<agent_0>{LEFT}</agent_0><agent_1>{RIGHT}</agent_1>"
    assert runtime([joint], batch_items=[ITEM]) == reward([LEFT], [RIGHT])
    parsed = runtime.split_eval([[[joint]]], [ITEM])
    logger = setup.get_eval_logging(dataset)["eval_logger"]
    assert logger(agent_completions_turns=parsed) == logger(
        agent_completions_turns=[[[LEFT]], [[RIGHT]]]
    )


def _run_entrypoint(monkeypatch, tmp_path, extra):
    import sys
    import train_magrpo as entrypoint

    observed = {}

    def capture(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(train=lambda: None)

    monkeypatch.setattr(entrypoint, "CentralizedMAGRPOTrainer", capture)
    monkeypatch.setattr(
        entrypoint, "load_dataset", lambda *a, **kw: [{"prompt": "Task"}]
    )
    monkeypatch.setattr(
        entrypoint.AutoTokenizer,
        "from_pretrained",
        lambda *a, **kw: SimpleNamespace(pad_token="pad", eos_token="eos"),
    )
    config = Path(__file__).resolve().parents[1] / "configs/magrpo_tldr_config.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            str(config),
            "--override",
            "magrpo.collaboration_mode=centralized",
            f"output.base_dir={tmp_path}",
            "output.save_final_model=false",
            "wandb.enabled=false",
            *extra,
        ],
    )
    entrypoint.main()
    return observed


@pytest.mark.parametrize(
    "extra,adapter_name",
    [
        ([], "WritingCentralizedComparatorAdapter"),
        (
            ["magrpo.centralized_prompt=concatenated"],
            "WritingCentralizedComparatorAdapter",
        ),
        (["magrpo.centralized_prompt=native"], "WritingNativeCentralizedAdapter"),
    ],
)
def test_entrypoint_selects_the_centralized_prompt(
    extra, adapter_name, monkeypatch, tmp_path
):
    observed = _run_entrypoint(monkeypatch, tmp_path, extra)
    assert type(observed["centralized_adapter"]).__name__ == adapter_name


def test_entrypoint_rejects_unknown_prompt_mode(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="centralized_prompt"):
        _run_entrypoint(monkeypatch, tmp_path, ["magrpo.centralized_prompt=joint"])
