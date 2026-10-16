"""Launch wiring is exercised without model downloads or training jobs."""

import copy
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from checkpoint_launch import checkpoint_train_options, prepare_wandb_config
from config import Config


ENTRYPOINTS = [
    ("magrpo", "MAGRPOTrainer"),
    ("grpo", "MAGRPOTrainer"),
    ("ac", "IACTrainer"),
    ("iac", "IACTrainer"),
    ("maac", "MAACTrainer"),
    ("madpo", "MADPOTrainer"),
    ("marlhf", "MARLHFTrainer"),
    ("madpo_iter", "MADPOIterTrainer"),
    ("marlhf_iter", "MARLHFIterTrainer"),
]


def config_from(data):
    config = object.__new__(Config)
    config.data = data
    return config


def test_unconfigured_launch_needs_no_checkpoint_api_or_io(monkeypatch, tmp_path):
    import comlrl.utils

    monkeypatch.delattr(comlrl.utils, "read_checkpoint_metadata")
    config = config_from({"wandb": {"project": "original"}})
    base = {"project": "original", "name": "fresh"}
    assert checkpoint_train_options(config) == {}
    assert prepare_wandb_config(config, base, {}) == base
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "section",
    [
        False,
        [],
        "checkpoint-1",
        {"unknown": 1},
        {"checkpoint_interval": True},
        {"checkpoint_interval": 1.0},
        {"checkpoint_interval": 0},
        {"checkpoint_interval": None},
        {"checkpoint_total_limit": False},
        {"checkpoint_total_limit": -1},
        {"checkpoint_total_limit": "2"},
        {"resume_from_checkpoint": ""},
        {"checkpoint_dir": []},
    ],
)
def test_invalid_checkpoint_options_are_rejected_without_mutation(section):
    config = config_from({"checkpoint": section})
    before = copy.deepcopy(config.data)
    with pytest.raises(ValueError, match="checkpoint"):
        checkpoint_train_options(config)
    assert config.data == before


@pytest.mark.parametrize(
    "explicit", [{"id": None}, {"resume": None}, {"id": "new"}, {"resume": "never"}]
)
def test_explicit_identity_key_presence_skips_metadata(monkeypatch, explicit):
    import comlrl.utils

    def forbidden(*args, **kwargs):
        raise AssertionError("Explicit identity must skip metadata reads.")

    monkeypatch.setattr(comlrl.utils, "read_checkpoint_metadata", forbidden)
    base = {"project": "chosen", "entity": "team", "name": "display"}
    result = prepare_wandb_config(
        config_from({"wandb": explicit}), base, {"resume_from_checkpoint": "saved"}
    )
    assert result == {**base, **explicit}
    assert base == {"project": "chosen", "entity": "team", "name": "display"}


@pytest.mark.parametrize(
    "built_identity",
    [{"id": None}, {"resume": None}, {"id": "built-id"}, {"resume": "must"}],
)
def test_built_wandb_identity_also_skips_metadata(monkeypatch, built_identity):
    import comlrl.utils

    def forbidden(*args, **kwargs):
        raise AssertionError("Built W&B identity must skip metadata reads.")

    monkeypatch.setattr(comlrl.utils, "read_checkpoint_metadata", forbidden)
    base = {"project": "chosen", "name": "display", **built_identity}
    result = prepare_wandb_config(
        config_from({"wandb": {"project": "chosen"}}),
        base,
        {"resume_from_checkpoint": "saved"},
    )
    assert result == base


@pytest.mark.parametrize(
    "section,base",
    [
        (None, {}),
        (False, {}),
        ({"enabled": False}, {}),
        ({"mode": "disabled"}, {}),
        ({}, None),
    ],
)
def test_disabled_wandb_does_not_read_metadata(monkeypatch, section, base):
    import comlrl.utils

    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled W&B must skip metadata reads.")

    monkeypatch.setattr(comlrl.utils, "read_checkpoint_metadata", forbidden)
    assert (
        prepare_wandb_config(
            config_from({"wandb": section}), base, {"resume_from_checkpoint": "saved"}
        )
        is None
    )


@pytest.fixture
def launch_stub(monkeypatch, tmp_path):
    import comlrl.utils
    import transformers
    import wandb
    import preference_train_common

    def launch(
        section, trainer_name, overrides=(), identity=None, forbid_metadata=False
    ):
        entry = importlib.import_module(f"train_{section}")
        setup = (
            preference_train_common
            if section in {"madpo", "marlhf", "madpo_iter", "marlhf_iter"}
            else entry
        )
        events = []
        observed = {}

        def read_metadata(path):
            if forbid_metadata:
                raise AssertionError("Unexpected checkpoint metadata read.")
            events.append("metadata")
            observed["metadata_path"] = path
            return {"wandb": identity}

        def create_trainer(**kwargs):
            events.append("construct")
            observed["constructor"] = kwargs

            def train(**options):
                events.append("train")
                observed["train"] = options

            return SimpleNamespace(train=train)

        def dataset(*args, **kwargs):
            events.append("dataset")
            return [{"prompt": "Task", "abstract_text": "Abstract"}]

        monkeypatch.setattr(comlrl.utils, "read_checkpoint_metadata", read_metadata)
        monkeypatch.setattr(entry, trainer_name, create_trainer)
        monkeypatch.setattr(setup, "load_dataset", dataset)
        monkeypatch.setattr(
            transformers.AutoTokenizer,
            "from_pretrained",
            lambda *a, **kw: SimpleNamespace(
                pad_token="pad", eos_token="eos", add_special_tokens=lambda *a: None
            ),
        )
        monkeypatch.setattr(
            transformers.AutoModelForCausalLM,
            "from_pretrained",
            lambda *a, **kw: object(),
        )
        monkeypatch.setattr(wandb, "run", None)
        output = tmp_path / "output"
        config = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / f"{section}_tldr_config.yaml"
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train",
                "--config",
                str(config),
                "--override",
                f"output.base_dir={output}",
                "output.save_final_model=false",
                "wandb.name=display-name",
                "wandb.tags=['writing','resume']",
                "wandb.dir=chosen-wandb-dir",
                *overrides,
            ],
        )
        return entry.main, events, observed, output

    return launch


@pytest.mark.parametrize("section,trainer_name", ENTRYPOINTS)
@pytest.mark.parametrize(
    "mode",
    [
        "default",
        "resume",
        "resume_only",
        "legacy",
        "explicit_id",
        "explicit_resume",
        "disabled",
        "null",
    ],
)
def test_every_entrypoint_prepares_wandb_before_constructor_and_forwards_options(
    launch_stub, section, trainer_name, mode
):
    expected = (
        {} if mode == "default" else {"resume_from_checkpoint": "saved/checkpoint-1"}
    )
    overrides = (
        []
        if mode == "default"
        else ["checkpoint.resume_from_checkpoint=saved/checkpoint-1"]
    )
    if mode == "resume":
        expected.update(
            checkpoint_dir="new/checkpoints",
            checkpoint_interval=2,
            checkpoint_total_limit=3,
        )
        overrides += [
            "checkpoint.checkpoint_dir=new/checkpoints",
            "checkpoint.checkpoint_interval=2",
            "checkpoint.checkpoint_total_limit=3",
        ]
    if mode == "explicit_id":
        overrides.append("wandb.id=chosen-id")
    if mode == "explicit_resume":
        overrides.append("wandb.resume=never")
    if mode == "disabled":
        overrides.append("wandb.enabled=false")
    if mode == "null":
        overrides.append("wandb=null")
    read_expected = mode in {"resume", "resume_only", "legacy"}
    identity = (
        None
        if mode == "legacy"
        else {"id": "saved-id", "project": "saved-project", "entity": "saved-team"}
    )
    main, events, observed, _ = launch_stub(
        section, trainer_name, overrides, identity, forbid_metadata=not read_expected
    )
    main()
    assert observed["train"] == expected
    assert events[-2:] == ["construct", "train"]
    if read_expected:
        assert events.index("metadata") < events.index("construct")
        assert observed["metadata_path"] == "saved/checkpoint-1"
    else:
        assert "metadata" not in events
    config = observed["constructor"]["wandb_config"]
    if mode in {"disabled", "null"}:
        assert config is None
        return
    assert config["name"] == "display-name"
    assert config["tags"] == ["writing", "resume"]
    assert config["dir"] == "chosen-wandb-dir"
    assert set(config["config_sections"]) == {
        "dataset",
        "agent_model",
        "output",
        "trainer",
    }
    if mode in {"resume", "resume_only"}:
        assert {key: config[key] for key in identity} == identity
        assert config["resume"] == "allow"
    elif mode == "explicit_id":
        assert config["id"] == "chosen-id"
        assert "resume" not in config
    elif mode == "explicit_resume":
        assert config["resume"] == "never"
        assert "id" not in config
    else:
        assert "id" not in config and "resume" not in config


@pytest.mark.parametrize("section,trainer_name", ENTRYPOINTS)
def test_every_entrypoint_validates_options_before_loading_or_creating_output(
    launch_stub, section, trainer_name
):
    main, events, _, output = launch_stub(
        section,
        trainer_name,
        ["checkpoint.checkpoint_interval=0"],
        forbid_metadata=True,
    )
    with pytest.raises(ValueError, match="checkpoint_interval"):
        main()
    assert events == []
    assert not output.exists()
