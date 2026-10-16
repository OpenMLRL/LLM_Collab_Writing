"""Optional checkpoint and W&B resume settings for training entrypoints."""

from numbers import Integral
from os import PathLike, fspath


_CHECKPOINT_KEYS = {
    "resume_from_checkpoint",
    "checkpoint_dir",
    "checkpoint_interval",
    "checkpoint_total_limit",
}


def checkpoint_train_options(config):
    """Validate explicit YAML options without reading or creating checkpoints."""
    section = config.get("checkpoint")
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError("checkpoint must be a mapping of training checkpoint options.")
    unknown = set(section) - _CHECKPOINT_KEYS
    if unknown:
        raise ValueError(
            f"Unknown checkpoint option(s): {', '.join(map(str, unknown))}"
        )
    options = dict(section)
    for name in ("resume_from_checkpoint", "checkpoint_dir"):
        value = options.get(name)
        if value is not None and (
            not isinstance(value, (str, PathLike))
            or not isinstance(fspath(value), str)
            or not fspath(value).strip()
        ):
            raise ValueError(f"checkpoint.{name} must be a non-empty path or null.")
    for name in ("checkpoint_interval", "checkpoint_total_limit"):
        if name not in options or (
            name == "checkpoint_total_limit" and options[name] is None
        ):
            continue
        value = options[name]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"checkpoint.{name} must be a positive integer.")
    return options


def prepare_wandb_config(config, wandb_config, checkpoint_options):
    """Resolve run identity before the trainer can initialize W&B."""
    section = config.get("wandb", {})
    if (
        wandb_config is None
        or section is None
        or section is False
        or isinstance(section, dict)
        and (section.get("enabled") is False or section.get("mode") == "disabled")
    ):
        return None
    section = section if isinstance(section, dict) else {}
    result = dict(wandb_config)
    for name in ("id", "resume"):
        if name in section:
            result[name] = section[name]
    if "id" in result or "resume" in result:
        return result
    checkpoint = checkpoint_options.get("resume_from_checkpoint")
    if checkpoint is not None:
        # Import lazily so ordinary launches also work with older CoMLRL builds.
        from comlrl.utils import read_checkpoint_metadata

        identity = read_checkpoint_metadata(checkpoint).get("wandb")
        if identity:
            result.update(
                {
                    name: identity[name]
                    for name in ("id", "project", "entity")
                    if name in identity
                }
            )
            result["resume"] = "allow"
    return result
