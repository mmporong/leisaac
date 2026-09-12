"""Run LeRobot ACT fine-tuning without replacing checkpoint normalization stats.

Accepts the same CLI arguments as lerobot-train. Use a local --policy.path and
--resume=false for a fresh optimizer with pretrained weights and processors.
The installed LeRobot package is not modified.
"""

from contextlib import contextmanager
from pathlib import Path


def fixed_processor_kwargs(kwargs: dict) -> dict:
    """Remove only dataset-stat overrides, retaining device/features/rename options."""
    policy_cfg = kwargs.get("policy_cfg")
    if getattr(policy_cfg, "type", None) != "act":
        raise ValueError("fixed-normalizer fine-tuning supports ACT only")
    path = kwargs.get("pretrained_path")
    if path is None:
        raise ValueError("a local pretrained --policy.path is required")
    checkpoint = Path(path).expanduser().resolve()
    for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
        if not (checkpoint / filename).is_file():
            raise ValueError(f"checkpoint processor file is missing: {checkpoint / filename}")

    result = dict(kwargs)
    result.pop("dataset_stats", None)
    for key, normalizer in (
        ("preprocessor_overrides", "normalizer_processor"),
        ("postprocessor_overrides", "unnormalizer_processor"),
    ):
        overrides = {name: dict(values) for name, values in result.get(key, {}).items()}
        if normalizer in overrides:
            overrides[normalizer].pop("stats", None)
        result[key] = overrides
    return result


@contextmanager
def preserve_checkpoint_normalizers(train_module):
    """Scope the trainer's processor-factory compatibility override to this run."""
    original = train_module.make_pre_post_processors

    def make_fixed_processors(*args, **kwargs):
        if args:
            raise TypeError("unsupported LeRobot processor factory call: expected named arguments")
        return original(**fixed_processor_kwargs(kwargs))

    train_module.make_pre_post_processors = make_fixed_processors
    try:
        yield
    finally:
        train_module.make_pre_post_processors = original


def main():
    from lerobot.scripts import lerobot_train

    with preserve_checkpoint_normalizers(lerobot_train):
        lerobot_train.main()


if __name__ == "__main__":
    main()
