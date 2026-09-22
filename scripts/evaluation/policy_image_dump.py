"""Lossless, opt-in recording of the exact pre-action ACT input images."""
import hashlib
from pathlib import Path

from PIL import Image
import numpy as np


def validate_dump_options(every, span, horizon, trace_steps):
    if every < 0:
        raise ValueError("dump-policy-images-every cannot be negative")
    if span is not None and (len(span) != 2 or not 1 <= span[0] <= span[1] <= horizon):
        raise ValueError("dump-policy-images-range must be a one-based inclusive interval within horizon")
    if every == 0:
        if span is not None:
            raise ValueError("image range requires --dump-policy-images-every > 0")
        return None
    bounds = span if span is not None else [1, horizon]
    if trace_steps < bounds[1]:
        raise ValueError("trace-steps must cover the complete policy image dump interval")
    return tuple(bounds)


def should_dump(step, every, bounds):
    return bool(every > 0 and bounds is not None and bounds[0] <= step <= bounds[1]
                and (step - bounds[0]) % every == 0)


def write_policy_images(output_dir, trial, step, state_before, images):
    state = np.asarray(state_before)
    if trial < 1 or step < 1 or state.shape != (6,) or not np.isfinite(state).all():
        raise ValueError("invalid one-based dump index or joint state")
    if set(images) != {"front", "wrist"}:
        raise ValueError("front and wrist inputs are both required")
    for pixels in images.values():
        if pixels.dtype != np.uint8 or pixels.shape != (224, 224, 3):
            raise ValueError("policy image dumps require uint8 RGB (224,224,3) arrays")
    directory = Path(output_dir) / "policy_images" / f"rollout_{trial:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    paths = {name: directory / f"step_{step:06d}_{name}.png" for name in images}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("refusing to overwrite policy image dump")
    row = {"step": step, "observation_index": step - 1, "state_before": state.tolist()}
    for name, pixels in images.items():
        path = paths[name]
        # RGB array is identical to the byte payload sent to the inference server.
        with path.open("xb") as stream:
            Image.fromarray(pixels).save(stream, format="PNG")
        row[name] = {"path": str(path.relative_to(output_dir)),
                     "shape": list(pixels.shape), "dtype": str(pixels.dtype),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()}
    return row
