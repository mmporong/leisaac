"""Derive raw MimicGen shards whose first two camera frames show the recorded initial state.

Legacy generation recorded frames 0 and 1 from the renderer's previous output (the last
frame of the previous attempt). This tool copies the selected demos of each raw shard into
a new shard directory, loads every demo's stored ``initial_state`` with ``env.reset_to``,
refreshes the cameras exactly like the fixed generator (``refresh_reset_observations``)
and overwrites ``obs/front[0:2]`` and ``obs/wrist[0:2]`` with the refreshed observation.
Everything else (actions, states, joint observations, later frames, attributes) is copied
unchanged. Originals are never modified; the output directory must not exist.
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--selection-manifest", type=Path, required=True,
                    help="candidate_selection.json listing raw shards and selected_demo_names")
parser.add_argument("--output-dir", type=Path, required=True, help="new directory receiving shard_*.hdf5")
parser.add_argument("--refresh-frames", type=int, default=4)
parser.add_argument("--observation-image-size", type=int, default=224)
parser.add_argument("--render-width", type=int, default=640)
parser.add_argument("--render-height", type=int, default=480)
parser.add_argument("--seed", type=int, default=43)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app

import hashlib
import json
import sys

import gymnasium as gym
import h5py
import numpy as np
import torch
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils import parse_env_cfg

import leisaac  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_options import refresh_reset_observations, wrap_policy_camera_observations  # noqa: E402

CAMERAS = ("front", "wrist")
TASK = "LeIsaac-SO101-PickCubeIntoBox-v0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def main() -> None:
    if args_cli.refresh_frames <= 0:
        raise ValueError("refresh-frames must be positive; the tool exists to replace stale frames")
    output = args_cli.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to write into an existing directory: {output}")
    manifest = json.loads(args_cli.selection_manifest.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    shards = manifest["shards"]
    if not shards:
        raise ValueError("selection manifest has no shards")
    output.mkdir(parents=True)
    size = args_cli.observation_image_size

    cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
    cfg.use_teleop_device("mimic_so101leader")
    cfg.recorders = None
    cfg.terminations = {}
    cfg.observations.policy.concatenate_terms = False
    cfg.seed = args_cli.seed
    for camera in CAMERAS:
        getattr(cfg.scene, camera).width = args_cli.render_width
        getattr(cfg.scene, camera).height = args_cli.render_height
    wrap_policy_camera_observations(cfg, size)
    env = gym.make(TASK, cfg=cfg).unwrapped
    robot = env.scene["robot"]

    provenance = {
        "selection_manifest": str(args_cli.selection_manifest.expanduser().resolve()),
        "selection_manifest_sha256": sha256_file(args_cli.selection_manifest.expanduser().resolve()),
        "refresh_frames": args_cli.refresh_frames,
        "observation_image_size": size,
        "render_dimensions": [args_cli.render_width, args_cli.render_height],
        "seed": args_cli.seed,
        "task": TASK,
        "replaced_frames": [0, 1],
        "code_sha256": {
            "scripts/mimic/rerender_initial_frames.py": sha256_file(Path(__file__).resolve()),
            "scripts/mimic/runtime_options.py": sha256_file(Path(__file__).resolve().with_name("runtime_options.py")),
        },
        "shards": [],
    }
    try:
        env.reset()
        with torch.inference_mode():
            for item in shards:
                raw_path = Path(item["raw_path"]).resolve(strict=True)
                raw_hash = sha256_file(raw_path)
                if raw_hash != item["raw_sha256"]:
                    raise ValueError(f"raw shard hash differs from the selection manifest: {raw_path}")
                names = list(item["selected_demo_names"])
                dst_path = output / raw_path.name
                handler = HDF5DatasetFileHandler()
                handler.open(str(raw_path))
                demos = []
                try:
                    with h5py.File(raw_path, "r") as src, h5py.File(dst_path, "x") as dst:
                        data = dst.create_group("data")
                        for key, value in src["data"].attrs.items():
                            if key != "total":
                                data.attrs[key] = value
                        total = 0
                        for name in names:
                            src.copy(src[f"data/{name}"], data, name=name)
                            demo = data[name]
                            old_front = src[f"data/{name}/obs/front"][:3]
                            old_wrist = src[f"data/{name}/obs/wrist"][:3]
                            episode = handler.load_episode(name, env.device)
                            env.reset_to(episode.get_initial_state(), torch.tensor([0], device=env.device),
                                         seed=args_cli.seed, is_relative=True)
                            robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel))
                            env.scene.write_data_to_sim()
                            refresh_reset_observations(env, args_cli.refresh_frames)
                            joint_error = float(np.abs(
                                robot.data.joint_pos[0].cpu().numpy() - src[f"data/{name}/obs/joint_pos"][0]
                            ).max())
                            new = {cam: env.obs_buf["policy"][cam][0].detach().cpu().numpy().astype(np.uint8)
                                   for cam in CAMERAS}
                            for cam in CAMERAS:
                                if new[cam].shape != (size, size, 3):
                                    raise ValueError(f"unexpected refreshed image shape {new[cam].shape}")
                                demo[f"obs/{cam}"][0:2] = np.stack([new[cam], new[cam]])
                            frames = int(demo["actions"].shape[0])
                            total += frames
                            record = {
                                "name": name, "frames": frames, "joint_pos_error_rad": joint_error,
                                "cube_xyz": env.scene["cube"].data.root_pos_w[0].cpu().tolist(),
                                "front": {"mae_new0_vs_old0": mae(new["front"], old_front[0]),
                                          "mae_new0_vs_old2": mae(new["front"], old_front[2])},
                                "wrist": {"mae_new0_vs_old0": mae(new["wrist"], old_wrist[0]),
                                          "mae_new0_vs_old2": mae(new["wrist"], old_wrist[2])},
                            }
                            demos.append(record)
                            print(f"{raw_path.name}/{name}: wrist new0 vs old0 {record['wrist']['mae_new0_vs_old0']:.2f}"
                                  f" vs old2 {record['wrist']['mae_new0_vs_old2']:.2f}", flush=True)
                        data.attrs["total"] = total
                finally:
                    handler.close()
                output_hash = sha256_file(dst_path)
                # The batch converter requires a completed manifest beside every raw shard. This one states
                # plainly that the shard is derived (frames 0/1 re-rendered), not a fresh generation.
                source_manifest = raw_path.with_suffix(".generation.json")
                dst_path.with_suffix(".generation.json").write_text(json.dumps({
                    "completed": True,
                    "derived_shard": True,
                    "derivation": "frames 0 and 1 re-rendered from initial_state; other data copied unchanged",
                    "selected_demo_names": names,
                    "source_output_file": str(raw_path), "source_sha256": raw_hash,
                    "source_manifest": str(source_manifest) if source_manifest.is_file() else None,
                    "output_file": str(dst_path), "output_sha256": output_hash,
                    "refresh_frames": args_cli.refresh_frames,
                    "rerender_provenance": str(output / "rerender_provenance.json"),
                }, indent=2) + "\n", encoding="utf-8")
                provenance["shards"].append({
                    "filename": raw_path.name, "source_path": str(raw_path), "source_sha256": raw_hash,
                    "output_path": str(dst_path), "output_sha256": output_hash,
                    "demos": demos,
                })
                (output / "rerender_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    finally:
        env.close()
    all_demos = [d for s in provenance["shards"] for d in s["demos"]]
    provenance["summary"] = {
        "demo_count": len(all_demos),
        "max_joint_pos_error_rad": max(d["joint_pos_error_rad"] for d in all_demos),
        **{f"{cam}_{key}_median": float(np.median([d[cam][key] for d in all_demos]))
           for cam in CAMERAS for key in ("mae_new0_vs_old0", "mae_new0_vs_old2")},
        **{f"{cam}_mae_new0_vs_old2_max": max(d[cam]["mae_new0_vs_old2"] for d in all_demos) for cam in CAMERAS},
    }
    (output / "rerender_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance["summary"], indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
