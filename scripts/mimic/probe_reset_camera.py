"""Measure whether the camera observations right after a reset show the reset scene.

Measurement script for the Mimic reset camera fix, not meant to be merged. For each camera it records:

* whether the image returned by the upstream reset (``ManagerBasedRLMimicEnv.reset``, i.e. without the fix)
  is bit-identical to the last observation before the reset,
* the MAE to a converged image (10 extra renders and a forced camera read) after k = 0..4 renders,
* whether the physics state changes while re-rendering,
* the result of ``env.reset()`` / ``env.reset_to()`` of the env class that is checked out.

Writes ``<out>.json`` with all values and ``<out>.txt`` with a short summary.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import statistics
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="LeIsaac-SO101-LiftCube-Mimic-v0")
parser.add_argument("--trials", type=int, default=3)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--move_steps", type=int, default=30)
parser.add_argument("--label", default="run")
parser.add_argument("--out", default=None, help="output path without extension")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import isaaclab_mimic.envs  # noqa: E402,F401
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLMimicEnv  # noqa: E402
from isaaclab.sensors import Camera  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import leisaac  # noqa: E402

out = args.out or f"probe_reset_camera_{args.label}_{args.task.split('-')[2].lower()}"
env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env_cfg.task_type = "so101leader"
env = gym.make(args.task, cfg=env_cfg).unwrapped
cameras = {name: s for name, s in env.scene.sensors.items() if isinstance(s, Camera)}


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


results = {
    "label": args.label,
    "task": args.task,
    "num_envs": env.num_envs,
    "leisaac_path": leisaac.__file__,
    "env_class": f"{type(env).__name__} ({inspect.getsourcefile(type(env))})",
    "isaaclab": version("isaaclab"),
    "isaacsim": version("isaacsim"),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "cameras": list(cameras),
    "has_num_rerenders_on_reset": hasattr(env.cfg, "num_rerenders_on_reset"),
    "trials": [],
}
print(f"[probe] {json.dumps({k: v for k, v in results.items() if k != 'trials'})}")


def images(env_id=0):
    return {name: env.obs_buf["policy"][name][env_id].detach().float().cpu().clone() for name in cameras}


def sha(t):
    return hashlib.sha256(t.to(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def mae(a, b):
    return round(float((a - b).abs().mean()), 3)


def physics_digest():
    h = hashlib.sha256()
    state = env.scene.get_state(is_relative=False)
    for kind in sorted(state):
        for name in sorted(state[kind]):
            for key in sorted(state[kind][name]):
                h.update(state[kind][name][key].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def force_read(env_ids, renders):
    for _ in range(renders):
        env.sim.render()
    for cam in cameras.values():
        cam.reset(env_ids)
        cam.update(0.0, force_recompute=True)
    env.obs_buf = env.observation_manager.compute()


def move_arm():
    """Move the end effector with regular env steps so that the scene before the reset differs."""
    ee = env.obs_buf["policy"]["ee_frame_state"][:, :7].clone()
    target = ee.clone()
    target[:, 0] += 0.05
    target[:, 2] += 0.08
    for _ in range(args.move_steps):
        action = torch.cat([target, torch.ones(env.num_envs, 1, device=env.device)], dim=1)
        env.step(action)


env_ids0 = torch.tensor([0], dtype=torch.int64, device=env.device)

with torch.inference_mode():
    env.reset()
    force_read(None, 10)
    reset_state = env.scene.get_state(is_relative=True)

    for trial in range(args.trials):
        rows = []
        for k in range(5):
            move_arm()
            before = images()
            ManagerBasedRLMimicEnv.reset(env, env_ids=env_ids0)  # upstream behaviour, without the fix
            returned = images()
            digest0 = physics_digest()
            force_read(env_ids0, k)
            after_k = images()
            force_read(env_ids0, 10)
            converged = images()
            row = {"k": k, "physics_unchanged": digest0 == physics_digest()}
            for name in cameras:
                row[name] = {
                    "reset_equals_before": sha(returned[name]) == sha(before[name]),
                    "mae_reset_vs_before": mae(returned[name], before[name]),
                    "mae_reset_vs_converged": mae(returned[name], converged[name]),
                    "mae_k_vs_converged": mae(after_k[name], converged[name]),
                    "k_equals_before": sha(after_k[name]) == sha(before[name]),
                }
            rows.append(row)
        checked = {}
        for method in ("reset", "reset_to"):
            move_arm()
            before = images()
            t0 = time.perf_counter()
            if method == "reset":
                env.reset(env_ids=env_ids0)
            else:
                env.reset_to(reset_state, env_ids0, is_relative=True)
            elapsed_ms = round((time.perf_counter() - t0) * 1e3, 1)
            returned = images()
            force_read(env_ids0, 10)
            converged = images()
            checked[method] = {"ms": elapsed_ms}
            for name in cameras:
                checked[method][name] = {
                    "equals_before": sha(returned[name]) == sha(before[name]),
                    "mae_vs_before": mae(returned[name], before[name]),
                    "mae_vs_converged": mae(returned[name], converged[name]),
                }
        results["trials"].append({"trial": trial, "k_rows": rows, "checked_out": checked})
        print(f"[probe] trial {trial} done")

    if env.num_envs > 1:
        move_arm()
        env1_before = images(1)
        env.reset(env_ids=env_ids0)
        env1_after = images(1)
        results["other_env_unchanged"] = {name: sha(env1_before[name]) == sha(env1_after[name]) for name in cameras}

# summary over trials
lines = [
    f"# {args.label} {args.task} isaaclab={results['isaaclab']} isaacsim={results['isaacsim']} gpu={results['gpu']}"
]
lines.append(f"# env class: {results['env_class']}")
n = len(results["trials"])
for name in cameras:
    stale = sum(t["k_rows"][0][name]["reset_equals_before"] for t in results["trials"])
    lines.append(f"[{name}] upstream reset returns the pre-reset image: {stale}/{n} trials")
    lines.append(f"[{name}] k renders | still equals pre-reset | median MAE vs converged | physics unchanged")
    for k in range(5):
        rows = [t["k_rows"][k] for t in results["trials"]]
        equal = sum(r[name]["k_equals_before"] for r in rows)
        med = statistics.median(r[name]["mae_k_vs_converged"] for r in rows)
        phys = sum(r["physics_unchanged"] for r in rows)
        lines.append(f"[{name}]   k={k} | {equal}/{n} | {med} | {phys}/{n}")
    for method in ("reset", "reset_to"):
        vals = [t["checked_out"][method][name] for t in results["trials"]]
        equal = sum(v["equals_before"] for v in vals)
        med = statistics.median(v["mae_vs_converged"] for v in vals)
        ms = statistics.median(t["checked_out"][method]["ms"] for t in results["trials"])
        lines.append(
            f"[{name}] checked-out env.{method}(): equals pre-reset {equal}/{n}, median MAE vs converged {med}"
        )
        lines.append(f"[{name}]   median time {ms} ms")
if "other_env_unchanged" in results:
    lines.append(f"other env unchanged after env.reset(env_ids=[0]): {results['other_env_unchanged']}")

with open(f"{out}.json", "w") as f:
    json.dump(results, f, indent=1)
with open(f"{out}.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"[probe] wrote {out}.json and {out}.txt")
env.close()
simulation_app.close()
