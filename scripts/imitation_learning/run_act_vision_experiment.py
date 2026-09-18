"""Train one ACT model and evaluate it offline and in fresh simulation processes.

This Linux runner uses no LLM calls and never controls physical hardware.
Existing outputs are refused; interruption leaves checkpoints and logs intact.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.imitation_learning.run_mimic_image_batch import atomic_json, check_free_space, sha256
from scripts.evaluation.compare_act_rollouts import _load_rollout


def validate_split(root: Path) -> dict:
    marker = json.loads((root / "split_provenance.json").read_text())
    if marker.get("schema_version") != 1 or marker.get("success") is not True:
        raise ValueError("split marker does not record successful validation")
    source = Path(marker["source"])
    if sha256(source / "meta/info.json") != marker["source_info_sha256"]:
        raise ValueError("source metadata changed after splitting")
    groups = marker["splits"]
    train_ids = groups["train"]["original_episode_indices"]
    valid_ids = groups["valid"]["original_episode_indices"]
    if len(train_ids) != 400 or len(valid_ids) != 100:
        raise ValueError("this experiment requires 400 train and 100 validation episodes")
    if set(train_ids) & set(valid_ids) or sorted(train_ids + valid_ids) != list(range(500)):
        raise ValueError("train/validation overlap or incomplete coverage")
    for name in ("train", "valid"):
        part = root / name
        info = json.loads((part / "meta/info.json").read_text())
        expected = groups[name]
        if Path(expected["root"]).resolve() != part.resolve():
            raise ValueError(f"split root does not match provenance: {name}")
        if any(expected["stats_validation"].get(key) != "passed" for key in
               ("episode_stats_aggregate", "independent_full_frame_mean_std")):
            raise ValueError(f"split statistics were not verified: {name}")
        if info["total_episodes"] != expected["episode_count"] or info["total_frames"] != expected["frame_count"]:
            raise ValueError(f"split count mismatch: {name}")
        if sha256(part / "meta/stats.json") != expected["stats_sha256"]:
            raise ValueError(f"split stats changed after validation: {name}")
        for camera in ("front", "wrist"):
            if info["features"][f"observation.images.{camera}"]["shape"] != [224, 224, 3]:
                raise ValueError(f"wrong split image shape: {name}/{camera}")
    return marker


def training_command(split: Path, output: Path, repo_id: str, steps: int, batch_size: int) -> list[str]:
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch-size must be positive")
    return [
        sys.executable, "-u", "-m", "lerobot.scripts.lerobot_train",
        f"--dataset.root={split}", f"--dataset.repo_id={repo_id}",
        "--dataset.use_imagenet_stats=true", "--dataset.eval_split=0",
        "--policy.type=act", "--policy.device=cuda", "--policy.push_to_hub=false",
        "--policy.repo_id=local/so101_act_vision224_train400", "--policy.chunk_size=30",
        "--policy.n_action_steps=30", "--policy.dim_model=256", "--policy.n_heads=8",
        "--policy.dim_feedforward=1024", "--policy.n_encoder_layers=4", "--policy.use_vae=true",
        "--policy.optimizer_lr=0.0001", "--policy.optimizer_lr_backbone=0.00001",
        f"--batch_size={batch_size}", "--num_workers=2", f"--steps={steps}",
        f"--save_freq={min(5000, steps)}", "--log_freq=100", "--eval_steps=0",
        "--seed=43", "--cudnn_deterministic=true", "--wandb.enable=false", f"--output_dir={output}",
    ]


def verify_checkpoint(checkpoint: Path, train: Path, steps: int) -> dict:
    from safetensors.torch import load_file
    from lerobot.utils.constants import IMAGENET_STATS
    import torch

    config = json.loads((checkpoint / "config.json").read_text())
    training_state = json.loads((checkpoint.parent / "training_state/training_step.json").read_text())
    if training_state["step"] != steps:
        raise ValueError("training did not reach its planned step count")
    if not all(torch.isfinite(value).all() for value in load_file(checkpoint / "model.safetensors").values()):
        raise ValueError("checkpoint contains non-finite parameters")
    stats = json.loads((train / "meta/stats.json").read_text())
    checked = []
    for processor_name in ("policy_preprocessor", "policy_postprocessor"):
        processor = json.loads((checkpoint / f"{processor_name}.json").read_text())
        normalizers = [s for s in processor["steps"] if s["registry_name"] in
                       ("normalizer_processor", "unnormalizer_processor")]
        if len(normalizers) != 1:
            raise ValueError("expected one normalizer per processor")
        tensors = load_file(checkpoint / normalizers[0]["state_file"])
        keys = ("action", "observation.state") if processor_name == "policy_preprocessor" else ("action",)
        for key in keys:
            for stat in ("mean", "std"):
                np.testing.assert_allclose(tensors[f"{key}.{stat}"].numpy(), stats[key][stat], rtol=1e-6, atol=1e-7)
                checked.append(f"{processor_name}/{key}.{stat}")
        if processor_name == "policy_preprocessor":
            for camera in ("front", "wrist"):
                key = f"observation.images.{camera}"
                if config["input_features"][key]["shape"] != [3, 224, 224]:
                    raise ValueError("checkpoint is not a 224 image policy")
                for stat in ("mean", "std"):
                    np.testing.assert_allclose(tensors[f"{key}.{stat}"].numpy(), IMAGENET_STATS[stat], rtol=1e-6)
    return {"steps": steps, "model_sha256": sha256(checkpoint / "model.safetensors"),
            "train_only_state_action_stats": True, "visual_stats": "fixed ImageNet", "checked": checked}


def validate_offline_report(report: dict, valid: dict) -> None:
    if report["episodes"] != valid["episode_count"] or report["frames"] != valid["frame_count"]:
        raise ValueError("offline validation dataset count mismatch")
    for name in ("first_frames", "uniform_frames"):
        part = report[name]
        if part["samples"] <= 0 or not np.isfinite([part["rmse"], part["mae"], *part["per_joint_rmse"]]).all():
            raise ValueError("invalid or non-finite offline errors")


def validate_video(output: Path, report: dict) -> None:
    import cv2

    result = report["results"][0]
    suffix = "success" if result["success"] else "failure"
    path = output / f"rollout_001_{suffix}.mp4"
    video = cv2.VideoCapture(str(path))
    try:
        if (not video.isOpened() or int(video.get(cv2.CAP_PROP_FRAME_COUNT)) != result["steps"]
                or int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) != 1280
                or int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) != 480
                or abs(video.get(cv2.CAP_PROP_FPS) - 60) > 0.01):
            raise ValueError(f"rollout video contract mismatch: {path}")
    finally:
        video.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--isaac-python", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=list(range(4000, 4010)))
    parser.add_argument("--min-free-gib", type=float, default=8)
    args = parser.parse_args()
    split = args.split_root.expanduser().resolve(strict=True)
    marker = validate_split(split)
    isaac_python = args.isaac_python.expanduser().resolve(strict=True)
    root = args.output_dir.expanduser().resolve()
    command = training_command(split / "train", root / "model", marker["splits"]["train"]["repo_id"],
                               args.steps, args.batch_size)
    if not args.eval_seeds or len(set(args.eval_seeds)) != len(args.eval_seeds) or min(args.eval_seeds) < 0:
        raise ValueError("evaluation seeds must be unique and nonnegative")
    check_free_space(root.parent, args.min_free_gib)
    root.mkdir(exist_ok=False)
    (root / "logs").mkdir()
    plan = {"split_root": str(split), "split_provenance_sha256": sha256(split / "split_provenance.json"),
            "training_command": command, "eval_seeds": args.eval_seeds, "steps": args.steps,
            "batch_size": args.batch_size, "min_free_gib": args.min_free_gib,
            "checkpoint_selection": "predeclared final step; no selection on rollout results",
            "render_dimensions": [640, 480], "policy_image_size": 224, "horizon": 1200,
            "reset_render_frames": 4, "gripper_effort_mode": "task", "trace_steps": 60,
            "code_sha256": {name: sha256(REPO / name) for name in (
                "scripts/imitation_learning/run_act_vision_experiment.py",
                "scripts/imitation_learning/prepare_mimic_act_split.py",
                "scripts/imitation_learning/serve_lerobot_act.py",
                "scripts/evaluation/lerobot_act_offline.py",
                "scripts/evaluation/lerobot_act_so101.py",
            )},
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()}
    atomic_json(root / "plan.json", plan)
    status = {"status": "running", "stage": "prepared", "eval_completed": 0}

    def update(**values):
        status.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(root / "progress.json", status)

    def run_stage(name: str, cmd: list[str]) -> None:
        if (root / "STOP").exists():
            raise RuntimeError("STOP file found at stage boundary; partial outputs preserved")
        check_free_space(root, args.min_free_gib)
        update(stage=name, status="running")
        with (root / "logs" / f"{name}.log").open("x") as log:
            process = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            update(child_pid=process.pid)
            try:
                while True:
                    try:
                        code = process.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        check_free_space(root, args.min_free_gib)
                        update(child_pid=process.pid)
                if code:
                    raise subprocess.CalledProcessError(code, cmd)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                update(child_pid=None)

    try:
        run_stage("train", command)
        checkpoint = root / "model/checkpoints" / f"{args.steps:06d}" / "pretrained_model"
        atomic_json(root / "checkpoint_validation.json", verify_checkpoint(checkpoint, split / "train", args.steps))
        run_stage("offline_valid", [sys.executable, "-u", "scripts/evaluation/lerobot_act_offline.py",
                  "--checkpoint", str(checkpoint), "--dataset-root", str(split / "valid"),
                  "--repo-id", marker["splits"]["valid"]["repo_id"], "--output", str(root / "offline_valid.json"),
                  "--max-samples", "500", "--device", "cuda"])
        validate_offline_report(json.loads((root / "offline_valid.json").read_text()), marker["splits"]["valid"])
        reports = []
        for seed in args.eval_seeds:
            output = root / "rollouts" / f"seed_{seed}"
            run_stage(f"rollout_{seed}", [str(isaac_python), "-u", "scripts/evaluation/lerobot_act_so101.py",
                      "--checkpoint", str(checkpoint), "--num-rollouts", "1", "--horizon", "1200",
                      "--seed", str(seed), "--n-action-steps", "30", "--server-device", "cpu",
                      "--server-seed", "0", "--server-python", sys.executable, "--video-count", "1",
                      "--render-width", "640", "--render-height", "480", "--policy-image-size", "224",
                      "--reset-render-frames", "4", "--gripper-effort-mode", "task", "--trace-steps", "60",
                      "--output-dir", str(output), "--headless", "--device", "cuda:0"])
            checked = _load_rollout(root / "rollouts", seed, "vision224")
            if checked["checkpoint"] != str(checkpoint) or checked["conditions"]["policy_image_size"] != 224:
                raise ValueError("rollout checkpoint/image contract mismatch")
            report = json.loads((output / "evaluation.json").read_text())
            validate_video(output, report)
            reports.append(report)
            update(eval_completed=len(reports), successes=sum(r["successes"] for r in reports))
        summary = {"checkpoint": str(checkpoint), "num_rollouts": len(reports),
                   "successes": sum(r["successes"] for r in reports), "evaluations": reports,
                   "offline": json.loads((root / "offline_valid.json").read_text()),
                   "physical_robot_tested": False}
        summary["success_rate"] = summary["successes"] / summary["num_rollouts"]
        atomic_json(root / "evaluation_summary.json", summary)
        update(status="complete", stage="evaluated", summary=str(root / "evaluation_summary.json"))
    except BaseException as error:
        update(status="failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
