"""Safely continue screened Mimic replay, conversion, split, and ACT evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.imitation_learning.action_contract import (  # noqa: E402
    atomic_json,
    sha256_file,
    validate_aggregate_contract,
)


POLL_SECONDS = 15
EXPECTED_CANDIDATES = 76
EXPECTED_REPLAY = 20
BOOLEAN_RESULT_FIELDS = (
    "source_success", "omitted_final_transition", "success_by_common_horizon",
    "success", "final_success",
)
DEMO_NAME = re.compile(r"demo_[0-9]+")


def group_has_live_processes(pgid: int) -> bool:
    """Linux-only: zombies cannot run work and do not need another signal."""
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(") ", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] not in ("Z", "X"):
                return True
        except (FileNotFoundError, ProcessLookupError):
            continue
    return False


def terminate_owned_group(pgid: int, grace_s: float = 15) -> None:
    """Only call with the PGID created by this runner's start_new_session child."""
    if not group_has_live_processes(pgid):
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_s
    while group_has_live_processes(pgid) and time.monotonic() < deadline:
        time.sleep(.05)
    if group_has_live_processes(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def install_termination_handlers() -> dict:
    previous = {}
    requested = False
    def terminate(signum, _frame):
        nonlocal requested
        if not requested:
            requested = True
            raise SystemExit(128 + signum)
    for signum in (signal.SIGTERM, signal.SIGHUP):
        handler = signal.getsignal(signum)
        if signum == signal.SIGHUP and handler == signal.SIG_IGN:
            continue  # Preserve nohup's intentional disconnect handling.
        previous[signum] = handler
        signal.signal(signum, terminate)
    return previous


def wait_for_process_exit(pid: int, deadline: float, stop: Path) -> None:
    """Wait without signalling the pre-existing replay process; detect PID reuse."""
    stat = Path(f"/proc/{pid}/stat")
    try:
        original = stat.read_text().rsplit(") ", 1)[1].split()[19]
    except FileNotFoundError:
        return
    while time.monotonic() < deadline:
        if stop.exists():
            raise RuntimeError("STOP file found while waiting for replay process exit")
        try:
            fields = stat.read_text().rsplit(") ", 1)[1].split()
        except FileNotFoundError:
            return
        if fields[0] in ("Z", "X") or fields[19] != original:
            return
        time.sleep(min(POLL_SECONDS, max(0., deadline - time.monotonic())))
    raise TimeoutError("broad replay report is complete but process cleanup did not finish")


def check_free_space(path: Path, minimum_gib: float) -> None:
    if not math.isfinite(minimum_gib) or minimum_gib < 8:
        raise ValueError("minimum free space must be finite and at least 8 GiB")
    if shutil.disk_usage(path).free < minimum_gib * 2**30:
        raise RuntimeError(f"free space fell below {minimum_gib} GiB: {path}")


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _partial_json(path: Path) -> dict | None:
    try:
        return _read_json(path)
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _finite_numbers(value, location: str = "root") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError(f"non-finite number at {location}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_numbers(item, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_numbers(item, f"{location}.{key}")


def manifest_identities(path: Path, *, verify_raw: bool) -> tuple[list[tuple[str, str, str]], dict]:
    manifest = _read_json(path)
    if (type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
            or manifest.get("action_source") != "recorded_target"):
        raise ValueError("invalid recorded_target selection manifest")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("selection manifest must have nonempty shards")
    identities = []
    seen = set()
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise ValueError(f"invalid manifest shard {index}")
        raw_path = shard.get("raw_path")
        raw_hash = shard.get("raw_sha256")
        names = shard.get("selected_demo_names")
        if (not isinstance(raw_path, str) or not Path(raw_path).is_absolute()
                or not isinstance(raw_hash, str) or len(raw_hash) != 64
                or not isinstance(names, list) or not names):
            raise ValueError(f"invalid manifest source fields at shard {index}")
        raw = Path(raw_path).resolve(strict=verify_raw)
        if verify_raw and (not raw.is_file() or sha256_file(raw) != raw_hash):
            raise ValueError(f"manifest raw hash mismatch: {raw}")
        for name in names:
            if not isinstance(name, str) or DEMO_NAME.fullmatch(name) is None:
                raise ValueError(f"invalid demo name at shard {index}")
            identity = (str(raw), raw_hash, name)
            logical_identity = (raw_hash, name)
            if logical_identity in seen:
                raise ValueError("duplicate raw/demo identity in selection manifest")
            seen.add(logical_identity)
            identities.append(identity)
    return identities, manifest


def validate_audit_root(root: Path) -> dict:
    plan = _read_json(root / "plan.json")
    summary = _read_json(root / "summary.json")
    if (type(plan.get("schema_version")) is not int or plan["schema_version"] != 1
            or plan.get("action_source") != "recorded_target"
            or plan.get("require_unclipped_targets") is not True):
        raise ValueError("strict audit plan contract mismatch")
    if (summary.get("status") != "complete" or summary.get("screen_pass") != EXPECTED_CANDIDATES
            or summary.get("replay_cohort_size") != EXPECTED_REPLAY
            or summary.get("training_started") is not False):
        raise ValueError("strict audit summary is incomplete or has unexpected counts")
    reference = Path(plan.get("reference_path", "")).resolve(strict=True)
    if plan.get("reference_sha256") != sha256_file(reference):
        raise ValueError("strict audit reference hash mismatch")
    reference_report = _read_json(reference)
    limits = plan.get("joint_limits", {})
    if (reference_report.get("success_criteria") != plan.get("success_criteria")
            or reference_report.get("control_dt_s") != plan.get("control_dt_s")
            or reference_report.get("joint_names") != limits.get("joint_names")
            or reference_report.get("joint_lower_limits_rad") != limits.get("joint_lower_limits_rad")
            or reference_report.get("joint_upper_limits_rad") != limits.get("joint_upper_limits_rad")):
        raise ValueError("strict audit plan differs from its common reference report")
    candidates = root / "candidate_selection.json"
    replay = root / "replay_cohort.json"
    candidate_ids, candidate_manifest = manifest_identities(candidates, verify_raw=True)
    replay_ids, _ = manifest_identities(replay, verify_raw=True)
    if len(candidate_ids) != EXPECTED_CANDIDATES or len(replay_ids) != EXPECTED_REPLAY:
        raise ValueError("strict audit manifest counts differ from declared cohorts")
    if not set(replay_ids) <= set(candidate_ids):
        raise ValueError("replay cohort must be a subset of screened training candidates")
    diagnostics_path = root / "episode_diagnostics.json"
    if sha256_file(diagnostics_path) != summary.get("episode_diagnostics_sha256"):
        raise ValueError("quality diagnostics hash mismatch")
    diagnostics = _read_json(diagnostics_path).get("episodes")
    if not isinstance(diagnostics, list) or len(diagnostics) != summary.get("episode_count"):
        raise ValueError("quality diagnostics episode count mismatch")
    sources = {shard["filename"]: shard for shard in candidate_manifest["shards"]}
    screened = []
    for episode in diagnostics:
        if not isinstance(episode, dict) or type(episode.get("screen_pass")) is not bool:
            raise ValueError("invalid quality screening decision")
        if episode["screen_pass"]:
            source = sources.get(episode.get("shard"))
            if (source is None or episode.get("accepted") is not True or "diagnostic_error" in episode
                    or episode.get("recorded_release", {}).get("final_stable") is not True
                    or type(episode.get("jumps", {}).get("clipped_frames")) is not int
                    or episode["jumps"]["clipped_frames"] != 0):
                raise ValueError("selected episode violates strict quality criteria")
            screened.append((str(Path(source["raw_path"]).resolve()), episode.get("raw_sha256"),
                             episode.get("name")))
    if len(screened) != len(set(screened)) or set(screened) != set(candidate_ids):
        raise ValueError("candidate selection differs from strict episode diagnostics")
    _finite_numbers(plan, "audit_plan")
    return {"plan": plan, "summary": summary, "candidate_manifest": candidates,
            "candidate_manifest_sha256": sha256_file(candidates), "candidate_ids": candidate_ids,
            "replay_manifest": replay, "replay_manifest_sha256": sha256_file(replay),
            "replay_ids": replay_ids}


def _result_identity(result: dict) -> tuple[str, str, str]:
    dataset = result.get("dataset")
    raw_hash = result.get("raw_sha256")
    episode = result.get("episode")
    if not isinstance(dataset, str) or not isinstance(raw_hash, str) or not isinstance(episode, str):
        raise ValueError("replay result identity has wrong types")
    return str(Path(dataset).resolve()), raw_hash, episode


def validate_replay_conditions(report: dict, audit_plan: dict) -> None:
    expected_limits = audit_plan["joint_limits"]
    if (report.get("success_criteria") != audit_plan.get("success_criteria")
            or report.get("control_dt_s") != audit_plan.get("control_dt_s")
            or report.get("joint_names") != expected_limits.get("joint_names")
            or report.get("joint_lower_limits_rad") != expected_limits.get("joint_lower_limits_rad")
            or report.get("joint_upper_limits_rad") != expected_limits.get("joint_upper_limits_rad")):
        raise ValueError("replay criteria, control dt, or joint limits differ from audit plan")


def validate_replay_report(report_path: Path, manifest_path: Path, audit_plan: dict) -> dict:
    expected, _ = manifest_identities(manifest_path, verify_raw=True)
    report = _read_json(report_path)
    if (report.get("dataset") is not None
            or Path(report.get("selection_manifest", "")).resolve() != manifest_path.resolve()
            or report.get("selection_manifest_sha256") != sha256_file(manifest_path)
            or report.get("selection_action_source") != "recorded_target"
            or report.get("mode") != "recorded_target"
            or report.get("gripper_effort_mode") != "task"):
        raise ValueError("replay report source or mode contract mismatch")
    validate_replay_conditions(report, audit_plan)
    results = report.get("results")
    if not isinstance(results, list) or len(results) != len(expected):
        raise ValueError("replay result count does not match selection manifest")
    observed = [_result_identity(result) for result in results if isinstance(result, dict)]
    if len(observed) != len(results) or len(set(observed)) != len(observed) or set(observed) != set(expected):
        raise ValueError("replay result identities are missing, duplicated, or unexpected")
    for result in results:
        _finite_numbers(result, f"replay.{result.get('episode', '?')}")
        if any(type(result.get(field)) is not bool for field in BOOLEAN_RESULT_FIELDS):
            raise ValueError("replay result boolean field has non-boolean type")
        if result["final_success"] is not True:
            raise ValueError("replay cohort contains a final failure")
        if (not all(result[field] for field in BOOLEAN_RESULT_FIELDS)
                or result.get("mode") != "recorded_target"):
            raise ValueError("replay success flags or result mode are inconsistent")
        steps = result.get("steps")
        first = result.get("first_success_step")
        required_steps = math.ceil(audit_plan["success_criteria"]["hold_time_s"]
                                   / audit_plan["control_dt_s"] - 1e-9)
        if (type(steps) is not int or steps < required_steps
                or type(first) is not int or not required_steps <= first <= steps
                or type(result.get("source_frame_count")) is not int
                or result["source_frame_count"] != steps + 1
                or type(result.get("common_horizon")) is not int or result["common_horizon"] != steps):
            raise ValueError("replay duration or success timing is inconsistent")
        if type(result.get("clipped_steps")) is not int or result["clipped_steps"] != 0:
            raise ValueError("replay cohort contains clipped target commands")
        error = result.get("initial_joint_max_error_rad")
        if type(error) not in (int, float) or not math.isfinite(error) or not 0 <= error <= 1e-6:
            raise ValueError("replay initial joint error exceeds tolerance")
    return report


def validate_candidate_contract(contract: dict, expected_ids: list[tuple[str, str, str]], limits: dict) -> None:
    if (contract.get("action_source") != "recorded_target"
            or contract.get("episode_count") != len(expected_ids)
            or contract.get("joint_limits") != limits):
        raise ValueError("aggregate action contract source, count, or joint limits mismatch")
    episodes = contract.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("aggregate action contract episodes must be a list")
    observed = []
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("invalid aggregate episode record")
        observed.append((str(Path(episode.get("raw_path", "")).resolve()),
                         episode.get("raw_sha256"), episode.get("raw_demo")))
    if len(observed) != len(set(observed)) or set(observed) != set(expected_ids):
        raise ValueError("aggregate candidate identities are missing, duplicated, or unexpected")


def known_candidate_failures(report: dict, candidate_ids: list[tuple[str, str, str]]) -> list[dict]:
    candidates = set(candidate_ids)
    failures = []
    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError("broad replay results must be a list")
    for result in results:
        if not isinstance(result, dict) or type(result.get("final_success")) is not bool:
            raise ValueError("broad replay final_success must be a strict boolean")
        identity = _result_identity(result)
        if identity in candidates and result["final_success"] is False:
            failures.append({"dataset": identity[0], "raw_sha256": identity[1],
                             "episode": identity[2], "final_success": False})
    return failures


def validate_batch_output(batch_root: Path, audit: dict) -> dict:
    progress = _read_json(batch_root / "progress.json")
    aggregate = batch_root / "lerobot_all"
    if (progress.get("status") != "complete" or progress.get("target") != EXPECTED_CANDIDATES
            or progress.get("completed_episodes") != EXPECTED_CANDIDATES
            or Path(progress.get("dataset_root", "")).resolve() != aggregate.resolve()):
        raise ValueError("batch progress does not prove complete 76-episode aggregation")
    batch_plan = _read_json(batch_root / "plan.json")
    if (Path(batch_plan.get("selection_manifest", "")).resolve()
            != audit["candidate_manifest"].resolve()
            or batch_plan.get("selection_manifest_sha256") != audit["candidate_manifest_sha256"]):
        raise ValueError("batch selection manifest differs from strict audit candidates")
    contract = validate_aggregate_contract(aggregate)
    validate_candidate_contract(contract, audit["candidate_ids"], audit["plan"]["joint_limits"])
    return contract


def wait_for_replay(directory: Path, manifest: Path, deadline: float, stop: Path, audit_plan: dict) -> dict:
    expected, _ = manifest_identities(manifest, verify_raw=True)
    if len(expected) != EXPECTED_REPLAY:
        raise ValueError("broad replay manifest must contain exactly 20 fixed cohort episodes")
    expected_set = set(expected)
    report_path = directory / "evaluation.json"
    while time.monotonic() < deadline:
        if stop.exists():
            raise RuntimeError("STOP file found while waiting for broad replay")
        report = _partial_json(report_path)
        if report is not None:
            if (Path(report.get("selection_manifest", "")).resolve() != manifest.resolve()
                    or report.get("selection_manifest_sha256") != sha256_file(manifest)
                    or report.get("selection_action_source") != "recorded_target"
                    or report.get("mode") != "recorded_target" or report.get("gripper_effort_mode") != "task"):
                raise ValueError("broad replay manifest or control mode mismatch")
            validate_replay_conditions(report, audit_plan)
            results = report.get("results")
            if not isinstance(results, list):
                raise ValueError("broad replay results must be a list")
            observed = [_result_identity(item) for item in results if isinstance(item, dict)]
            if len(observed) != len(results) or len(observed) != len(set(observed)):
                raise ValueError("broad replay contains invalid or duplicate identities")
            if not set(observed) <= expected_set:
                raise ValueError("broad replay contains identities outside its manifest")
            for result in results:
                _finite_numbers(result)
                if (result.get("mode") != "recorded_target"
                        or any(type(result.get(field)) is not bool for field in BOOLEAN_RESULT_FIELDS)):
                    raise ValueError("broad replay result mode or boolean contract mismatch")
            if len(observed) == len(expected):
                return report
        time.sleep(min(POLL_SECONDS, max(0.0, deadline - time.monotonic())))
    raise TimeoutError("timed out waiting for broad replay cohort")


def wait_for_batch(batch_root: Path, deadline: float, stop: Path, audit: dict) -> dict:
    progress_path = batch_root / "progress.json"
    while time.monotonic() < deadline:
        if stop.exists():
            raise RuntimeError("STOP file found while waiting for screened conversion")
        progress = _partial_json(progress_path)
        if progress is not None:
            status = progress.get("status")
            if status == "failed":
                raise RuntimeError(f"screened conversion failed: {progress.get('error')}")
            if status in ("paused", "rejected"):
                raise RuntimeError(f"screened conversion stopped with status {status}")
            if status == "complete":
                return validate_batch_output(batch_root, audit)
        time.sleep(min(POLL_SECONDS, max(0.0, deadline - time.monotonic())))
    raise TimeoutError("timed out waiting for screened conversion")


class Pipeline:
    def __init__(self, args, root: Path, audit: dict):
        self.args = args
        self.root = root
        self.audit = audit
        self.status = {"status": "running", "stage": "prepared", "training_started": False}

    def update(self, **values) -> None:
        self.status.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(self.root / "progress.json", self.status)
        print(json.dumps({key: self.status.get(key) for key in ("status", "stage")}), flush=True)

    def boundary(self, stage: str) -> None:
        if (self.root / "STOP").exists():
            raise RuntimeError(f"STOP file found before {stage}")
        check_free_space(self.root, self.args.min_free_gib)

    def run_child(self, stage: str, command: list[str]) -> None:
        self.boundary(stage)
        self.update(status="running", stage=stage, child_pid=None)
        with (self.root / "logs" / f"{stage}.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            lingering = False
            try:
                self.update(child_pid=process.pid)
                while True:
                    try:
                        code = process.wait(timeout=POLL_SECONDS)
                        break
                    except subprocess.TimeoutExpired:
                        check_free_space(self.root, self.args.min_free_gib)
                        self.update(child_pid=process.pid)
                if code:
                    raise subprocess.CalledProcessError(code, command)
                lingering = group_has_live_processes(process.pid)
            finally:
                terminate_owned_group(process.pid)
                process.wait()
                self.update(child_pid=None)
            if lingering:
                raise RuntimeError(f"{stage} leader exited with live descendants; owned group terminated")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--wait-replay-dir", type=Path, required=True)
    parser.add_argument("--wait-replay-manifest", type=Path, required=True)
    parser.add_argument("--wait-replay-pid", type=int,
                        help="Wait for a still-running replay interpreter; omit only after confirmed exit.")
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--isaac-python", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=list(range(5000, 5010)))
    parser.add_argument("--wait-timeout-seconds", type=float, default=3600)
    parser.add_argument("--min-free-gib", type=float, default=8.0)
    args = parser.parse_args()
    if ((args.wait_replay_pid is not None and args.wait_replay_pid <= 0)
            or args.steps <= 0 or args.batch_size <= 0 or not args.eval_seeds
            or len(args.eval_seeds) != len(set(args.eval_seeds)) or min(args.eval_seeds) < 0
            or not math.isfinite(args.wait_timeout_seconds) or args.wait_timeout_seconds <= 0
            or not math.isfinite(args.min_free_gib) or args.min_free_gib < 8):
        raise ValueError("invalid bounded pipeline numeric arguments")
    audit_root = args.audit_root.expanduser().resolve(strict=True)
    wait_replay_dir = args.wait_replay_dir.expanduser().resolve(strict=True)
    wait_replay_manifest = args.wait_replay_manifest.expanduser().resolve(strict=True)
    batch_root = args.batch_root.expanduser().resolve(strict=True)
    isaac_python = args.isaac_python.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    check_free_space(output.parent, args.min_free_gib)
    audit = validate_audit_root(audit_root)
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    strict_replay = output / "strict_replay"
    split = output / "split"
    act = output / "act"
    replay_command = [str(isaac_python), "-u", "scripts/evaluation/replay_joint_contract.py",
                      "--selection-manifest", str(audit["replay_manifest"]), "--mode", "recorded_target",
                      "--gripper-effort-mode", "task", "--video-count", str(EXPECTED_REPLAY),
                      "--output-dir", str(strict_replay), "--headless", "--device", "cuda:0"]
    split_command = [sys.executable, "-u", "scripts/imitation_learning/prepare_mimic_act_split.py",
                     "--dataset-root", str(batch_root / "lerobot_all"),
                     "--repo-id", "local/so101_mimic_vision224_76", "--output-dir", str(split),
                     "--shard-size", "25", "--valid-per-shard", "5"]
    act_command = [sys.executable, "-u", "scripts/imitation_learning/run_act_vision_experiment.py",
                   "--split-root", str(split), "--output-dir", str(act),
                   "--isaac-python", str(isaac_python), "--steps", str(args.steps),
                   "--batch-size", str(args.batch_size), "--eval-seeds", *map(str, args.eval_seeds),
                   "--min-free-gib", str(args.min_free_gib)]
    plan = {"schema_version": 1, "audit_root": str(audit_root),
            "audit_plan_sha256": sha256_file(audit_root / "plan.json"),
            "candidate_manifest": str(audit["candidate_manifest"]),
            "candidate_manifest_sha256": audit["candidate_manifest_sha256"],
            "strict_replay_manifest": str(audit["replay_manifest"]),
            "strict_replay_manifest_sha256": audit["replay_manifest_sha256"],
            "wait_replay_dir": str(wait_replay_dir), "wait_replay_manifest": str(wait_replay_manifest),
            "wait_replay_pid": args.wait_replay_pid,
            "wait_replay_manifest_sha256": sha256_file(wait_replay_manifest),
            "batch_root": str(batch_root), "isaac_python": str(isaac_python),
            "steps": args.steps, "batch_size": args.batch_size, "eval_seeds": args.eval_seeds,
            "wait_timeout_seconds": args.wait_timeout_seconds, "min_free_gib": args.min_free_gib,
            "commands": {"strict_replay": replay_command, "split": split_command, "act": act_command}}
    atomic_json(output / "plan.json", plan)
    pipeline = Pipeline(args, output, audit)
    previous_handlers = install_termination_handlers()
    try:
        pipeline.boundary("broad_replay_wait")
        pipeline.update(stage="waiting_broad_replay")
        broad_report = wait_for_replay(wait_replay_dir, wait_replay_manifest,
                                       time.monotonic() + args.wait_timeout_seconds, output / "STOP",
                                       audit["plan"])
        known_failures = known_candidate_failures(broad_report, audit["candidate_ids"])
        pipeline.update(known_candidate_failures=known_failures)
        if args.wait_replay_pid is not None:
            wait_for_process_exit(args.wait_replay_pid, time.monotonic() + args.wait_timeout_seconds,
                                  output / "STOP")
        pipeline.run_child("strict_replay", replay_command)
        try:
            validate_replay_report(strict_replay / "evaluation.json", audit["replay_manifest"], audit["plan"])
            if known_failures:
                raise ValueError("broad replay already disproved one or more screened training candidates")
        except BaseException as error:
            pipeline.update(status="rejected", stage="strict_replay_gate", training_started=False,
                            error=f"{type(error).__name__}: {error}")
            raise
        pipeline.boundary("batch_wait")
        pipeline.update(stage="waiting_screened_conversion")
        wait_for_batch(batch_root, time.monotonic() + args.wait_timeout_seconds, output / "STOP", audit)
        pipeline.run_child("split", split_command)
        pipeline.update(training_started=True)
        pipeline.run_child("act", act_command)
        pipeline.update(status="complete", stage="evaluated", training_started=True)
    except BaseException as error:
        if pipeline.status.get("status") != "rejected":
            pipeline.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
