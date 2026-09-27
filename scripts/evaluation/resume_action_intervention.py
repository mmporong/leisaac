"""Resume an audited intervention prefix without overwriting any completed case."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from run_action_intervention import (
    FOLLOWUPS, REPO, SEEDS, allowed_followups, audit_case, atomic_json,
    build_command, load_preflight, require_gpu_idle, run_case, sha256_file,
    summarize_launches,
)


def validate_saved_runs(root, manifest, preflight, args):
    expected = [(seed, condition) for seed in SEEDS for condition in ("teacher_all", *FOLLOWUPS)]
    actual = [(run["audit"]["seed"], run["audit"]["condition"]) for run in manifest["runs"]]
    if not actual or actual != expected[:len(actual)]:
        raise ValueError("completed cases must be a nonempty preregistered prefix")
    for run in manifest["runs"]:
        seed, condition = run["audit"]["seed"], run["audit"]["condition"]
        output = root / f"seed_{seed}" / condition
        command = build_command(args, condition, seed, output)
        command_hash = hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest()
        if run["command"] != command or run["command_sha256"] != command_hash:
            raise ValueError("saved command changed")
        log = root / "logs" / f"seed_{seed}_{condition}.log"
        if run["log"] != str(log) or run["log_sha256"] != sha256_file(log):
            raise ValueError("saved log changed")
        audit = audit_case(output, condition, seed, preflight)
        if audit != run["audit"]:
            raise ValueError("saved case no longer matches its full audit")
        if condition == "teacher_all" and not allowed_followups(audit):
            raise ValueError("cannot continue a failed positive control")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true", help="Read-only audit of the existing prefix.")
    cli = parser.parse_args()
    root = cli.output_root.expanduser().resolve(strict=True)
    if (root / "manifest.sha256").exists():
        raise FileExistsError("diagnostic already finalized")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("stage") != "action_intervention_diagnostic":
        raise ValueError("not an intervention manifest")
    saved = manifest["preflight"]
    if saved.get("gripper_effort_mode") != "fixed":
        raise ValueError("continuation requires the fixed-effort protocol")
    preflight = load_preflight(Path(saved["raw_path"]), saved["demo"],
                              Path(saved["model"]), Path(saved["reference_path"]))
    for key, value in preflight.items():
        if key != "targets" and saved.get(key) != value:
            raise ValueError(f"saved preflight changed: {key}")
    expected_files = {"runner": REPO / "scripts/evaluation/run_action_intervention.py",
                      "evaluator": REPO / "scripts/evaluation/lerobot_act_so101.py",
                      "intervention_helper": REPO / "scripts/evaluation/action_intervention.py"}
    if set(saved["implementation"]) != set(expected_files):
        raise ValueError("unexpected implementation manifest")
    for label, path in expected_files.items():
        if saved["implementation"][label] != {"path": str(path), "sha256": sha256_file(path)}:
            raise ValueError(f"implementation changed: {label}; use a new experiment")
    preflight.update(gripper_effort_mode="fixed", implementation=saved["implementation"])
    user = os.environ.get("USER") or Path.home().name
    args = argparse.Namespace(
        isaac_python=(Path("/data") / user / "conda-envs/leisaac/bin/python").resolve(strict=True),
        server_python=(Path.home() / "miniforge3/envs/lerobot/bin/python").resolve(strict=True),
        evaluator=expected_files["evaluator"], checkpoint=Path(saved["model"]),
        raw_path=Path(saved["raw_path"]), demo=saved["demo"],
        gripper_effort_mode="fixed", gripper_effort_limit=saved["effort"][0],
    )
    validate_saved_runs(root, manifest, preflight, args)
    if cli.check_only:
        print(json.dumps({"prefix_audit_pass": True, "completed_runs": len(manifest["runs"]),
                          "gpu_checked": False, "outputs_written": False}))
        return
    require_gpu_idle()
    index = len(list(root.glob("continuation_*.json"))) + 1
    snapshot = root / f"resume_snapshot_{index:03d}.json"
    with snapshot.open("xb") as stream:
        stream.write(manifest_path.read_bytes())
    continuation = {
        "script": str(Path(__file__).resolve()), "script_sha256": sha256_file(Path(__file__)),
        "argv": sys.argv, "snapshot": str(snapshot), "snapshot_sha256": sha256_file(snapshot),
        "previous_run_count": len(manifest["runs"]),
    }
    with (root / f"continuation_{index:03d}.json").open("x") as stream:
        json.dump(continuation, stream, indent=2)
    completed = {(run["audit"]["seed"], run["audit"]["condition"]) for run in manifest["runs"]}
    for seed in SEEDS:
        for condition in ("teacher_all", *FOLLOWUPS):
            if (seed, condition) in completed:
                continue
            if condition != "teacher_all":
                teacher = next(run["audit"] for run in manifest["runs"]
                               if run["audit"]["seed"] == seed and run["audit"]["condition"] == "teacher_all")
                if not allowed_followups(teacher):
                    break
            output = root / f"seed_{seed}" / condition
            log = root / "logs" / f"seed_{seed}_{condition}.log"
            if output.exists() or log.exists():
                raise FileExistsError("unfinished unmanifested case exists; preserve it and use a new experiment")
            run = run_case(args, root, preflight, condition, seed)
            manifest["runs"].append(run)
            atomic_json(manifest_path, manifest)
    manifest["launch_comparison"] = summarize_launches(manifest["runs"])
    manifest["raw_model_unchanged"] = True
    atomic_json(manifest_path, manifest)
    digest = sha256_file(manifest_path)
    with (root / "manifest.sha256").open("x") as stream:
        stream.write(f"{digest}  manifest.json\n")
    print(json.dumps({"output": str(root), "runs": len(manifest["runs"]), "manifest_sha256": digest}))


if __name__ == "__main__":
    main()
