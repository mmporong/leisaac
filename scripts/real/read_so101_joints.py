"""Read-only SO101 follower joint readback: raw ticks and candidate simulator joint angles.

Opens the serial bus, pings the six motors, reads Present_Position and Torque_Enable, and closes the
port without writing anything (no torque, goal, EEPROM or lock writes; disconnect skips torque-off).
Optionally grabs the wrist frame served by the robot dashboard right after each reading.
Run with the LeRobot Python environment.
"""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time
import urllib.request

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
# LeIsaac's linear map between LeRobot normalized values and USD joint limits (degrees),
# source/leisaac/leisaac/assets/robots/lerobot.py and utils/robot_utils.py.
USD_LIMITS_DEG = {
    "shoulder_pan": (-110.0, 110.0), "shoulder_lift": (-100.0, 100.0), "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0), "wrist_roll": (-160.0, 160.0), "gripper": (-10.0, 100.0),
}
MOTOR_LIMITS = {name: (-100.0, 100.0) for name in MOTORS[:-1]} | {"gripper": (0.0, 100.0)}
TICKS_PER_TURN = 4096


def candidate_angles(raw: dict[str, int], calibration: dict) -> dict[str, dict[str, float]]:
    """Per joint: LeRobot normalized value, LeIsaac linear-map angle and tick angles from 2047 and range middle."""
    result = {}
    for name in MOTORS:
        cal, value = calibration[name], raw[name]
        lo, hi = cal["range_min"], cal["range_max"]
        span = (100.0 if name == "gripper" else 200.0)
        norm = (value - lo) / (hi - lo) * span + MOTOR_LIMITS[name][0]
        if cal["drive_mode"]:
            norm = -norm if name != "gripper" else 100.0 - norm
        m_lo, m_hi = MOTOR_LIMITS[name]
        j_lo, j_hi = USD_LIMITS_DEG[name]
        leisaac_deg = (norm - m_lo) / (m_hi - m_lo) * (j_hi - j_lo) + j_lo
        result[name] = {
            "raw": value,
            "lerobot_norm": norm,
            "leisaac_linear_rad": math.radians(leisaac_deg),
            "ticks_from_2047_rad": (value - 2047) * 2 * math.pi / TICKS_PER_TURN,
            "ticks_from_range_mid_rad": (value - (lo + hi) / 2) * 2 * math.pi / TICKS_PER_TURN,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14032243-if00")
    parser.add_argument("--calibration", type=Path, default=Path.home()
                        / ".cache/huggingface/lerobot/calibration/robots/so_follower/tabletop_follower_20261026.json")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--frame-url", default="http://127.0.0.1:8765/frame.jpg",
                        help="Dashboard wrist frame to save after the readings; empty string to skip.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="pose")
    parser.add_argument("--watch-seconds", type=float, default=0.0,
                        help="Keep reading for this long and save every pose held still with the cube in view.")
    parser.add_argument("--max-captures", type=int, default=4)
    parser.add_argument("--teach-poses", default=None,
                        help="Ordered name:needs_cube list, e.g. hover:1,grasp:1,box:0,rest:0. Captures one pose per "
                             "name, writes teach_poses.json and skips the cube analysis.")
    args = parser.parse_args()
    if args.samples <= 0 or args.interval < 0 or args.watch_seconds < 0 or args.max_captures <= 0:
        raise ValueError("samples and max-captures must be positive, interval and watch-seconds non-negative")
    calibration = json.loads(args.calibration.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bus = FeetechMotorsBus(
        port=args.port,
        motors={name: Motor(calibration[name]["id"], "sts3215", MotorNormMode.RANGE_M100_100) for name in MOTORS},
    )

    def read_once() -> dict:
        raw = bus.sync_read("Present_Position", normalize=False)
        torque = bus.sync_read("Torque_Enable", normalize=False)
        return {"t": time.time(), "raw": {k: int(v) for k, v in raw.items()},
                "torque_enable": {k: int(v) for k, v in torque.items()}}

    bus.connect(handshake=True)
    try:
        if args.watch_seconds == 0:
            samples = []
            for _ in range(args.samples):
                samples.append(read_once())
                time.sleep(args.interval)
            save_record(args, calibration, args.label, samples)
        else:
            watch(args, calibration, read_once)
    finally:
        bus.disconnect(disable_torque=False)


def red_pixels(jpeg: bytes) -> int:
    import cv2
    import numpy as np

    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    b, g, r = (bgr[..., i].astype(np.int16) for i in range(3))
    return int(((r > 50) & (r - np.maximum(g, b) > 40)).sum())


def grab_frame(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read()


def save_record(args, calibration, label: str, samples: list[dict], frame: bytes | None = None) -> None:
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    out = args.output_dir / f"{stamp}_{label}"
    frame_file = None
    if frame is None and args.frame_url:
        frame = grab_frame(args.frame_url)
    if frame is not None:
        frame_file = out.with_suffix(".jpg")
        frame_file.write_bytes(frame)
    raw_spread = {n: max(s["raw"][n] for s in samples) - min(s["raw"][n] for s in samples) for n in MOTORS}
    median_raw = {n: sorted(s["raw"][n] for s in samples)[len(samples) // 2] for n in MOTORS}
    record = {
        "label": label, "time": stamp, "port": args.port, "calibration_file": str(args.calibration),
        "writes_performed": False, "samples": samples, "raw_spread_ticks": raw_spread,
        "median_raw": median_raw, "angles": candidate_angles(median_raw, calibration),
        "frame_file": str(frame_file) if frame_file else None,
    }
    out.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"label": label, "median_raw": median_raw, "raw_spread_ticks": raw_spread,
                      "torque_enable": samples[-1]["torque_enable"],
                      "leisaac_linear_rad": {n: round(a["leisaac_linear_rad"], 3) for n, a in record["angles"].items()},
                      "json": str(out.with_suffix(".json"))}, ensure_ascii=False), flush=True)


def write_status(path: Path, status: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, ensure_ascii=False))
    temporary.replace(path)


def watch(args, calibration, read_once, still_ticks: int = 6, still_seconds: float = 1.5,
          moved_ticks: int = 120, min_red_pixels: int = 800) -> None:
    """Save a record each time the arm is held still with the cube in view, then wait for it to move away.

    The starting pose is treated as already captured, so the arm has to move away from it first.
    Progress goes to <output-dir>/status.json for the capture dashboard; at the end the captures are
    analysed with scripts/real/estimate_real_cube.py.
    """
    if not args.frame_url:
        raise ValueError("watch mode needs --frame-url to check that the cube is in view")
    status_path = args.output_dir / "status.json"
    started = time.time()
    deadline = started + args.watch_seconds
    teach = [(n.split(":")[0], n.split(":")[1] == "1") for n in args.teach_poses.split(",")] if args.teach_poses else None
    if teach:
        args.max_captures = len(teach)
    status = {"phase": "move", "captures": [], "target": args.max_captures, "started": started,
              "labels": [name for name, _ in teach] if teach else None,
              "deadline": deadline, "hold_fraction": 0.0, "moved_from_last": False, "cube_pixels": None,
              "result": None, "message": "팔을 시작 자세에서 옮겨 주세요"}
    window, last_saved, captures = [], None, []
    print(f"WATCH: hold the arm still about 2 s where the wrist camera sees the cube ({args.max_captures} poses)", flush=True)
    while time.time() < deadline and len(captures) < args.max_captures:
        sample = read_once()
        if last_saved is None:
            last_saved = sample["raw"]
        window = [s for s in window if sample["t"] - s["t"] <= still_seconds] + [sample]
        span = max(max(s["raw"][n] for s in window) - min(s["raw"][n] for s in window) for n in MOTORS)
        still_for = sample["t"] - window[0]["t"] if span <= still_ticks else 0.0
        if span > still_ticks:
            window = [sample]
        # The gripper counts too, so opening or closing the jaws alone is a new pose.
        away = max(abs(sample["raw"][n] - last_saved[n]) for n in MOTORS) >= moved_ticks
        status.update({"raw": sample["raw"], "now": sample["t"], "moved_from_last": away,
                       "hold_fraction": min(still_for / still_seconds, 1.0) if away else 0.0})
        if not away:
            status.update(phase="move", message="다음 자세로 팔을 옮겨 주세요" if captures else "팔을 시작 자세에서 옮겨 주세요")
        elif still_for < still_seconds * 0.9:
            status.update(phase="hold", message="그대로 2초 멈춰 주세요")
        else:
            frame = grab_frame(args.frame_url)
            pixels = red_pixels(frame)
            status["cube_pixels"] = pixels
            needs_cube = teach[len(captures)][1] if teach else True
            if needs_cube and pixels < min_red_pixels:
                status.update(phase="no_cube", message="큐브가 화면에 보이지 않습니다")
            else:
                label = teach[len(captures)][0] if teach else f"{args.label}_{len(captures) + 1}"
                save_record(args, calibration, label, window, frame)
                saved = sorted(args.output_dir.glob(f"*_{label}.json"))[-1]
                captures.append({"label": label, "json": str(saved), "jpg": saved.with_suffix(".jpg").name,
                                 "time": time.time(), "cube_pixels": pixels})
                last_saved = sample["raw"]
                status.update(phase="captured", captures=captures, message=f"{label} 포착 — 다음 자세로 옮겨 주세요")
                print(f"CAPTURED {len(captures)}/{args.max_captures} (red pixels {pixels}); move to the next pose", flush=True)
        write_status(status_path, status)
        time.sleep(args.interval)
    print(f"WATCH DONE: {len(captures)} captures", flush=True)
    if teach:
        complete = len(captures) == len(teach)
        poses = {c["label"]: json.loads(Path(c["json"]).read_text())["median_raw"] for c in captures}
        (args.output_dir / "teach_poses.json").write_text(json.dumps(
            {"poses": poses, "order": [name for name, _ in teach], "complete": complete,
             "captures": captures, "calibration_file": str(args.calibration)}, indent=2) + "\n")
        status.update(phase="done" if complete else "timeout", complete=complete,
                      message="교시 완료 — 팔을 휴식 자세에 두셔도 됩니다" if complete
                      else f"시간 종료 — {len(captures)}/{len(teach)} 자세만 기록")
        write_status(status_path, status)
        return
    if len(captures) < 2:
        status.update(phase="timeout", message=f"시간 종료 — 포착 {len(captures)}개로 분석할 수 없습니다")
        write_status(status_path, status)
        return
    status.update(phase="analyzing", message="분석 중입니다 — 팔을 내려놓으셔도 됩니다")
    write_status(status_path, status)
    import importlib.util
    import sys

    path = Path(__file__).resolve().parent / "estimate_real_cube.py"
    spec = importlib.util.spec_from_file_location("estimate_real_cube_watch", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    result = module.analyze([Path(c["json"]) for c in captures], overlay_dir=args.output_dir / "overlay")
    complete = len(captures) >= args.max_captures
    status.update(phase="done", result=result, complete=complete,
                  message="완료 — 팔을 내려놓으셔도 됩니다" if complete else f"시간 종료 — 포착 {len(captures)}개로 분석했습니다")
    write_status(status_path, status)
    print("ANALYSIS", json.dumps(result["mappings"]), flush=True)


if __name__ == "__main__":
    main()
