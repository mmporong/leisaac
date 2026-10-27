"""Taught-pose pick and place on the physical SO101 follower: hover -> grasp -> close -> lift -> box -> open -> rest.

Poses come from read_so101_joints.py --teach-poses (raw Present_Position ticks recorded with torque off),
so no simulator joint mapping is involved. Without --execute the plan is printed and the serial port is
not opened. With --execute:
  - only RAM registers are written (Goal_Position, Torque_Enable, Goal_Velocity, gripper Torque_Limit);
    no EEPROM, calibration or lock writes;
  - torque is enabled only after Goal_Position is set to the present position, so enabling does not jump;
  - each move is interpolated in 20 ms steps at a capped joint speed;
  - every step checks tracking error and every few steps the load; a stall or overload holds the arm where
    it is (goal = present) and stops the sequence with torque on, because cutting torque mid-air drops it;
  - the gripper closes until it stops on the cube, then the goal is set just past the contact so the
    servo does not keep pushing at full effort;
  - torque is turned off only after reaching the taught rest pose.
Run with the LeRobot Python environment.
"""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import signal
import time

MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ARM = MOTORS[:-1]
TICKS_PER_DEG = 4096 / 360.0
DT = 0.02
ALLOWED_WRITES = {"Goal_Position", "Torque_Enable", "Goal_Velocity", "Torque_Limit"}


class Stop(RuntimeError):
    """The sequence stopped with the arm held in place."""


def load_magnitude(raw: int) -> int:
    """STS Present_Load: bit 10 is the direction, the low 10 bits are 0..1000 (per mille of max torque)."""
    return int(raw) & 0x3FF


def build_plan(poses: dict, close_target_ticks: int, hold_squeeze_ticks: int) -> list[dict]:
    for name in ("hover", "grasp", "box", "rest"):
        if name not in poses:
            raise ValueError(f"teach file has no '{name}' pose")
    hover, grasp, box, rest = (poses[n] for n in ("hover", "grasp", "box", "rest"))
    open_grip = max(hover["gripper"], grasp["gripper"])
    close_target = int(close_target_ticks)

    def arm(p):
        return {m: int(p[m]) for m in ARM}

    return [
        {"name": "hover", "arm": arm(hover), "gripper": open_grip},
        {"name": "grasp", "arm": arm(grasp), "gripper": open_grip},
        {"name": "close", "arm": arm(grasp), "gripper": close_target, "close": True,
         "squeeze_ticks": hold_squeeze_ticks},
        {"name": "lift", "arm": arm(hover), "gripper": None},
        {"name": "box", "arm": arm(box), "gripper": None},
        {"name": "open", "arm": arm(box), "gripper": open_grip},
        {"name": "rest", "arm": arm(rest), "gripper": int(rest["gripper"])},
    ]


def check_targets(plan: list[dict], calibration: dict, poses: dict, margin: int = 120) -> None:
    """Every target must lie inside the calibrated range widened to what was taught by hand."""
    for motor in MOTORS:
        taught = [int(p[motor]) for p in poses.values()]
        lo = min(calibration[motor]["range_min"], min(taught)) - margin
        hi = max(calibration[motor]["range_max"], max(taught)) + margin
        for step in plan:
            value = step["gripper"] if motor == "gripper" else step["arm"][motor]
            if value is not None and not lo <= value <= hi:
                raise ValueError(f"{step['name']}: {motor} target {value} outside [{lo}, {hi}]")


class Runner:
    def __init__(self, bus, speed_deg_s: float, gripper_speed_deg_s: float, log: list):
        self.bus, self.log = bus, log
        self.arm_step = max(1, int(round(speed_deg_s * DT * TICKS_PER_DEG)))
        self.grip_step = max(1, int(round(gripper_speed_deg_s * DT * TICKS_PER_DEG)))
        self.goal: dict[str, int] = {}
        self.stall_ticks, self.stall_seconds, self.overload, self.overload_seconds = 150, 0.6, 800, 0.5

    def write(self, register: str, values: dict) -> None:
        assert register in ALLOWED_WRITES, register
        self.bus.sync_write(register, values, normalize=False)

    def present(self) -> dict:
        return {k: int(v) for k, v in self.bus.sync_read("Present_Position", normalize=False).items()}

    def hold(self, why: str):
        here = self.present()
        self.write("Goal_Position", here)
        self.goal = dict(here)
        self.log.append({"t": time.time(), "event": "hold", "why": why, "present": here})
        raise Stop(why)

    def start(self, gripper_torque_limit: int) -> dict:
        torque = self.bus.sync_read("Torque_Enable", normalize=False)
        if any(int(v) for v in torque.values()):
            raise Stop(f"torque already on at start {dict(torque)}; refusing to take over an energized arm")
        here = self.present()
        self.write("Goal_Position", here)
        self.write("Goal_Velocity", {m: 0 for m in MOTORS})
        self.write("Torque_Limit", {"gripper": gripper_torque_limit})
        self.write("Torque_Enable", {m: 1 for m in MOTORS})
        self.goal = dict(here)
        self.log.append({"t": time.time(), "event": "torque_on", "present": here})
        return here

    def move(self, name: str, arm: dict | None, gripper: int | None, settle_ticks: int = 15, settle_s: float = 2.0):
        target = dict(self.goal)
        if arm:
            target.update(arm)
        if gripper is not None:
            target["gripper"] = int(gripper)
        steps = max(math.ceil(max(abs(target[m] - self.goal[m]) for m in ARM) / self.arm_step),
                    math.ceil(abs(target["gripper"] - self.goal["gripper"]) / self.grip_step), 1)
        start = dict(self.goal)
        lagging_since = None
        heavy_since = None
        for i in range(1, steps + 1):
            t0 = time.monotonic()
            goal = {m: int(round(start[m] + (target[m] - start[m]) * i / steps)) for m in MOTORS}
            self.write("Goal_Position", goal)
            self.goal = goal
            here = self.present()
            lag = max(abs(goal[m] - here[m]) for m in ARM)
            now = time.monotonic()
            lagging_since = (lagging_since or now) if lag > self.stall_ticks else None
            if lagging_since and now - lagging_since > self.stall_seconds:
                self.hold(f"{name}: arm tracking error {lag} ticks for {self.stall_seconds}s")
            if i % 5 == 0:
                load = {m: load_magnitude(v) for m, v in self.bus.sync_read("Present_Load", normalize=False).items()}
                worst = max(load[m] for m in ARM)
                heavy_since = (heavy_since or now) if worst > self.overload else None
                if heavy_since and now - heavy_since > self.overload_seconds:
                    self.hold(f"{name}: arm load {worst}/1000 for {self.overload_seconds}s")
            time.sleep(max(0.0, DT - (time.monotonic() - t0)))
        deadline = time.monotonic() + settle_s
        while True:
            here = self.present()
            err = max(abs(self.goal[m] - here[m]) for m in ARM)
            if err <= settle_ticks or time.monotonic() > deadline:
                break
            time.sleep(DT)
        if err > self.stall_ticks:
            self.hold(f"{name}: arm still {err} ticks from the goal after {settle_s}s")
        self.log.append({"t": time.time(), "event": "reached", "step": name, "goal": self.goal, "present": here,
                         "max_arm_error_ticks": err})
        return here

    def close_gripper(self, target: int, squeeze_ticks: int, timeout_s: float = 6.0) -> int:
        """Close toward target; once the gripper stops on the cube, park the goal just past contact."""
        start = self.goal["gripper"]
        steps = max(math.ceil(abs(target - start) / self.grip_step), 1)
        last, still = None, 0
        t_end = time.monotonic() + timeout_s
        for i in range(1, steps + 1):
            goal = int(round(start + (target - start) * i / steps))
            self.write("Goal_Position", {"gripper": goal})
            self.goal["gripper"] = goal
            time.sleep(DT)
            here = self.present()["gripper"]
            still = still + 1 if last is not None and abs(here - last) <= 2 and abs(goal - here) > 30 else 0
            last = here
            if still >= 10 or time.monotonic() > t_end:
                break
        here = self.present()["gripper"]
        hold_goal = here - squeeze_ticks if target < start else here + squeeze_ticks
        self.write("Goal_Position", {"gripper": hold_goal})
        self.goal["gripper"] = hold_goal
        self.log.append({"t": time.time(), "event": "gripper_closed", "contact_ticks": here, "hold_goal": hold_goal})
        return here

    def torque_off(self):
        self.write("Torque_Enable", {m: 0 for m in MOTORS})
        self.log.append({"t": time.time(), "event": "torque_off"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teach", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Open the serial port and move the arm.")
    parser.add_argument("--port", default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14032243-if00")
    parser.add_argument("--calibration", type=Path, default=Path.home()
                        / ".cache/huggingface/lerobot/calibration/robots/so_follower/tabletop_follower_20261026.json")
    parser.add_argument("--speed-deg-s", type=float, default=20.0)
    parser.add_argument("--gripper-speed-deg-s", type=float, default=40.0)
    parser.add_argument("--close-target-ticks", type=int, default=None,
                        help="Gripper goal when closing (default: calibrated fully closed); the cube stops the fingers first.")
    parser.add_argument("--squeeze-ticks", type=int, default=40)
    parser.add_argument("--gripper-torque-limit", type=int, default=500, help="RAM Torque_Limit for the gripper (0..1000).")
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/so101_real_pick_runs"))
    args = parser.parse_args()
    if not 1.0 <= args.speed_deg_s <= 45.0:
        raise ValueError("speed-deg-s must be between 1 and 45")
    teach = json.loads(args.teach.read_text())
    calibration = json.loads(args.calibration.read_text())
    close_target = args.close_target_ticks if args.close_target_ticks is not None else calibration["gripper"]["range_min"]
    plan = build_plan(teach["poses"], close_target, args.squeeze_ticks)
    check_targets(plan, calibration, teach["poses"])
    print("PLAN (raw ticks)")
    for step in plan:
        print(f"  {step['name']:6s} arm {step['arm']} gripper {step['gripper']}")
    if not args.execute:
        print("dry run: serial port not opened; add --execute to move the arm")
        return

    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    bus = FeetechMotorsBus(port=args.port, motors={m: Motor(calibration[m]["id"], "sts3215", MotorNormMode.RANGE_M100_100)
                                                   for m in MOTORS})
    log: list = []
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"{stamp}_pick.json"
    interrupted = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: interrupted.update(flag=True))
    outcome = "not_started"
    bus.connect(handshake=True)
    runner = Runner(bus, args.speed_deg_s, args.gripper_speed_deg_s, log)
    try:
        runner.start(args.gripper_torque_limit)
        for step in plan:
            if interrupted["flag"]:
                runner.hold("operator interrupt")
            print(f"STEP {step['name']}", flush=True)
            if step.get("close"):
                runner.close_gripper(step["gripper"], step["squeeze_ticks"])
            else:
                runner.move(step["name"], step["arm"], step["gripper"])
        runner.torque_off()
        outcome = "completed"
    except Stop as stop:
        outcome = f"stopped: {stop}"
        print(f"STOPPED with the arm held in place (torque on): {stop}", flush=True)
    finally:
        log_path.write_text(json.dumps({"teach": str(args.teach), "plan": plan, "outcome": outcome,
                                        "speed_deg_s": args.speed_deg_s, "events": log}, indent=2) + "\n")
        bus.disconnect(disable_torque=False)
        print(f"OUTCOME {outcome}; log {log_path}", flush=True)


if __name__ == "__main__":
    main()
