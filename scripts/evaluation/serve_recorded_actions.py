"""Diagnostic-only playback server. --checkpoint is a trace manifest, NOT a model."""

import argparse
import json
from pathlib import Path
import socket

import numpy as np

from run_action_intervention import sha256_file
from scripts.imitation_learning.serve_lerobot_act import receive_message, send_message


def load_commands(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("kind") != "historical_applied_action_replay":
        raise ValueError("only an explicit diagnostic replay manifest is accepted")
    if type(manifest.get("horizon")) is not int or manifest["horizon"] < 1:
        raise ValueError("recorded horizon must be a positive integer")
    path = Path(manifest["trace"])
    if sha256_file(path) != manifest["trace_sha256"]:
        raise ValueError("source trace hash changed")
    trace = json.loads(path.read_text())
    if len(trace) != manifest["horizon"] or [r["step"] for r in trace] != list(range(1, len(trace) + 1)):
        raise ValueError("recorded command steps are not a complete contiguous sequence")
    commands = np.asarray([r["applied_action"] for r in trace], dtype=np.float32)
    if commands.shape != (len(trace), 6) or not np.isfinite(commands).all():
        raise ValueError("recorded applied commands must be finite six-joint values")
    return commands


class RecordedCommands:
    def __init__(self, commands):
        self.commands, self.index = commands, 0

    def reset(self):
        self.index = 0

    def next(self):
        if self.index >= len(self.commands):
            raise ValueError("recorded horizon exhausted; extending or holding is forbidden")
        action = self.commands[self.index].tolist()
        self.index += 1
        return action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--n-action-steps", type=int, default=30)
    args = parser.parse_args()
    playback = RecordedCommands(load_commands(args.checkpoint))
    print("RECORDED_ACTION_SERVER_READY diagnostic_only=true model_loaded=false", flush=True)
    with socket.create_server((args.host, args.port), reuse_port=False) as server:
        connection, _ = server.accept()
        with connection:
            while True:
                request = receive_message(connection)
                if request.get("command") == "reset":
                    playback.reset()
                    send_message(connection, {"ok": True})
                elif request.get("command") == "shutdown":
                    send_message(connection, {"ok": True})
                    return
                elif request.get("command") == "predict":
                    send_message(connection, {"action": playback.next()})
                else:
                    raise ValueError("unsupported diagnostic playback request")


if __name__ == "__main__":
    main()
