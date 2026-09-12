"""Serve a local LeRobot ACT checkpoint to the Isaac Sim evaluator."""

import argparse
import pickle
import socket
import struct
from pathlib import Path

import numpy as np
import torch
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy


HEADER = struct.Struct("!Q")
MAX_MESSAGE_BYTES = 1_048_576
EXPECTED_STATE_SHAPE = (6,)
EXPECTED_IMAGE_SHAPE = (84, 84, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-action-steps", type=int, default=None)
    return parser.parse_args()


def receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("client disconnected")
        chunks.extend(chunk)
    return bytes(chunks)


def receive_message(connection: socket.socket):
    size = HEADER.unpack(receive_exact(connection, HEADER.size))[0]
    if size > MAX_MESSAGE_BYTES:
        raise ValueError(f"request is too large: {size} bytes")
    return pickle.loads(receive_exact(connection, size))


def send_message(connection: socket.socket, value) -> None:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    connection.sendall(HEADER.pack(len(payload)) + payload)


def build_observation(request: dict, device: torch.device) -> dict[str, torch.Tensor]:
    state = np.asarray(request["state"], dtype=np.float32)
    if state.shape != EXPECTED_STATE_SHAPE or not np.isfinite(state).all():
        raise ValueError(f"invalid state: shape={state.shape}, finite={np.isfinite(state).all()}")
    observation = {
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device)
    }
    for camera in ("front", "wrist"):
        image_value = request[camera]
        if isinstance(image_value, bytes):
            expected_bytes = int(np.prod(EXPECTED_IMAGE_SHAPE))
            if len(image_value) != expected_bytes:
                raise ValueError(
                    f"invalid {camera} byte count: {len(image_value)}, expected {expected_bytes}"
                )
            image_array = np.frombuffer(image_value, dtype=np.uint8).reshape(EXPECTED_IMAGE_SHAPE).copy()
        else:
            image_array = np.asarray(image_value)
        if image_array.shape != EXPECTED_IMAGE_SHAPE or image_array.dtype != np.uint8:
            raise ValueError(
                f"invalid {camera} image: shape={image_array.shape}, dtype={image_array.dtype}"
            )
        image = torch.from_numpy(image_array)
        observation[f"observation.images.{camera}"] = (
            image.permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(device)
        )
    return observation


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = ACTPolicy.from_pretrained(checkpoint).to(device).eval()
    if args.n_action_steps is not None:
        if not 1 <= args.n_action_steps <= model.config.chunk_size:
            raise ValueError(
                f"n-action-steps must be between 1 and chunk_size={model.config.chunk_size}"
            )
        model.config.n_action_steps = args.n_action_steps
    device_override = {"device": str(device)}
    preprocessor, postprocessor = make_pre_post_processors(
        model.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": device_override},
        postprocessor_overrides={"device_processor": device_override},
    )

    with socket.create_server((args.host, args.port), reuse_port=False) as server:
        print(
            f"ACT_SERVER_READY {args.host}:{args.port} n_action_steps={model.config.n_action_steps}",
            flush=True,
        )
        connection, _ = server.accept()
        with connection:
            while True:
                request = receive_message(connection)
                command = request.get("command")
                if command == "shutdown":
                    send_message(connection, {"ok": True})
                    return
                if command == "reset":
                    model.reset()
                    send_message(connection, {"ok": True})
                    continue
                if command != "predict":
                    raise ValueError(f"unsupported command: {command}")

                observation = build_observation(request, device)
                with torch.inference_mode():
                    action = postprocessor(model.select_action(preprocessor(observation)))
                action_array = action.squeeze(0).detach().cpu().numpy()
                if action_array.shape != EXPECTED_STATE_SHAPE or not np.isfinite(action_array).all():
                    raise RuntimeError(f"policy returned invalid action: {action_array}")
                # A plain list keeps the protocol compatible across the Isaac
                # (NumPy 1.x) and LeRobot (NumPy 2.x) Python environments.
                send_message(connection, {"action": action_array.tolist()})


if __name__ == "__main__":
    main()
