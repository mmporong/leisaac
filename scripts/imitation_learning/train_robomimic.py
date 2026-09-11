"""Train a Robomimic policy without launching Isaac Sim."""

import argparse
import json
from pathlib import Path

from robomimic.config import config_factory
from robomimic.scripts.train import train
from robomimic.utils.torch_utils import get_torch_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/robomimic"))
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.expanduser().open(encoding="utf-8") as config_file:
        external_config = json.load(config_file)

    config = config_factory(external_config["algo_name"])
    with config.values_unlocked():
        config.update(external_config)
        config.train.data = str(args.dataset.expanduser().resolve())
        config.train.output_dir = str(args.output_dir.expanduser().resolve())
        if args.debug:
            config.experiment.epoch_every_n_steps = 3
            config.experiment.validation_epoch_every_n_steps = 3
            config.train.num_epochs = 2
            config.experiment.name = f"{config.experiment.name}_debug"

    config.lock()
    device = get_torch_device(try_to_use_cuda=config.train.cuda)
    train(config, device=device)


if __name__ == "__main__":
    main()
