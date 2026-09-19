"""CPU-only tests for action-contract lineage and fail-closed validation."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.imitation_learning.action_contract import (
    CONTRACT_FILENAME,
    atomic_json,
    build_aggregate_contract,
    build_split_contract,
    sha256_file,
    validate_aggregate_contract,
    validate_conversion_provenance,
    validate_split_contract,
)


class ActionContractTest(unittest.TestCase):
    def _aggregate(self, root: Path, action_source: str = "recorded_target") -> Path:
        aggregate = root / "aggregate"
        aggregate.mkdir()
        raw = root / "shard.hdf5"
        raw.write_bytes(b"raw-demo")
        limits = root / "limits.json"
        limit_payload = {
            "joint_names": ["a", "b", "c", "d", "e", "f"],
            "joint_lower_limits_rad": [-1.0] * 6,
            "joint_upper_limits_rad": [1.0] * 6,
        }
        limits.write_text(json.dumps(limit_payload))
        provenance = {
            "input_sha256": sha256_file(raw),
            "action_source": action_source,
            "image_size": 224,
            "episode_count": 3,
            "frame_count": 9,
            "episodes": [
                {"name": f"demo_{index}", "frame_count": 3} for index in range(3)
            ],
        }
        if action_source == "recorded_target":
            provenance.update({
                "target_alignment": "action[t] = obs/joint_pos_target[t+1]",
                "joint_limits": {
                    "path": str(limits), "sha256": sha256_file(limits),
                    **limit_payload,
                },
            })
        provenance_path = root / "conversion_provenance.json"
        atomic_json(provenance_path, provenance)
        contract = build_aggregate_contract(
            aggregate,
            action_source,
            [{"raw_path": str(raw), "conversion_provenance_path": str(provenance_path)}],
            3,
            9,
        )
        atomic_json(aggregate / CONTRACT_FILENAME, contract)
        return aggregate

    def test_recorded_target_lineage_survives_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aggregate = self._aggregate(root)
            source = validate_aggregate_contract(aggregate)
            self.assertEqual(source["episodes"][2]["raw_demo"], "demo_2")
            split = root / "split"
            split.mkdir()
            atomic_json(
                split / CONTRACT_FILENAME,
                build_split_contract(aggregate, source, {"train": [0, 2], "valid": [1]}),
            )
            validated = validate_split_contract(split)
            self.assertEqual(validated["splits"]["train"]["source_episode_indices"], [0, 2])

    def test_live_raw_hash_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aggregate = self._aggregate(root)
            (root / "shard.hdf5").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "raw source changed"):
                validate_aggregate_contract(aggregate)

    def test_legacy_contract_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            aggregate = self._aggregate(Path(directory), "next_observed")
            with self.assertRaisesRegex(ValueError, "explicit opt-in"):
                validate_aggregate_contract(aggregate)
            self.assertEqual(
                validate_aggregate_contract(aggregate, allow_legacy=True)["action_source"],
                "next_observed",
            )

    def test_aggregate_rejects_different_valid_limits_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aggregate = self._aggregate(root)
            contract_path = aggregate / CONTRACT_FILENAME
            contract = json.loads(contract_path.read_text())
            limits = root / "other_limits.json"
            payload = json.loads((root / "limits.json").read_text())
            payload["joint_upper_limits_rad"] = [2.0] * 6
            atomic_json(limits, payload)
            contract["joint_limits"] = {"path": str(limits), "sha256": sha256_file(limits), **payload}
            atomic_json(contract_path, contract)
            with self.assertRaisesRegex(ValueError, "source joint limits differ"):
                validate_aggregate_contract(aggregate)

    def test_split_rejects_relabelled_legacy_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aggregate = self._aggregate(root, "stored")
            source = validate_aggregate_contract(aggregate, allow_legacy=True)
            split = root / "split"
            split.mkdir()
            contract = build_split_contract(aggregate, source, {"train": [0, 2], "valid": [1]})
            contract["action_source"] = "next_observed"
            contract["alignment"] = "action[t] = obs/joint_pos[min(t+1, T-1)]"
            atomic_json(split / CONTRACT_FILENAME, contract)
            with self.assertRaisesRegex(ValueError, "split action_source differs"):
                validate_split_contract(split, allow_legacy=True)

    def test_aggregate_rejects_wrong_legacy_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            aggregate = self._aggregate(Path(directory))
            path = aggregate / CONTRACT_FILENAME
            contract = json.loads(path.read_text())
            contract["legacy_action_source"] = True
            atomic_json(path, contract)
            with self.assertRaisesRegex(ValueError, "legacy action-source flag"):
                validate_aggregate_contract(aggregate)

    def test_conversion_requires_explicit_alignment_and_matching_embedded_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._aggregate(root)
            path = root / "conversion_provenance.json"
            original = json.loads(path.read_text())
            for mutation in ("alignment", "limits"):
                payload = json.loads(json.dumps(original))
                if mutation == "alignment":
                    payload.pop("target_alignment")
                else:
                    payload["joint_limits"]["joint_upper_limits_rad"][0] = 2.0
                atomic_json(path, payload)
                with self.assertRaises(ValueError):
                    validate_conversion_provenance(path, root / "shard.hdf5", "recorded_target")

    def test_aggregate_rejects_wrong_or_duplicated_episode_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            aggregate = self._aggregate(Path(directory))
            path = aggregate / CONTRACT_FILENAME
            original = json.loads(path.read_text())
            payload = json.loads(json.dumps(original))
            payload["episodes"][0]["raw_demo"] = "demo_unknown"
            atomic_json(path, payload)
            with self.assertRaisesRegex(ValueError, "episode lineage"):
                validate_aggregate_contract(aggregate)
            payload = json.loads(json.dumps(original))
            payload["sources"].append(payload["sources"][0])
            atomic_json(path, payload)
            with self.assertRaisesRegex(ValueError, "duplicate raw-sha/demo"):
                validate_aggregate_contract(aggregate)

    def test_split_rejects_wrong_episode_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aggregate = self._aggregate(root)
            source = validate_aggregate_contract(aggregate)
            split = root / "split"
            split.mkdir()
            contract = build_split_contract(aggregate, source, {"train": [0, 2], "valid": [1]})
            contract["splits"]["train"]["episodes"][0] = source["episodes"][1]
            atomic_json(split / CONTRACT_FILENAME, contract)
            with self.assertRaisesRegex(ValueError, "train episode lineage"):
                validate_split_contract(split)


if __name__ == "__main__":
    unittest.main()
