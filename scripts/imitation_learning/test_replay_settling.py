import unittest

from scripts.evaluation.replay_settling import observation_steps, settling_summary


class SettlingTest(unittest.TestCase):
    def test_bounded_duration(self):
        self.assertEqual(observation_steps(0, 1 / 60), 0)
        self.assertEqual(observation_steps(1, 1 / 60), 60)
        for seconds in (-1, 5.01, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                observation_steps(seconds, 1 / 60)
        with self.assertRaises(ValueError):
            observation_steps(1, 0)

    def test_original_failure_is_preserved_when_settling_succeeds(self):
        result = settling_summary(False, [{"success": True}] * 60, 1 / 60)
        self.assertFalse(result["source_horizon_final_success"])
        self.assertTrue(result["final_success"])
        self.assertEqual(result["first_success_step"], 1)
        self.assertEqual(result["classification"], "settled_during_observation")

    def test_no_observation_cannot_claim_success(self):
        for source in (False, True):
            result = settling_summary(source, [], 1 / 60)
            self.assertIsNone(result["final_success"])
            self.assertEqual(result["classification"], "not_observed")

    def test_transient_and_persistent_failures_are_distinct(self):
        for source, values, expected in (
            (False, [False, False], "unresolved_after_observation"),
            (False, [True, False], "unstable_during_observation"),
            (True, [False, False], "unstable_during_observation"),
            (True, [True, True], "stable_at_source_horizon"),
            (True, [False, True], "stable_at_source_horizon"),
            (False, [True, False, True], "settled_during_observation"),
        ):
            self.assertEqual(settling_summary(source, [{"success": x} for x in values], 1/60)
                             ["classification"], expected)

    def test_non_boolean_success_is_rejected(self):
        with self.assertRaises(ValueError):
            settling_summary(False, [{"success": 1}], 1 / 60)


if __name__ == "__main__":
    unittest.main()
