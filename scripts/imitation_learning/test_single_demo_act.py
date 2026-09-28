"""Regression checks for the single-demo autonomous-success experiment."""

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from scripts.imitation_learning import run_single_demo_act as experiment
from scripts.imitation_learning import test_action_intervention_runner as fixtures

from scripts.imitation_learning.run_single_demo_act import (
    CANDIDATES, FINAL_SEEDS, SCREEN_SEEDS, choose_candidate, final_success,
    single_training_command, validate_subset_arrays,
    validate_manifest,
    validate_output_location, validate_selection,
    validate_completed_stage, validate_screen_order,
)
import numpy as np


class SingleDemoActTest(unittest.TestCase):
    def test_training_selects_only_one_episode_without_fake_validation(self):
        command = single_training_command(Path('/new/model'), CANDIDATES[0])
        for option in ('--dataset.episodes=[20]', '--dataset.eval_split=0',
                       '--steps=3000', '--save_freq=3000', '--policy.use_vae=true',
                       '--policy.dropout=0.1', '--policy.chunk_size=30',
                       '--policy.n_action_steps=30', '--policy.push_to_hub=false'):
            self.assertIn(option, command)
        self.assertFalse(any('policy.path=' in value for value in command))

    def test_deterministic_candidate_is_explicit_and_preserves_other_settings(self):
        command = single_training_command(Path('/new/model'), CANDIDATES[2])
        self.assertIn('--policy.use_vae=false', command)
        self.assertIn('--policy.dropout=0.0', command)
        self.assertIn('--steps=10000', command)
        self.assertIn('--save_freq=10000', command)

    def test_subset_contract_rejects_shift_or_another_episode(self):
        states = np.arange(676 * 6, dtype=np.float32).reshape(676, 6)
        validate_subset_arrays([20] * 675, states[:-1], states[1:], states, states)
        for ids, observed, action in (([19] * 675, states[:-1], states[1:]),
                                       ([20] * 675, states[1:], states[1:]),
                                       ([20] * 675, states[:-1], states[:-1])):
            with self.assertRaises(ValueError):
                validate_subset_arrays(ids, observed, action, states, states)

    def test_screening_selection_is_fixed_and_never_uses_final_seeds(self):
        self.assertFalse(set(SCREEN_SEEDS) & set(FINAL_SEEDS))
        def case(name, wins, lifts, error, order):
            return {'candidate': name, 'queue': 30, 'successes': wins,
                    'lifts': lifts, 'grasp_rmse': error, 'order': order}
        self.assertIsNone(choose_candidate([case('p3k', 0, 3, 0.01, 0)]))
        first = case('p3k', 1, 2, 0.05, 0)
        better = case('p10k', 2, 2, 0.1, 2)
        self.assertEqual(choose_candidate([first, better]), better)
        complete = case('d10k', 3, 3, 0.2, 4)
        self.assertEqual(choose_candidate([first, complete]), complete)

    def test_final_requires_all_ten_policy_only_results(self):
        rows = [{'seed': seed, 'success': i < 8, 'condition': 'policy',
                 'audit_pass': True} for i, seed in enumerate(FINAL_SEEDS)]
        self.assertTrue(final_success(rows))
        rows[7]['success'] = False
        self.assertFalse(final_success(rows))
        with self.assertRaises(ValueError):
            final_success(rows[:-1])
        rows[0]['condition'] = 'teacher_all'
        with self.assertRaises(ValueError):
            final_success(rows)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(experiment, 'preflight', return_value={}), \
                patch.object(experiment.sys, 'argv', ['runner', '--stage', 'prepare', '--output-root', folder]):
            with self.assertRaises((FileExistsError, ValueError)):
                experiment.main()

    def test_output_cannot_be_inside_or_above_protected_inputs(self):
        for path in (experiment.TRAIN / 'new', experiment.SPLIT / 'new',
                     experiment.ORIGINAL_MODEL / 'new', experiment.RAW.parent / 'new', experiment.REPO):
            with self.assertRaises(ValueError):
                validate_output_location(path)
        validate_output_location(experiment.REPO / 'outputs/act_single_demo_new_test')

    def test_selection_payload_and_checkpoint_must_match_screening(self):
        root = Path('/new')
        case = {'candidate': 'p3k', 'queue': 30, 'successes': 3, 'lifts': 3,
                'grasp_rmse': 0.01, 'order': 0,
                'checkpoint': '/new/p3k/model/checkpoints/003000/pretrained_model'}
        self.assertEqual(validate_selection(dict(case, selection_sha256='sha'), [case], root), CANDIDATES[0])
        with self.assertRaises(ValueError):
            validate_selection(dict(case, queue=2), [case], root)
        with self.assertRaises(ValueError):
            validate_selection(dict(case, checkpoint='/other/model'), [case], root)

    def test_resume_rejects_changed_registration_or_inputs(self):
        base = {'protected': {'original': 'sha'}}
        manifest = {'schema_version': 1, 'experiment': 'single_demo_act_autonomous_success',
                    'preflight': base, 'candidates': list(CANDIDATES),
                    'screen_seeds': list(SCREEN_SEEDS), 'final_seeds': list(FINAL_SEEDS)}
        validate_manifest(manifest, base)
        for key in ('preflight', 'candidates', 'screen_seeds', 'final_seeds'):
            bad = dict(manifest, **{key: []})
            with self.assertRaises(ValueError):
                validate_manifest(bad, base)

    def test_completed_stage_is_bound_to_command_log_and_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / 'stage.log'
            log.write_text('done')
            stored = {'command': ['train'], 'log_sha256': experiment.sha256_file(log),
                      'result': {'model_sha256': 'model'}}
            self.assertEqual(validate_completed_stage(stored, ['train'], log,
                             lambda: {'model_sha256': 'model'}), stored['result'])
            for value, command, result in ((None, ['train'], stored['result']),
                                          (stored, ['other'], stored['result']),
                                          (stored, ['train'], {'model_sha256': 'changed'})):
                with self.assertRaises(ValueError):
                    validate_completed_stage(value, command, log, lambda: result)
            log.write_text('partial or changed')
            with self.assertRaises(ValueError):
                validate_completed_stage(stored, ['train'], log, lambda: stored['result'])

    def test_screening_order_and_early_stop_cannot_be_changed(self):
        cases = [{'candidate': 'p3k', 'queue': 30, 'order': 0, 'successes': 1},
                 {'candidate': 'p3k', 'queue': 1, 'order': 1, 'successes': 3}]
        validate_screen_order(cases)
        for rows in ([], cases[:1], cases[::-1], cases[1:], cases + [dict(cases[0])],
                     [dict(cases[0], successes=3), cases[1]]):
            with self.assertRaises(ValueError):
                validate_screen_order(rows)

    def test_queue_one_is_opt_in_and_default_thirty_is_preserved(self):
        helper = fixtures.runner
        with tempfile.TemporaryDirectory() as folder:
            case, base = fixtures.ActionInterventionRunnerTest().fixture(Path(folder), 'policy')
            path = case / 'evaluation.json'
            value = json.loads(path.read_text())
            value['n_action_steps'] = 1
            path.write_text(json.dumps(value))
            old_hash = helper.sha256_file
            def digest(path):
                if str(path).endswith('model.safetensors'):
                    return 'model'
                return 'raw' if str(path) == '/raw' else old_hash(path)
            with patch.object(helper, 'sha256_file', side_effect=digest):
                self.assertTrue(helper.audit_autonomous_case(case, 4101, base, n_action_steps=1)['audit_pass'])
                with self.assertRaisesRegex(ValueError, 'n_action_steps'):
                    helper.audit_case(case, 'policy', 4101, base)


if __name__ == '__main__':
    unittest.main()
