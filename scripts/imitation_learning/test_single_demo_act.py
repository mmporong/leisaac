"""Regression checks for the single-demo autonomous-success experiment."""

from pathlib import Path
import json
import copy
import tempfile
import unittest
from unittest.mock import MagicMock, patch

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
import h5py


class SingleDemoActTest(unittest.TestCase):
    def test_training_selects_only_one_episode_without_fake_validation(self):
        command = single_training_command(Path('/new/model'), CANDIDATES[0])
        for option in ('--dataset.episodes=[20]', '--dataset.eval_split=0',
                       '--steps=3000', '--save_freq=3000', '--policy.use_vae=true',
                       '--policy.dropout=0.1', '--policy.chunk_size=30',
                       '--policy.kl_weight=10',
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

    def test_raw_and_reference_changed_together_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            raw, reference = Path(folder) / 'raw.hdf5', Path(folder) / 'reference.json'
            raw.write_bytes(b'changed raw')
            reference.write_text(json.dumps({'results': [{'raw_sha256': experiment.sha256_file(raw)}]}))
            with patch.object(experiment, 'RAW', raw), patch.object(experiment, 'REFERENCE', reference), \
                    patch.object(experiment, 'load_preflight') as load:
                with self.assertRaisesRegex(ValueError, 'original RAW sha256'):
                    experiment.preflight()
                load.assert_not_called()

    def test_full_training_config_rejects_seed_lr_kl_and_external_writes(self):
        # A saved real config is not required for this pure contract fixture.
        config = {'seed': 43, 'steps': 3000, 'batch_size': 8, 'num_workers': 2,
                  'save_freq': 3000, 'log_freq': 100, 'eval_steps': 0, 'cudnn_deterministic': True,
                  'resume': False, 'save_checkpoint': True, 'save_checkpoint_to_hub': False,
                  'wandb': {'enable': False},
                  'dataset': {'root': str(experiment.TRAIN), 'repo_id': experiment.REPO_ID,
                              'episodes': [20], 'eval_split': 0, 'image_transforms': {'enable': False},
                              'use_imagenet_stats': True, 'video_backend': 'pyav'},
                  'policy': {'type': 'act', 'device': 'cuda', 'push_to_hub': False, 'pretrained_path': None,
                             'chunk_size': 30, 'n_action_steps': 30, 'n_obs_steps': 1, 'dim_model': 256,
                             'n_heads': 8, 'dim_feedforward': 1024, 'n_encoder_layers': 4,
                             'kl_weight': 10, 'use_vae': True, 'dropout': .1,
                             'optimizer_lr': .0001, 'optimizer_lr_backbone': .00001,
                             'temporal_ensemble_coeff': None}}
        experiment.validate_training_config(config, CANDIDATES[0])
        for keys, value in ((('seed',), 44), (('policy', 'optimizer_lr'), .001),
                            (('policy', 'kl_weight'), 1), (('policy', 'push_to_hub'), True)):
            bad = copy.deepcopy(config); section = bad
            for key in keys[:-1]:
                section = section[key]
            section[keys[-1]] = value
            with self.assertRaises(ValueError):
                experiment.validate_training_config(bad, CANDIDATES[0])

    def test_loss_rows_and_nonfinite_metrics_are_rejected(self):
        candidate = dict(CANDIDATES[0], steps=200)
        good = 'step:100 loss:0.2 grdn:1 l1_loss:0.1 kld_loss:0.01\nstep:200 loss:0.1 grdn:1 l1_loss:0.1 kld_loss:0\n'
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / 'train.log'; log.write_text(good)
            self.assertEqual(experiment.audit_training_log(log, candidate)['loss_rows'], 2)
            for content in (good.splitlines()[0], good.replace('loss:0.2', 'loss:nan'),
                            good.replace('grdn:1', 'grdn:inf'), good.replace('step:200', 'step:199')):
                log.write_text(content)
                with self.assertRaises(ValueError):
                    experiment.audit_training_log(log, candidate)

    def test_early_failed_autonomous_rollout_is_rejected(self):
        with patch.object(fixtures.runner, '_audit_action_case',
                          return_value={'success': False, 'steps': 674}):
            with self.assertRaisesRegex(ValueError, 'complete horizon'):
                fixtures.runner.audit_autonomous_case(Path('/new'), 4201, {}, 30)
        with patch.object(fixtures.runner, '_audit_action_case',
                          return_value={'success': False, 'steps': 675}):
            self.assertEqual(fixtures.runner.audit_autonomous_case(Path('/new'), 4201, {}, 30)['steps'], 675)

    def test_raw_target_group_is_rejected_before_array_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / 'outputs/mimic_vision224_500_20260913/raw/shard_009.hdf5'
            model = root / 'outputs/act_reset_fixed_76_20260921/model/checkpoints/010000/pretrained_model'
            reference = root / 'reference.json'
            raw.parent.mkdir(parents=True)
            model.mkdir(parents=True)
            reference.write_text('{}')
            with h5py.File(raw, 'w') as stream:
                stream.create_group('data/demo_9/obs/joint_pos_target')
            with patch.object(fixtures.runner, 'REPO', root), \
                    patch.object(fixtures.runner, 'sha256_file', return_value=fixtures.runner.MODEL_SHA256):
                with self.assertRaisesRegex(ValueError, 'target must be an HDF5 dataset'):
                    fixtures.runner.load_preflight(raw, 'demo_9', model, reference)

    def test_offline_rejects_non_single_sample_before_prediction(self):
        from scripts.evaluation import lerobot_act_offline as offline
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = offline.argparse.Namespace(checkpoint=root / 'model', dataset_root=root / 'train',
                output=root / 'report.json', repo_id='local/test', episodes=None, max_samples=1,
                device='cpu', grasp_window=None)
            dataset = MagicMock()
            dataset.__len__.return_value = 1
            dataset.hf_dataset = {'episode_index': [0]}
            dataset.__getitem__.return_value = [{}]
            with patch.object(offline, 'parse_args', return_value=args), \
                    patch.object(offline, 'LeRobotDataset', return_value=dataset), \
                    patch.object(offline.ACTPolicy, 'from_pretrained'), \
                    patch.object(offline, 'make_pre_post_processors', return_value=(None, None)), \
                    patch.object(offline, 'predict') as predict:
                with self.assertRaisesRegex(TypeError, 'single sample mapping'):
                    offline.main()
                predict.assert_not_called()
                self.assertFalse(args.output.exists())

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
