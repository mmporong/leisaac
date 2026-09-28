"""Train a single verified demonstration and audit autonomous ACT rollouts.

No physical hardware is controlled. Original datasets and models are read-only.
Use prepare, screen, then final; subsequent invocations require --resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import cv2
import h5py
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.evaluation.run_action_intervention import (
    HORIZON, audit_autonomous_case, build_command, load_preflight, require_gpu_idle,
)
from scripts.imitation_learning.action_contract import atomic_json, sha256_file
from scripts.imitation_learning.run_act_vision_experiment import (
    run_owned_stage, training_command, validate_split, verify_checkpoint,
)
from scripts.imitation_learning.run_screened_mimic_pipeline import install_termination_handlers

SPLIT = REPO / 'outputs/act_reset_fixed_76_20260921_split'
TRAIN = SPLIT / 'train'
REPO_ID = 'local/so101_mimic_vision224_76_train'
ORIGINAL_MODEL = REPO / 'outputs/act_reset_fixed_76_20260921/model/checkpoints/010000/pretrained_model'
RAW = REPO / 'outputs/mimic_vision224_500_20260913/raw/shard_009.hdf5'
RESET_RAW = REPO / 'outputs/mimic_reset_fixed_76_20260921/raw/shard_009.hdf5'
REFERENCE = REPO / 'outputs/evaluation/control_diagnosis_20260921/shard009_recorded_target/evaluation.json'
SCREEN_SEEDS = (4201, 4202, 4203)
FINAL_SEEDS = tuple(range(4301, 4311))
CANDIDATES = (
    {'name': 'p3k', 'steps': 3000, 'use_vae': True, 'dropout': 0.1},
    {'name': 'p10k', 'steps': 10000, 'use_vae': True, 'dropout': 0.1},
    {'name': 'd10k', 'steps': 10000, 'use_vae': False, 'dropout': 0.0},
)


def single_training_command(output, candidate):
    command = training_command(TRAIN, output, REPO_ID, candidate['steps'], 8)
    command = [f"--save_freq={candidate['steps']}" if value.startswith('--save_freq=')
               else f"--policy.use_vae={str(candidate['use_vae']).lower()}"
               if value.startswith('--policy.use_vae=') else value for value in command]
    return command + ['--dataset.episodes=[20]', '--dataset.image_transforms.enable=false',
                      '--dataset.video_backend=pyav', f"--policy.dropout={candidate['dropout']}"]


def validate_subset_arrays(ids, state, action, raw_state, target):
    if (len(ids) != 675 or set(ids) != {20} or state.shape != (675, 6)
            or action.shape != (675, 6) or raw_state.shape != (676, 6)
            or target.shape != (676, 6) or not np.isfinite(state).all()
            or not np.isfinite(action).all() or not np.array_equal(state, raw_state[:-1])
            or not np.array_equal(action, target[1:])):
        raise ValueError('single-demo episode/state/recorded-target t+1 contract changed')


def choose_candidate(cases):
    complete = [case for case in cases if case['successes'] == 3]
    positive = [case for case in cases if case['successes'] > 0]
    if complete:
        return min(complete, key=lambda case: case['order'])
    return min(positive, key=lambda case: (-case['successes'], -case['lifts'],
                                         case['grasp_rmse'], case['order'])) if positive else None


def final_success(rows):
    if ([row['seed'] for row in rows] != list(FINAL_SEEDS)
            or any(row.get('condition') != 'policy' or row.get('audit_pass') is not True for row in rows)):
        raise ValueError('final evaluation requires all ten preregistered autonomous audited runs')
    return sum(row['success'] is True for row in rows) >= 8


def validate_manifest(manifest, saved_base):
    if (manifest.get('schema_version') != 1
            or manifest.get('experiment') != 'single_demo_act_autonomous_success'
            or manifest.get('preflight') != saved_base
            or manifest.get('candidates') != list(CANDIDATES)
            or manifest.get('screen_seeds') != list(SCREEN_SEEDS)
            or manifest.get('final_seeds') != list(FINAL_SEEDS)):
        raise ValueError('saved implementation/input/preregistration changed; preserve the root')


def validate_output_location(root):
    root = root.resolve()
    protected = (SPLIT, ORIGINAL_MODEL.parents[3], RAW.parent.parent,
                 RESET_RAW.parent.parent, REPO / 'datasets')
    if (any(root == path.resolve() or root.is_relative_to(path.resolve())
            or path.resolve().is_relative_to(root) for path in protected)
            or root.parent != (REPO / 'outputs').resolve()
            or not root.name.startswith('act_single_demo_')):
        raise ValueError('output must be a dedicated outputs/act_single_demo_* sibling, outside protected trees')


def validate_selection(selection, cases, root):
    expected = choose_candidate(cases)
    payload = {key: value for key, value in selection.items() if key != 'selection_sha256'}
    if expected is None or payload != expected or selection['queue'] not in (30, 1):
        raise ValueError('locked selection differs from the deterministic preregistered screening result')
    candidate = next((item for item in CANDIDATES if item['name'] == selection['candidate']), None)
    if candidate is None:
        raise ValueError('unknown locked candidate')
    path = root / candidate['name'] / 'model/checkpoints' / f"{candidate['steps']:06d}" / 'pretrained_model'
    if Path(selection['checkpoint']).resolve() != path.resolve():
        raise ValueError('selection checkpoint is not its screened candidate')
    return candidate


def validate_completed_stage(stored, command, log, verify):
    if stored is None:
        raise ValueError('required screening stage is incomplete')
    if stored['command'] != command or stored['log_sha256'] != sha256_file(log):
        raise ValueError('completed stage command/log changed')
    result = verify()
    if stored['result'] != result:
        raise ValueError('completed stage artifact changed')
    return result


def validate_screen_order(cases):
    expected = [(candidate['name'], queue, order * 2 + qorder)
                for order, candidate in enumerate(CANDIDATES)
                for qorder, queue in enumerate((30, 1))]
    actual = [(row['candidate'], row['queue'], row['order']) for row in cases]
    if (not actual or actual != expected[:len(actual)]
            or any(row['successes'] == 3 for row in cases[:-1])):
        raise ValueError('screening cases differ from the preregistered prefix/stop rule')
    if cases[-1]['successes'] != 3 and len(cases) != len(expected):
        raise ValueError('screening prefix is incomplete')


def screening_case(candidate, order, queue, error, audits, checkpoint):
    if ([row['seed'] for row in audits] != list(SCREEN_SEEDS)
            or any(row['condition'] != 'policy' or row['audit_pass'] is not True
                   or row['queue'] != queue for row in audits)):
        raise ValueError('screening requires three autonomous audited runs')
    return {'candidate': candidate['name'], 'checkpoint': str(checkpoint),
            'model_sha256': sha256_file(checkpoint / 'model.safetensors'), 'queue': queue,
            'successes': sum(row['success'] for row in audits),
            'lifts': sum(row['first_lift_step'] is not None for row in audits),
            'grasp_rmse': error['grasp_rmse'], 'order': order}


def preflight():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.sampler import EpisodeAwareSampler
    validate_split(SPLIT)
    result = load_preflight(RAW, 'demo_9', ORIGINAL_MODEL, REFERENCE)
    d = LeRobotDataset(REPO_ID, root=TRAIN, episodes=[20],
                      delta_timestamps={'action': [i / 60 for i in range(30)]}, video_backend='pyav')
    indices = list(EpisodeAwareSampler(
        d.meta.episodes['dataset_from_index'], d.meta.episodes['dataset_to_index'],
        episode_indices_to_use=d.episodes, absolute_to_relative_idx=d.absolute_to_relative_idx))
    if len(d) != 675 or sorted(indices) != list(range(675)):
        raise ValueError('sampler must contain exactly the 675 selected relative frame indices')
    state = np.stack([np.asarray(x) for x in d.hf_dataset['observation.state']])
    action = np.stack([np.asarray(x) for x in d.hf_dataset['action']])
    with h5py.File(RESET_RAW, 'r') as stream:
        group = stream['data/demo_9']
        validate_subset_arrays(np.asarray(d.hf_dataset['episode_index']).tolist(), state, action,
                               group['obs/joint_pos'][:], group['obs/joint_pos_target'][:])
    marker = json.loads((SPLIT / 'split_provenance.json').read_text())
    if marker['splits']['train']['original_to_new_episode_index'].get('24') != 20:
        raise ValueError('aggregate24 -> train20 mapping changed')
    result.update(gripper_effort_mode='fixed', n_action_steps=30,
                  subset={'episodes': [20], 'frames': 675, 'stats_scope': 'train61'},
                  protected={str(path): sha256_file(path) for path in (
                      TRAIN / 'meta/stats.json', TRAIN / 'meta/info.json',
                      TRAIN / d.meta.get_data_file_path(20), RESET_RAW, RAW,
                      ORIGINAL_MODEL / 'model.safetensors')},
                  implementation={label: {'path': str(path), 'sha256': sha256_file(path)}
                      for label, path in {
                          'runner': Path(__file__).resolve(),
                          'training_helper': REPO / 'scripts/imitation_learning/run_act_vision_experiment.py',
                          'audit_helper': REPO / 'scripts/evaluation/run_action_intervention.py',
                          'evaluator': REPO / 'scripts/evaluation/lerobot_act_so101.py',
                          'server': REPO / 'scripts/imitation_learning/serve_lerobot_act.py',
                          'offline': REPO / 'scripts/evaluation/lerobot_act_offline.py',
                      }.items()})
    for camera in ('front', 'wrist'):
        path = TRAIN / d.meta.get_video_file_path(20, f'observation.images.{camera}')
        result['protected'][str(path)] = sha256_file(path)
    import lerobot
    import subprocess
    package = Path(lerobot.__file__).resolve()
    checkout = package.parents[2]
    revision = subprocess.run(['git', '-C', str(checkout), 'rev-parse', 'HEAD'],
                              check=True, capture_output=True, text=True).stdout.strip()
    result['lerobot_source'] = {'import_path': str(package), 'revision': revision}
    return result


def guard_resources(root):
    if shutil.disk_usage(root).free < 8 * 2**30 or shutil.disk_usage(Path.home()).free < 2**30:
        raise RuntimeError('new stage requires at least 8GiB data and 1GiB home free; originals preserved')
    size = sum(path.stat().st_size for path in root.rglob('*') if path.is_file() and not path.is_symlink())
    if size > 4 * 2**30:
        raise RuntimeError('new experiment exceeded its 4GiB output guard')


def verify_training(checkpoint, candidate):
    report = verify_checkpoint(checkpoint, TRAIN, candidate['steps'])
    config = json.loads((checkpoint / 'train_config.json').read_text())
    policy = config['policy']
    if (config['dataset']['episodes'] != [20] or config['dataset']['eval_split'] != 0
            or config['dataset']['image_transforms']['enable']
            or Path(config['dataset']['root']).resolve() != TRAIN.resolve()
            or config['batch_size'] != 8 or config['steps'] != candidate['steps']
            or policy['use_vae'] != candidate['use_vae'] or policy['dropout'] != candidate['dropout']
            or policy['chunk_size'] != 30 or policy['n_action_steps'] != 30):
        raise ValueError('trained checkpoint differs from the preregistered single-demo candidate')
    report['train_config_sha256'] = sha256_file(checkpoint / 'train_config.json')
    return report


def verify_offline(path):
    value = json.loads(path.read_text())
    if value['episodes'] != 1 or value['frames'] != 675 or value['uniform_frames']['samples'] != 675:
        raise ValueError('offline self-fit must cover every selected frame')
    window = value['grasp_window']
    if (window['frame_start'], window['frame_end'], window['samples']) != (240, 329, 90):
        raise ValueError('offline grasp interval must correspond to steps241..330')
    if not np.isfinite([window['rmse'], value['uniform_frames']['rmse'], *window['per_joint_rmse']]).all():
        raise ValueError('non-finite offline predictions')
    return {'sha256': sha256_file(path), 'grasp_rmse': window['rmse'],
            'uniform_rmse': value['uniform_frames']['rmse'], 'grasp_per_joint_rmse': window['per_joint_rmse']}


def audit_rollout(output, queue, seed, base, checkpoint):
    case = dict(base, model=str(checkpoint), model_sha256=sha256_file(checkpoint / 'model.safetensors'))
    audit = audit_autonomous_case(output, seed, case, n_action_steps=queue)
    trace = json.loads((output / 'trace_001.json').read_text())
    for row in trace:
        if any(np.asarray(row[key]).shape != (6,) or not np.isfinite(row[key]).all()
               for key in ('state_before', 'state_after')) or not np.isfinite(row['cube_xyz']).all():
            raise ValueError('invalid physical trace state')
    inputs = json.loads((output / 'policy_images_001.json').read_text())
    expected = list(range(241, min(330, audit['steps']) + 1))
    if [row['step'] for row in inputs] != expected:
        raise ValueError('pre-action input window changed')
    for row in inputs:
        if row['state_before'] != trace[row['step'] - 1]['state_before']:
            raise ValueError('input state differs from pre-action trace')
        for camera in ('front', 'wrist'):
            record = row[camera]
            path = (output / record['path']).resolve(strict=True)
            if not path.is_relative_to(output.resolve()) or sha256_file(path) != record['sha256']:
                raise ValueError('input PNG path/hash changed')
            with Image.open(path) as image:
                pixels = np.asarray(image)
            if (pixels.shape != (224, 224, 3) or pixels.dtype != np.uint8
                    or hashlib.sha256(pixels.tobytes()).hexdigest() != record['pixel_sha256']):
                raise ValueError('policy input pixels changed')
    video = cv2.VideoCapture(audit['video'])
    try:
        if (not video.isOpened() or int(video.get(cv2.CAP_PROP_FRAME_COUNT)) != audit['steps']
                or int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) != 1280
                or int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) != 480
                or abs(video.get(cv2.CAP_PROP_FPS) - 60) > 0.01):
            raise ValueError('video frame/size/fps contract changed')
    finally:
        video.release()
    audit.update(queue=queue, input_frames=len(inputs),
                 input_manifest_sha256=sha256_file(output / 'policy_images_001.json'))
    return audit


def main():
    install_termination_handlers()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=REPO / 'outputs/act_single_demo_success_v3_20260928')
    parser.add_argument('--stage', choices=('prepare', 'screen', 'final'), required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    base = preflight()
    saved_base = {key: value for key, value in base.items() if key != 'targets'}
    root = args.output_root.expanduser().resolve()
    if args.check_only:
        print(json.dumps({'preflight_pass': True, 'subset': base['subset'], 'writes': False}))
        return
    validate_output_location(root)
    if args.resume:
        manifest = json.loads((root / 'manifest.json').read_text())
        validate_manifest(manifest, saved_base)
    else:
        if args.stage != 'prepare' or root.exists():
            raise FileExistsError('first invocation must prepare a new output root; use --resume for later stages')
        root.mkdir(parents=True)
        (root / 'logs').mkdir()
        manifest = {'schema_version': 1, 'experiment': 'single_demo_act_autonomous_success',
                    'preflight': saved_base, 'candidates': CANDIDATES, 'screen_seeds': SCREEN_SEEDS,
                    'final_seeds': FINAL_SEEDS, 'stages': {}, 'cases': [], 'selection': None,
                    'goal_success': False}
    # Both training and inference subprocesses inherit this explicit CPU thread bound.
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    manifest['environment'] = {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}

    def update(**values):
        guard_resources(root)
        manifest['progress'] = values
        atomic_json(root / 'manifest.json', manifest)

    def stage(name, command, verify, required_completed=False):
        stored = manifest['stages'].get(name)
        log = root / 'logs' / f'{name}.log'
        if stored is not None:
            return validate_completed_stage(stored, command, log, verify)
        if required_completed:
            raise ValueError(f'required screening stage is incomplete: {name}')
        guard_resources(root)
        if (root / 'STOP').exists():
            raise RuntimeError('STOP requested; partial experiment preserved')
        require_gpu_idle(wait_seconds=10)
        print(f'START {name}', flush=True)
        run_owned_stage(root, 8.0, update, name, command)
        result = verify()
        manifest['stages'][name] = {'command': command, 'log_sha256': sha256_file(log), 'result': result}
        update(stage=name, status='complete')
        print(f'DONE {name} {json.dumps(result)}', flush=True)
        return result

    def checkpoint_path(candidate):
        return root / candidate['name'] / 'model/checkpoints' / f"{candidate['steps']:06d}" / 'pretrained_model'

    def rollout(candidate, queue, seed, label, required_completed=False):
        checkpoint = checkpoint_path(candidate)
        output = root / candidate['name'] / f'{label}_q{queue}_seed{seed}'
        user = Path.home().name
        ns = argparse.Namespace(isaac_python=Path('/data') / user / 'conda-envs/leisaac/bin/python',
                                server_python=Path.home() / 'miniforge3/envs/lerobot/bin/python',
                                evaluator=REPO / 'scripts/evaluation/lerobot_act_so101.py',
                                checkpoint=checkpoint, raw_path=RAW, demo='demo_9',
                                gripper_effort_mode='fixed', gripper_effort_limit=base['effort'][0])
        command = build_command(ns, 'policy', seed, output)
        command[command.index('--n-action-steps') + 1] = str(queue)
        command += ['--dump-policy-images-every', '1', '--dump-policy-images-range', '241', '330']
        return stage(f"{candidate['name']}_{label}_q{queue}_seed{seed}", command,
                     lambda: audit_rollout(output, queue, seed, base, checkpoint), required_completed)

    def train_offline(candidate, required_completed=False):
        checkpoint = checkpoint_path(candidate)
        stage(f"{candidate['name']}_train",
              single_training_command(root / candidate['name'] / 'model', candidate),
              lambda: verify_training(checkpoint, candidate), required_completed)
        offline = root / f"offline_{candidate['name']}.json"
        command = [sys.executable, '-u', str(REPO / 'scripts/evaluation/lerobot_act_offline.py'),
                   '--checkpoint', str(checkpoint), '--dataset-root', str(TRAIN), '--repo-id', REPO_ID,
                   '--episodes', '20', '--max-samples', '675', '--device', 'cpu',
                   '--grasp-window', '240', '329', '--output', str(offline)]
        return stage(f"{candidate['name']}_offline", command,
                     lambda: verify_offline(offline), required_completed)

    update(stage=args.stage, status='starting')
    if args.stage == 'prepare':
        print(json.dumps({'prepared': str(root), 'subset': base['subset'], 'goal_success': False}))
        return
    if args.stage == 'screen':
        if manifest['selection'] is not None:
            raise RuntimeError('selection is locked; screening cannot use final outcomes for reselection')
        for order, candidate in enumerate(CANDIDATES):
            checkpoint = checkpoint_path(candidate)
            error = train_offline(candidate)
            for qorder, queue in enumerate((30, 1)):
                audits = [rollout(candidate, queue, seed, 'screen') for seed in SCREEN_SEEDS]
                case = screening_case(candidate, order * 2 + qorder, queue, error, audits, checkpoint)
                if case not in manifest['cases']:
                    manifest['cases'].append(case)
                update(stage='screen', status='candidate_complete')
                if case['successes'] == 3:
                    break
            if choose_candidate(manifest['cases']) and choose_candidate(manifest['cases'])['successes'] == 3:
                break
        selection = choose_candidate(manifest['cases'])
        if selection is not None:
            path = root / 'selection.json'
            if path.exists():
                raise FileExistsError('refusing to replace locked selection')
            atomic_json(path, selection)
            manifest['selection'] = dict(selection, selection_sha256=sha256_file(path))
        update(stage='screen', status='selected' if selection else 'no_success_candidate')
        print(json.dumps({'selection': manifest['selection'], 'goal_success': False,
                          'next': 'final' if selection else 'review next actual training intervention'}))
        return
    selection = manifest['selection']
    if selection is None or sha256_file(root / 'selection.json') != selection['selection_sha256']:
        raise ValueError('final requires an unchanged locked screening selection')
    validate_screen_order(manifest['cases'])
    reconstructed = []
    errors = {}
    for saved in manifest['cases']:
        candidate = next(item for item in CANDIDATES if item['name'] == saved['candidate'])
        if candidate['name'] not in errors:
            errors[candidate['name']] = train_offline(candidate, required_completed=True)
        audits = [rollout(candidate, saved['queue'], seed, 'screen', required_completed=True)
                  for seed in SCREEN_SEEDS]
        reconstructed.append(screening_case(candidate, saved['order'], saved['queue'],
                             errors[candidate['name']], audits, checkpoint_path(candidate)))
    if reconstructed != manifest['cases']:
        raise ValueError('screening cases differ from the re-audited completed artifacts')
    candidate = validate_selection(selection, reconstructed, root)
    if sha256_file(checkpoint_path(candidate) / 'model.safetensors') != selection['model_sha256']:
        raise ValueError('locked model changed')
    rows = [rollout(candidate, selection['queue'], seed, 'final') for seed in FINAL_SEEDS]
    manifest['goal_success'] = final_success(rows)
    manifest['final_summary'] = {'executions': 10, 'successes': sum(row['success'] for row in rows),
                                 'lifts': sum(row['first_lift_step'] is not None for row in rows),
                                 'scope': 'fixed training scene; no position/generalization/real validation'}
    update(stage='final', status='goal_success' if manifest['goal_success'] else 'goal_not_met')
    print(json.dumps(manifest['final_summary'] | {'goal_success': manifest['goal_success']}))


if __name__ == '__main__':
    main()
