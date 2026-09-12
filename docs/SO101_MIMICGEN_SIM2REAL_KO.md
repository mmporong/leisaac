# SO101 리더 10회에서 MimicGen과 실물 검증까지

## 결론

이 프로젝트는 **LeIsaac을 실행 기반으로 사용하고, ROBOTIS `cyclo_lab`은 파이프라인 설계의 비교 자료로 사용한다.** 두 저장소를 동시에 실행하거나 코드를 섞는 방식은 아니다.

- LeIsaac을 쓰는 이유: SO101 리더 입력, SO101 시뮬레이션 모델, 관절↔말단 자세 변환이 이미 연결되어 있다.
- `cyclo_lab`을 참고하는 이유: ROBOTIS가 공개한 `실물 시연 → MimicGen → BC 학습 → 실물 추론` 구조를 확인할 수 있다.
- 그대로 쓰지 않는 이유: `cyclo_lab`의 실물 인터페이스와 데이터 변환기는 OMY·FFW 계열용이며 SO101 포트, 관절 수, 카메라 키와 맞지 않는다.
- MimicGen은 강화학습이 아니다. 사람이 만든 시연을 분할·재조합해 시연 데이터를 늘리는 도구다.
- ACT는 MimicGen의 필수 구성요소는 아니다. 이 프로젝트에서는 BC-RNN 0/5 실패 뒤 action chunking 비교 모델로 추가했다. 기존 50회 모델에서 관찰한 1/5 성공은 같은 조건의 반복 실행에서 재현되지 않아 안정적인 성공 근거로 사용하지 않는다.

ROBOTIS의 공개 예제도 같은 큰 흐름을 사용한다. OMY 예제는 실물 시연 10회를 받고 MimicGen 500회를 생성한 뒤 학습과 실물 추론으로 연결한다. 다만 대상 로봇이 SO101이 아니므로 구조만 참고한다.

- [`cyclo_lab` README의 Sim2Real imitation-learning 절차](https://github.com/ROBOTIS-GIT/cyclo_lab/blob/42dcd8256651f7ccad64d7ba0c9bd34bb878e7d6/README.md#sim2real)
- [`cyclo_lab` 저장소](https://github.com/ROBOTIS-GIT/cyclo_lab)

## 전체 흐름

```text
SO101 리더 1대
  ↓ 관절값으로 시뮬레이션 팔 조작
Isaac Sim 원본 성공 시연 10회
  ↓ joint action → end-effector pose action
IK 데이터 10회
  ↓ pick_cube / place_cube 자동 판정
MimicGen 주석 데이터 10회
  ↓ 물체 초기 위치 변경 + 서브태스크 재조합 + 작은 동작 노이즈
합성 시연 N회
  ↓ IK action → joint action
학습용 HDF5
  ↓ BC-RNN 기준선 → ACT action-chunking 비교
시뮬레이션 롤아웃 평가
  ↓ 카메라·관절 규격과 안전 범위를 맞춘 추론 어댑터
SO101 팔로워 1대 실물 평가
```

`10회`는 증강 전 사람이 제공하는 원본 시연 수다. 10회가 자동으로 정확히 50회가 되는 규칙은 없다. 현재 설정은 `generation_guarantee=True`이므로 생성 명령의 `--generation_num_trials`은 성공 데이터 목표 수로 동작한다. 현재 IsaacLab 생성 루프는 `max_num_failures`를 종료 조건으로 사용하지 않으므로 성공 목표에 도달하거나 실행 오류·사용자 중단이 발생할 때까지 재시도한다.

## 현재 태스크 정의

- 물체: 한 변 4 cm인 빨간 큐브
- 목표: 파란 박스 내부
- 완료 동작: 큐브를 박스 안에 놓고 그리퍼를 연다.
- 원위치 복귀: 필요 없다.
- 원본 녹화 완료: 사용자가 `N`으로 직접 확정한다.
- 실패 및 재시작: `R`
- 각 에피소드 조작 시작: `B`
- MimicGen 서브태스크:
  1. `pick_cube`: 큐브를 잡는다.
  2. `place_cube`: 큐브가 박스 안에 있고 그리퍼가 열린다.

성공 판정은 큐브 위치뿐 아니라 그리퍼 해제까지 확인한다. 큐브 파지 판정이 `0.26 rad 미만`을 사용하므로 배치 판정은 같은 경계의 반대인 `0.26 rad 초과`를 사용한다.

## 현재 실행 결과와 증거

2026-09-11~12 실행 기준이다. 실행 결과는 코드 주석이 아니라 이 문서와 산출물에 기록한다.

| 단계 | 결과 | 파일 |
|---|---:|---|
| 원본 녹화 | 성공 10회 + 초기 실패 1회 | `datasets/pick_cube_into_box_source_10_20260911.hdf5` |
| 원본 성공 영상 | MP4 10개 | `outputs/portfolio/pick_cube_into_box/source_demos/` |
| 관절→IK 변환 | 성공 10회 | `datasets/pick_cube_into_box_source_10_ik_20260911.hdf5` |
| 자동 주석 | 10/10, 두 서브태스크 신호 확인 | `datasets/pick_cube_into_box_annotated_10_20260911.hdf5` |
| MimicGen 시험 생성 | 성공 10회, 총 6,123프레임 | `datasets/pick_cube_into_box_mimic_generated_10_20260911.hdf5` |
| 생성 데이터 관절 복원 | 성공 10회, action 8차원→6차원 | `datasets/pick_cube_into_box_mimic_joint_10_20260911.hdf5` |
| 생성 성공 영상 | MP4 10개 | `outputs/portfolio/pick_cube_into_box/mimic_generated/` |
| 원본·생성 비교 영상 | 1개 | `outputs/portfolio/pick_cube_into_box/source_vs_mimic.mp4` |
| 손목카메라 재생·재주석 | 성공 10회, 총 5,640프레임, `front`·`wrist` 동시 기록 | `datasets/pick_cube_into_box_annotated_wrist_10_20260911.hdf5` |
| 손목카메라 재생 영상 | MP4 10개와 전면·손목 비교 시트 | `outputs/portfolio/pick_cube_into_box/wrist_replay/` |
| 손목카메라 MimicGen 본 생성 | 성공 50회·실패 75회, 성공률 40%, 성공 데이터 27,217프레임 | `datasets/pick_cube_into_box_mimic_generated_wrist_50_20260911.hdf5` |
| 본 생성 실패 궤적 | 실패 75회, 학습 입력에서 제외 | `datasets/pick_cube_into_box_mimic_generated_wrist_50_20260911_failed.hdf5` |
| 본 생성 확인 영상 | 전면·손목 나란히 보기 MP4 3개와 비교 시트 | `outputs/portfolio/pick_cube_into_box/mimic_generated_wrist_50/` |
| 6관절 복원 | 성공 50회, 27,217프레임, action 8차원→6차원 | `datasets/pick_cube_into_box_mimic_joint_wrist_50_20260911.hdf5` |
| 학습·검증 분리 | 공간 분산 방식 train 40회 / validation 10회 | `datasets/pick_cube_into_box_mimic_joint_front_wrist_50_84x84_20260912.split.json` |
| BC-RNN 학습 | Robomimic 0.4.0, 전면+손목+6관절 문맥, 최저 validation loss 0.0972 | `outputs/robomimic/so101_pick_cube_into_box_bc_rnn_front_wrist_context6_84x84_long/20260912021237/` |
| 미학습 시드 평가 | 5회 중 성공 0회, 관절 제한 clipping 0회 | `outputs/evaluation/so101_bc_rnn_context6_long_5seeds/evaluation.json` |
| 정책 실패 영상 | 새 시드 1000·1001 각 1개 | `outputs/evaluation/so101_bc_rnn_context6_long_5seeds/rollout_00{1,2}_failure.mp4` |
| LeRobot v3 변환 | 성공 50회·27,217프레임, 전면+손목 84×84, 상태·행동 6차원 | `datasets/lerobot/so101_mimic_joint_next_state_front_wrist_50_20260912/` |
| ACT 설정 게이트 | 2회 과적합 비교에서 VAE+학습률 1e-4 채택 | `outputs/lerobot/so101_act_front_wrist_overfit2_vae_lr1e4_3000/offline_evaluation.json` |
| ACT 관절 라벨 점검 | IK 목표는 8/50 에피소드에서 프레임 간 1 rad 초과, 다음 관측 상태는 0/50 | `datasets/pick_cube_into_box_mimic_joint_next_state_context6_front_wrist_50_84x84_20260912.hdf5` |
| ACT 본 학습 | train 40회, VAE, chunk 100, 30,000 step, 약 10.9 epoch | `outputs/lerobot/so101_act_next_state_mimic50_chunk100_30000/` |
| 기존 ACT 미학습 시드 평가 | 첫 실행 1/5 뒤 추가 실행 2/5, 동일 seed 반복 0/2로 성공 재현 실패 | `outputs/evaluation/so101_act_next_state_chunk100_replan30_seeded_repeat2/evaluation.json` |
| 실행 설정 대조 | 10-step 재계획 0/5, 30-step 정착 0/5, AA Off 0/5, temporal ensemble 0/2 | `outputs/evaluation/so101_act_next_state_chunk100_replan10_matched_resize_5seeds/` 외 3개 디렉터리 |
| 독립 시드 추가 생성 | seed 43, 성공 100회·실패 115회, 총 215회 시도, 성공률 46.5% | `datasets/pick_cube_into_box_mimic_generated_wrist_100_seed43_20260912.generation.json` |
| 추가 생성 무결성 | 성공 100회·58,074프레임, 초기 위치/궤적 100개 고유, 기존 seed 42 데이터와 궤적 해시 중복 0개 | `datasets/pick_cube_into_box_mimic_generated_wrist_100_seed43_20260912.hdf5` |
| 확장 학습 데이터 | 성공 100회·58,074프레임, train 80 / validation 20, 전면+손목 84×84 | `datasets/lerobot/so101_mimic_next_state_front_wrist_100_seed43_20260912/` |
| 확장 ACT 학습 | train 80회·45,920프레임, VAE, chunk 30, 30,000 step, 약 5.23 epoch | `outputs/lerobot/so101_act_next_state_mimic100_seed43_chunk30_30000/` |
| 확장 ACT 오프라인 검증 | 30,000-step validation 균등 RMSE 0.0415 rad, MAE 0.0285 rad | `outputs/lerobot/so101_act_next_state_mimic100_seed43_chunk30_30000/offline_validation_030000.json` |
| 확장 ACT 폐루프 관찰값 | 시드 3000~3009에서 2/10 성공, 전 회차 영상 저장 | `outputs/evaluation/so101_act_mimic100_seed43_chunk30_30k_{5seeds,seeds3005_3009}/` |
| 성공 재현성 게이트 | 성공 seed 3003·3006의 독립 반복은 모두 실패, 0/2 | `outputs/evaluation/so101_act_mimic100_seed43_chunk30_30k_seed{3003,3006}_repeat/` |

원본 HDF5의 초기 실패 1회는 관절→IK 변환 단계에서 자동 제외됐다. 삭제하거나 성공 데이터로 바꾸지 않았다.

생성 10개의 큐브 초기 위치와 관절 궤적 해시는 모두 달랐다. 따라서 같은 영상을 복사한 것이 아니라 서로 다른 초기 조건과 동작 궤적을 가진 합성 시연임을 확인했다.

## Linux 재현 명령

저장소 루트와 Python 경로를 먼저 지정한다.

```bash
export LEISAAC_ROOT="${LEISAAC_ROOT:-/data/$USER/leisaac}"
export LEISAAC_PYTHON="${LEISAAC_PYTHON:-/data/$USER/conda-envs/leisaac/bin/python}"
export LEISAAC_ASSETS_ROOT="${LEISAAC_ASSETS_ROOT:-/data/$USER/leisaac-assets/assets}"
cd "$LEISAAC_ROOT"
```

MimicGen 도구의 UI 모듈 요구사항을 설치할 때는 Isaac Sim이 요구하는 `psutil` 버전을 유지한다.

```bash
"$LEISAAC_PYTHON" -m pip install \
  'ipython>=8,<9' \
  'ipywidgets>=8,<9' \
  'psutil==5.9.8'
```

### 1. 리더로 원본 10회 녹화

```bash
"$LEISAAC_PYTHON" scripts/environments/teleoperation/teleop_se3_agent.py \
  --task LeIsaac-SO101-PickCubeIntoBox-v0 \
  --teleop_device so101leader \
  --port /dev/serial/by-id/<SO101_LEADER_BOARD_ID> \
  --num_envs 1 \
  --device cuda:0 \
  --enable_cameras \
  --record \
  --dataset_file ./datasets/pick_cube_into_box_source_10_20260911.hdf5 \
  --num_demos 10
```

각 회차는 `B → 집기 → 박스에 놓기 → 그리퍼 열기 → N` 순서다. 잘못된 회차는 `R`을 누르고 다시 `B`로 시작한다.

### 2. 관절 동작을 IK 동작으로 변환

```bash
"$LEISAAC_PYTHON" scripts/mimic/eef_action_process.py \
  --input_file ./datasets/pick_cube_into_box_source_10_20260911.hdf5 \
  --output_file ./datasets/pick_cube_into_box_source_10_ik_20260911.hdf5 \
  --to_ik \
  --headless
```

MimicGen은 물체 기준의 말단 자세 궤적을 변형하므로 이 변환이 필요하다.

### 3. 서브태스크 자동 주석

```bash
"$LEISAAC_PYTHON" scripts/mimic/annotate_demos.py \
  --device cuda:0 \
  --task LeIsaac-SO101-PickCubeIntoBox-Mimic-v0 \
  --input_file ./datasets/pick_cube_into_box_source_10_ik_20260911.hdf5 \
  --output_file ./datasets/pick_cube_into_box_annotated_wrist_10_20260911.hdf5 \
  --auto \
  --headless \
  --enable_cameras
```

완료 후 모든 에피소드에 `pick_cube=True`, `place_cube=True` 구간과 `obs/wrist` 영상이 존재하는지 확인한다. 사람이 `N`을 누른 사실만으로 자동 주석 성공을 가정하지 않는다.

### 4. MimicGen 생성

작은 시험은 성공 10회, 본 생성은 성공 50회 이상으로 나눈다. `generation_guarantee=True`에서는 성공 수가 목표에 도달할 때까지 재시도한다. 추가 데이터는 앞선 실행과 다른 `--datagen-seed`를 지정한다. 생성기는 기존 성공·실패 HDF5와 완료 manifest를 덮어쓰지 않으며, 목표 성공 수에 도달한 경우에만 `.generation.json`을 남긴다. 비정상 종료나 디스크·GPU 오류가 있을 수 있으므로 manifest와 결과 HDF5의 성공 에피소드 수를 함께 확인한다.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$LEISAAC_PYTHON" scripts/mimic/generate_dataset.py \
  --device cuda:0 \
  --task LeIsaac-SO101-PickCubeIntoBox-Mimic-v0 \
  --num_envs 1 \
  --generation_num_trials 50 \
  --input_file ./datasets/pick_cube_into_box_annotated_wrist_10_20260911.hdf5 \
  --output_file ./datasets/pick_cube_into_box_mimic_generated_wrist_50_20260911.hdf5 \
  --headless \
  --enable_cameras
```

기존 seed 42 데이터와 분리한 확장 생성은 다음처럼 실행했다.

```bash
"$LEISAAC_PYTHON" scripts/mimic/generate_dataset.py \
  --device cuda:0 \
  --task LeIsaac-SO101-PickCubeIntoBox-Mimic-v0 \
  --num_envs 1 \
  --generation_num_trials 100 \
  --datagen-seed 43 \
  --input_file ./datasets/pick_cube_into_box_annotated_wrist_10_20260911.hdf5 \
  --output_file ./datasets/pick_cube_into_box_mimic_generated_wrist_100_seed43_20260912.hdf5 \
  --headless \
  --enable_cameras
```

RTX 5050 Laptop 8GB 환경에서는 `--num_envs 1`을 사용한다. 640×480 전면·손목 영상을 함께 기록하면 에피소드 내보내기의 `torch.stack`에서 약 958MB 추가 할당이 발생해 OOM으로 실패했다. 두 카메라를 320×240으로 낮추고 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`를 적용한 뒤 성공 50회를 생성했다. 환경 수를 크게 올리는 것은 속도 설정이지 데이터 의미를 높이는 조건이 아니다.

2026-09-11 본 생성은 총 125회 시도 중 성공 50회, 실패 75회로 성공률 40%였다. `generation_keep_failed=True`이므로 실패 궤적은 `_failed.hdf5`에 분리됐다. 학습에는 성공 파일만 사용하고, 실패 파일은 실패 원인 분석에만 사용한다.

2026-09-12 seed 43 확장 생성은 총 215회 시도 중 성공 100회, 실패 115회로 성공률 46.5%였다. 성공 데이터는 58,074프레임이며 초기 위치와 관절 궤적 해시는 100개 모두 고유했다. 기존 seed 42 성공 50회와의 궤적 해시 중복도 0개였다. 실패 115회는 동일하게 학습에서 제외했다.

### 5. IK 동작을 관절 동작으로 복원

```bash
"$LEISAAC_PYTHON" scripts/mimic/eef_action_process.py \
  --input_file ./datasets/pick_cube_into_box_mimic_generated_wrist_50_20260911.hdf5 \
  --output_file ./datasets/pick_cube_into_box_mimic_joint_wrist_50_20260911.hdf5 \
  --to_joint \
  --headless
```

이 파일이 정책 학습 또는 시뮬레이션 재생에 쓰이는 최종 관절 명령 데이터다.

### 6. BC-RNN 학습 데이터 준비

Robomimic 0.4.0을 설치한다. LeIsaac의 `mimic` extra에도 같은 버전이 고정돼 있다.

```bash
"$LEISAAC_PYTHON" -m pip install \
  'robomimic @ git+https://github.com/ARISE-Initiative/robomimic.git@v0.4.0'
```

320×240 전면·손목 영상을 84×84로 축소한다. 큰 영상에서 작은 영역을 바로 무작위 crop하면 큐브나 박스가 관측에서 빠질 수 있기 때문이다.

```bash
"$LEISAAC_PYTHON" scripts/imitation_learning/resize_robomimic_images.py \
  --input ./datasets/pick_cube_into_box_mimic_joint_wrist_50_20260911.hdf5 \
  --output ./datasets/pick_cube_into_box_mimic_joint_front_wrist_50_84x84_20260912.hdf5 \
  --height 84 \
  --width 84 \
  --compression lzf \
  --batch-size 32
```

프레임을 묶어 읽고 쓰도록 전처리기를 바꿔 58,074프레임 변환을 완료했다. 표본으로 첫·중간·마지막 에피소드의 축소 영상을 직접 `cv2.INTER_AREA` 결과와 픽셀 단위로 대조했다.

IK→관절 복원 결과의 각 에피소드 0번 행동에는 이전 상태가 섞인 이상치가 있었다. 다음 관절 목표로 0번 행동을 복구하고, 8차원 IK `obs/actions`는 실물에서도 얻을 수 있는 직전 6관절 목표로 바꾼다. 원래 관절 목표는 `joint_position_targets`에 보존된다.

```bash
"$LEISAAC_PYTHON" scripts/imitation_learning/prepare_so101_bc_dataset.py \
  --dataset ./datasets/pick_cube_into_box_mimic_joint_front_wrist_50_84x84_20260912.hdf5 \
  --validation-count 10

"$LEISAAC_PYTHON" scripts/imitation_learning/convert_robomimic_joint_actions.py \
  --representation joint_target \
  --input ./datasets/pick_cube_into_box_mimic_joint_front_wrist_50_84x84_20260912.hdf5 \
  --output ./datasets/pick_cube_into_box_mimic_joint_target_context6_front_wrist_50_84x84_20260912.hdf5
```

분리는 초기 큐브 XY에 대한 farthest-point sampling으로 공간 전체를 덮는 validation 10회를 고른다. 생성 HDF5에는 원본 시연 계보가 남아 있지 않으므로 **원본 계보가 완전히 분리됐다고 주장할 수 없다.** 최종 일반화 판정은 별도 시드의 시뮬레이터 롤아웃으로 한다.

### 7. Robomimic BC-RNN 학습

입력은 `front`, `wrist`, 현재 관절 위치·속도, 직전 6관절 명령이다. 출력은 다음 6관절 목표다. ACT나 강화학습은 이 단계에 사용하지 않았다.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$LEISAAC_PYTHON" scripts/imitation_learning/train_robomimic.py \
  --config source/leisaac/leisaac/tasks/pick_cube_into_box/agents/robomimic/bc_rnn_front_wrist.json \
  --dataset ./datasets/pick_cube_into_box_mimic_joint_target_context6_front_wrist_50_84x84_20260912.hdf5
```

실행에서는 45 epoch까지 확인했고 30 epoch에서 최저 validation loss 0.0971587을 기록했다. 이후 학습 loss가 약 0.10에서 정체돼 최종 설정은 40 epoch로 제한했다. GMM 행동 출력과 관절 delta 출력도 시험했지만 각각 오프라인 오차와 누적 drift가 커서 최종 경로에서 제외했다.

### 8. 미학습 큐브 위치 평가

```bash
CHECKPOINT=./outputs/robomimic/so101_pick_cube_into_box_bc_rnn_front_wrist_context6_84x84_long/20260912021237/models/model_epoch_30_best_validation_0.09715865477919579.pth

"$LEISAAC_PYTHON" scripts/evaluation/robomimic_so101.py \
  --checkpoint "$CHECKPOINT" \
  --action-mode joint_target \
  --num-rollouts 5 \
  --horizon 1200 \
  --seed 1000 \
  --output-dir ./outputs/evaluation/so101_bc_rnn_context6_long_5seeds \
  --video-count 2 \
  --headless \
  --enable_cameras \
  --device cuda:0
```

2026-09-12 결과는 **0/5 성공**이다. 모든 시도에서 관절 제한 clipping은 0회였고, 시드 1001에서는 큐브를 약 8mm 밀었지만 파지하지 못했다. 영상에서는 팔이 큐브 부근까지 접근하고 손목 카메라가 큐브를 포착하지만 정밀 파지로 이어지지 않는다. 따라서 현재 증거로 말할 수 있는 범위는 다음과 같다.

- MimicGen 성공 50회 생성, 6관절 학습 변환, BC-RNN 학습, 새 시드 자율 평가까지 파이프라인은 실행된다.
- 현재 50회 데이터와 BC-RNN 설정은 자율 집기 성공에 충분하지 않다.
- 이 결과를 “모방학습 성공” 또는 “실물 전이 준비 완료”라고 표현하면 안 된다.
- 다음 비교 실험은 ACT 같은 action chunking, 더 많은 독립 원본 시연, 파지 구간 가중 샘플링 순서가 적절하다. 강화학습은 이 실패를 자동으로 해결한 단계가 아니며 아직 수행하지 않았다.

### 9. ACT용 관절 라벨 정제와 LeRobot 변환

BC-RNN 실패 뒤 원본 관절 목표를 검사했다. MimicGen의 IK→관절 변환 결과는 물리적으로 성공한 에피소드여도 IK 해가 바뀌면서 관절 목표가 불연속일 수 있다. 실제로 50회 중 8회에서 연속 프레임 행동 차이의 L2 norm이 1 rad를 넘었고 최댓값은 3.87 rad였다. 성공한 물리 궤적의 다음 관측 관절 위치를 목표로 쓰면 1 rad 초과 에피소드는 0회이며 최댓값은 0.320 rad다.

```bash
"$LEISAAC_PYTHON" scripts/imitation_learning/convert_robomimic_joint_actions.py \
  --representation joint_next_state \
  --input ./datasets/pick_cube_into_box_mimic_joint_front_wrist_50_84x84_20260912.hdf5 \
  --output ./datasets/pick_cube_into_box_mimic_joint_next_state_context6_front_wrist_50_84x84_20260912.hdf5

export LEROBOT_PYTHON="${LEROBOT_PYTHON:-$HOME/miniforge3/envs/lerobot/bin/python}"

"$LEROBOT_PYTHON" scripts/imitation_learning/convert_hdf5_to_lerobot.py \
  --input ./datasets/pick_cube_into_box_mimic_joint_next_state_context6_front_wrist_50_84x84_20260912.hdf5 \
  --output-root ./datasets/lerobot/so101_mimic_joint_next_state_front_wrist_50_20260912 \
  --repo-id local/so101_mimic_joint_next_state_front_wrist_50 \
  --fps 60 \
  --max-action-step-norm 1.0
```

변환기는 성공 플래그, 6차원 상태·행동, finite 값, 두 카메라의 84×84 RGB 규격을 확인하고 1 rad를 넘는 행동 점프가 있으면 출력 생성 전에 중단한다.

### 10. ACT 학습

현재 LeRobot 소스는 Python 3.12 문법을 사용하므로 Isaac Sim의 Python 3.11 환경에 억지로 섞지 않는다. 데이터 변환·ACT 학습·ACT 추론 서버는 LeRobot 환경에서 실행하고, Isaac 평가는 로컬 TCP로 6관절 행동만 받는다.

설정 게이트에서는 2개 에피소드에 3,000 step 과적합해 VAE 사용 여부와 학습률을 비교했다. `VAE + lr 1e-4`가 첫 프레임 RMSE 0.088 rad, 균등 샘플 RMSE 0.057 rad로 가장 좋아 본 학습에 사용했다. 본 학습은 공간 분산 validation 10회를 제외한 기존 40회만 사용했다.

```bash
CUDA_VISIBLE_DEVICES=0 "$HOME/miniforge3/envs/lerobot/bin/lerobot-train" \
  --dataset.repo_id=local/so101_mimic_joint_next_state_front_wrist_50 \
  --dataset.root="$LEISAAC_ROOT/datasets/lerobot/so101_mimic_joint_next_state_front_wrist_50_20260912" \
  --dataset.episodes='[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,17,19,20,21,22,26,27,28,29,30,32,33,35,37,38,39,40,41,42,44,45,46,47,48,49]' \
  --policy.type=act \
  --policy.device=cuda \
  --policy.repo_id=local/so101_act_next_state_mimic50_chunk100 \
  --policy.push_to_hub=false \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --policy.dim_model=256 \
  --policy.n_heads=8 \
  --policy.dim_feedforward=1024 \
  --policy.n_encoder_layers=4 \
  --policy.use_vae=true \
  --policy.optimizer_lr=0.0001 \
  --policy.optimizer_lr_backbone=0.00001 \
  --batch_size=8 \
  --num_workers=2 \
  --steps=30000 \
  --eval_steps=0 \
  --save_freq=5000 \
  --output_dir="$LEISAAC_ROOT/outputs/lerobot/so101_act_next_state_mimic50_chunk100_30000" \
  --wandb.enable=false \
  --cudnn_deterministic=true
```

30,000-step 모델의 validation 균등 샘플 RMSE는 0.084 rad였다. 5,000-step 모델의 0.133 rad보다 낮다. 첫 프레임 RMSE는 0.235 rad이며 그리퍼 오차가 0.509 rad로 가장 크므로 초기 상태 가중 보강은 남은 개선 항목이다.

### 11. ACT 폐루프 평가

학습 입력과 같은 투영을 유지하려면 카메라는 320×240으로 렌더한 뒤 OpenCV `INTER_AREA`로 84×84에 축소한다. 카메라 자체를 84×84로 바꾸면 종횡비와 투영이 달라지고 Isaac의 저해상도 DLSS 보정까지 들어간다. 평가기는 이 전처리를 자동 수행한다.

```bash
CHECKPOINT="$LEISAAC_ROOT/outputs/lerobot/so101_act_next_state_mimic50_chunk100_30000/checkpoints/030000/pretrained_model"

"$LEISAAC_PYTHON" scripts/evaluation/lerobot_act_so101.py \
  --checkpoint "$CHECKPOINT" \
  --num-rollouts 5 \
  --horizon 1200 \
  --seed 3000 \
  --n-action-steps 30 \
  --server-device cuda \
  --video-count 2 \
  --output-dir ./outputs/evaluation/so101_act_next_state_chunk100_replan30_matched_resize_5seeds \
  --headless
```

동일한 100-step 예측 모델에서 100개를 모두 열린 고리로 실행하면 0/5였다. 30-step마다 다시 관측하고 계획한 최초 실행에서는 1/5가 성공했지만, 이후 seed와 추론 난수를 고정한 평가에서 2/5 뒤 동일 성공 seed 반복이 0/2였다. 10-step 재계획, 물리 정착, AA Off, temporal ensemble도 각각 0/5, 0/5, 0/5, 0/2였다. 따라서 이 결과는 **폐루프 성공 장면이 관찰됐다는 증거**이지 재현 가능한 정책 성공의 증거가 아니다.

### 12. 독립 시드 100회 확장과 chunk 30 재학습

실행 설정만 바꾸는 대조군이 모두 실패해 데이터 다양성을 먼저 늘렸다. 기존 seed 42 성공 50회를 복제하거나 합치지 않고, 원본 리더 시연 10회에서 seed 43으로 성공 합성 시연 100회를 새로 만들었다. 행동 청크도 실제 재계획 주기와 같은 30으로 줄였다.

```bash
"$LEISAAC_PYTHON" scripts/mimic/eef_action_process.py \
  --input_file ./datasets/pick_cube_into_box_mimic_generated_wrist_100_seed43_20260912.hdf5 \
  --output_file ./datasets/pick_cube_into_box_mimic_joint_wrist_100_seed43_20260912.hdf5 \
  --to_joint \
  --headless

"$LEISAAC_PYTHON" scripts/imitation_learning/resize_robomimic_images.py \
  --input ./datasets/pick_cube_into_box_mimic_joint_wrist_100_seed43_20260912.hdf5 \
  --output ./datasets/pick_cube_into_box_mimic_joint_front_wrist_100_seed43_84x84_20260912.hdf5 \
  --height 84 \
  --width 84 \
  --compression lzf \
  --batch-size 32

"$LEISAAC_PYTHON" scripts/imitation_learning/prepare_so101_bc_dataset.py \
  --dataset ./datasets/pick_cube_into_box_mimic_joint_front_wrist_100_seed43_84x84_20260912.hdf5 \
  --validation-count 20

"$LEISAAC_PYTHON" scripts/imitation_learning/convert_robomimic_joint_actions.py \
  --representation joint_next_state \
  --input ./datasets/pick_cube_into_box_mimic_joint_front_wrist_100_seed43_84x84_20260912.hdf5 \
  --output ./datasets/pick_cube_into_box_mimic_joint_next_state_front_wrist_100_seed43_84x84_20260912.hdf5

export LEROBOT_PYTHON="${LEROBOT_PYTHON:-$HOME/miniforge3/envs/lerobot/bin/python}"

"$LEROBOT_PYTHON" scripts/imitation_learning/convert_hdf5_to_lerobot.py \
  --input ./datasets/pick_cube_into_box_mimic_joint_next_state_front_wrist_100_seed43_84x84_20260912.hdf5 \
  --output-root ./datasets/lerobot/so101_mimic_next_state_front_wrist_100_seed43_20260912 \
  --repo-id local/so101_mimic_next_state_front_wrist_100_seed43 \
  --fps 60 \
  --max-action-step-norm 1.0
```

변환 결과는 성공 100회·58,074프레임이며 train은 80회·45,920프레임, validation은 20회·12,154프레임이다. 축소 영상은 첫·중간·마지막 표본을 직접 OpenCV 결과와 픽셀 단위로 대조했고, LeRobot 변환 뒤 에피소드·프레임 수와 분할의 교집합이 없음을 다시 확인했다.

```bash
CUDA_VISIBLE_DEVICES=0 "$HOME/miniforge3/envs/lerobot/bin/lerobot-train" \
  --dataset.repo_id=local/so101_mimic_next_state_front_wrist_100_seed43 \
  --dataset.root="$LEISAAC_ROOT/datasets/lerobot/so101_mimic_next_state_front_wrist_100_seed43_20260912" \
  --dataset.episodes='[2,3,4,5,8,10,11,12,13,14,15,16,17,19,20,21,22,23,24,26,27,28,29,30,31,32,33,34,35,36,37,38,39,41,42,43,45,47,48,49,50,51,52,53,54,55,57,58,59,60,61,63,64,65,66,67,69,70,71,72,73,76,77,78,79,80,82,83,84,86,87,88,89,90,91,93,96,97,98,99]' \
  --policy.type=act \
  --policy.device=cuda \
  --policy.repo_id=local/so101_act_next_state_mimic100_seed43_chunk30 \
  --policy.push_to_hub=false \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.dim_model=256 \
  --policy.n_heads=8 \
  --policy.dim_feedforward=1024 \
  --policy.n_encoder_layers=4 \
  --policy.use_vae=true \
  --policy.optimizer_lr=0.0001 \
  --policy.optimizer_lr_backbone=0.00001 \
  --batch_size=8 \
  --num_workers=2 \
  --steps=30000 \
  --eval_steps=0 \
  --save_freq=5000 \
  --output_dir="$LEISAAC_ROOT/outputs/lerobot/so101_act_next_state_mimic100_seed43_chunk30_30000" \
  --wandb.enable=false \
  --cudnn_deterministic=true
```

5,000~30,000-step 체크포인트를 validation 20회에서 같은 500프레임으로 비교했다. 균등 RMSE는 순서대로 0.0616, 0.0576, 0.0528, 0.0528, 0.0461, 0.0415 rad였고 30,000-step 모델이 가장 낮았다. 기존 50회·chunk 100 모델의 0.084 rad보다 오프라인 오차는 줄었지만 이것만으로 폐루프 성공을 뜻하지 않는다.

```bash
CHECKPOINT="$LEISAAC_ROOT/outputs/lerobot/so101_act_next_state_mimic100_seed43_chunk30_30000/checkpoints/030000/pretrained_model"

"$LEISAAC_PYTHON" scripts/evaluation/lerobot_act_so101.py \
  --checkpoint "$CHECKPOINT" \
  --num-rollouts 5 \
  --horizon 1200 \
  --seed 3000 \
  --n-action-steps 30 \
  --server-device cuda \
  --server-seed 0 \
  --video-count 5 \
  --output-dir ./outputs/evaluation/so101_act_mimic100_seed43_chunk30_30k_5seeds \
  --headless
```

시드 3000~3009의 최초 순차 실행에서는 seed 3003과 3006이 성공해 관찰 성공률은 2/10이었다. 하지만 두 성공 seed를 새 프로세스에서 각각 반복하자 모두 실패했다. 초기 큐브 위치와 관절 상태는 같아도 RTX 카메라 픽셀 해시와 첫 행동이 미세하게 달랐고, 그 차이가 긴 폐루프에서 증폭됐다. 렌더 워밍업, 버리는 프라이밍 롤아웃, PhysX enhanced determinism도 반복 성공을 만들지 못해 최종 코드에는 넣지 않았다. 따라서 현재 판정은 **오프라인 학습 성능 개선, 폐루프 성공 장면 2개 확보, 재현 가능한 자율 파지 미달, 실물 실행 금지**다. 강화학습은 아직 수행하지 않았다.

## 실행 게이트 상태

초기 10회 생성은 **MimicGen 파이프라인 검증**이었고, 이후 seed 42 성공 50회와 독립 seed 43 성공 100회를 생성했다. 50회 BC-RNN은 0/5였고, 100회 ACT는 오프라인 오차가 개선됐지만 폐루프 성공 2/10이 독립 반복 0/2로 재현되지 않았다. 데이터 생성과 정책 학습 파이프라인은 완주했지만 안정적 자율 집기와 실물 팔로워 검증은 완료되지 않았다.

| 게이트 | 수행 내용 | 완료 증거 | 현재 상태 |
|---|---|---|---|
| G0 증강 검증 | 원본 10회로 MimicGen 성공 10회 생성 | HDF5 무결성, 서로 다른 초기 위치·궤적, 비교 영상 | 완료 |
| G1 관측 일치 | 시뮬레이션 SO101에 손목 카메라 추가하고 기존 시연 재생 | HDF5 10/10에 `obs/wrist`, 총 5,640프레임, 영상 10개 | 완료 |
| G2 본 데이터 생성 | 주석된 원본 10회에서 독립 seed로 성공 합성 시연 생성 | seed 42 성공 50회와 seed 43 성공 100회, 두 집합 간 궤적 해시 중복 0개 | 완료 |
| G3 관절 복원·데이터 분리 | IK action을 6관절 action으로 복원하고 공간 분산 train/validation 분리 | 확장 데이터 action 6차원, train 80 / validation 20, 계보 정보 부재 명시 | 완료 |
| G4 모방학습 | BC-RNN 기준선과 ACT action-chunking 모델 학습 | 설정·체크포인트·오프라인 validation 결과 | 완료 |
| G5 시뮬레이션 평가 | 학습에 쓰지 않은 시드의 큐브 위치에서 자율 롤아웃 | BC-RNN 0/5, 확장 ACT 관찰 2/10·성공 seed 반복 0/2, 성공 영상 2개 | 완료·성능 미달 |
| G6 실물 평가 | 카메라 입력과 SO101 팔로워 출력을 잇는 추론 어댑터로 단계 평가 | 접근·집기·배치 단계별 실측 결과와 영상 | 이번 작업 제외 |

### 단계별 수행 기록

1. **완료:** SO101 손목 링크에 LeIsaac 기본 SO101 손목카메라를 복원했다. 원본 재생 검증은 640×480 RGB로 수행했다. 8GB GPU에서 두 카메라를 포함한 MimicGen 생성 시 내보내기 OOM이 확인되어 본 생성과 학습용 관측은 전면·손목 모두 320×240으로 조정했다.
2. **완료:** 기존 원본 10회를 새 손목 시점으로 재생·재주석했다. 성공 10/10, 총 5,640프레임이며 모든 에피소드에 `obs/front`와 `obs/wrist`가 함께 있다. 이 검증 HDF5는 640×480이고, 50회 생성 결과는 320×240으로 기록한다.
3. **완료:** 손목 영상이 포함된 `pick_cube_into_box_annotated_wrist_10_20260911.hdf5`로 seed 42 성공 50회와 seed 43 성공 100회를 만들었다. 확장 성공 파일은 58,074프레임이고 100개의 초기 위치와 관절 궤적이 모두 고유하며 기존 50회와 궤적 해시 중복이 없다.
4. **완료:** 확장 데이터의 8차원 IK action을 SO101 6관절 목표로 복원하고 train 80회 / validation 20회로 나눴다. 원본 계보가 HDF5에 없어 계보 독립성은 검증하지 못했으며 이 한계를 split manifest에 기록했다.
5. **완료:** Robomimic 0.4.0 BC-RNN 학습기·설정·데이터 전처리를 추가하고 최저 validation loss 0.0972 체크포인트를 만들었다.
6. **완료·성능 미달:** BC-RNN은 보지 않은 시드에서 0/5였다. IK 목표의 불연속을 발견해 다음 실제 관절 상태로 라벨을 정제하고 ACT를 학습했다. 확장 100회·chunk 30 모델은 오프라인 validation RMSE 0.0415 rad까지 개선됐고 시드 10개에서 성공 장면 2개를 만들었지만 성공 seed의 독립 반복은 0/2였다.
7. **이번 작업 제외:** 실물 팔로워는 실행하지 않았다. 시뮬레이션 성공 기준을 통과하기 전에는 이 정책을 실물 성공 정책으로 취급하지 않는다.

G0의 비교 영상은 데이터 증강 검증 자료로 바로 사용할 수 있다. 포트폴리오에서는 이를 “MimicGen으로 정책 학습을 완료했다”가 아니라 **“리더 시연 10회를 물체 위치와 궤적이 다른 성공 시연으로 증강하는 파이프라인을 검증했다”**고 설명한다.

현재 손목카메라 위치·렌즈 값은 LeIsaac의 SO101 기본 설정이다. 실물 손목카메라의 실제 해상도, 시야각과 장착 변환을 아직 측정해 맞춘 것은 아니다. 시뮬레이션 100회 생성에는 사용할 수 있지만 실물 전이 전에는 실측값으로 보정하고 필요하면 데이터 생성부터 다시 수행한다.

## 학습: 강화학습이 아니라 먼저 모방학습

현재 데이터에 바로 맞는 첫 학습은 BC 계열이다.

1. 입력: 실물과 시점을 맞춘 시뮬레이션 손목 카메라 영상과 현재 관절 상태
2. 출력: 다음 SO101 관절 목표
3. 학습: BC 또는 BC-RNN
4. 평가: 보지 않은 큐브 초기 위치에서 시뮬레이션 성공률 측정
5. 통과 후: 실물 카메라 입력과 팔로워 출력을 연결

ACT는 긴 시간 구간의 동작 묶음을 예측한다. 이 프로젝트에서는 BC-RNN 0/5 뒤 실제 비교 대상으로 추가했고, 확장 100회·chunk 30 모델의 최초 시드 10개에서 성공 장면 2개를 얻었다. 두 성공 seed의 독립 반복은 모두 실패했으므로 이는 모방학습 파이프라인과 폐루프 성공 장면의 증거이지 안정적인 정책이나 실물 준비 완료의 증거가 아니다.

강화학습은 별도 실험이다. MimicGen 데이터 자체가 강화학습을 수행하지 않는다. 필요하다면 BC 정책을 초기 정책으로 사용하거나, 동일 태스크의 보상·종료 조건으로 시뮬레이션 RL을 추가할 수 있지만 현재 실물 연결의 필수 단계는 아니다.

## 실물 팔로워로 연결할 때 필요한 것

MimicGen HDF5를 팔로워에 그대로 재생하는 것은 실물 정책 배포가 아니다. 실제 연결에는 다음 네 가지가 더 필요하다.

1. **관측 규격 일치**
   - 학습에 전면 카메라를 썼다면 실물에도 대응 카메라가 있어야 한다.
   - 해상도, 크롭, 색 순서, 정규화, 프레임 속도를 학습과 맞춘다.
   - 손목 카메라만 쓸 경우 시뮬레이션에도 같은 손목 시점 카메라를 추가하고 데이터를 다시 만든다.
2. **행동 규격 일치**
   - 정책 출력 관절 순서와 SO101 팔로워 관절 순서를 대조한다.
   - 정규화 범위와 실제 캘리브레이션 범위를 변환한다.
3. **실물 추론 어댑터**
   - 카메라·현재 관절 상태를 읽는다.
   - 정책을 실행한다.
   - 속도·관절 범위·통신 실패 처리 후 팔로워에 목표값을 쓴다.
4. **단계적 평가**
   - 시뮬레이션 비학습 위치 성공률
   - 팔로워 무부하 저속 동작
   - 큐브 접근만 수행
   - 집기
   - 박스 배치와 그리퍼 해제

현재까지 검증된 것은 리더 시연 10회, MimicGen seed 42 성공 50회와 seed 43 성공 100회, 6관절 변환, BC-RNN 0/5, 확장 ACT 관찰 2/10과 성공 seed 반복 0/2까지다. 실제 팔로워는 실행하지 않았다.

실물 완료 기준은 횟수를 숨기지 않고 기록하는 것이다. 고정 위치와 변경 위치 각각에서 총 시도 수, 성공 수, 실패 유형을 남긴다. 실제 측정 전에는 목표 성공률을 달성했다고 쓰지 않는다.

## `cyclo_lab`과의 관계

| 항목 | 현재 SO101 프로젝트 | ROBOTIS `cyclo_lab` |
|---|---|---|
| 실행 저장소 | LeIsaac | cyclo_lab |
| 주 대상 | SO101 | OMY, FFW 등 ROBOTIS 로봇 |
| 원본 시연 | SO101 리더 → Isaac Sim | 키보드 또는 ROBOTIS 실물 인터페이스 |
| 증강 | IsaacLab MimicGen | IsaacLab MimicGen |
| 정책 예제 | 별도 BC 경로를 연결해야 함 | Robomimic BC 예제 포함 |
| 실물 통신 | SO101 Feetech/LeRobot 어댑터 필요 | ROBOTIS DDS SDK 사용 |
| 그대로 코드 재사용 | 해당 없음 | SO101에는 관절·카메라·DDS 규격 불일치 |

따라서 “ROBOTIS와 같이 한다”는 말은 **파이프라인 구조를 참고한다는 의미에서는 맞고, `cyclo_lab`을 SO101 실행 저장소로 같이 돌린다는 의미에서는 틀리다.** 현재 구현은 `cyclo_lab`의 개념적 흐름을 SO101용 LeIsaac 태스크로 옮긴 것이다.

## Windows 전달

Linux의 MimicGen·모방학습 실험과 Windows의 강화학습 실험은 병행할 수 있다.
Windows 작업은 동일한 큐브·박스·SO101 자산, 관절 순서, 제어 주기와 성공 판정을 먼저 재현한 뒤
보상과 PPO 학습 환경을 검증하는 별도 작업이다. 현재 이 문서의 IL 태스크만으로 PPO가 준비됐다는 뜻은 아니다.
ACT 체크포인트를 PPO actor에 바로 로드할 수 있다고 가정하지 않는다. 모델 구조·관측·행동이 맞는
BC actor를 따로 학습하거나 정책 증류 어댑터를 구현한 후에 BC 초기화 PPO와 순수 PPO를 비교한다.
Windows의 실제 설치·학습 실행은 해당 PC에서 별도로 검증해야 한다.

Windows에는 Git 저장소와 대용량 산출물을 분리해서 전달한다.

- Git 저장소: 태스크 코드, MimicGen 호환 패치, 이 문서
- 별도 복사: `datasets/*.hdf5`, `outputs/portfolio/pick_cube_into_box/**/*.mp4`, 선택한 `outputs/robomimic/**/*.pth`, `outputs/evaluation/**/*.json`, `outputs/evaluation/**/*.mp4`
- 캘리브레이션 JSON: 보드별 파일이므로 저장소에 포함하지 않고 해당 Linux 장비에 보관

HDF5와 MP4는 Git에 넣지 않는다. 복사 후 양쪽에서 SHA-256을 비교해 파일 손상을 확인한다.
Windows는 결과 열람·보관 외에 별도 강화학습 작업을 맡을 수 있다. 리더를 이용한 원본 녹화는
현재 검증된 Linux 환경에서 수행한다. 복구 상태 `outputs/evaluation/*.pt`와 ACT의
`pretrained_model` 디렉터리도 필요한 경우 별도 복사한다.

## 실패 상태에서 이어지는 복구 데이터 실험 — 2026-09-12

공식 [MimicGen 결과](https://mimicgen.github.io/)는 소수 인간 시연에서 새 물체 배치의
시연을 생성하고 BC로 학습하는 흐름을 보여준다. 이 프로젝트의 10회 원본 증강은 그 흐름을 따른다.
[ROBOTIS cyclo_lab](https://github.com/ROBOTIS-GIT/cyclo_lab)은 참고 구현이며 SO101 지원을 그대로 보장하지 않는다.

이번에 추가한 상태 머신은 **MimicGen 원본 알고리즘이 아니라 별도의 privileged-state 복구 오라클**이다.
시뮬레이터의 큐브·박스 위치를 읽어 ACT가 방문한 실패 상태에서 새 성공 궤적을 만드는 실험이다.
DAgger의 상태 분포 보완 아이디어를 참고하지만, 데이터 병합·재학습·반복 평가까지 수행한 DAgger 결과는 아직 아니다.
학습된 ACT의 성공률과 오라클 복구 성공률은 구분한다.

### 확인된 상태 복원 오류

Isaac Lab `InteractiveScene.reset_to()`는 저장된 `joint_velocity`를 실제 관절 속도뿐 아니라
PD 제어기의 목표 속도로도 복원한다. 위치 제어로 이어갈 때 목표 속도가 남아 관절이 계속 밀렸다.
`generate.py`는 복원 직후 목표 속도만 0으로 바꾼다. 물리적인 관절 속도·위치·물체 상태는 보존한다.
회귀 테스트는 이 구분과 마지막 녹화 에피소드의 저장을 검증한다.

### 실행 순서

저장소 루트에서 해당 PC의 Isaac Lab Python 환경을 활성화한 뒤 실행한다.
아래 checkpoint·snapshot·HDF5 경로는 저장소 루트 기준이며 대용량 파일은 Git에 포함되지 않는다.

```bash
python scripts/evaluation/lerobot_act_so101.py \
  --checkpoint outputs/lerobot/so101_act_next_state_mimic100_seed43_chunk30_30000/checkpoints/030000/pretrained_model \
  --num-rollouts 1 --horizon 1200 --seed 3000 --n-action-steps 30 \
  --video-count 0 --output-dir outputs/evaluation/act_failure_capture \
  --failure-state-file outputs/evaluation/act_failure_capture.pt \
  --failure-state-interval 120 --device cuda:0 --headless

python scripts/datagen/state_machine/generate.py \
  --task LeIsaac-SO101-PickCubeIntoBox-v0 --num_envs 1 \
  --initial_state_file outputs/evaluation/act_failure_capture.pt \
  --num_demos 0 --max_attempts 10 --step_hz 240 \
  --summary_file outputs/evaluation/recovery_audit.json \
  --device cuda:0 --enable_cameras --headless

python scripts/imitation_learning/test_recovery_state.py
```

`--record --dataset_file datasets/recovery.hdf5`를 추가하면 시도 전체를 기록한다.
HDF5는 실패도 포함할 수 있으므로 각 `data/demo_*` 그룹의 `success` 속성을 확인하고 성공만 학습에 사용한다.
`step_hz`는 실행 속도 상한이며 물리 제어 주기(현재 60Hz)를 바꾸지 않는다.
`LEISAAC_SM_DIAGNOSTICS=1`은 단계별 턱–큐브 거리와 관절 목표를 출력한다.

이 실험의 snapshot은 한 ACT rollout의 120~1200스텝 10개다. 독립 초기조건 10회가 아니다.
오라클은 시뮬레이션 상태를 사용하고 로봇 중력을 비활성화하므로 복구 데이터의 정책 전이 성능은
별도로 검증해야 한다. 복구 성공만으로 ACT 개선이나 실물 성공을 주장하지 않는다.

### 실측 결과 및 산출물

| 검증 | 결과 | 해석 |
|---|---|---|
| 수정 전 한 실패 rollout의 중간 상태 10개 연속 복구 | 0/10 | 복원된 목표 속도 잔류 상태 |
| 목표 속도 수정 후 같은 10개 상태 연속 복구 | 9/10 | step 120 실패, step 240~1200 성공 |
| step 240 새 프로세스 단독 녹화 | 0/1 | 연속 실행 성공이 단독 성공을 보장하지 않음 |
| step 1200 새 프로세스 단독 녹화 | 1/1 | 성공 속성 및 810프레임 저장 확인 |
| step 1200 또 다른 새 프로세스 단독 재실행 | 1/1 | 동일 상태 반복 성공, 독립 초기조건 평가는 아님 |

- 연속 평가: `outputs/evaluation/oracle_recovery_seed3000_zero_velocity_target.json`
- 단독 녹화 결과: `outputs/evaluation/oracle_recovery_seed3000_step1200_recorded.json`
- 단독 반복: `outputs/evaluation/oracle_recovery_seed3000_step1200_repeat.json`
- 성공 복구 데이터: `datasets/pick_cube_into_box_recovery_seed3000_step1200.hdf5`
- 비교용 실패 데이터: `datasets/pick_cube_into_box_recovery_seed3000_step240.hdf5` (`success=false`, 학습 제외)
- 복구 영상: `outputs/portfolio/pick_cube_into_box/recovery_seed3000_step1200.mp4`

성공 HDF5의 전면·손목 RGB는 각각 810×240×320×3, 관절 상태는 810×6,
action은 810×8(IK 자세 7 + 이진 그리퍼 1)이다. action 전체가 유한값이고 마지막 영상에서
빨간 큐브가 파란 박스 안에 있는 것을 확인했다. 영상은 60fps 기준 13.5초다.
이 영상은 **학습된 ACT가 성공한 영상이 아니라 ACT 실패 상태를 오라클이 복구한 영상**으로 설명한다.

다음 학습 전 조건은 다른 rollout seed에서도 단독 복구가 재현되는 성공 데이터 확보,
6관절 행동 규격 변환, 원본·MimicGen·복구의 출처 구분 및 rollout 단위 분리다.
같은 실패 rollout의 snapshot을 train과 validation 양쪽에 나누면 안 된다.
성공 복구 1회를 병합한 결과는 아래에 기록한다. ACT 재학습·성능 개선은 아직 검증하지 않았다.

### 다른 초기조건 검증과 성공 데이터 병합 — 2026-09-12 후속

ACT seed 3100~3103을 각각 1200스텝 실행한 결과는 0/4였다. 각 rollout의
600·1200스텝 상태를 추출하고, **snapshot마다 새 Isaac Sim 프로세스**에서 오라클을
실행했다. 결과는 **0/8**이다. 같은 rollout 안의 두 snapshot은 서로 독립 표본이 아니므로
독립 rollout 4개에서 나온 복구 시도 8개로 표기한다. 이전 seed 3000 연속 실행의 9/10을
일반적인 복구 성공률로 해석하면 안 된다.

| seed 3101, step 1200 파지 비교 | 결과 | 변경 조건 |
|---|---|---|
| 손목 목표 고정, 후보 offset | 0/1 | (-0.020, 0, 0.095) m |
| 손목 목표 고정, 기존 offset | 0/1 | (-0.012, 0.020, 0.090) m |
| 기존 offset + 열린 그리퍼로 120스텝 대기 | 0/1 | 접근 지연 확인 후 대기 추가 |
| 같은 대기 + 낮은 offset | 0/1 | (-0.012, 0.020, 0.075) m |
| 원본 demo 7의 파지 자세 + 같은 대기 | 0/1 | offset (-0.004, 0.019, 0.090) m, RPY (-0.197, 0.110, 0.334) rad |

마지막 조건은 원본 `pick_cube_into_box_annotated_wrist_10_20260911.hdf5`의
demo 7, frame 199에서 관찰한 자세를 반올림해 쓴 실험값이다. 하드웨어 캘리브레이션 값이 아니다.
`--cube_grasp_alignment fixed_wrist`, `--cube_grasp_offset`, `--cube_grasp_rpy`는
실패 비교를 재현하기 위한 실험 옵션이다. 기본 `live_jaw` 동작과 810스텝 길이는 유지한다.
fixed_wrist는 열린 상태로 대기하는 120스텝이 추가되어 930스텝이다.
위 비교는 모두 같은 상태이므로 일반화 평가로 세지 않는다.

새 성공 데이터는 확보하지 못했다. 기존 seed 3000 step 1200 성공 1회만 병합 도구 검증에 사용하고,
새 실패 8회는 제외했다. 동일 성공 상태의 반복 녹화는 추가 시연으로 세지 않았다.

- HDF5: `datasets/pick_cube_into_box_mimic100_recovery1_next_state_84.hdf5`
- 출처·SHA-256·제외 사유·분리 목록: 같은 이름의 `.manifest.json`
- LeRobot: `datasets/lerobot/so101_mimic100_recovery1_next_state_84`
- 복구 상태 및 시도별 결과: `outputs/evaluation/recovery_round1/`
- ACT 실패 상태 원본: `outputs/evaluation/recovery_round1_states_3100_3103.pt`
- 새 조건 실패 영상: `outputs/portfolio/pick_cube_into_box/recovery_round1_failed_seed3101_step1200.mp4`

병합 결과는 **101회, 58,884프레임, HDF5 train 81회 / valid 20회**다.
복구 action은 IK의 8차원 값이 아니라 `joint_pos[t+1]` 6차원이며 마지막 프레임은 반복한다.
전면·손목 영상은 84×84 RGB로 변환했다. 복구 에피소드의 처음·중간·마지막 프레임에서
LeRobot 로더의 관절·action 값이 HDF5와 일치하고 영상이 정상 디코딩되는 것을 확인했다.

병합 도구는 성공 bool 속성, 유한값, 관절·영상 shape를 검사하고 같은 관절/action 궤적의
중복을 제외한다. 복구는 train에만 추가하며 기존 valid 목록은 변경하지 않는다.
이 중복 검사는 동일 궤적 탐지일 뿐 같은 rollout의 다른 snapshot을 독립 데이터로 만들어 주지 않는다.

```bash
# 저장소 안의 터미널에서 루트로 이동한다. Isaac Lab 환경에서 실행한다.
cd "$(git rev-parse --show-toplevel)"
python scripts/imitation_learning/merge_recovery_dataset.py \
  --baseline datasets/pick_cube_into_box_mimic_joint_next_state_front_wrist_100_seed43_84x84_20260912.hdf5 \
  --recovery datasets/pick_cube_into_box_recovery_seed3000_step1200.hdf5 \
  --output datasets/mimic100_recovery1_rebuild.hdf5
python scripts/imitation_learning/test_merge_recovery_dataset.py
python scripts/imitation_learning/test_recovery_state.py

# LeRobot 환경에서는 추가된 demo_100만 변환하여 기존 100회 영상의 재인코딩을 피할 수 있다.
python scripts/imitation_learning/convert_hdf5_to_lerobot.py \
  --input datasets/mimic100_recovery1_rebuild.hdf5 --start-episode 100 \
  --output-root datasets/lerobot/recovery1_rebuild --repo-id local/recovery1_rebuild
```

**주의: LeRobot 변환·aggregate는 HDF5의 train/valid mask를 자동 보존하지 않는다.**
전체 101회 root를 그대로 학습하면 valid 20회도 포함된다. 학습 전에 manifest의 train 목록을
LeRobot episode index로 명시하고 정규화 통계도 학습 subset 기준인지 확인해야 한다.
현재 101회 파일은 데이터 연결 검증용이며 새로운 ACT 학습이나 정책 선택에 사용하지 않았다.
복구 1회를 추가한 것만으로 성능이 개선됐다고 주장하지 않는다.

## 포트폴리오에서 보여줄 증거

1. 원본 사람이 조작한 시연 영상
2. 같은 태스크에서 초기 큐브 위치가 달라진 MimicGen 생성 영상
3. 원본 10회와 생성 데이터 수, 자동 주석 통과 수
4. 시뮬레이션의 보지 않은 위치 성공률과 실패 영상
5. GMM·관절 delta·절대 관절 목표 모델의 실패 원인과 수정 과정
6. 이후 성공 정책이 확보됐을 때 동일 정책의 실물 팔로워 결과

현재 포트폴리오 표현은 `리더 원본 10회 → 독립 seed MimicGen 성공 50회·100회 생성 → IK 관절 목표 불연속 발견·다음 상태 라벨 정제 → ACT 오프라인 RMSE 0.0415 rad → 미학습 시드 성공 장면 2/10, 성공 seed 반복 0/2`가 정확하다. “안정적 자율 파지”나 “실물 전이 완료”로 표현하면 안 된다.
