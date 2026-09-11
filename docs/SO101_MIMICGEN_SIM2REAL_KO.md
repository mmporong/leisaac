# SO101 리더 10회에서 MimicGen과 실물 검증까지

## 결론

이 프로젝트는 **LeIsaac을 실행 기반으로 사용하고, ROBOTIS `cyclo_lab`은 파이프라인 설계의 비교 자료로 사용한다.** 두 저장소를 동시에 실행하거나 코드를 섞는 방식은 아니다.

- LeIsaac을 쓰는 이유: SO101 리더 입력, SO101 시뮬레이션 모델, 관절↔말단 자세 변환이 이미 연결되어 있다.
- `cyclo_lab`을 참고하는 이유: ROBOTIS가 공개한 `실물 시연 → MimicGen → BC 학습 → 실물 추론` 구조를 확인할 수 있다.
- 그대로 쓰지 않는 이유: `cyclo_lab`의 실물 인터페이스와 데이터 변환기는 OMY·FFW 계열용이며 SO101 포트, 관절 수, 카메라 키와 맞지 않는다.
- MimicGen은 강화학습이 아니다. 사람이 만든 시연을 분할·재조합해 시연 데이터를 늘리는 도구다.
- ACT는 필수가 아니다. 우선 BC 또는 BC-RNN으로 검증할 수 있다. 강화학습은 별도 선택 단계다.

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
  ↓ BC 또는 BC-RNN 학습
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

2026-09-11 실행 기준이다. 실행 결과는 코드 주석이 아니라 이 문서와 산출물에 기록한다.

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

작은 시험은 성공 10회, 본 생성은 성공 50회 이상으로 나눈다. `generation_guarantee=True`에서는 성공 수가 목표에 도달할 때까지 재시도한다. 비정상 종료나 디스크·GPU 오류가 있을 수 있으므로 종료 로그만 믿지 않고 결과 HDF5의 성공 에피소드 수를 다시 센다.

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

RTX 5050 Laptop 8GB 환경에서는 `--num_envs 1`을 사용한다. 640×480 전면·손목 영상을 함께 기록하면 에피소드 내보내기의 `torch.stack`에서 약 958MB 추가 할당이 발생해 OOM으로 실패했다. 두 카메라를 320×240으로 낮추고 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`를 적용한 뒤 성공 50회를 생성했다. 환경 수를 크게 올리는 것은 속도 설정이지 데이터 의미를 높이는 조건이 아니다.

2026-09-11 본 생성은 총 125회 시도 중 성공 50회, 실패 75회로 성공률 40%였다. `generation_keep_failed=True`이므로 실패 궤적은 `_failed.hdf5`에 분리됐다. 학습에는 성공 파일만 사용하고, 실패 파일은 실패 원인 분석에만 사용한다.

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
  --width 84
```

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

## 실행 게이트 상태

초기 10회 생성은 **MimicGen 파이프라인 검증**이었고, 이후 본 생성에서 전면·손목 카메라가 함께 든 성공 50회를 확보했다. 이 50회로 BC-RNN을 학습했으나 새 시드 자율 평가는 0/5였다. 정책 학습 파이프라인은 완주했지만 자율 집기 성능과 실물 팔로워 검증은 완료되지 않았다.

| 게이트 | 수행 내용 | 완료 증거 | 현재 상태 |
|---|---|---|---|
| G0 증강 검증 | 원본 10회로 MimicGen 성공 10회 생성 | HDF5 무결성, 서로 다른 초기 위치·궤적, 비교 영상 | 완료 |
| G1 관측 일치 | 시뮬레이션 SO101에 손목 카메라 추가하고 기존 시연 재생 | HDF5 10/10에 `obs/wrist`, 총 5,640프레임, 영상 10개 | 완료 |
| G2 본 데이터 생성 | 주석된 원본 10회에서 성공 합성 시연 50회 생성 | 성공 50/50, 초기 위치·궤적 50개 고유, HDF5 무결성, 표본 영상 | 완료 |
| G3 관절 복원·데이터 분리 | IK action을 6관절 action으로 복원하고 공간 분산 train/validation 분리 | action 6차원, train 40 / validation 10, 계보 정보 부재 명시 | 완료 |
| G4 모방학습 | 전면·손목 영상과 관절 문맥으로 다음 6개 관절 목표를 예측하는 BC-RNN 학습 | 설정 파일, 체크포인트, 학습·검증 로그 | 완료 |
| G5 시뮬레이션 평가 | 학습에 쓰지 않은 시드의 큐브 위치에서 자율 롤아웃 | 0/5 성공, 실패 영상 2개, clipping 0회 | 완료·성능 미달 |
| G6 실물 평가 | 카메라 입력과 SO101 팔로워 출력을 잇는 추론 어댑터로 단계 평가 | 접근·집기·배치 단계별 실측 결과와 영상 | 이번 작업 제외 |

### 단계별 수행 기록

1. **완료:** SO101 손목 링크에 LeIsaac 기본 SO101 손목카메라를 복원했다. 원본 재생 검증은 640×480 RGB로 수행했다. 8GB GPU에서 두 카메라를 포함한 MimicGen 생성 시 내보내기 OOM이 확인되어 본 생성과 학습용 관측은 전면·손목 모두 320×240으로 조정했다.
2. **완료:** 기존 원본 10회를 새 손목 시점으로 재생·재주석했다. 성공 10/10, 총 5,640프레임이며 모든 에피소드에 `obs/front`와 `obs/wrist`가 함께 있다. 이 검증 HDF5는 640×480이고, 50회 생성 결과는 320×240으로 기록한다.
3. **완료:** 손목 영상이 포함된 `pick_cube_into_box_annotated_wrist_10_20260911.hdf5`로 MimicGen 성공 50회를 만들었다. 성공 파일은 27,217프레임이며 50개의 초기 큐브 위치와 관절 궤적이 모두 서로 달랐다.
4. **완료:** 8차원 IK action을 SO101 6관절 목표로 복원하고 train 40회 / validation 10회로 나눴다. 원본 계보가 HDF5에 없어 계보 독립성은 검증하지 못했으며 이 한계를 manifest에 기록했다.
5. **완료:** Robomimic 0.4.0 BC-RNN 학습기·설정·데이터 전처리를 추가하고 최저 validation loss 0.0972 체크포인트를 만들었다.
6. **완료·성능 미달:** 보지 않은 시드 5개에서 자율 평가했으나 0/5였다. 실패는 큐브 접근 후 파지 실패 유형이다.
7. **이번 작업 제외:** 실물 팔로워는 실행하지 않았다. 시뮬레이션 성공 기준을 통과하기 전에는 이 정책을 실물 성공 정책으로 취급하지 않는다.

G0의 비교 영상은 데이터 증강 검증 자료로 바로 사용할 수 있다. 포트폴리오에서는 이를 “MimicGen으로 정책 학습을 완료했다”가 아니라 **“리더 시연 10회를 물체 위치와 궤적이 다른 성공 시연으로 증강하는 파이프라인을 검증했다”**고 설명한다.

현재 손목카메라 위치·렌즈 값은 LeIsaac의 SO101 기본 설정이다. 실물 손목카메라의 실제 해상도, 시야각과 장착 변환을 아직 측정해 맞춘 것은 아니다. 50회 생성에는 사용할 수 있지만 실물 전이 전에는 실측값으로 보정하고 필요하면 데이터 생성부터 다시 수행한다.

## 학습: 강화학습이 아니라 먼저 모방학습

현재 데이터에 바로 맞는 첫 학습은 BC 계열이다.

1. 입력: 실물과 시점을 맞춘 시뮬레이션 손목 카메라 영상과 현재 관절 상태
2. 출력: 다음 SO101 관절 목표
3. 학습: BC 또는 BC-RNN
4. 평가: 보지 않은 큐브 초기 위치에서 시뮬레이션 성공률 측정
5. 통과 후: 실물 카메라 입력과 팔로워 출력을 연결

ACT는 긴 시간 구간의 동작 묶음을 예측하는 선택지다. 이 프로젝트의 첫 증명에는 필수가 아니다. BC-RNN이 실패하거나 장시간 동작의 흔들림이 클 때 비교 대상으로 추가한다.

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

현재까지 검증된 것은 리더 시연 10회, MimicGen 성공 50회, 6관절 변환, BC-RNN 학습, 미학습 시드 5회 평가까지다. 자율 평가는 0/5였고 실제 팔로워는 실행하지 않았다.

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

Windows에는 Git 저장소와 대용량 산출물을 분리해서 전달한다.

- Git 저장소: 태스크 코드, MimicGen 호환 패치, 이 문서
- 별도 복사: `datasets/*.hdf5`, `outputs/portfolio/pick_cube_into_box/**/*.mp4`, 선택한 `outputs/robomimic/**/*.pth`, `outputs/evaluation/**/*.json`, `outputs/evaluation/**/*.mp4`
- 캘리브레이션 JSON: 보드별 파일이므로 저장소에 포함하지 않고 해당 Linux 장비에 보관

HDF5와 MP4는 Git에 넣지 않는다. 복사 후 양쪽에서 SHA-256을 비교해 파일 손상을 확인한다. Windows는 우선 결과 열람·보관·포트폴리오 편집 대상으로 사용하고, 리더를 이용한 원본 녹화는 현재 검증된 Linux 환경에서 수행한다.

## 포트폴리오에서 보여줄 증거

1. 원본 사람이 조작한 시연 영상
2. 같은 태스크에서 초기 큐브 위치가 달라진 MimicGen 생성 영상
3. 원본 10회와 생성 데이터 수, 자동 주석 통과 수
4. 시뮬레이션의 보지 않은 위치 성공률과 실패 영상
5. GMM·관절 delta·절대 관절 목표 모델의 실패 원인과 수정 과정
6. 이후 성공 정책이 확보됐을 때 동일 정책의 실물 팔로워 결과

현재 포트폴리오 표현은 `10회 원본 → 성공 50회 증강 → BC-RNN 학습 → 미학습 위치 0/5와 실패 분석`까지가 정확하다. 실물 검증이나 모방학습 성공은 다음 결과가 나오기 전까지 포함하지 않는다.
