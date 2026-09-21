# 500회 Mimic 데이터 → 224 ACT 학습·평가 인계

## 2026-09-21 사용자 요청 소규모 ACT 진단 학습

사용자의 학습 진행 요청으로 기존 76개 후보의 진단 학습과 평가를 완료했다.
아래 과거 기록의 본학습 보류는 유지한다. strict replay 승인 게이트는 변경하지 않으며,
이번 실행은 그 게이트 통과 주장이나 실물 배포 승인이 아니다.

- 행동 표현: `recorded_target` 6D 절대 관절 목표각(rad). 원래 Mimic 8D 말단 자세 명령을
  학습하는 것이 아니다. 실행도 SO101 관절 위치 제어이며 관절 제한을 적용한다.
- 새 데이터 분리: seed 43, 전체 76개에서 학습 61개/26,527프레임,
  검증 15개/7,998프레임. 에피소드 중복 없이 분리하고 상태·행동 통계를 학습 집합에서 산출했다.
  같은 원본 10회에서 파생된 에피소드 분리이므로 원본 시연 단위 일반화 검증은 아니다.
- 설정: ACT 2,000 updates, batch 8, 앞/손목 224 영상, chunk 30, 실행 30스텝마다 재관측.
  사전 지정한 마지막 체크포인트를 사용하며 평가 성공률로 모델을 고르지 않는다.
- 평가: 오프라인 검증 후 seed 4100/4101/4102 각각 새 Isaac 프로세스에서 최대 1,200스텝.
  기존 안정 놓기 판정과 영상 저장을 유지한다. 3회는 진단이며 신뢰할 만한 일반 성공률 추정은 아니다.
- 원본 데이터·과거 모델 보존. 새 결과는 `outputs/act_pilot_76_20260921/`,
  분리는 `outputs/act_pilot_76_20260921_split/`에 저장한다. 디스크 여유 8 GiB 미만이면 중단한다.

```bash
cd "/data/$USER/leisaac"
"$HOME/miniforge3/envs/lerobot/bin/python" -u scripts/imitation_learning/run_act_vision_experiment.py \
  --split-root outputs/act_pilot_76_20260921_split \
  --output-dir outputs/act_pilot_76_20260921 \
  --isaac-python "/data/$USER/conda-envs/leisaac/bin/python" \
  --steps 2000 --batch-size 8 --eval-seeds 4100 4101 4102 --min-free-gib 8
```

같은 출력 경로는 재사용하지 않는다. 완료 여부는 `progress.json`과
`evaluation_summary.json`, 체크포인트 검증 및 각 단계 종료 코드로 확인한다.

학습은 2,000스텝을 마쳤고 체크포인트의 파라미터 유한성과 학습 전용 정규화 통계가
검증됐다. 검증 집합의 균등 500프레임에서 관절 목표각 RMSE는 0.088081 rad,
15개 시작 프레임에서는 0.293607 rad다. 두 지표는 정책의 실제 집기 성공률이 아니다.
2,000×8/26,527 ≈ 0.60만큼의 프레임 샘플 수에 해당하는 짧은 학습이므로
수렴이나 데이터 충분성 판단에 사용하지 않는다. 기존 진단 대상 002/demo_17과
009/demo_9는 각각 원본 합본 인덱스 5, 24로 학습 집합에 포함됐다.

최종 시뮬레이션 평가는 **0/3 성공**이다. seed 4100/4101/4102 모두 1,200스텝 동안
큐브 상승 0 m(`no_lift`), 관절 목표 제한 적용 0스텝이었다. 학습·체크포인트 검증·
오프라인 평가·새 프로세스 3회 실행을 담당한 runner는 exit 0으로 끝났다.
영상 3개는 각각 1280×480·60fps·1,200프레임이며 결과와 프레임 수가 일치한다.
회귀 테스트 146개 통과. 원본 데이터 부족이나 Mimic 무효를 입증한 결과는 아니며,
이 짧은 학습 정책으로는 아직 집기를 수행하지 못한다.

다음 실험은 현재 모델을 기준점으로 보존하고 시작 상태에서의 큰 예측 오차와
학습 수렴 여부를 확인하는 것이다. 평가 결과에 맞춰 성공 기준을 바꾸지 않는다.
추가 장시간 학습이나 제어 표현 변경은 이번 실행에 포함하지 않았다.
[학습·평가 근거](evidence/act_pilot_76_20260921.json)에 설정·분리·오차·영상 해시를 보존했다.

> **2026-09-21 제어 비교 갱신:** 기존 집기 실패 두 시연을 각각 새 프로세스에서 비교했다.
> `002/demo_17`은 고정 관절 목표각 재생에서 실패했지만 원래 Mimic 명령에서는 성공했다.
> `009/demo_9`는 이번 단독 실행에서 두 방식 모두 성공했다. 데이터 전체의 불량이나
> 학습 준비 완료로 단정하지 않는다. 아래 제어 비교가 최신 해석이며 본학습은 보류한다.

## 2026-09-21 집기 실패 제어 비교

두 모드 모두 관절 PD 제어가 있다. `recorded_target`은 저장된 관절 목표각 순서를
그대로 사용하고, `mimic_action`은 원래 말단 자세 명령과 현재 상태로 IK 목표각을
매 스텝 다시 계산한다. 같은 시연의 초기 물리 상태 해시는 일치했고, 비교 구간은
공통 675스텝으로 제한했다. Mimic의 마지막 추가 1스텝 때문에 성공한 것이 아니다.

| 시연 | 관절 목표각 재생 | 원래 Mimic 명령 | Mimic 첫 성공 스텝 |
| --- | --- | --- | --- |
| shard_002/demo_17 | 실패, 최대 상승 0 m | 성공, 최대 상승 0.113816 m | 643 |
| shard_009/demo_9 | 성공, 최대 상승 0.115449 m | 성공, 최대 상승 0.114470 m | 644 |

확인된 사실:

- 두 관절 목표각 재생에서 실제 적용 목표와 raw 저장 목표의 최대 차이는 0이다.
- 002는 두 모드 모두 287스텝에서 원래 관절 궤적과 1e-6 rad를 넘는 차이가 처음 생겼다.
  Mimic은 이후 현재 상태를 반영해 목표각을 조정하며 성공했다.
  공통 구간 관절 RMSE는 고정 목표 0.028090 rad, Mimic 0.003619 rad다.
- 002 고정 목표 재생의 그리퍼 힘 제한은 500스텝에서 바뀌었다. 궤적 차이보다 늦으므로
  힘 제한 변경을 최초 실패 원인으로 단정할 수 없다.
- 009는 과거 연속 평가에서 실패했지만 이번 새 프로세스의 고정 목표 재생에서 성공했다.
  실행 순서·초기화 맥락과 수치적 변동을 분리하는 추가 검증이 필요하다.
- 4개 영상은 모두 1280×480이며 프레임 수는 각 실행의 675 또는 676스텝과 일치한다.
  진단 JSONL도 같은 행 수를 갖는다. 로그의 `actuator_reported_torque_nm`은
  implicit actuator의 제한 적용된 PD 모델 추정값이며 실측 토크나 접촉력이 아니다.

해석과 다음 순서:

1. 이번 결과만으로 원본 데이터를 폐기하거나 리더 시연을 다시 수집하지 않는다.
   76개 후보 전수의 성공이나 학습 정책 성능을 검증한 결과는 아니다.
2. 009의 과거 실행 순서를 재현하고 새 프로세스 실행과 비교해 초기화 영향을 분리한다.
3. 학습 라벨과 실행 제어기의 행동 표현을 맞추는 검증을 먼저 한다.
   고정 관절 목표각 재생 실패를 원래 Mimic 시연 실패와 동일하게 취급하지 않는다.
4. 이후 고정된 검증 조건으로 후보를 평가하고 소규모 학습 및 미학습 조건 평가로
   데이터 충분성을 판단한다. 기존 학습 게이트·성공 판정은 이번에 완화하지 않았다.

최초 미세 편차의 물리적 원인과 009의 실행 맥락 영향은 아직 확정하지 못했다.
원본 데이터·기존 모델을 보존했고 새 학습·강화학습·실물 실행은 하지 않았다.
실험 후 변경은 추정 토크 설명과 로그 파일 열기 실패 시 영상 자원 해제 보완뿐이다.
실험 구현 해시는 [측정 근거](evidence/mimic_control_diagnosis_20260921.json)에 보존했다.
영상과 상세 로그는 `outputs/evaluation/control_diagnosis_20260921/`에 있다.

> **2026-09-21 선행 안정화 검증:** 고정 표본 20개에서 시연 구간 성공 18개,
> 1초 추가 관찰 종료 성공 19개였다. 해당 고정 목표각 평가에서 남은 집기 실패는
> 위 제어 비교로 추가 조사했다. [안정화 관찰 구간 분리](#2026-09-21-안정화-관찰-구간-분리)는 선행 검증 기록이다.

> **2026-09-19 갱신:** 현재 배치 기본 라벨은 `recorded_target`이며, 성공 판정은
> `stable_release_v2`다. 이전 500회 데이터·30,000-step 모델은 보존된 과거 실험이다.
> **현재 상태:** 500개는 보존 raw 전수다. 최신 후보는 76개이며 실제 재생 불일치가 발견돼
> 본 학습은 보류한다. 최신 절차는 [급변 시연 전수 검수](#2026-09-19-급변-시연-전수-검수)가 우선이다.
> 아래 400/100 분리·seed 4000~4009 계획은 과거 기록이다.
> 앞선 3회 통합 검증은 [파이프라인 보완](#2026-09-19-파이프라인-보완)에 남겨뒀다.

## 과거 500개 실험의 범위

증강 데이터 500회 생성은 끝났다. 이번 작업은 **ACT 모방학습과 시뮬레이션 평가**다.
강화학습이나 실물 팔로워 실행은 포함하지 않는다. 큐브·박스·로봇·카메라 위치와
기존 놓기/그리퍼 해제 성공 판정은 아래 과거 실험에 해당한다. 최신 판정은 안정 유지 조건을 추가했다.

## 생성 데이터 검증

- 원본 리더 시연 10회로 총 1,308회 시도하여 500성공·808실패, 생성 성공률 약 38.2%.
- 성공 데이터 500회·277,447프레임, 관절 궤적 SHA256 중복 없음.
- 전체 관절·원시 행동의 유한성 및 관절 연속성 검사 통과.
- 500회 **전체 프레임**의 LeRobot 상태가 원시 관측 관절과 같고, 행동이 다음 관측
  관절과 같음을 대조한다. 마지막 프레임의 행동은 마지막 관절값 반복이다.
- 앞/손목 관측 각각 uint8 224×224×3. 렌더는 640×480에서 AREA 축소한다.

근거: [500회 전수 검증](evidence/mimic_vision224_500_complete_20260913.json).
생성 데이터 성공률과 학습된 정책의 집기 성공률은 다른 지표다.

## 과거 실험의 데이터 분리

`prepare_mimic_act_split.py`가 seed 43으로 각 25회 생성 묶음에서 20회 학습,
5회 검증을 뽑는다. 최종 **train 400 / valid 100**, 에피소드 중복·누락 없음.
이 분할은 생성 seed를 통째로 보류하는 OOD 평가가 아니라 같은 생성 분포의
에피소드 holdout이다. 두 집합은 같은 원본 시연 10회에서 파생됐다.

LeRobot SDK의 `split_dataset()`으로 별도 데이터 루트를 만든다. `dataset.episodes`
필터만 사용하면 정규화 통계는 여전히 전체 500회 통계여서 이번 목적에 맞지 않는다.
원본 episode 통계 재집계와 원본 parquet 전체 행동·상태 mean/std를 독립 대조한 뒤
`split_provenance.json`을 생성한다. 원본→분할 후 episode 번호 매핑도 기록한다.

- 관절 상태/행동 정규화: **train 400회 통계만** 사용.
- 영상 정규화: 기존 ResNet 설정과 같은 **ImageNet 고정 mean/std** 사용.
  검증 영상에서 평균/분산을 구해 학습에 넣지 않는다.
- checkpoint에 저장된 활성 정규화 텐서를 위 값과 다시 대조한다.

## 과거 실험의 사전 고정 학습·평가 계획

| 항목 | 설정 |
| --- | --- |
| 모델 | 새 ACT, ResNet18 ImageNet backbone, 84 정책 이어학습 아님 |
| 입력 | state 6D, 앞/손목 RGB 각각 224×224 |
| 학습 | 30,000 updates, batch 8, seed 43 |
| ACT | chunk 30 / n_action_steps 30, dim 256, encoder 4, heads 8, VAE 사용 |
| 학습률 | policy 1e-4 / backbone 1e-5 |
| 저장 | 5,000 updates마다 checkpoint, 이전 실험 모델 보존 |
| 평가할 모델 | 사전 지정한 최종 30,000-step 모델 |
| 오프라인 | valid 100회 첫 프레임 + 균등 최대 500프레임의 관절 RMSE/MAE |
| 폐루프 | seed 4000~4009, 각 seed마다 새 Isaac 프로세스, 최대 1,200 steps |
| 추론 | CPU 서버, seed 0, 30행동 실행 후 재관측 |
| 카메라 | 640×480 렌더 → 224×224 입력 |
| 영상 | 10회 성공/실패 모두 기록, 1280×480·60fps |

30,000 updates는 첫 비교 실행의 고정 예산이지 수렴 보장이 아니다. 데이터가 커졌으므로
이전 80회 학습보다 에피소드당 반복 방문은 적다. 검증 오차와 폐루프 결과를 보고
추가 학습 필요성을 판단한다. rollout 성적을 보고 여러 checkpoint 중 유리한 것만
골라 보고하지 않는다. 기존 평가 seed는 역사적 비교용이며 미사용 test seed라고 부르지 않는다.

## 자동 실행·중단 조건

Linux의 현재 저장소는 `/data/lim/leisaac`이다. 아래 경로는 현재 기기 배치 기준이다.
아래 명령은 과거 실행 기록이다. 현재 코드는 출처 계약이 없는 기존 split을 기본 거부하므로
새 실험에는 아래 최신 절차와 별도 출력 경로를 사용한다.

```bash
cd "/data/$USER/leisaac"
"$HOME/miniforge3/envs/lerobot/bin/python" scripts/imitation_learning/prepare_mimic_act_split.py \
  --dataset-root outputs/mimic_vision224_500_20260913/lerobot_all \
  --repo-id local/so101_mimic_vision224_500 \
  --output-dir outputs/mimic_vision224_500_split_20260913

"$HOME/miniforge3/envs/lerobot/bin/python" scripts/imitation_learning/run_act_vision_experiment.py \
  --split-root outputs/mimic_vision224_500_split_20260913 \
  --output-dir outputs/lerobot/act_vision224_500_20260913 \
  --isaac-python "/data/$USER/conda-envs/leisaac/bin/python"
```

이미 생성된 출력에는 같은 명령을 덮어 실행하지 않는다. 이 runner는 기존 출력이 있으면
거부한다. 중단 시 checkpoint를 보존하지만 자동 resume은 구현하지 않았다. 재개는
마지막 정상 checkpoint와 optimizer 상태를 확인한 후 별도 실행으로 진행한다.

runner는 학습 → checkpoint/정규화 검사 → 오프라인 평가 → 10회 시뮬레이션 평가를
LLM 호출 없이 순서대로 수행한다. 학습과 Isaac은 동시에 실행하지 않는다.
여유 공간 8GiB 미만이면 runner 소유 자식 프로세스 그룹만 종료하고 오류로 기록한다.
원본 데이터·기존 모델은 삭제하지 않는다. `STOP` 파일은 단계 시작 전에 확인하며
이미 시작한 학습의 즉시 중단 기능은 아니다.

`outputs/lerobot/act_vision224_500_20260913/` 아래:

- `plan.json`: 학습 명령, 분할 provenance 해시, 평가 조건, 코드 commit
- `progress.json`: 진행 단계·heartbeat·오류, `status=complete`가 전체 평가 완료
- `logs/train.log`: 학습 step/loss/학습률
- `model/checkpoints/`: 정책과 optimizer 상태
- `checkpoint_validation.json`: 유한한 모델 가중치와 정규화 검증
- `offline_valid.json`: 검증 데이터 오차
- `rollouts/seed_400*/`: 각 평가 JSON과 성공/실패 영상
- `evaluation_summary.json`: 최종 결과 묶음

데이터·모델·영상은 로컬 출력이며 코드 push에는 포함되지 않는다. 이전 실험과
데이터 수·해상도·정규화가 달라, 결과가 좋아져도 **500회 데이터 수만의 효과**라고
단정할 수 없다. 실물 파지 성능은 별도 실물 검증 전까지 주장하지 않는다.

## 본 학습 전 검증 결과

- 실제 분리: train 400회·223,267프레임, valid 100회·54,180프레임.
- 원본과 두 split의 **모든 상태·행동 프레임이 정확히 일치**한다.
  양쪽에서 처음·중간·마지막 이미지도 실제 디코딩해 224 입력을 확인했다.
- CPU 회귀 테스트 64개, 새 실행 파일/테스트 컴파일, 공백 검사 통과.
- 독립 검증에서 분할·정규화·실제 CLI 인자/SDK 호환성 확인.
- full-runner smoke: **batch 8 / 2 학습 step**으로 실제 400회 데이터를 로드하여
  GPU 학습·checkpoint 저장·정규화 텐서 검사를 통과했다.
- 같은 smoke 모델에서 valid 첫 100 + 균등 500프레임 평가, seed 4999의
  1,200-step 시뮬레이션·1280×480/60fps 영상 검증까지 자동 완주했다.
- smoke 결과 0/1, 상승 기준 미달이었다. 두 번 업데이트한 모델의 **실행 연결 검사**이며
  본 모델 집기 성공률이나 학습 실패의 근거로 사용하지 않는다.
- smoke 이후 소스 SHA 기록과 자식 PID 상태 갱신만 보강했다.

근거: [분리·학습 실행 전 검증 JSON](evidence/act_vision224_500_launch_validation_20260913.json).
본 30,000-step 학습과 최종 10회 평가는 별도 `act_vision224_500_20260913` 실행이며,
이 문서의 smoke 완료를 본 실험 완료로 해석하지 않는다.

## 2026-09-18: 추가 학습 전에 생성·평가 제어 계약 수정

500회로 학습한 기존 ACT의 역사적 평가 결과는 **1/10 성공**이었다.
현재 데이터 수를 더 늘리거나 조명 증강부터 적용하지 않는다. 성공 시연을 같은
초기 상태에서 재생하는 검사에서 생성기와 평가기의 차이가 발견됐다.

1. **초기 카메라 관측:** Isaac Lab의 `rerender_on_reset=False` 기본값에서는 이전
   RTX 출력이 초기 관측에 남을 수 있다. 평가기에 물리 스텝 없이 4회 렌더한 뒤
   front/wrist 센서 캐시를 갱신하는 처리를 추가했다. 물리 상태 해시가 달라지면
   중단한다. `--reset-render-frames 0`은 이전 관측 방식 재현용이다.
2. **그리퍼 힘 제한:** Mimic 환경은 매 스텝
   `dynamic_reset_gripper_effort_limit_sim`을 호출하지만 기존 ACT 평가기는
   일반 환경을 사용하면서 이 호출이 빠져 있었다. `--gripper-effort-mode task`로
   같은 처리를 적용한다. `fixed`는 기존 고정 제한 방식 재현용이다.
3. **행동 라벨:** 기존 `next_observed`는 `q[t+1]`이라는 물리 결과를 학습한다.
   평가기는 이를 모터 목표각으로 해석한다. 원본 recorder의 목표각은
   `obs/joint_pos_target[t+1]`이다. 숫자의 단위가 같아도 제어 의미는 다르다.

첫 두 수정은 평가 기본값에 반영했다. 기존 500회 데이터·모델은 덮어쓰지 않는다.
카메라 수정만 적용한 seed 4001/4007은 둘 다 `no_lift`였다. 따라서 초기 영상
갱신만으로 정책 성능이 해결됐다고 주장하지 않는다.

### 학습 없이 제어 계약을 검사하는 방법

`scripts/evaluation/replay_joint_contract.py`는 같은 HDF5 초기 상태를 복원하여
다음 명령들을 비교한다. 실물이나 리더 연결은 사용하지 않는다.

- `mimic_action`: 원본 IK 행동 전체 T개. 원본 성공 재현 대조군.
- `recorded_target`: 저장된 목표각 `target[1:T]`를 관절 제한 안으로 잘라 적용.
  마지막 목표각이 기록되지 않아 T−1개만 실행한다.
- `next_observed`: 현재 학습 계약과 같은 `q[1:T] + q[-1]`, 총 T개.

성공 비교에는 공통 T−1 구간의 `success_by_common_horizon`을 사용한다.
T와 T−1 전체 RMSE를 섞지 않고 `common_horizon_joint_rmse_rad`를 비교한다.
마지막 동작을 빠뜨려 원본 성공을 실패로 오판하지 않도록 원본 IK 대조군은
T개 전부 실행한다. 초기 상태 해시와 관절 오차, 실제 그리퍼 힘 제한도 기록한다.

```bash
cd "/data/$USER/leisaac"
"/data/$USER/conda-envs/leisaac/bin/python" scripts/evaluation/replay_joint_contract.py \
  --dataset outputs/mimic_vision224_500_20260913/raw/shard_000.hdf5 \
  --episodes 0 1 2 --mode recorded_target --gripper-effort-mode task \
  --output-dir outputs/evaluation/replay_target_new --headless --device cuda:0
```

실험 영상·로그·JSON은 `outputs/evaluation/act_diagnostics_20260918/`에 보존한다.
현재 이 경로가 이미 있으므로 재실행 시 새 출력 디렉터리를 사용한다.
생성 시연의 재생 성공은 학습된 ACT의 자율 성공률과 다른 지표다.

수치·출력 JSON 해시·97개 검수 대상 목록은
[제어 계약 진단 근거](evidence/act_action_contract_20260918.json)에 보존했다.

### 대조 실험 결과와 새 라벨 변환

첫 shard의 `demo_0/1/2`를 사전 선택했다. 각 시연의 초기 물리 상태 해시는
조건 간 동일했고, 원본 첫 관절 관측과 복원 상태의 오차는 0이었다.

| 명령 | 그리퍼 힘 설정 | 공통 T−1 구간 성공 |
| --- | --- | --- |
| 원본 IK 행동 | Mimic과 같은 task 방식 | 3/3 |
| 원본 목표각 + 관절 제한 | 기존 고정 방식 | 0/3 |
| 원본 목표각 + 관절 제한 | Mimic과 같은 task 방식 | 3/3 |
| 기존 next_observed 학습 라벨 | Mimic과 같은 task 방식 | 0/3 |

목표각+task 방식의 첫 성공 스텝은 211/185/605로 마지막 전이 이전이었다.
공통 구간 관절 RMSE는 목표각 방식에서 약 0.003107/0/0 rad,
기존 라벨 방식에서 약 0.046943/0.047568/0.030391 rad였다.
목표각 재생에서는 그리퍼 제한이 0.066667로 유지됐다. 이는 이 시뮬레이션의
물체 질량 기반 설정이며 실물 그리퍼의 캘리브레이션 값으로 복사하지 않는다.

이번 표는 **제어 계약의 재생 진단**이다. 새 ACT 학습 성공률, 500회 전체의
재생 성공률, 미관측 위치 일반화나 실물 성공률을 뜻하지 않는다.
기존 30,000-step ACT를 카메라·그리퍼 수정 후 seed 4001에서 평가했을 때도
`no_lift`였다. 평가 설정을 고쳤다는 사실과 기존 정책이 개선됐다는 주장을 구분한다.

변환기에 `--action-source recorded_target`을 추가했다.
상태·RGB의 원본 인덱스 0..T−2와, 제한을 적용한 목표각 인덱스 1..T−1을
쌍으로 저장한다. 마지막 목표각은 기록에 없으므로 마지막 프레임을 제외한다.
관절 순서·상하한·유한성, pre/post-step 정렬을 검사하고, limits 파일 해시와
실제 제한 적용량을 provenance에 남긴다. `stored`/`next_observed`는 기존 동작을 유지한다.

```bash
cd "/data/$USER/leisaac"
"$HOME/miniforge3/envs/lerobot/bin/python" scripts/imitation_learning/convert_hdf5_to_lerobot.py \
  --input outputs/mimic_vision224_500_20260913/raw/shard_000.hdf5 \
  --output-root outputs/evaluation/recorded_target_pilot_new \
  --repo-id local/so101_recorded_target_pilot \
  --image-size 224 --max-episodes 3 --action-source recorded_target \
  --joint-limits-file outputs/evaluation/act_diagnostics_20260918/render4_seed4001/evaluation.json
```

파일럿은 **3회·1,148프레임**이다. 모든 상태·행동을 원본과 대조했고,
LeRobot SDK에서 첫/마지막 front·wrist 224×224 영상을 디코딩했다.
기존 500회 데이터·기존 모델과 별도 출력이다. 전체 500회 재변환이나
새 라벨 ACT 학습이 완료된 상태는 아니다.
최종 파일럿 경로는 `outputs/evaluation/act_diagnostics_20260918/target_label_pilot_v2/`다.
제한 적용은 30프레임·31개 값, 최대 보정량은 약 0.330775 rad였다.
CPU 회귀 테스트 79개, Python 컴파일, diff 검사를 통과했고 별도 리뷰에서
차단 수정 사항은 남지 않았다. LSP·정적 타입 검사 도구는 없어 실행하지 않았다.

### 확대 전 남은 품질 조건

500회 원본 목표각을 전수 검사하면 마지막 프레임 제외 후 276,947프레임이다.
관절 제한을 적용한 뒤에도 **97회**는 기존 연속성 검사
`||action[t+1]−action[t]||₂ ≤ 1 rad`를 초과하며, 최대는 약 4.1642 rad였다.
나머지 403회는 이 검사 항목의 통과 후보일 뿐, 재생·학습 품질 전체가 검증된 것은 아니다.
이 기준은 기존 변환기의 품질 기준이지 실물 관절 속도의 안전 한계를 뜻하지 않는다.

다음 순서는 **명령 급변 시연 검수 → 보존된 raw에서 새 라벨로 별도 배치 변환
→ 원 시연 식별자를 유지한 train/valid 분리 → ACT 재학습 → 새 평가 조건에서
폐루프 검증**이다. 97회를 몰래 버리거나 허용치를 올려 통과시키지 않는다.
2026-09-18 당시 `run_mimic_image_batch.py`는 `next_observed` 경로였다.
2026-09-19 변경으로 새 라벨 배치 경로를 연결했다. 조명·배경 증강과 강화학습은
제어 계약이 맞는 정책의 기준 성능을 확보한 뒤 별도 비교한다.

## 2026-09-19 파이프라인 보완

### 변경한 동작

1. 배치 기본값은 `recorded_target`이다. `--joint-limits-file`이 필수이며,
   기존 라벨 사용에는 `--allow-legacy-action-source`를 명시해야 한다.
   단독 변환기의 기존 기본값 `stored`는 호환성을 위해 유지한다.
2. `--raw-dir --convert-only`로 원본을 복제·재생성하지 않고 재변환한다.
   변환기의 `--audit-only --audit-output`은 에피소드별 통과/실패 이유를 기록한다.
   배치 `--selection-manifest`는 raw 파일·검사 결과의 해시와 선택한 demo 이름을 검증한다.
   누락·급변 시연을 자동으로 제외하지 않으며, 선택 후에도 변환 검사는 다시 적용한다.
   audit의 schema·관절 제한·행동 급변 기준·통과/실패 개수는 현재 변환 설정과 일치해야 한다.
3. `action_contract.json`에 raw SHA256·원 demo·라벨 정렬·관절 제한·프레임 수를 남긴다.
   aggregate → split → training에서 출처와 학습/검증 매핑을 다시 검사한다.
   출처 계약이 없는 기존 split은 기본 거부한다. 예외 허용은 과거 실험 재현용이지
   기존 데이터가 새 목표각 라벨로 바뀌었다는 뜻이 아니다.
   계약 없는 예외 경로도 provenance에 `stored` 또는 `next_observed`가 명시돼야 한다.
   라벨 필드가 없거나 `recorded_target`이면 예외 옵션을 줘도 거부한다.
4. 성공은 박스 영역 안에서 그리퍼를 열고 **0.5초 연속 안정 상태**를 유지해야 한다.
   선속도 ≤ 0.03 m/s, 각속도 ≤ 0.5 rad/s다. 기존 영역 범위는 유지한다.
   생성·annotation 재생·상태기계·평가에서 매 제어 스텝 관찰하며,
   중복 호출·reset·관측 누락으로 유지 시간이 늘어나지 않게 했다.
5. 평가 비교는 카메라 reset 렌더 횟수, 그리퍼 힘 방식, 제어 주기,
   성공 기준, 관절 순서/제한까지 일치해야 한다. 이 값이 없는 과거 결과는
   새 결과와 자동 비교하지 않는다. 과거 JSON에 새 기준을 소급해서 적지 않는다.

안정 유지 값은 현재 태스크의 운영 기준이며 실물 캘리브레이션 값이 아니다.
기존 raw 500개의 `success=true`는 과거 판정 결과다. 이번 변경만으로
500개 모두 새 기준에 통과했다고 간주하지 않는다.

### 실제 통합 검증

작업 출력 루트: `outputs/evaluation/contract_hardening_20260918/`.
9월 18일 재생 검증을 시작한 경로이며, 변환·학습 smoke는 9월 19일 수행했다.

| 확인 항목 | 결과 |
| --- | --- |
| 새 성공 기준으로 원본 demo 0/1/2 목표각 재생 | 3/3 성공, 첫 성공 스텝 247/220/640 |
| 첫 raw shard 25개 사전 검사 | 22개 통과·3개 불통과, 자동 제외 없음 |
| 명시 선택 demo 0/1/2 변환·aggregate | 3개·1,148프레임 |
| train/valid 분리 | demo 0/2: 928프레임 / demo 1: 220프레임 |
| aggregate와 split의 모든 상태·행동 대조 | 총 2,296행, 원본 상태/제한 적용 목표각과 정확히 일치 |
| ACT 연결 검사 | 2 updates·batch 2, checkpoint 및 train-only 정규화 검사 통과 |
| 오프라인 검증 | 보류 220프레임 평가 완료, RMSE 약 0.7581 rad |
| Isaac 폐루프·영상 | seed 4998, 1,200스텝·1280×480·60fps 저장 및 검사 완료 |
| 해당 2-step 모델 집기 | 0/1, `no_lift` — 성능 검증용 학습이 아님 |
| 회귀 검증 | CPU 단위 테스트 114개 통과, Python 컴파일·diff 검사 통과 |

`act_smoke/progress.json`의 `complete`는 실행·산출물 검증 완료를 뜻한다.
집기 성공 여부는 `evaluation_summary.json`의 `successes`를 별도로 읽는다.
재생 3/3은 원본 명령 재현 결과이고 ACT 자율 성공률이 아니다.
재생 JSON은 `control_dt_s` 기록 추가 전에 생성돼 그 필드가 없다.
최종 ACT 평가 JSON에는 1/60초 및 새 성공 기준이 기록됐다.

실행에 사용한 핵심 인수는 다음과 같다. 기존 출력은 보존하며 재실행에는 새 경로가 필요하다.

```bash
cd "/data/$USER/leisaac"
"$HOME/miniforge3/envs/lerobot/bin/python" scripts/imitation_learning/run_mimic_image_batch.py \
  --input datasets/pick_cube_into_box_annotated_wrist_10_20260911.hdf5 \
  --output-dir outputs/evaluation/contract_hardening_20260918/batch_smoke \
  --raw-dir outputs/mimic_vision224_500_20260913/raw --convert-only \
  --selection-manifest outputs/evaluation/contract_hardening_20260918/smoke_selection.json \
  --joint-limits-file outputs/evaluation/act_diagnostics_20260918/render4_seed4001/evaluation.json \
  --total 3 --chunk 3
```

분리는 `--shard-size 3 --valid-per-shard 1`, 학습 runner는
`--steps 2 --batch-size 2 --eval-seeds 4998`로 실행했다.
검사·선택·계약 JSON, 영상 및 해시 근거는
[통합 검증 근거](evidence/act_pipeline_hardening_20260919.json)에 연결했다.

### 남은 일과 제약

- 97개 급변 시연 검수 → 확정한 선택 목록으로 별도 재변환 → 새 split → 본 학습 순서다.
  403개 후보도 안정 놓기 재생 등 품질 검증을 마쳐야 한다.
- 선택된 시연을 이어 붙인 뒤의 split은 고정 크기 묶음 기준이다. raw 생성 seed나
  원 리더 시연을 통째로 보류한 OOD 분리가 아니다.
- 이번 smoke에서 성공 여부로 모델·평가 seed를 고르지 않았다. seed 4998은 이제 사용 이력이 있다.
- 새 계약의 경로는 현재 Linux 절대경로이며 원본 raw·limits·변환 provenance가 있어야 검증된다.
  Windows로 폴더만 복사해 동일 계약 검증이 된다고 보장하지 않는다. 이식 시 경로 재지정과
  원본 해시 보존을 위한 별도 절차가 필요하다.
- 전체 500개 재변환, 본 ACT 재학습, 강화학습, 실물 검증은 이번 완료 범위가 아니다.
  기존 원본·모델·영상을 삭제하거나 덮어쓰지 않았다.

## 2026-09-19 급변 시연 전수 검수

`audit_mimic_quality.py`로 원본 20개 shard·500개 시연을 읽기 전용 검사했다.
결과는 `outputs/evaluation/mimic_quality_20260919/`에 있으며, 위 3회 smoke와 별개다.
목표각 변화 기준 `||u[t+1]−u[t]||₂ ≤ 1 rad`와 `stable_release_v2`는 바꾸지 않았다.

| 분류 | 개수 | 처리 |
| --- | --- | --- |
| 목표각 급변 기준 불통과 | 97 | 원본 보존, 이번 학습 후보에서 제외 |
| 급변 기준 통과하지만 마지막 안정 놓기 불충족 | 109 | 원본 보존, 이번 학습 후보에서 제외 |
| 두 조건 모두 통과 | 294 | 178,239프레임, 실제 재생 확인 전 후보 |

급변 97개에서 기준을 넘는 전이는 15,685개다. 최대 변화 관절 기준으로
팔꿈치 15,662개, 어깨 올림 23개였으며, 그리퍼 개폐만으로 걸린 시연은 없었다.
해당 전이의 목표각 변화 norm 중앙값은 약 2.0138 rad, 기록된 관절 상태 변화
중앙값은 약 0.2560 rad다. EEF 위치 명령 변화의 중앙값은 약 0.00675 m다.
관측 변화량은 새 목표각을 적용한 전이에 맞춰 계산했으며, 별도 리뷰에서 찾은
진단용 인덱스 한 스텝 오차를 수정한 뒤 다시 계산했다. 이 0.2560 rad 수치의 출처는
`outputs/evaluation/mimic_quality_verified_20260919/episode_diagnostics.json`이다.
최초 broad 출력은 수정 전 0.2562 rad 및 294개 선정의 보존 기록이다.
인덱스 수정은 선택된 시연 집합에는 영향이 없다.
pre/post-step 기록 정렬을 포함한 기존 변환 검사 결과는 403개 통과·97개 급변 불통과로
이전 전수 조사와 일치했다. 이는 시연별 목표각 급변을 확인한 결과이며,
IK의 어떤 설정이 급변을 일으켰는지까지 인과적으로 검증한 것은 아니다.

현재 Mimic 경로는 EEF pose → absolute DLS IK → 관절 목표각이다.
관련 코드는 `devices/action_process.py`, `enhance/envs/manager_based_rl_leisaac_mimic_env.py`,
Isaac Lab의 `DifferentialInverseKinematicsAction.apply_actions()`다.
subtask 시작점을 이전 실제 pose가 아닌 이전 목표 pose로 보간하는 설정도 있지만,
급변 발생 원인으로 확정하지 않았다. 원인을 분리하려면 같은 시점의 Jacobian·IK 오차와
subtask 경계를 추가로 대조해야 한다. 이번에는 명령을 평활화하거나 허용치를 올리지 않았다.

### 기록 상태 검사와 실제 재생의 구분

기록 상태 검사는 저장된 cube/box pose·cube 속도·그리퍼 각도로 공용 성공 함수를 적용한다.
제어 주기는 보존된 평가 JSON의 `control_dt_s`를 참조하며 출처 해시를 남긴다.
학습 가능한 목표각에 대응하는 **T−1개 post-step 상태만** 사용하므로,
라벨이 없는 마지막 전이로 0.5초 조건을 채우지 않는다.
학습 후보는 마지막에도 30스텝 연속 안정 상태여야 한다.

이 검사는 새 물리 재생의 성공 보장이 아니다. 이전 demo 0의 기록상 첫 성공은
249스텝, 목표각 재생에서는 247스텝이었다. 관절 제한 적용 등으로 재생 궤적이
조금 달라질 수 있다. demo 1/2는 각각 220/640스텝으로 일치했다.

후보 294개 목록은 `candidate_selection.json`에 명시했다. 별도 replay cohort는
각 raw shard의 숫자순 통과 후보 목록에서 중앙 항목 1개씩, 총 20개를 선정했다.
선정은 재생 결과를 보기 전에 고정했으며, `replay_cohort.json`의 해시로 추적한다.
`replay_joint_contract.py --selection-manifest`는 이 20개를 한 Isaac 프로세스에서
순서대로 재생하고 raw 파일 해시·시연 이름·초기 상태·최종 성공 여부와 영상을 남긴다.
cohort는 학습 정책 성능 평가나 미관측 테스트셋이 아니라 데이터 재현 점검용이다.

### 관절 제한 보정 없는 76개 후보와 자동 진행 조건

첫 cohort의 `shard_001/demo_14`는 기록 검사에서는 통과했지만 목표각 재생에서는
집지 못했다. 손목 관절에 27프레임·최대 약 0.06828 rad의 제한 보정이 적용됐다.
이 보정이 실패의 원인이라는 결론은 아직 내리지 않았다. 다만 원본과 실행 명령이
달라지는 시연은 보수적인 기준 모델에서 분리하기로 했다. 첫 cohort는 20회 재생을
완료했고 17회 최종 성공·3회 실패였다. 성공·실패 모두 영상 20개를 보존했다.
실패 시연을 성공 집계에서 빼지 않는다.

`--require-unclipped-targets`를 추가해 기준을 강화했다. 기본값은 꺼짐이며,
켜면 원본 `joint_pos_target[1:]`를 관절 제한으로 한 값이라도 바꾸는 시연을 제외한다.
검사 기준을 완화하거나 원본 명령을 평활화한 것이 아니다.

- 기존 두 조건 통과: 294개
- 그중 관절 제한 보정이 필요한 시연: 218개, 이번 기준 모델 후보에서 제외
- 보정 없이 통과: **76개·34,525프레임**, raw 20개 shard 모두 포함
- 최종 검사 경로: `outputs/evaluation/mimic_quality_verified_20260919/`
- 후보 변환 경로: `outputs/mimic_target_unclipped_76_20260919/`

앞선 검사 출력은 보존했다. 최종 `verified` 디렉터리의 plan·diagnostics·selection을
이후 단계의 기준으로 사용한다. 상태 기록 검사에 통과했다는 뜻이며, 경로 이름만으로
76개 전부 실제 재생 검증까지 끝났다고 해석하지 않는다.

자동 runner는 다음 순서로 진행한다.

1. 기존 broad cohort 재생이 완료될 때까지 기다린다. 이 결과의 성패는 지우지 않는다.
2. 새 76개 후보에서 raw shard별 중앙 항목 1개씩, 새 cohort 20개를 재생한다.
3. 명시한 raw SHA·시연·성공 기준·관절 제한이 일치하고 **20개 모두 최종 성공,
   목표각 보정 0회**이며, 별도 재생에서 확인된 후보 실패도 없을 때만 다음 단계로 넘어간다.
4. 76개 변환 완료와 원본→변환 계약을 검증한 뒤 train/valid를 분리한다.
5. 사전 고정한 ACT 30,000 updates·batch 8을 실행하고 마지막 checkpoint를 평가한다.
   평가 seed는 5000~5009로 고정한다. 결과를 보고 좋은 checkpoint만 고르지 않는다.

실패·누락·변조·여유 공간 8GiB 미만이면 학습 단계로 넘어가지 않는다.
현재 실행은 실물 로봇을 사용하지 않으며, 다른 시뮬레이터를 종료하지 않는다.
20개 cohort 검증은 76개 전수 재생이나 실물 성공 보장이 아니다.

**현재 추가 차단 사유:** broad replay의 `shard_009/demo_9`는 보정 0회인데도
집지 못했다. 관절 RMSE는 약 0.02473 rad였고 이 시연은 후보 76개에 포함된다.
따라서 관절 제한 보정만으로 실패를 설명할 수 없다. 이 시연이 새로운 20개 표본에
없더라도 `known_candidate_failures`로 남겨 학습을 막는다. 이미 확인된 실패를
좋은 표본 결과로 덮거나 목록에서 숨기지 않는다. 후보의 실제 재현성을 더 점검해야 한다.
재생 초기 velocity가 원인인지도 점검했지만, 이 시연과 대조 시연의 원본 초기 관절 속도는
모두 0이었다. 이 값만으로 실패 원인을 설명할 근거는 없다.

### 완료된 broad 재생과 후속 실행

첫 cohort 결과는 `outputs/evaluation/mimic_quality_20260919/replay_cohort_run/evaluation.json`이다.
실패 목록은 `shard_001/demo_14`(보정 27회·들기 없음), `shard_009/demo_9`(보정 0회·들기 없음),
`shard_016/demo_13`(보정 1회·최대 0.16946 m 들었지만 최종 안정 놓기 불충족)이다.
17/20은 **시연 재현 결과**이며 학습 정책 성공률이 아니다.
요약과 결과 파일 해시는 [검수 근거](evidence/mimic_quality_screen_20260919.json)에 보존한다.

후속 runner는 `scripts/imitation_learning/run_screened_mimic_pipeline.py`다.
기존 broad 프로세스의 정상 종료를 확인한 뒤 아래 명령으로 실행한다. 아직 실행 중이면
실제 Isaac Python PID를 `--wait-replay-pid`로 넘겨 결과 저장 후 앱 정리까지 기다린다.
같은 출력 경로로 재실행하지 않는다. 기존 결과를 덮어쓰지 않도록 거부한다.

```bash
cd "/data/$USER/leisaac"
"$HOME/miniforge3/envs/lerobot/bin/python" -u scripts/imitation_learning/run_screened_mimic_pipeline.py \
  --audit-root outputs/evaluation/mimic_quality_verified_20260919 \
  --wait-replay-dir outputs/evaluation/mimic_quality_20260919/replay_cohort_run \
  --wait-replay-manifest outputs/evaluation/mimic_quality_20260919/replay_cohort.json \
  --batch-root outputs/mimic_target_unclipped_76_20260919 \
  --output-dir outputs/evaluation/screened_mimic_pipeline_20260919 \
  --isaac-python "/data/$USER/conda-envs/leisaac/bin/python"
```

상태는 출력 디렉터리의 `progress.json`, 고정 계획은 `plan.json`, 새 영상과 재생 결과는
`strict_replay/`, 실행 로그는 `logs/strict_replay.log`에 남긴다.
현재 데이터에서는 이미 알려진 후보 실패 때문에 새 표본이 모두 성공해도 `rejected`로
끝나며 ACT를 시작하지 않는다. 새 표본 재생은 실패 범위를 확인하는 진단 작업이다.
원인을 해결한 뒤에는 새 증거와 새 후보 목록을 만들어 별도 실험으로 검증해야 한다.

`STOP` 파일은 단계 경계·외부 작업 대기에서 확인한다. 진행 중인 시연 한가운데를
강제로 멈추는 버튼은 아니다. runner에 SIGTERM을 보내면 해당 runner가 생성한
자식 프로세스 그룹만 정리한다. 별도 시뮬레이터나 외부 변환 프로세스에는 신호를 보내지 않는다.

## 2026-09-21 안정화 관찰 구간 분리

9월 19일 진단에서 strict `shard_019/demo_16`은 원래 시연 구간 끝에 안정 상태가
29프레임 이어졌다. 같은 목표를 1프레임 더 유지하자 기존 30프레임 기준을 통과했다.
반면 strict `shard_002/demo_17`은 단독 재생과 1초 추가 관찰에서도 큐브를 들지 못했다.
7개 trace·2,452프레임에서 독립 계산과 판정기 출력이 일치했다.
근거는 `outputs/evaluation/verdict_diagnostic_20260919/summary.json`이다.
이는 판정식을 느슨하게 바꿀 근거가 아니라, 실제 실패와 관찰 시간 부족을 구분할 근거다.

`replay_joint_contract.py --settling-seconds 1`은 마지막 제한 적용 목표각을
1초 동안 유지하며 같은 성공 함수를 호출한다. 기본값 0은 기존 실행을 유지한다.
허용 범위는 0~5초이며 `recorded_target` 모드에서만 관찰 구간을 켤 수 있다.
성공 기준·원본 HDF5·학습 라벨은 바꾸지 않는다.

- 기존 `steps`, `success`, `final_success`, RMSE와 최종 좌표는 원래 시연 구간 결과다.
- `source_horizon_release`에는 시연 끝의 위치·속도·그리퍼 각도·연속 안정 프레임을 남긴다.
- `settling`에는 별도 관찰 구간의 시간·첫 성공 시점·최종 성공·프레임별 상태를 남긴다.
- 원래 성공했거나 관찰 중 성공했더라도 관찰 종료 시 실패하면
  `unstable_during_observation`으로 분류한다. 중간에 불안정해졌다가 다시 안정되면
  최종 성공으로 기록하되, 중간 변화는 trace에 보존한다.
- 관찰 후 처음 안정되면 `settled_during_observation`, 계속 미충족이면
  `unresolved_after_observation`이다. 미충족을 곧바로 불량 원본 데이터라고 부르지 않는다.
- 원래 영상과 `_settling.mp4`를 나눠 저장한다. 관찰 구간은 학습 데이터에 덧붙이지 않는다.

연속 재생에서는 관찰 구간 이후 다음 시연을 reset하므로 이전 실행과 물리 컨텍스트가
같다고 보장할 수 없다. 같은 표본·순서로 다시 실행하더라도 과거 18/20을 소급 수정하지
않으며, 새 프로토콜의 시연 구간 결과와 추가 관찰 결과를 각각 보고한다.
학습 runner는 추가 관찰을 사용한 보고서를 **진단 전용**으로 거부한다.
실제 파지 실패의 해결 및 후보별 재검증 전까지 본학습은 보류한다.

재검증 명령(기존 strict 표본 20개·순서 유지):

```bash
cd "/data/$USER/leisaac"
"/data/$USER/conda-envs/leisaac/bin/python" -u scripts/evaluation/replay_joint_contract.py \
  --selection-manifest outputs/evaluation/mimic_quality_verified_20260919/replay_cohort.json \
  --mode recorded_target --gripper-effort-mode task --settling-seconds 1 \
  --video-count 20 --output-dir outputs/evaluation/settling_cohort_20260921 \
  --headless --device cuda:0
```

결과는 해당 출력 경로의 `evaluation.json`에 시연마다 갱신된다. `results`가 20개이고
프로세스가 정상 종료된 것을 확인하기 전에는 전수 재검증 완료라고 보고하지 않는다.

### 고정 표본 20개 재검증 완료 결과

구현 커밋 `e54194e`로 20개를 원래 manifest 순서대로 실행했고 정상 종료(exit 0)했다.
이는 시연 재생 검사이며 학습된 정책이나 실물 로봇의 성공률이 아니다.

- 새 실행의 원래 시연 구간: 18/20 성공.
- 각 시연 뒤 마지막 목표각을 1초 유지한 관찰 구간 종료: 19/20 성공.
- `shard_019/demo_16`: 시연 끝의 연속 안정 상태 29프레임. 추가 1프레임 뒤 성공,
  관찰 구간 종료까지 성공 유지. 원래 시연 구간의 실패 기록은 그대로 보존.
- `shard_002/demo_17`: 큐브 들기 0 m. 추가 관찰 60프레임 뒤에도 실패.
- 관찰 trace 1,200프레임의 독립 카운트·성공값 대조 불일치 0건.
- 영상 40개(원래 시연 20개 + 관찰 20개), 1280×480, 각 프레임 수 대조 통과.
- 회귀 테스트 145개, compileall, diff 검증 통과. 독립 산출물 검증 통과.
  LSP/flake8은 환경에 설치되지 않아 실행하지 않았다.

학습 승인 검사는 추가 관찰 보고서를 진단 전용으로 거부하는 것을 실제 결과로 확인했다.
원본 500개와 후보 76개 변환 결과는 유지하며, 본학습·강화학습·실물 실행은 하지 않았다.
다음 단계는 미해결 파지 실패의 원인 비교와 후보별 재현성 검사다.
이 표본 결과만으로 후보 76개가 모두 검증됐다고 간주하지 않는다.

근거: [안정화 관찰 검증 기록](evidence/mimic_settling_20260921.json).
