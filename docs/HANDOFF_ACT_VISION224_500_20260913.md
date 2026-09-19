# 500회 Mimic 데이터 → 224 ACT 학습·평가 인계

> **2026-09-19 갱신:** 현재 배치 기본 라벨은 `recorded_target`이며, 성공 판정은
> `stable_release_v2`다. 이전 500회 데이터·30,000-step 모델은 보존된 과거 실험이다.
> 최신 변경과 3회 통합 검증은 아래 [파이프라인 보완](#2026-09-19-파이프라인-보완)을 따른다.

## 범위

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

## 누출 없는 데이터 분리

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

## 사전 고정한 학습·평가 계획

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
