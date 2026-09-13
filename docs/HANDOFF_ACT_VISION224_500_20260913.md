# 500회 Mimic 데이터 → 224 ACT 학습·평가 인계

## 범위

증강 데이터 500회 생성은 끝났다. 이번 작업은 **ACT 모방학습과 시뮬레이션 평가**다.
강화학습이나 실물 팔로워 실행은 포함하지 않는다. 큐브·박스·로봇·카메라 위치와
기존 놓기/그리퍼 해제 성공 판정은 유지한다.

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
