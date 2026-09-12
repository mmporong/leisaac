# ACT 복구 데이터 비교 완료·다음 작업 기록 — 2026-09-12

## 현재 상태

**추가 학습 2개와 평가 20/20회를 완료했다. 대조군 0/10, 복구 추가 모델 1/10이다.**
영상 20개도 프레임 수·크기·fps를 검증했다. 남아 있는 학습·평가 프로세스는 없으며,
이번 비교를 이어 돌리거나 같은 모델을 다시 학습할 필요는 없다.

15회 시점의 정리 요청으로 한 번 멈췄다가, 후속 요청에 따라 남은 5회를 완료했다.
당시 진행 중이던 `recovery84 / seed 4007` 결과는 보존했다. 첫 반복 셸의 종료 코드 143은
사용자 요청에 따른 의도적인 중단이고, 마지막 5회 실행 셸은 정상 종료했다.
실물 로봇은 조작하지 않았다.

- 저장소: `mmporong/leisaac`, 원격 이름 `mmporong`, 브랜치 `main`
- `origin`은 `LightwheelAI/leisaac` 상류이므로 개인 작업 push 대상으로 사용하지 않는다.
- 실행 환경: Linux, Isaac Sim 5.1 / Isaac Lab 환경과 LeRobot 환경을 분리 사용
- 저장된 모델·데이터·영상은 로컬 `outputs/`, `datasets/`에 있으며 Git 추적 대상이 아니다.
- 코드·문서·작은 검증 JSON만 리포에 반영한다. 다른 PC에서는 Git clone만으로 모델/데이터가 복원되지 않는다.

## 무엇을 비교하는가

원본 리더 시연 10회 → MimicGen 생성 100회 → train 80 / valid 20으로 분리한 데이터가 기반이다.
별도의 ACT 실패 rollout 3100~3103에서 공식 MimicGen 궤적으로 복구 성공 4회를 확보했다.
손작성 오라클의 기존 성공 1회는 이번 4회에 섞지 않았다.

| 조건 | 대조군 `baseline80` | 복구 추가 `recovery84` |
|---|---|---|
| 데이터 | 기존 train 80회 | 같은 train 80 + 복구 4회 |
| 프레임 | 45,920 | 48,159 |
| 시작 | 동일 ACT 30,000스텝 모델 | 동일 |
| 추가 학습 | 3,000스텝, batch 8, seed 43 | 동일 |
| optimizer | 새 optimizer, LR 1e-5 / backbone 1e-6 | 동일 |
| 정규화 | 기존 체크포인트 값 유지 | 동일 |

정규화는 시작 모델의 원래 100회 통계다. 양쪽의 새 데이터 통계로 서로 다르게 바꾸지 않았다.
실제 모델 텐서는 양쪽 각각 153/234개 변경됐고, ACT 활성 feature mean/std의 shape·값은
원본과 같다. 학습된 두 모델의 SHA-256도 서로 다르다.

- 상세 설정·모델 해시: [학습 검증 JSON](evidence/act_recovery_comparison_round1_training.json)
- 전체 과정: [SO101 MimicGen → Sim2Real 문서](SO101_MIMICGEN_SIM2REAL_KO.md)

이것은 **증강 데이터를 이용한 ACT 모방학습 비교**다. 강화학습이나 실물 전이 검증은 하지 않았다.
시뮬레이션 큐브는 시각 3cm / 충돌 3.0154cm다. 사용자가 원하는 실물 4cm 큐브 실험과 구분한다.

## 최종 결과

seed 4000~4009, 모델별 10회, 총 **20회 완료**다. 모든 seed에서 두 모델을 비교했다.

| seed | baseline80 | recovery84 |
|---|---|---|
| 4000 | 상승 기준 미달 | 상승 기준 미달 |
| 4001 | 상승 후 마지막에는 낮음 | 성공, 753스텝 |
| 4002 | 상승 기준 미달 | 상승 기준 미달 |
| 4003 | 상승 기준 미달 | 상승 기준 미달 |
| 4004 | 들고 있지만 미완료 | 상승 기준 미달 |
| 4005 | 상승 기준 미달 | 상승 기준 미달 |
| 4006 | 상승 기준 미달 | 상승 기준 미달 |
| 4007 | 상승 기준 미달 | 상승 기준 미달 |
| 4008 | 상승 기준 미달 | 상승 기준 미달 |
| 4009 | 상승 기준 미달 | 상승 기준 미달 |

**대조군 0/10 대 복구 추가 1/10, 관측 차이 +10%p**다. 복구 모델만 성공한 조건은 4001이다.
학습 seed 하나의 소규모 비교이므로 통계적 우월성이나 반복 재현성은 입증하지 못했다.
초기 물리 상태 전체 SHA-256은 10쌍 모두 일치했다. 초기 RGB는 전면 0/10, 손목 1/10만
해시가 같았으므로 동일 seed라도 정책에 들어간 픽셀까지 완전히 동일한 비교는 아니다.

- [10쌍 최종 비교 JSON](evidence/act_recovery_comparison_round1_final_10pairs.json)
- [7쌍 중간 기록](evidence/act_recovery_comparison_round1_partial_7pairs.json)은 이력 보존용이며 최종 결과 대신 쓰지 않는다.
- 성공 판정: 큐브 중심이 지정된 박스 내부 범위에 있고 그리퍼가 열린 해당 스텝에서 종료
- 성공 후 정착·유지 시간은 평가하지 않음
- 상승 분류 기준: 큐브 초기 높이보다 0.02m 이상 높아졌는지 여부
- 대조군: 상승 기준 미달 8회 / 상승 후 낮음 1회 / 들고 있지만 미완료 1회.
- 복구 추가: 성공 1회 / 상승 기준 미달 9회. 정확한 원인은 이 분류만으로 단정하지 않는다.

## 보존된 파일

저장소 루트 기준 경로다.

- 원본 시작 모델: `outputs/lerobot/so101_act_next_state_mimic100_seed43_chunk30_30000/checkpoints/030000/pretrained_model`
- 추가 학습 모델: `outputs/lerobot/act_recovery_comparison_round1/{baseline80,recovery84}/checkpoints/003000/pretrained_model`
- 학습 계획·로그: `outputs/lerobot/act_recovery_comparison_round1/plan.json`, `logs/`
- 결과·영상: `outputs/evaluation/act_recovery_comparison_round1/{baseline80,recovery84}/seed_400*/`
- 평가 로그: `outputs/evaluation/act_recovery_comparison_round1/logs/`
- 성공 영상: `outputs/evaluation/act_recovery_comparison_round1/recovery84/seed_4001/rollout_001_success.mp4`
- 대조 실패 영상: `outputs/evaluation/act_recovery_comparison_round1/baseline80/seed_4001/rollout_001_failure.mp4`
- 들고 있지만 미완료: `outputs/evaluation/act_recovery_comparison_round1/baseline80/seed_4004/rollout_001_failure.mp4`

위 성공 영상은 이번에 추가 학습한 **ACT 정책**의 결과다. 앞서 저장한 MimicGen 복구 생성
영상과 혼동하지 않는다. 성공·실패 영상 모두 전면과 손목 카메라를 좌우로 붙여 저장했다.

## 나중에 이어갈 작업

이번 비교는 끝났지만 안정적인 자율 파지는 아직 해결하지 못했다. **단순 500회 확대나 RL 전환보다
실패 원인 진단이 우선**이다. 아래는 후속 작업 후보이며 이번에 실행하지 않았다.

1. 성공 4001과 파지 실패 영상을 비교하고, 접근 위치·그리퍼 닫힘 시점·관절별 명령 clipping을
   계측한다. 현재 기록은 clipping 스텝 수만 있어 어느 관절이 원인인지 단정할 수 없다.
2. 후속 학습용 seed를 따로 정해 파지 직전·접촉 중 복구 조건을 진단한다. 4000~4009 평가 seed를
   학습에 넣고 다시 미학습 성능이라고 보고하지 않는다.
3. 성공 1회의 반복 재현성과 방출 후 유지 판정을 검증한다. 즉시 완료 판정과 안정적 배치를 구분한다.
4. 실물 목표 4cm 큐브는 별도 씬·출력 경로에서 검증한다. 기존 3cm 데이터의 출처나 자산을
   덮어쓰지 않으며, 실제 팔로워 실행은 별도 실행 요청 범위에서만 수행한다.

재개 시 이 문서와 최종 JSON을 먼저 읽고 기존 학습·20회 평가를 반복하지 않는다.
아래 명령은 **기존 결과를 다시 집계하기만** 한다. 시뮬레이터나 로봇을 실행하지 않는다.
비교 도구는 CLI로 전달된 seed만 검사하므로 10개를 빠짐없이 지정해야 한다.
누락·조건 불일치는 오류로 보고되고 기존 출력은 덮어쓰지 않는다.

```bash
cd "/data/$(id -un)/leisaac"
"$HOME/miniforge3/envs/lerobot/bin/python" scripts/evaluation/compare_act_rollouts.py \
  --baseline-dir outputs/evaluation/act_recovery_comparison_round1/baseline80 \
  --recovery-dir outputs/evaluation/act_recovery_comparison_round1/recovery84 \
  --seeds 4000 4001 4002 4003 4004 4005 4006 4007 4008 4009 \
  --output outputs/evaluation/act_recovery_comparison_round1/comparison_rechecked_10pairs.json
```

## 검증 및 해석의 경계

- 관련 CPU 회귀 테스트 30개 통과. 신규 비교·정규화·상승 분류·물리 해시 테스트 포함.
- 별도 리뷰에서 코드 차단 이슈 없음. 문서의 comparator 범위와 repo_id 안내 오류는 수정했다.
- AST·공백 검사 통과. 정식 lint/typecheck 도구는 현재 두 Python 환경에 없어 실행하지 못했다.
- 완료 증거는 20개 evaluation JSON, 20개 영상, 학습 체크포인트 2개, 검증 JSON 3개다.
- 실험 완료와 정책 성능 목표 달성은 다르다. 현재 결과를 안정적인 자율 파지·실물 전이 완료로 표현하지 않는다.
