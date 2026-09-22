# 팔·그리퍼 명령 분리 진단 (2026-09-22 사전 등록)

## 목적과 기존 근거

사용자가 정책 명령과 물리 실행 문제를 구분하는 후속 진단을 승인했다.
`outputs/evaluation/control_diagnosis_20260921/shard009_recorded_target/evaluation.json`에서
동일 초기 물리 상태 `375aaad2…`의 시연 목표각 재생은 642스텝에 안정 놓기를 달성했다.
물리적으로 불가능한 장면이라는 가설보다 ACT 명령의 어느 부분이 실패에 기여하는지 검사한다.

이 진단은 plan_r5의 미결인 1.2/1.1.7을 통과했다고 간주하지 않는다.
4.1의 학습됨/암기 판정을 대신하지 않고, 2.2 실행 여부도 여기서 결정하지 않는다.
과거 reference·성공 기준·최소 표본 규칙은 보존한다. 초기 렌더 RGB 재현성은 여전히 별도 미결이다.

## 고정 조건

- 모델: `outputs/act_reset_fixed_76_20260921/model/checkpoints/010000/pretrained_model`, SHA `ebed7e54…`.
- 원본: `outputs/mimic_vision224_500_20260913/raw/shard_009.hdf5`, `demo_9`.
- 초기 상태는 원본 복원, velocity target 0, 물리 상태 해시 확인.
- 관측·추론: 640×480 렌더 → 224×224 RGB, reset render 4, CPU ACT, n_action_steps 30, server seed 0.
- 제어: task gripper effort, 기존 joint clipping, control dt 1/60 s.
- horizon **675**: source 676프레임의 `joint_pos_target[1:]`에 대응. 마지막 값 반복·추가 settling 없음.
- 성공: 기존 stable_release_v2 유지. 성공 시 evaluator의 기존 중단 동작 유지.
- 각 조건은 별도 Isaac 실행. GPU 점유 확인 후 하나씩 실행하며 실물은 사용하지 않는다.

## 2×2 조건

| 조건 | 팔 5축 | 그리퍼 1축 | 용도 |
| --- | --- | --- | --- |
| policy | ACT | ACT | 같은 675스텝 범위의 현재 기준선 |
| teacher_gripper | ACT | 시연 target[t+1] | 그리퍼 명령 교체의 효과 |
| teacher_arm | 시연 target[t+1] | ACT | 접근·운반 명령 교체의 효과 |
| teacher_all | 시연 target[t+1] | 시연 target[t+1] | 새 교체 경로의 양성 대조 |

네 조건 모두 실제 현재 관측으로 같은 ACT 서버를 계속 질의한다. 반환된 원래 정책 명령과 교체 후 명령을 따로 기록한다.
teacher 입력은 항상 현재 rollout t와 원본 t+1의 시간 인덱스로 맞춘다. 상태를 시연 위치로 강제 이동하지 않는다.
섞은 명령이 시연 때와 다른 상태에서 적용될 수 있으므로, 혼합 조건 실패만으로 해당 채널이 무관하다고 판단하지 않는다.

## 실행 순서와 반복

1. 테스트와 별도 코드 검토 후 각 seed(4101, 4102)에서 `teacher_all`을 먼저 실행한다.
2. 해당 seed의 대조가 stable_release_v2 성공에 도달하지 못하면 그 seed의 나머지 조건은 실행하지 않는다. lift 2cm 미달과 lift 후 안정 놓기 실패를 구분해 구현·초기화·제어 차이를 검사하고 채널 효과 판정에서 제외한다.
3. 양성 대조가 성공한 seed에서만 policy → teacher_gripper → teacher_arm 순서로 실행한다.
4. 같은 초기 물리 상태를 쓰므로 두 seed는 초기 배치 다양성이 아니라 launch/reset 변동 확인이다. 두 번의 결과가 불일치하면 회복이라고 판정하지 않고 launch-sensitive diagnostic으로 남긴다.
5. 성공한 조건만 골라 보고하지 않는다. 별도 launch의 초기 RGB 차이는 혼란변수이므로, 두 seed에서 일관돼도 채널 영향의 지지 근거이지 완전한 인과 분리가 아니다.

교체 검증 계약:

- 모든 실행의 매 스텝에 원래 `policy_action`, 교체 후 `requested_action`(preclip), `applied_action`(postclip)을 기록한다. 성공으로 조기 종료하면 실행한 prefix 전체를 검사한다.
- teacher로 지정한 채널의 preclip 값은 raw target[t+1]와 최대 차이 0, 나머지는 policy_action과 최대 차이 0이어야 한다. teacher_all의 clipping은 0이어야 한다.
- source length 676, 최대 horizon 675, raw/model SHA 전후 동일을 검사한다.
- teacher_all은 기존 recorded_target 근거와 초기 physical SHA, success criteria, dt, effort 범위, joint limits를 대조한다.
- 각 실행의 초기 front/wrist pixel SHA와 첫 정책 명령을 보존하고 조건 간 RGB·첫 정책 명령 차이를 보고한다.

## 보고와 해석 한계

- 조건별 2회 결과, 최대 큐브 높이, 최초 2cm lift 스텝, 최초 안정 놓기 스텝, 실제 그리퍼 최소 명령, clipping, trace·영상 경로.
- 팔을 교체한 조건에서만 회복되면 팔 궤적/그로 인해 얻는 관측의 영향이 크다는 근거다. 그리퍼 명령 자체만의 효과로 분리하지 않는다.
- 그리퍼를 교체한 조건에서만 회복되면 닫힘·열림 명령/시점의 영향이 크다는 근거다. 전체 그리퍼 시계열을 바꾸므로 닫힘 한 순간의 원인이라고 좁혀 말하지 않는다.
- 둘 다 교체해야만 회복되면 결합 오차 또는 시간 정렬 문제로 남긴다.
- 표본 2회로 일반화 성능·성공률 개선을 주장하지 않는다. 혼합/teacher_all은 시연의 미래 명령을 쓰므로 자율 정책 성능이 아니다.
- 학습·데이터 재생성·성공 판정 완화·원본 변경은 하지 않는다.

산출물은 `outputs/action_intervention_20260922/`의 조건·seed별 새 디렉터리와 Git의 근거 JSON·인계 갱신이다.

## 대안 모델 비교 준비 (사용자 추가 질문 반영)

ACT는 고정된 최종 선택이 아니다. 다음 모델 대조 후보는 Diffusion Policy다.
공식 구현과 설명: [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/),
[LeRobot 학습 예제](https://github.com/huggingface/lerobot/blob/main/examples/tutorial/diffusion/diffusion_training_example.py).

현재 LeRobot 환경에서 DiffusionConfig/Policy와 diffusers import를 확인했다.
수정 76개 LeRobot 데이터·기존 train/valid 분할·raw target[t+1] 라벨을 재사용할 수 있다.
다만 학습 진입·추론 서버·오프라인 평가·checkpoint 검증은 ACT에 묶여 있어 정책별 분기가 필요하다.
Diffusion의 기본 MIN_MAX 정규화와 관측/행동 시간창을 ACT 설정으로 잘못 검증하면 안 된다.
후보 연결 설정은 n_obs_steps=1, horizon=32, n_action_steps=30, drop_n_last_frames=2이며, 성능 최적값이라는 주장은 아니다.
현재 GPU 8GiB에서 두 카메라 224×224 입력의 batch1 학습/추론 메모리 스모크가 선행해야 한다.
이번 진단에서는 새 모델을 학습하지 않는다. ACT보다 더 나은지 아직 확인하지 않았으며, 같은 평가 조건에서 비교해야 한다.
