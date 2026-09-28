# 팔 시연 보조 종료 시점 진단 (2026-09-28 사전 등록)

## 완료 조건과 범위

기존 fixed-effort 진단에서 시연 팔 + ACT 그리퍼는 같은 초기 장면의 두 launch 모두 성공했다.
이번에는 팔 시연 명령을 언제까지 사용해야 하는지 좁힌다.
원본·모델·성공 기준·관절 제한·제어 주기는 유지하고 새 root에 모든 성공·실패·영상을 보존한다.
12조건의 전구간 명령·해시·영상·입력 PNG 감사를 통과하고 결과를 기록하면 이 진단은 완료다.
실물·새 본학습·데이터 증강은 이번 진단에 포함하지 않는다.

## 고정 조건

- 모델·raw·initial state·stable_release_v2·fixed effort는 [앞선 프로토콜](action_intervention_20260922.md)과 같다.
- horizon 675, raw target[t+1], 두 seed 4101/4102, ACT action queue 30, 60Hz, server CPU/seed0.
- 단계 번호는 완료 step 기준 1-based. cutoff N이면 step 1..N은 시연 팔, N+1..675는 ACT 팔이다.
- 그리퍼는 보조 조건에서 항상 ACT를 사용한다. 상태 teleport나 큐브 부착은 없다.
- ACT는 시연 팔 구간에서도 매 step 실제 관측으로 질의하며 기존 action queue를 소비한다.
- cutoff 240/300/420은 모두 queue 30의 경계다. 정책으로 전환하는 N+1에서 기존 규칙대로 새 chunk를 예측한다. 추가 queue reset은 하지 않는다.
- 이 숫자는 실험 설계 경계이며 모든 시연에 공통인 물리 이벤트 시점이라는 뜻은 아니다.
- 원본 raw 정렬은 유지한다. 정책으로 전환한 뒤 시연 target을 현재 상태로 재정렬하지 않는다.

## 실행 순서와 gate

각 seed에서 아래 순서로 실행한다.

| 조건 | 팔 | 그리퍼 |
| --- | --- | --- |
| teacher_all | 전체 시연 | 전체 시연 |
| teacher_arm | 전체 시연 | ACT |
| policy | 전체 ACT | ACT |
| teacher_arm_until_240 | step 240까지 시연, 이후 ACT | ACT |
| teacher_arm_until_300 | step 300까지 시연, 이후 ACT | ACT |
| teacher_arm_until_420 | step 420까지 시연, 이후 ACT | ACT |

teacher_all과 teacher_arm 두 양성 대조 모두 성공·명령 감사를 통과한 seed에서만 후속 조건을 실행한다.
대조 실패는 기록한 뒤 진단을 중단한다. 성공 조건만 골라서 이어가지 않는다.
GPU 점유 시 다른 프로세스를 종료하지 않으며 기존 출력과 성공 판정을 바꾸지 않는다.
부분 실행에서 재개할 경우 완료된 연속 prefix를 재감사하고 덮어쓰지 않는다.
코드·모델·raw 해시가 달라지면 같은 root의 재개를 거부한다.

## 검사와 수집

- 원래 ACT 명령·teacher target·교체 후 명령·clip 후 명령·실제 effort·상태·cube xyz를 매 step 기록한다.
- teacher 팔 구간의 5축은 해당 raw target[t+1], 나머지와 전환 이후는 ACT 명령과 오차 0이어야 한다.
- 매 step 적용 source를 기록하고 cutoff의 포함/제외 경계를 감사한다.
- 초기 물리 상태와 관절 한계·success criteria·dt·effort를 기존 reference와 대조한다.
- 양성 대조의 clip은 0이어야 한다. 혼합 조건의 clip은 기록하되 결과를 숨기지 않는다.
- 정확한 pre-action front/wrist 224 RGB·state를 step 211..451에서 30step마다 PNG로 저장한다.
  이는 queue 재예측 step 211,241,271,301,331,361,391,421,451에 해당한다.
- PNG가 trace.state_before와 같고 SHA·pixel SHA·shape가 맞는지 검사한다. 영상은 1280×480/60fps·실행 step 수를 검사한다.
- 원본·모델·진단 코드의 SHA를 실행 전후 확인한다. 기존 2026-09-27 artifact의 frozen SHA는 당시 커밋 `90eac48`의 역사 근거이며 새 코드를 과거 실행으로 소급하지 않는다.

## 결과 해석과 다음 분기

동일 장면 두 launch의 성공·lift·놓기·clip·전환 직후 명령 점프·구간별 팔 오차를 모두 보고한다.
실패와 불일치도 보존한다. 두 launch가 다르면 launch-sensitive 미결로 남긴다.
전환 이후의 성공은 시연 보조 진단이지 ACT 자율 성능이 아니다.
prefix 보조는 팔 궤적뿐 아니라 이후 관측·그리퍼 예측도 바꾼다. 특정 관절이나 RGB만의 원인이라고 단정하지 않는다.

- 240부터 성공: 해당 장면에서는 초기 접근 보조만으로 회복 가능한 근거.
- 240 실패, 300 성공: 필요한 보조 구간이 그 사이로 좁혀지는 근거. 단일 이벤트 원인 확정은 아니다.
- 300 실패, 420 성공: 파지·lift 전후까지의 보조가 필요한 근거.
- 420까지 실패하고 전체 시연 팔만 성공: lift 이후 제어·접촉 유지·운반 구간도 분리할 필요가 있다.
- 대조 실패 또는 반복 불일치: 학습 개선 판단보다 초기화·관측·실행 차이를 먼저 조사한다.

결과로 필요한 구간이 좁혀진 뒤 학습 입력·라벨·재생 품질과 소규모 학습 대조를 설계한다.
결과 없이 데이터 수 증가·ACT 폐기·Diffusion 본학습으로 넘어가지 않는다.
기존 plan_r5의 미결 gate는 그대로 보존한다.
