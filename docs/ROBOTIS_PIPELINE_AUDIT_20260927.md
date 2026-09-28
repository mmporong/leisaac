# ROBOTIS OMY Mimic 파이프라인과 SO101 비교

조회일: 2026-09-27. 공개 코드로 확인한 절차와 로컬 실행 근거를 구분한다.

후속 갱신(2026-09-28): 새로운 launch의 teacher_arm 양성 대조가 실패했다.
성공·실패 명령의 고정 재생6회는 각각의 결과·큐브·관절 상태를 재현했다.
아래 7절은 당시 성공 근거로 보존하고, 안정적인 회복 보장이나 팔만의 문제로 확대하지 않는다.
현재 결과·다음 분기는 [접촉 재현성 진단](ARM_HANDOFF_CONTACT_DIAGNOSIS_20260928.md)을 따른다.

## 1. 확인 범위와 버전

- `ROBOTIS-GIT/cyclo_lab`: `f4c0470a5e0af54a18327cf96967e8716d64dbc0`.
- 위 저장소의 Isaac Lab: `3c6e67bb5c7ada942a6d1884ab69338f57596f77`.
- `ROBOTIS-GIT/physical_ai_tools`: `27192daf75fa5f421a3bd1825a50ee63d649dccd`.
- 위 저장소의 LeRobot: `989f3d05ba47f872d75c587e76838e9cc574857a`.

README뿐 아니라 recorder, Mimic 종료 조건, 관절 변환, LeRobot 변환,
학습 manager, 추론 timer와 ACT action queue를 확인했다.
공식 OMY 예제를 SO101에 그대로 실행할 수 있다는 뜻은 아니다.

## 2. 공식 절차의 의미

공식 OMY 절차는 `수동 성공 처리한 시연 10개 → EEF/IK 변환 → 서브태스크 주석 → Mimic 성공 생성 목표 500개 → 관절 목표값 변환 → LeRobot 데이터 → 학습·추론`이다.
원본 기록은 leader의 절대 관절 위치 명령을 사용한다. 사용자가 `N`으로 성공을 표시하고 저장하며 `R`로 기록 버퍼를 초기화한다.
10개만으로 정책이 반드시 성공한다는 보장은 없다.

근거: [공식 절차](https://github.com/ROBOTIS-GIT/cyclo_lab/blob/f4c0470a5e0af54a18327cf96967e8716d64dbc0/README.md#sim2real),
[성공 처리·초기화](https://github.com/ROBOTIS-GIT/cyclo_lab/blob/f4c0470a5e0af54a18327cf96967e8716d64dbc0/scripts/sim2real/imitation_learning/recorder/record_demos.py),
[절대 관절 action](https://github.com/ROBOTIS-GIT/cyclo_lab/blob/f4c0470a5e0af54a18327cf96967e8716d64dbc0/source/cyclo_lab/cyclo_lab/manager_based/manipulation/pick_place/config/omy/joint_pos_env_cfg.py).

`generation_guarantee=True`인 OMY 설정에서 `generation_num_trials=500`은 성공 수를 기준으로 끝내는 목표다.
실패는 별도 HDF5에 저장할 수 있고 두 downstream 변환기가 `success=False`를 제외한다.
코드에 `max_num_failures=25`가 있어도 고정 Isaac Lab 실행 경로에서 소비되는 근거는 확인되지 않았다.
따라서 실패 25회 후 자동 중단한다고 안내하지 않는다.

근거: [OMY 생성 설정](https://github.com/ROBOTIS-GIT/cyclo_lab/blob/f4c0470a5e0af54a18327cf96967e8716d64dbc0/source/cyclo_lab/cyclo_lab/manager_based/manipulation/pick_place/config/omy/pick_place_mimic_env_cfg.py),
[성공 수 종료·실패 분리](https://github.com/isaac-sim/IsaacLab/blob/3c6e67bb5c7ada942a6d1884ab69338f57596f77/source/isaaclab_mimic/isaaclab_mimic/datagen/generation.py),
[관절 변환·실패 필터](https://github.com/ROBOTIS-GIT/cyclo_lab/blob/f4c0470a5e0af54a18327cf96967e8716d64dbc0/scripts/sim2real/imitation_learning/mimic/action_data_converter.py).

Mimic 생성 뒤의 학습은 모방학습이다. README의 병렬 RL 명령은 별도 태스크이며,
시연 10개·Mimic 500개의 필수 후속 단계로 RL이 자동 연결되는 구현은 아니다.

## 3. 관측·행동 정렬을 복사하면 안 되는 이유

공식 joint 변환은 `obs/joint_pos_target` 배열을 `actions`로 대입한다.
LeRobot 변환기는 같은 인덱스의 action, joint state, RGB를 저장한다.
`frame_skip`은 초반 몇 프레임을 제외하는 옵션이며 시간 간격을 줄이는 downsample이 아니다.

그런데 고정 Isaac Lab의 기록 순서는 다음과 같다.

```text
새 action 처리 → pre-step action·cached observation 기록
             → joint target 적용 → 물리 step → 새 observation 계산
```

따라서 pre-step 관측의 `joint_pos_target`은 새 action 적용 전의 목표값이다.
동일 배열 인덱스라는 사실과, 현재 상태에서 다음 전이에 사용할 명령이라는 의미는 구분해야 한다.
이는 공개 코드의 시간 관계를 설명한 것이며 ROBOTIS의 오류나 성능 원인이라고 단정하지 않는다.

근거: [동일 인덱스 변환](https://github.com/ROBOTIS-GIT/cyclo_lab/blob/f4c0470a5e0af54a18327cf96967e8716d64dbc0/scripts/sim2real/imitation_learning/data_converter/isaaclab2lerobot.py),
[pre-step recorder](https://github.com/isaac-sim/IsaacLab/blob/3c6e67bb5c7ada942a6d1884ab69338f57596f77/source/isaaclab/isaaclab/envs/mdp/recorders/recorders.py),
[step 순서](https://github.com/isaac-sim/IsaacLab/blob/3c6e67bb5c7ada942a6d1884ab69338f57596f77/source/isaaclab/isaaclab/envs/manager_based_rl_env.py),
[joint target 적용](https://github.com/isaac-sim/IsaacLab/blob/3c6e67bb5c7ada942a6d1884ab69338f57596f77/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py).

SO101은 전이 정렬을 raw pre/post 상태와 재생으로 검사했고
`action[t] = joint_pos_target[t+1]`를 채택했다.
이번 teacher_all 재생도 이 규약으로 성공했다. 공식 변환기에 맞춘다는 이유로 이 라벨을 변경하지 않는다.

## 4. 카메라·ACT 설정·공개 성능 근거

OMY의 Cyclo converter는 wrist/top RGB 두 개, 각 848×480과 7D action/state를 요구한다.
PAT의 OMY 실물 기본 설정에는 wrist 카메라만 활성화돼 있어,
두 카메라 모델을 그 설정으로 실행하려면 저장 모델의 feature와 실물 관측을 맞춰야 한다.
실제 공개 checkpoint를 확보한 것은 아니므로 카메라 불일치로 실패했다고 실측 주장하지 않는다.

근거: [Cyclo feature](https://github.com/ROBOTIS-GIT/cyclo_lab/blob/f4c0470a5e0af54a18327cf96967e8716d64dbc0/scripts/sim2real/imitation_learning/data_converter/isaaclab2lerobot.py),
[PAT OMY 카메라 설정](https://github.com/ROBOTIS-GIT/physical_ai_tools/blob/27192daf75fa5f421a3bd1825a50ee63d649dccd/physical_ai_server/config/omy_f3m_config.yaml).

PAT 새 학습 manager는 policy 종류와 일반 학습 옵션을 전달한다.
별도 resume 설정이 없는 ACT의 chunk/action-step 기본값은 고정 LeRobot에서 100/100이다.
이 값이 ROBOTIS 영상에 쓰인 실제 모델 설정이라는 근거는 없다.
추론은 task fps의 timer마다 `select_action`을 호출한다. ACT는 action queue를 소비하는 동안 새 관측으로 다시 예측하지 않는다.
데이터 FPS·제어 주기·queue 길이가 함께 맞아야 한다.

근거: [PAT 학습 옵션](https://github.com/ROBOTIS-GIT/physical_ai_tools/blob/27192daf75fa5f421a3bd1825a50ee63d649dccd/physical_ai_server/physical_ai_server/training/training_manager.py),
[ACT 기본값](https://github.com/huggingface/lerobot/blob/989f3d05ba47f872d75c587e76838e9cc574857a/src/lerobot/policies/act/configuration_act.py),
[ACT queue](https://github.com/huggingface/lerobot/blob/989f3d05ba47f872d75c587e76838e9cc574857a/src/lerobot/policies/act/modeling_act.py),
[추론](https://github.com/ROBOTIS-GIT/physical_ai_tools/blob/27192daf75fa5f421a3bd1825a50ee63d649dccd/physical_ai_server/physical_ai_server/inference/inference_manager.py),
[timer](https://github.com/ROBOTIS-GIT/physical_ai_tools/blob/27192daf75fa5f421a3bd1825a50ee63d649dccd/physical_ai_server/physical_ai_server/physical_ai_server.py).

확인한 두 저장소의 README·코드·changelog에는 OMY의 실제 500개 데이터, 영상 모델의 config/checkpoint,
시험 횟수와 실물 성공률이 확인되지 않았다. 공개 영상은 정량 비교 기준으로 쓰지 않는다.
PAT 새 학습 기본 경로는 평가 환경을 전달하지 않아 loss/checkpoint만으로 sim 성공률을 알 수 없다.

근거: [환경이 있어야 rollout 평가](https://github.com/ROBOTIS-GIT/physical_ai_tools/blob/27192daf75fa5f421a3bd1825a50ee63d649dccd/physical_ai_server/physical_ai_server/training/trainers/lerobot/lerobot_trainer.py).

## 5. 현재 SO101과의 차이

| 항목 | ROBOTIS 공개 OMY 절차·코드 | 현재 SO101 실행 근거 |
| --- | --- | --- |
| 원본 | 수동 성공 표시한 10개 | 리더 시연 10개 |
| 증강 | 성공 생성 목표 500개 | 1,308시도에서 500 성공 label, 808 실패 |
| 학습 데이터 | 실패 필터, 실제 공개 데이터 미확보 | 연속성·안정 놓기·관절 제한 선별 후보 76개 |
| 제어 차원 | 6 arm + 1 gripper | 5 arm + 1 gripper |
| 이미지 | converter wrist/top 848×480 | front/wrist 640×480 렌더→224×224 |
| 학습·평가 | 영상 모델의 실제 설정·성공률 미확인 | train61/valid15, ACT10k, 현재 reset-fixed 모델 자율 평가0/3 |
| action queue | 기본100, 실제 영상 설정 미확인 | 30명령/60Hz = 0.5초 |
| 물체 | OMY 병 pick-place | 시뮬 큐브 시각3cm/충돌약3.0154cm |

76개 전체가 재생 성공을 검증받은 데이터는 아니다. 학습·검증 분할도 원본 시연 계보가 완전히 분리된 것은 아니다.
실물 목표 4cm 큐브는 별도 검증 대상이다. 로봇·그리퍼·물체·접근 자유도가 다르므로
OMY 영상의 성공을 SO101 성공 보장으로 해석하지 않는다.

현재 모델은 `outputs/act_reset_fixed_76_20260921/model/checkpoints/010000/pretrained_model`,
split은 `outputs/act_reset_fixed_76_20260921_split`이다.
이전 `act_resume_10000_20260921`과 구분한다. 재렌더 raw의 초기 RGB 0/1 프레임을 수정한 뒤 10k 학습했으며,
자율 평가 0/3 결과는 그대로 남아 있다.

로컬 근거: `docs/evidence/mimic_vision224_500_complete_20260913.json`,
`outputs/mimic_reset_fixed_76_20260921/raw/rerender_provenance.json`,
`outputs/act_reset_fixed_76_20260921/{plan,checkpoint_validation,evaluation_summary}.json`,
`outputs/act_reset_fixed_76_20260921_split/split_provenance.json`,
[기존 태스크·크기 설명](SO101_MIMICGEN_SIM2REAL_KO.md).

## 6. 이번 진단의 보완

첫 실행 `outputs/action_intervention_20260927`에서는 다음을 관측했다.

- teacher_all: 642스텝 stable_release_v2 성공, 최대 lift 0.1154494m, effort 상수 0.06666666.
- policy: 675스텝 no_lift, 그리퍼 최소 명령 0.7542832rad.
- teacher_gripper: no_lift, effort 범위가 0.06666666~0.76800007로 변해 감사 실패.

태스크 `write_gripper_effort_limit_sim`은 가장 가까운 rigid object의 질량으로 effort를 결정한다.
혼합 명령이 만든 경로에서는 명령 채널 외에 effort도 변하므로 조건을 동등하게 비교할 수 없다.
중단된 manifest와 세 실행의 영상·trace·평가 파일을 보존한다. teacher_gripper는 감사 통과 결과로 집계하지 않는다.

수정 진단은 명시적 fixed effort 옵션으로 기존 성공 대조의 상수값을 매 step 적용한다.
모든 조건에 같은 값, 같은 초기 물리 상태, 같은 모델·관절 한계·성공 판정을 쓴다.
매 step 실제 effort와 summary min/max를 감사한다.
첫 fixed 실행 `outputs/action_intervention_fixed_effort_20260927`은 세 조건을 감사 통과한 뒤
다음 launch 직전 GPU 점유 감지로 중단됐다. 감지한 PID 4187916은 이어진 확인에서 이미 사라졌고 GPU도 비어 있었다.
Isaac 종료 뒤 드라이버의 일시적인 점유 잔류가 가능한 원인이지만 PID 소유를 확인하지 못해 확정하지 않는다.
기존 부분 결과를 보존하며, 실행 사이에 GPU 유휴 확인을 최대 10초 재시도하도록 보완한다.
재시도 뒤에도 점유가 있으면 중단하며 점유 프로세스를 종료하지 않는다.
최종 새 root는 `outputs/action_intervention_fixed_effort_v2_20260927`이다.
teacher_all 양성 대조가 성공한 seed만 나머지 조건을 진행한다.

이 fixed 진단의 결과는 기존 task-effort 평가와 분리한다. 성공 시연 명령을 넣은 조건은 자율 정책 성능이 아니다.
같은 초기 상태의 두 launch이므로 일반화 성공률을 평가하는 실험도 아니다.

## 7. 최종 8조건 결과와 다음 판단

`outputs/action_intervention_fixed_effort_v2_20260927`에서 네 조건 × 두 seed를 완료했다.
두 seed는 서로 다른 큐브 배치가 아니라 동일 물리 초기 장면을 별도 launch로 반복한 것이다.
모든 실제 step의 effort는 `0.06666666269302368`로 같고, 명령 교체·clipping 감사 오차는 모두 0이다.

| 조건 | seed 4101 | seed 4102 | 최대 lift: 4101 / 4102 |
| --- | --- | --- | --- |
| teacher_all: 시연 팔 + 시연 그리퍼 | 안정 놓기 성공, 642step | 안정 놓기 성공, 642step | 11.54 / 11.54cm |
| policy: ACT 팔 + ACT 그리퍼 | no_lift, 675step | no_lift, 675step | 0 / 0.83cm |
| teacher_gripper: ACT 팔 + 시연 그리퍼 | no_lift, 675step | no_lift, 675step | 0 / 0cm |
| teacher_arm: 시연 팔 + ACT 그리퍼 | 안정 놓기 성공, 639step | 안정 놓기 성공, 642step | 11.83 / 11.75cm |

`no_lift`는 최초 2cm lift에 도달하지 못한 결과다. 0.83cm 움직임을 집기 성공으로 바꾸지 않는다.
teacher_arm에서는 팔만 시연 명령이고, ACT는 실행 중의 상태·RGB로 그리퍼 명령을 계속 생성했다.
즉 이 장면에서는 **올바른 팔 궤적을 따라가면 현재 ACT의 그리퍼 명령으로 집기·안정 놓기가 가능했다.**
반대로 그리퍼 명령만 시연으로 바꿔도 ACT 팔 궤적에서 집기는 회복되지 않았다.

이는 팔 궤적과 그로 인해 얻는 관측의 영향이 크다는 지지 근거다.
집기 접근만이 단일 근본 원인이라고 확정하지 않는다. 전체 팔 시계열을 교체했기 때문에
접근·파지·운반 중 어느 구간이 결정적인지, 시각 특징·상태 분포·데이터 부족 중 무엇이 원인인지는 아직 분리되지 않았다.
launch 간 초기 RGB 차이도 남는다. teacher 명령을 넣은 성공을 ACT 자율 성공이나 일반화 성능으로 집계하지 않는다.

검증과 보존:

- 8개 실행의 stable_release_v2, 초기 물리 SHA, dt, joint limit, 명령·trace 전구간 감사 통과.
- 원본 raw·모델·실행 core 파일 SHA 전후 동일. 두 과거 부분 root도 보존하고 최종 결과에 섞지 않았다.
- 영상 8개: front/wrist 병합 1280×480, 60fps. 프레임 수가 실행 step과 같고 첫 프레임 디코딩 통과.
- 관련 회귀 테스트 205개 통과. 별도 verifier가 최종 8조건·해시·영상·감사를 승인했다.
- 다른 프로젝트 GPU 실행을 종료하지 않았다. 새 본학습·실물 실행은 없으며 진단의 자동 재개 작업도 종료됐다.

최종 매니페스트 SHA256:
`fda262faa08107fcd9c07e40bc982e8b100ac4f6ca2d6218ee886b6152b8465c`.
Git 보존 근거: [8조건 결과·영상 경로·해시·공식 버전](evidence/robotis_action_intervention_20260927.json).
성공 영상 예: `outputs/action_intervention_fixed_effort_v2_20260927/seed_4101/teacher_arm/rollout_001_success.mp4`.
자율 실패 대조: `outputs/action_intervention_fixed_effort_v2_20260927/seed_4101/policy/rollout_001_failure.mp4`.
대용량 영상은 로컬 outputs에 보존하고 Git에는 경로·해시·메타데이터를 반영한다.

### 7.1 중단·재개 기록

최종 v2 실행 중 `/home/lim/bimanual-robot`의 별도 컵 시뮬레이션 GPU 점유를 확인했다.
같은 core 코드·모델·raw를 유지하며 GPU가 비었을 때 남은 조건만 이어갔다.
새 `resume_action_intervention.py`는 저장된 실행이 사전 등록 순서의 연속 prefix인지,
완료 조건의 command·log·trace·원본·모델·코드 해시가 같은지 확인한다.
감사 완료 조건은 다시 실행하거나 덮어쓰지 않으며 부분 매니페스트와 재개 이력을 각각 보존한다.
3·4·5개 완료 시점의 스냅샷과 재개 3건이 최종 근거에 포함된다.
완료된 root는 재개를 거부한다. 사용 절차는 [진단 프로토콜](plans/remaining_causes_20260921/action_intervention_20260922.md)에 있다.

### 7.2 다음 단계: 전체 교체에서 구간별 교체로

다음은 **시연 팔 명령이 필요한 구간을 좁히는 진단**이다. 아직 구현·실행하지 않았다.

1. 접근·파지·운반 경계와 구간별 팔 교체 규약을 사전 등록한다. 현재 시연의 최초 2cm lift는 402step이고 ACT chunk는 30step이다.
2. 필요한 구간까지만 시연 팔을 사용한 뒤 ACT 팔로 복귀시키며 그리퍼는 계속 ACT로 실행한다. 같은 effort·상태·성공 기준·두 launch와 전체 명령 감사를 유지한다.
3. 어느 시점부터 ACT로 돌아가면 실패하는지와 궤적·관측·그리퍼 예측을 비교한다. 초기 RGB 변동과 실행 중 관측 차이도 함께 기록한다.
4. 구간이 좁혀지면 그 구간의 학습 입력·라벨·재생 품질을 검사하고, 소규모 학습 대조로 검증한다. 새 모델·데이터 수 증가의 효과는 같은 자율 평가로 비교한다.

현재 근거만으로 500개 확대, 그리퍼 loss 가중치 증가, ACT 폐기, Diffusion 본학습이 해결책이라고 결정하지 않는다.
Diffusion은 CPU 데이터 연결 검사까지만 통과했다. 본학습 전에 GPU 메모리와 정책별 추론·평가 계약을 확인해야 한다.
기존 plan_r5의 1.1.7·2.1·3.2·2.2 미결 항목은 이 진단으로 통과 처리하지 않는다.
