# SO101 Mimic → ACT 집기 실패 원인 조사 인계

이 문서를 다음 AI에게 전달하고 **12절의 작업부터 진행**하도록 요청하면 된다.
기준일은 2026-09-21이며, 학습 결과 기준 커밋은 `5f9697e`다.
이후 읽기 전용 조사에서 찾은 초기 영상 결함까지 이 문서에 반영했다.
**12절 1~6항은 같은 날 후속 조사로 수행했고 결과는 13절에 있다.** 다음 작업자는 13절부터 읽는다.

## 1. 목적과 현재 판단

SO101 단일 팔로 큐브를 집어 박스에 넣는 정책을 만든다.
리더 시연 10회 → Isaac Sim/Mimic 증강 → ACT 모방학습 → 시뮬레이션 평가 → 향후 실물 검증이 목표다.

현재 정책은 목표각 예측 오차가 줄어도 큐브를 들지 못한다.
**데이터 생성 시 초기 영상과 관절 상태가 불일치하는 결함을 확인했다.**
다만 이 결함을 수정하면 모든 집기 실패가 해결된다는 실험은 아직 없다.
추가 학습·데이터 재수집부터 시작하지 말고 관측과 제어 연결의 원인을 조사해야 한다.

## 2. 환경과 권한 범위

Linux에서 다음과 같이 저장소로 이동한다. 이 문서의 이후 파일 경로는 저장소 기준이다.

```bash
cd "/data/$USER/leisaac"
```

- 실제 조사 호스트: Linux, NVIDIA RTX 5050 Laptop GPU 8GB.
- 개인 원격 `mmporong`: `git@github.com:mmporong/leisaac.git`.
- `origin`은 upstream이므로 push하지 않는다.
- LeRobot Python: `$HOME/miniforge3/envs/lerobot/bin/python`.
- Isaac Python: `/data/$USER/conda-envs/leisaac/bin/python`.
- LeRobot 소스: `$HOME/lerobot/src/lerobot`.
- 다른 프로젝트가 GPU를 사용할 수 있다. 실행 전 확인하되 해당 프로세스를 종료하지 않는다.
- 실물 로봇 조작은 이 인계 작업 범위 밖이다.
- 원본 데이터·기존 모델·평가 영상을 보존한다. 실험 결과는 새 디렉터리에 저장한다.
- 학습 승인용 strict replay gate와 성공 판정을 편의상 완화하지 않는다.
- GitHub에는 문서·코드가 있고 대용량 `outputs/`는 로컬 산출물이다.
  다른 PC에서는 아래 파일의 존재를 먼저 확인한다. Git clone만으로 데이터와 모델이 복원되지 않는다.

## 3. 데이터 구성

원본 리더 시연 10회에서 Mimic 성공 데이터 500개를 생성했다.
연속성·안정 놓기·관절 제한 검사로 후보 76개를 선별했다.
**76개 전부의 실제 재생 성공이 검증된 것은 아니다.**

| 항목 | 설정 |
| --- | --- |
| 학습 | 61개, 26,527프레임 |
| 검증 | 15개, 7,998프레임 |
| 분리 | seed 43, 에피소드 중복 없음 |
| 분리 한계 | 같은 원본 10회에서 파생됨. 원본 시연 단위 독립 분리 아님 |
| 입력 | 관절 상태 6D + 앞/손목 RGB 224×224 |
| 출력 | 절대 관절 목표각 6D, rad |
| 행동 정렬 | `action[t] = obs/joint_pos_target[t+1]` |
| ACT | batch 8, chunk 30, `n_action_steps=30` |
| 평가 | seed 4100/4101/4102, 각 새 Isaac 프로세스, 최대 1,200스텝 |

raw와 합본 action 34,525프레임은 위 정렬로 전수 일치했다.
상태·행동 정규화는 학습 집합 통계를 사용하며 영상은 고정 ImageNet 통계를 사용한다.

## 4. 학습 결과

| 결과 | 2,000스텝 | 10,000스텝 |
| --- | --- | --- |
| 보류 500프레임 목표각 RMSE | 0.088081 rad | 0.050011 rad |
| 보류 15개 시작 프레임 RMSE | 0.293607 rad | 0.145189 rad |
| 실제 집기 성공 | 0/3 | 0/3 |
| 최대 큐브 상승 | 모두 0 m | 모두 0 m |
| 관절 제한 보정 | 모두 0스텝 | 4100/4101: 0, 4102: 290스텝 |

10,000스텝은 2,000스텝에서 가중치·옵티마이저·정규화·난수 상태·데이터 순서를
복원해 추가 8,000스텝 학습한 결과다. 새로운 옵티마이저로 시작한 fine-tuning이 아니다.
10,000스텝 seed 4102의 보정은 그리퍼에서 발생했고 최대 보정량은 0.0155400038 rad다.
다른 두 seed는 보정 없이 실패했으므로 클리핑만으로 모든 실패를 설명할 수 없다.

전후 같은 seed의 초기 물리 상태 해시는 일치하지만 렌더 영상은 비트 단위로 같지 않았다.
초기 PNG 픽셀 MAE(0~255)는 앞 카메라 0.069~0.131, 손목 0.017~0.033이었다.
이는 동일 설정의 비교이지 완전히 동일한 영상 입력을 통제한 실험은 아니다.

## 5. 기존 제어 방식 비교

| raw 시연 | 저장된 관절 목표각 재생 | 원래 Mimic 명령 재생 |
| --- | --- | --- |
| `shard_002/demo_17` | 집기 실패 | 성공 |
| `shard_009/demo_9` | 단독 실행에서는 성공 | 성공 |

`009/demo_9`는 과거 연속 평가에서는 실패했지만 새 프로세스 단독 재생에서는 성공했다.
실행 맥락의 영향은 아직 전부 규명하지 못했다.

원래 Mimic 행동은 **말단 자세 7D + 그리퍼 1D**이며 현재 상태에서 IK 목표각을 계산한다.
현재 ACT는 **6D 관절 목표각**을 학습한다. 둘을 같은 제어 방식으로 취급하면 안 된다.
관절 목표각 재생에도 PD 피드백이 있으므로 완전한 개방루프라고 부르지 않는다.

## 6. 최신 확정 사실: 초기 카메라 영상과 관절 상태 불일치

읽기 전용 조사와 별도 검토자가 아래 수치를 독립 재현했다.

1. 후보 76개 모두 첫 관절 상태는 `[0,0,0,0,0,0]`이다.
2. 앞·손목 카메라 각각 `frame 0 == frame 1`이 76/76개다.
3. 앞·손목 카메라 각각 `frame 1 == frame 2`는 0/76개다.
4. `shard_009/demo_9`의 시각적 비교:
   - frame 0·1: 팔이 박스 위로 내려가 있고 큐브가 박스 안에 보인다.
   - frame 2: 팔이 펼쳐지고 큐브가 테이블에 있는 시작 부근 장면으로 바뀐다.
   - frame 0 관절값은 이미 모두 0이다.
5. 이 영상은 raw에만 있는 것이 아니라 실제 학습 데이터에도 포함돼 있다.
   - 합본 원본 인덱스 24 → train episode 20 → global frame 8687.
   - 디코딩 영상과 raw frame 0의 MAE: front 1.871725 / wrist 2.101131.
   - raw frame 2와의 MAE: front 5.405685 / wrist 70.671967.

### 소스로 확인한 경로

- IsaacLab `ManagerBasedEnvCfg.rerender_on_reset` 기본값은 `False`다.
  해당 설정 주석에도 reset 이후 센서 데이터가 새 상태를 반영하지 않는다고 명시돼 있다.
- 생성 스크립트·해당 task 설정에서 이를 켜는 override를 찾지 못했다.
- reset 후 `obs_buf`를 계산하고 pre-step recorder가 그 관측을 저장한다.
- 평가의 `refresh_reset_images()`는 물리 상태를 진행시키지 않고 여러 번 렌더링한 뒤
  카메라 캐시를 갱신한다. 생성과 평가의 초기 관측 처리에 차이가 있다.

따라서 **reset 후 갱신되지 않은 카메라 관측이 학습에 기록되는 경로**가 확인됐다.
단, 오래된 영상이 정확히 어느 이전 시도에서 왔는지는 확정하지 않았다.
frame 0·1이 같은 세부 렌더 파이프라인 지연 메커니즘까지 규명한 것도 아니다.
첫 두 프레임의 문제를 확인한 것이며, 나머지 모든 프레임의 동기화가 정상이라고 인증한 것은 아니다.

## 7. 정책 입력 민감도 검사

10,000스텝 모델과 관절 상태 0을 고정하고 영상만 raw frame 0 → frame 2로 바꿨다.
비교 목표는 올바른 첫 행동인 `obs/joint_pos_target[1]`이다. 모델은 각 예측 전에 reset했다.
CPU에서 계산했고 별도 검토자가 같은 결과를 재현했다.

| 집합 | raw frame 0 입력 RMSE | raw frame 2 입력 RMSE | 개선 시연 |
| --- | --- | --- | --- |
| train 61개 | 0.137976453 rad | 0.122956291 rad | 44/61 |
| valid 15개 | 0.146206528 rad | 0.129999518 rad | 13/15 |

**제한:** frame 2는 정확한 초기 상태를 다시 렌더링한 영상이 아니다.
`frame2 image + q0 state + target1` 조합의 입력 민감도 검사다.
이 수치를 영상 수정의 인과 효과나 실제 rollout 성공률 개선으로 해석하지 않는다.
raw 영상을 입력했으므로 LeRobot 영상 디코딩을 사용한 기존 오프라인 RMSE와도 구분한다.

검증용 균등 500표본 중 `frame_index < 2`는 3개이며 모두 frame 0이다.
평균 오차만으로 시작 구간 문제를 판단하면 희석될 수 있다. 기존 별도 시작 프레임 지표도 함께 본다.

## 8. 첫 접근 명령도 이미 어긋남

평가 3개 모두 첫 팔 명령 5개 관절이 train 61개 **첫 목표각의 관절별 min/max 범위** 밖이다.
그리퍼 첫 명령은 범위 안이다. 이는 전체 관절 물리 한계 밖이라는 뜻은 아니다.

train 첫 목표각 범위:

| 관절 | 최소 rad | 최대 rad |
| --- | --- | --- |
| shoulder_pan | -0.038908 | 0.035562 |
| shoulder_lift | -0.114164 | 0.020658 |
| elbow_flex | -0.024772 | 0.136901 |
| wrist_flex | -0.022724 | 0.004112 |
| wrist_roll | -0.000003590 | 0.000002355 |
| gripper | 0.741859 | 1.348641 |

seed 4101 첫 정책 명령:
`[-0.085773, -0.264229, 0.278043, -0.077206, 0.068281, 0.963446]`.

가장 가까운 train 초기 큐브 위치의 시연은 `shard_009/demo_9`이며 XY 거리는 약 0.011973 m다.
이 시연의 올바른 첫 목표각(`target[1]`)은
`[-0.032017, -0.002645, 0.003171, -0.000527, -0.000001989, 0.815115]`이다.
물체 위치가 정확히 같지는 않으므로 이 차이를 곧바로 동일 조건의 정책 오차라고 부르지 않는다.

ACT는 현재 30개 행동을 큐에 넣고 소비한 뒤 다시 예측한다. 60Hz 기준 약 0.5초 간격이다.
매 스텝 이미지를 서버로 보내더라도 큐가 남아 있으면 새 예측에 사용되지 않는다.
이 구조가 초기 오류를 키우는지는 아직 별도 비교 실험을 하지 않았다.

## 9. 평가 위치 분포 차이

실제 train 61개를 split 원본 인덱스 → aggregate contract → raw demo로 매핑해 계산한 초기 범위:

- X: `[0.2790919244, 0.4207139015]` m.
- Y: `[-0.4366019368, -0.3610756993]` m.
- seed 4100: X가 범위 밖.
- seed 4101: 축별 bounding box 안.
- seed 4102: Y가 범위 밖.

500개 전체 범위를 train 61개 범위로 혼동하지 않는다.
bounding box 안이라는 조건도 결합분포의 충분한 학습 지원을 보장하지 않는다.
4101도 실패했으므로 OOD 위치만으로 세 실패를 모두 설명할 수 없다.

## 10. 반드시 피할 해석 오류

- `obs/joint_pos_target[0]`에는 이전 목표값이 남아 있다. 첫 행동 비교는 **target[1]**이다.
  기존 `target[1:]` 행동 정렬을 잘못된 정렬로 바꾸지 않는다.
- 그리퍼 약 0.8~1.0 rad는 이 구성에서 열림 방향이다. 0.963을 닫힘으로 해석하지 않는다.
- 첫 두 영상이 같다는 사실만으로 76개 전부 같은 잘못된 장면을 가진다고 단정하지 않는다.
- 초기 영상 문제를 모든 실패의 단독 원인으로 확정하지 않는다.
- 목표각 예측 RMSE, 원본 시연 재생 성공률, 학습 정책 성공률을 섞지 않는다.
- `actuator_reported_torque_nm`은 implicit actuator의 제한 적용 PD 추정값이며 실측 접촉력이 아니다.
- 시작 관절 상태가 같다고 초기 영상·물체 배치·물리 내부 상태까지 같다고 주장하지 않는다.

## 11. 파일 위치와 증거 수준

저장소 기준 경로다. 다른 기기에서 `outputs/`가 없으면 데이터·모델·영상은 별도로 전달받아야 한다.

| 용도 | 경로 |
| --- | --- |
| 기존 전체 인계 | `docs/HANDOFF_ACT_VISION224_500_20260913.md` |
| 제어 비교 근거 | `docs/evidence/mimic_control_diagnosis_20260921.json` |
| 2,000스텝 근거 | `docs/evidence/act_pilot_76_20260921.json` |
| 10,000스텝 근거 | `docs/evidence/act_resume_10000_20260921.json` |
| raw | `outputs/mimic_vision224_500_20260913/raw/shard_*.hdf5` |
| 후보 데이터 | `outputs/mimic_target_unclipped_76_20260919/lerobot_all` |
| 분리·매핑 | `outputs/act_pilot_76_20260921_split` |
| 현재 모델 | `outputs/act_resume_10000_20260921/model/checkpoints/010000/pretrained_model` |
| 평가 영상·초기 PNG·60스텝 trace | `outputs/act_resume_10000_20260921/rollouts/seed_4101` |
| 생성 | `scripts/mimic/generate_dataset.py` |
| 변환 | `scripts/imitation_learning/convert_hdf5_to_lerobot.py` |
| 추론 | `scripts/imitation_learning/serve_lerobot_act.py` |
| 평가·초기 카메라 갱신 | `scripts/evaluation/lerobot_act_so101.py` |
| task 관측·기본 설정 | `source/leisaac/leisaac/tasks/template/single_arm_env_cfg.py` |

설치된 IsaacLab 소스는 다음 위치 아래에서 확인했다.

```bash
cd "/data/$USER/conda-envs/leisaac/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab"
```

- `envs/manager_based_env_cfg.py`: `rerender_on_reset`.
- `envs/manager_based_env.py`: reset → render 조건 → observation 계산 순서.
- `envs/mdp/recorders/recorders.py`: `PreStepFlatPolicyObservationsRecorder`.

최신 초기 영상·민감도 조사는 코드·데이터를 수정하지 않고 수행했다.
수치와 영상 비교는 도구 출력 및 독립 검토로 재현했으며 이 문서가 최초 저장소 기록이다.
이 조사만의 실행 가능한 전용 진단 스크립트나 별도 JSON은 이 시점에는 없었다.
후속 조사에서 만든 스크립트와 근거 JSON은 13절에 있다.
기존 `docs/evidence/` JSON들이 최신 카메라 검사를 포함한다고 오해하지 않는다.

## 12. 다음 AI에게 요청할 작업과 완료 기준

1. 저장소 상태와 로컬 데이터·모델 존재를 확인한다.
2. 6~8절의 초기 영상 불일치를 재검증하고 실행 가능한 진단과 근거를 남긴다.
3. **물리 상태를 움직이지 않고 해당 상태의 영상을 갱신하는 방식**으로
   reset → camera 갱신 → observation 계산 → recording 순서를 검증한다.
4. 처음 두 프레임을 무작정 삭제하거나 전체 영상을 2프레임 이동하지 않는다.
   과거 raw 상태를 이용해 올바른 영상 확보가 가능한지 검토하고 원본은 보존한다.
5. 같은 초기 물리 장면에서 성공 시연과 정책의 첫 접근·손목·그리퍼 동작을 비교한다.
   현재 가까운 시연 비교는 큐브 위치가 약 1.2 cm 다르다는 한계가 있다.
6. 필요하면 같은 모델로 `n_action_steps=30`과 짧은 재관측 간격을 비교한다.
   모델·장면·성공 기준을 고정하고 학습 변경과 섞지 않는다.
7. 관측 결함과 시작 동작을 검증하기 전에는 추가 대규모 학습·500회 재생성·리더 재촬영으로 넘어가지 않는다.
8. 확인된 원인과 추정을 구분하고, 수정이 필요한 경우 작성·독립 검증 패스를 분리한다.

**완료 기준:** 추정 원인 목록을 늘리는 것이 아니라, 문제가 발생하는 경로를 재현하고
한 요소를 바꿨을 때 입력·정책 출력·실제 동작이 어떻게 달라지는지 보여준다.
오프라인 민감도 개선만으로 집기 성공 개선을 선언하지 않는다.

## 13. 후속 조사 결과 (2026-09-21, 12절 1~6항 수행)

근거 파일은 `docs/evidence/reset_camera_probe_20260921.json`(통합)과
`docs/evidence/initial_frame_sync_20260921.json`(76개 raw 전수 검사)이다.
로컬 산출물은 `outputs/initial_frame_diag_20260921/`(1.5 MB, JSON·224×224 PNG·로그, 영상 없음)에 있다.
학습·재생성·리더 재촬영은 하지 않았고 기존 데이터·모델은 수정하지 않았다.

### 13.1 원인 경로: 첫 두 프레임이 아니라 카메라 갱신 주기 전체의 문제

소스로 확인한 경로는 다음과 같다.

- 카메라 설정 `update_period = 1/30 s`, 물리 스텝 `1/60 s`, `decimation = 1`
  (`source/leisaac/leisaac/tasks/pick_cube_into_box/pick_cube_into_box_env_cfg.py`,
  `source/leisaac/leisaac/tasks/template/single_arm_env_cfg.py`).
- `sensor_base.py`의 `update(dt)`는 `timestamp - last_update >= update_period`일 때만 버퍼를 갱신한다.
  따라서 **카메라 영상은 제어 스텝 두 번마다 한 번 갱신**되고, 프레임 `(2k, 2k+1)`은 같은 영상이다.
- `reset()`은 `rerender_on_reset=False`면 렌더 없이 관측을 계산하고, `scene.reset()`이 센서를 outdated로
  표시하므로 관측은 **annotator에 남아 있던 직전 렌더**를 읽는다. 이것이 frame 0·1의 오래된 영상이다.
- `PreStepFlatPolicyObservationsRecorder`는 `obs_buf["policy"]`를 그대로 저장하므로 raw HDF5에 그 영상이 기록된다.
- 생성 스크립트·태스크 설정에는 `rerender_on_reset`·`update_period` override가 없다.
  재생·평가 스크립트만 `rerender_on_reset=True`와 4회 예열 렌더를 쓴다.

raw 전수 검사(`scripts/evaluation/inspect_initial_frames.py`)로 예측을 확인했다.

| 항목 | 앞 카메라 | 손목 카메라 |
| --- | --- | --- |
| frame 0 == frame 1 | 76/76 | 76/76 |
| frame 1 == frame 2 | 0/76 | 0/76 |
| 짝수 시작 쌍 (2k, 2k+1) 완전 일치 | 17,293/17,293 | 17,293/17,293 |
| 홀수 시작 쌍 (2k+1, 2k+2) 완전 일치 | 0/17,232 | 0/17,232 |
| frame 0 vs 같은 샤드 직전 demo 마지막 프레임 MAE 중앙값 | 3.22 (0.5 미만 13/71) | 29.06 |

직전 demo와의 MAE가 큰 경우는 그 사이에 버려진 실패 시도가 있었기 때문으로 설명되며, 실패 시도는 저장되지 않아 원본을 특정할 수 없다.
홀수 프레임의 영상은 관절 상태보다 한 스텝(1/60 s) 늦다. 이 구조는 학습과 평가에 똑같이 적용되므로 그 자체의 학습 영향은 측정하지 않았다.

### 13.2 Isaac 프로세스 재현 (`scripts/evaluation/probe_reset_camera_refresh.py`)

`shard_009/demo_9`를 저장된 초기 상태에서 원래 Mimic 행동으로 재생한 뒤 생성기와 같은 `env.reset()`을 호출했다.

- reset 직후 반환된 정책 영상은 **reset 직전 마지막 렌더와 비트 단위로 동일**했다(MAE 0.0, sha256 일치, 두 카메라 모두).
  같은 시점에 물리 상태 해시는 바뀌었고, 반환된 관측 dict는 레코더가 저장하는 `env.obs_buf` 그 객체였다.
- 프로세스 안 재생에서도 짝수 쌍 338/338 일치, 홀수 쌍 0/337 일치로 같은 패턴이 재현됐다.
- `rerender_on_reset=True`로 다시 실행하면 reset 내부의 렌더 1회로는 **앞 카메라가 여전히 비트 동일**했고 손목은 수렴 영상과 MAE 50.1 차이였다.
  즉 설정 플래그만으로는 고쳐지지 않는다.
- 물리를 진행하지 않고 `sim.render()`를 반복한 뒤 `camera.reset()`·`camera.update(0, force_recompute=True)`를 호출하면
  `reset_to` 뒤에는 두 번째 렌더부터, 재생 뒤 plain reset에서는 손목 카메라가 세 번째 렌더부터
  연속 렌더 간 차이가 0.1~0.3(노이즈 바닥)으로 수렴했고, 물리 상태 해시는 전후가 같았다.
  즉 최소 3회 렌더가 필요했고, 평가 스크립트의 `refresh_reset_images()`가 이미 이 방식(4회)이다.

### 13.3 저장된 상태로 올바른 영상 복원 가능 (12절 4항)

raw의 `states[k]`(스텝 k 이후 상태)를 `env.reset_to`로 올린 뒤 같은 갱신을 하면 기록 영상과 맞는다.
기록 프레임 `k`(짝수)는 스텝 `k-1` 이후 렌더이므로 `states[j]`는 프레임 `j+1`, `j+2`와 맞아야 한다.

| 올린 상태 | 비교 프레임 | 손목 MAE | 앞 MAE |
| --- | --- | --- | --- |
| states[1] | 1 (오래된 영상) | 69.66 | 4.69 |
| states[1] | 2 / 3 | 0.29 / 0.29 | 3.15 / 3.15 |
| states[1] | 4 | 2.97 | 3.67 |
| states[299] | 299 / 300 / 301 / 302 | 2.35 / 1.18 / 1.18 / 2.00 | 2.91 / 3.06 / 3.06 / 3.12 |

앞 카메라는 생성 프로세스와 이번 프로세스 사이에 약 3의 MAE 바닥(테이블 영역 가장자리·질감 잡음, 부호 평균 -0.19)이 있어 판별력이 낮고, 판정은 손목 카메라와 중복 쌍 개수로 한다.
따라서 frame 0·1은 삭제하거나 밀지 않고 **`initial_state`를 `reset_to`로 올려 다시 렌더링해 채우는 것이 가능**하다. 실제 데이터 수정은 하지 않았다.

### 13.4 같은 초기 물리 장면에서 정책과 성공 시연 비교 (12절 5·6항)

평가 스크립트에 opt-in 옵션 `--initial-state-hdf5`·`--initial-state-demo`를 추가해 `demo_9`의 저장된 초기 상태에서 정책을 실행했다.
초기 장면 해시 `375aaad2…`가 프로브와 같고, 관절 0, 큐브 XYZ가 시연과 동일하다. 큐브 위치 1.2 cm 차이 한계는 해소됐다.
모델·장면·성공 기준·초기 영상 갱신(4회)은 고정하고 `n_action_steps`만 바꿨다. 각 1회 롤아웃, seed 4101, 최대 1,200스텝.

| 항목 | 시연 demo_9 | 청크 30 | 청크 5 | 청크 1 |
| --- | --- | --- | --- | --- |
| 결과 | 성공 | no_lift | no_lift | no_lift |
| 그리퍼 목표 0.5 rad 미만 첫 스텝 | 278 | 138 | 없음(최소 0.78) | 없음(최소 0.76) |
| 큐브 XY 5 mm 이상 이동 첫 스텝 | 279 | 184 | 없음 | 없음 |
| 관절 제한 보정 스텝 | - | 0 | 1,125 (그리퍼 열림 방향) | 1,143 (그리퍼 열림 방향) |
| 명령 RMSE vs target (1/10/30/60 스텝) | - | 0.151/0.156/0.147/0.167 | 0.151/0.182/0.151/0.175 | 0.152/0.182/0.170/0.279 |
| 명령 RMSE vs target (120/240 스텝) | - | 0.365/0.498 | 0.424/0.592 | 0.483/0.619 |

- 첫 명령은 세 경우 모두 `[-0.069, -0.212, 0.215, -0.10, 0.063, 0.99]` 부근으로 같았고, 시연의 `target[1]`은 거의 0이다.
  Mimic 시연은 현재 자세에서 5스텝 보간으로 시작하는 반면 정책은 첫 스텝부터 큰 목표를 낸다.
- 10~60스텝 구간은 방향이 대체로 같다(pan·lift·wrist_flex). 60스텝 이후 차이가 커지며 청크 30은 시연보다 140스텝 먼저 잘못된 자세에서 그리퍼를 닫고 큐브를 밀었다.
- 재관측을 자주 할수록(청크 5·1) 그리퍼가 열림 한계(1.745 rad) 밖으로 포화한 채 닫히지 않았고 큐브에 닿지도 않았다.
  짧은 재관측 간격은 도움이 되지 않았다.
- 초기 큐브 z는 시연과 롤아웃 모두 첫 두 스텝에 0.0615 → 0.0560 m로 가라앉는다. 평가 결과의 `final_cube_lift_m ≈ -0.0055`는 접촉이 아니라 초기 정착이다.

### 13.5 확정과 추정의 구분

확정:
- 13.1의 메커니즘과 raw 전수 수치, 13.2의 비트 동일 재현, 13.3의 상태 기반 재렌더 정합.
- 올바른 초기 영상과 동일한 초기 장면을 주어도 현재 정책은 청크 30·5·1 모두 실패한다.

추정(검증 안 됨):
- 오래된 첫 두 프레임이 롤아웃 실패의 주요 원인이라는 가설. 동일 장면·올바른 영상에서도 실패하므로 단독 원인은 아니다.
- 30 Hz 카메라 중복이 학습을 해친다는 가설. 학습·평가에 동일하게 적용되므로 효과를 측정하지 않았다.
- reset 내부의 렌더 1회가 annotator에 반영되지 않는 정확한 이유. 두 번째 렌더부터 수렴한다는 실측만 있다.

한계:
- demo 1개, 청크 크기별 1회 롤아웃, 모델 1개. 성공률이 아니다.
- 데이터 재생성·재학습을 하지 않았으므로 초기 영상 수정이 학습에 주는 효과는 미검증이다.

### 13.6 다음 작업 제안 (순서 고정 아님)

1. 생성기에 평가와 같은 reset 후 영상 갱신(렌더 4회 + camera reset/force update, `obs_buf["policy"]` 영상 치환)을 넣거나,
   기존 raw의 `initial_state`로 frame 0·1을 재렌더링해 채운 파생 데이터셋을 새 디렉터리에 만든다. 원본은 보존한다.
2. 카메라 `update_period`를 물리 스텝과 맞출지(1/60 또는 0) 결정한다. 바꾸면 학습 데이터와 평가에 동시에 적용해야 한다.
3. 위 두 항목을 적용한 데이터로 학습하기 전에, 정책 실패의 나머지 원인(60스텝 이후 발산, 조기 파지)을 같은 초기 상태 비교로 계속 좁힌다.
   비교 도구: `scripts/evaluation/compare_first_approach.py`.
4. 검증 명령:

```bash
cd "/data/$USER/leisaac"
"/data/$USER/conda-envs/leisaac/bin/python" scripts/evaluation/inspect_initial_frames.py \
  --contract outputs/mimic_target_unclipped_76_20260919/lerobot_all/action_contract.json \
  --output /tmp/initial_frame_sync_check.json
"/data/$USER/conda-envs/leisaac/bin/python" scripts/evaluation/probe_reset_camera_refresh.py \
  --raw-path outputs/mimic_vision224_500_20260913/raw/shard_009.hdf5 --demo demo_9 \
  --output-dir outputs/<새 디렉터리> --headless
"$HOME/miniforge3/envs/lerobot/bin/python" -m unittest discover -s scripts/imitation_learning -p 'test_*.py' -q
```

### 13.7 생성기 수정과 스모크 검증 (13.6의 1항 수행)

`scripts/mimic/generate_dataset.py`에 opt-in `--reset-render-frames N`을 추가했다(기본 0 = 기존 동작).
N>0이면 매 reset 뒤 물리를 진행하지 않고 N회 렌더 → `camera.reset()`·`update(0, force_recompute=True)` →
`env.obs_buf = observation_manager.compute()` 순으로 갱신하고, 전후 물리 상태 해시가 다르면 예외를 낸다
(`runtime_options.refresh_reset_observations`). 이 옵션은 bounded 루프를 강제하며 매니페스트에 `reset_render_frames`로 기록된다.
`run_mimic_image_batch.py`에도 같은 이름의 전달 인자(기본 0)를 넣었다. 재생성할 때는 `--reset-render-frames 4`를 명시해야 한다.

스모크(seed 7101, 성공 2개/시도 7회, 근거 `docs/evidence/generator_reset_refresh_smoke_20260921.json`):

| 항목 | 기존 76개 후보 | 스모크 demo_0 / demo_1 |
| --- | --- | --- |
| 손목 MAE(frame 0, frame 2) | 41.9~76.8 (중앙값 54.8) | 3.39 / 3.68 |
| 앞 MAE(frame 0, frame 2) | 3.7~6.7 (중앙값 4.97) | 1.04 / 0.95 |
| frame 0 == frame 1 | 76/76 | 2/2 (카메라 주기 1/30 s 유지, 의도된 결과) |
| 짝수 쌍 중복 | 전부 | 전부 (같은 이유) |

즉 frame 0이 reset 장면을 담게 됐다. 30 Hz 중복은 별도 결정 사항이라 그대로 두었다.
단위 테스트 155개 통과. 500개 재생성과 학습은 하지 않았다.
재생성에는 raw 약 11 GB가 필요한데 `/data` 여유가 약 11 GB라 정리 없이는 들어가지 않는다.

