---
role: architect
status: review of pending plan
plan: .omc/plans/remaining-causes-20260921.md
date: 2026-09-21
mode: read-only investigation (no file, data, or model was modified)
---

# Architect 검토: 남은 원인 3건 조사 계획

코드와 데이터를 직접 대조해 검증했습니다. 계획의 전제 하나가 실측으로 뒤집혔습니다.

---

## 1. 과제별 아키텍처 건전성

### (a) 1.1 프로브의 전처리 경로와 `predict_action_chunk` 존재 여부

`predict_action_chunk`는 존재합니다. `$HOME/lerobot/src/lerobot/policies/act/modeling_act.py:126`이고 LeRobot 버전은 0.6.1입니다. 두 체크포인트 모두 `temporal_ensemble_coeff = None`, `chunk_size = 30`, `n_action_steps = 30`이므로 `select_action`이 큐에 넣는 슬라이스(`modeling_act.py:118`)와 `predict_action_chunk` 전체 반환이 같은 텐서입니다. 대안 경로는 필요 없습니다.

정규화 경로는 계획대로 맞습니다. 0.6에서는 정규화가 정책 밖 프로세서에 있고(`processor_act.py`, `processor/factory.py:158-175`), 저장된 파이프라인은 rename → to_batch → device → normalizer입니다. 배치 차원 처리도 안전합니다. `AddBatchDimension`은 차원 인식이라 1D state와 3D image만 unsqueeze합니다(`processor/batch_processor.py:104-119`). LeRobot 데이터셋 항목을 그대로 넣어도 `serve_lerobot_act.py:117-119`의 명시적 unsqueeze 경로와 같은 결과가 나옵니다.

다만 두 가지가 빠져 있습니다.

| 문제 | 근거 | 결과 |
| --- | --- | --- |
| device override 누락 | `policy_preprocessor.json`의 device_processor 값이 `"cuda"` | `--device cpu`를 줘도 정규화가 GPU를 잡습니다. "GPU 0분" 완료 기준과 충돌합니다 |
| 영상 소스 선택 | 학습 입력은 mp4 디코딩, 서빙 입력은 Isaac uint8 원본 | 가르려는 가설은 폐루프 동작이므로 서빙 분포가 본질입니다 |

`serve_lerobot_act.py:139-146`처럼 `preprocessor_overrides={"device_processor": {"device": "cpu"}}`를 명시해야 합니다. 그리고 계획이 부경로로 둔 원본 배열 경로를 close 구간 전 에피소드로 올리시기 바랍니다. 청크 예측이 0.034초/프레임이므로 추가 비용은 무시할 수준입니다.

해상도 경로는 일치합니다. 생성기(`runtime_options.py:66-80`)와 평가기(`lerobot_act_so101.py:169-175`)가 둘 다 640×480에서 `cv2.INTER_AREA`로 224를 만듭니다.

**여기서 가장 심각한 결함은 판정 지표가 거의 자동으로 통과한다는 점입니다.**

밴드 정의 자체는 정확합니다. c가 raw 인덱스이고 LeRobot 프레임 s의 청크가 raw target `s+1..s+30`이므로, 전이를 포함하는 프레임은 정확히 `s ∈ [c-30, c-1]`입니다. 계획의 close 밴드와 일치합니다.

문제는 `chunk_close_hit`과 `close_index_error`에 밴드 제한이 명시되지 않은 점입니다. close 스텝 이후의 모든 프레임은 정답 청크가 전부 닫힘이라 "정답 청크에 닫힘이 있다"는 조건을 만족합니다.

| 항목 | demo_9 기준 값 |
| --- | --- |
| 에피소드 길이 | 676 |
| close 스텝 | 278 |
| 양성 프레임 총수 | 약 398 |
| 그중 close 밴드 | 30 |
| 그중 post-close | 약 368 |

post-close 구간에서는 `observation.state`의 6번째 채널이 이미 닫힘을 담고 있고 `n_obs_steps = 1`이므로, 닫힘을 유지하기만 하면 맞습니다. `close_index_error`도 양쪽이 0이라 중앙값이 0이 됩니다. 즉 사전 등록한 "0.8 이상, ±5 이내면 학습된 것"은 현재 정의대로면 거의 확실히 통과합니다.

처방은 세 가지입니다. 두 지표를 close 밴드로 제한하고, pre-close 밴드에 오경보율을 짝으로 두고, 귀무모형을 hit 지표에도 돌리십시오. hit만 보면 "항상 닫힘 예측" 퇴화 모형이 1.0을 받습니다.

### (b) 정렬 규약의 일관성

정합합니다. `convert_hdf5_to_lerobot.py:134`가 `np.clip(joint_targets[1:], lower, upper)`이고, `:249`가 state와 RGB를 `[0:T-1]`로, `:285`가 출력 프레임 수를 `frame_count - 1`로 둡니다. 따라서 LeRobot 프레임 s는 state가 raw `joint_pos[s]`, 행동이 raw `joint_pos_target[s+1]`입니다. contract의 `"action[t] = obs/joint_pos_target[t+1]"`과 계획의 규약이 모두 같습니다. 청크가 raw target `s+1..s+30`에 대응한다는 것도 맞습니다.

### (c) 소스 귀속 주장의 취약성

닫힘 스텝 매핑의 단사성은 독립 재계산으로 확인했습니다. 소스 10개의 닫힘 스텝이 전부 서로 다릅니다.

| 소스 | 길이 | 소스 닫힘 | +6 |
| --- | --- | --- | --- |
| demo_0 | 825 | 463 | 469 |
| demo_1 | 205 | 100 | 106 |
| demo_2 | 1072 | 718 | 724 |
| demo_3 | 767 | 473 | 479 |
| demo_4 | 676 | 301 | 307 |
| demo_5 | 660 | 272 | 278 |
| demo_6 | 628 | 1 | 7 |
| demo_7 | 312 | 152 | 158 |
| demo_8 | 238 | 105 | 111 |
| demo_9 | 257 | 109 | 115 |

충돌이 없으므로 "관측된 6종이 6개 소스를 가리킨다"는 확정 주장은 지지됩니다. 다만 demo_6은 닫힘이 인덱스 1이라 퇴화 사례이고, 500개 중 close=1이 37개 관측되는 반면 close=7은 0개입니다. demo_6 계열에 대해서는 +6 규칙이 검증되지 않았습니다.

**그런데 raw 500개 전수로 같은 귀속을 돌리면 계획의 결론이 뒤집힙니다.**

| 소스 계열 | 생성 500 | 품질 스크린 통과 | 무클리핑 | 최종 76 |
| --- | --- | --- | --- | --- |
| demo_0 | 35 | 35 (100%) | 0 (0%) | 0 |
| demo_1 | 101 | 12 (12%) | 60 (59%) | 8 |
| demo_2 | 53 | 32 (60%) | 0 (0%) | 0 |
| demo_3 | 64 | 43 (67%) | 0 (0%) | 0 |
| demo_4 | 30 | 27 (90%) | 7 (23%) | 6 |
| demo_5 | 55 | 44 (80%) | 27 (49%) | 27 |
| demo_7 | 53 | 46 (87%) | 21 (40%) | 19 |
| demo_8 | 47 | 30 (64%) | 9 (19%) | 9 |
| demo_9 | 25 | 25 (100%) | 7 (28%) | 7 |
| close=1 계열 | 37 | 0 (0%) | 3 (8%) | 0 |

합계는 생성 500, 스크린 294, 최종 76으로 `docs/evidence/mimic_quality_screen_20260919.json`의 `continuity_and_recorded_release_pass = 294`, `unclipped_candidates = 76`과 정확히 맞습니다. 게이트 재구성이 맞다는 뜻입니다.

즉 Mimic은 소스 10개를 전부 썼습니다. 6개로 줄어든 것은 생성 단계가 아니라 감사 게이트에서 일어났습니다. 그리고 탈락의 지배 원인은 단일 관절입니다.

| 계열 | 지배 관절 | 중앙 최대 위반 (rad) | 위반 프레임 비율 |
| --- | --- | --- | --- |
| demo_0 | wrist_flex 35/35 | 0.0399 | 2.7% |
| demo_1 | wrist_flex 34/41, elbow_flex 7 | 0.3667 | 37.3% |
| demo_2 | wrist_flex 45/53, elbow_flex 8 | 0.1958 | 12.4% |
| demo_3 | wrist_flex 56/64, elbow_flex 8 | 0.1698 | 23.9% |
| demo_4 | wrist_flex 23/23 | 0.0279 | 0.4% |
| demo_5 | wrist_flex 21/28, elbow_flex 7 | 0.3404 | 14.2% |
| demo_7 | wrist_flex 32/32 | 0.0728 | 4.3% |
| demo_8 | wrist_flex 32/38, elbow_flex 6 | 0.3099 | 10.9% |
| demo_9 | wrist_flex 18/18 | 0.1221 | 9.4% |
| close=1 계열 | wrist_flex 19/34, elbow_flex 15 | 0.6578 | 25.4% |

관절 한계는 contract의 `joint_limits`에서 읽었습니다. `wrist_flex`는 [-1.6581, 1.6581] rad입니다.

demo_0은 35개 전부가 연속성과 안정 놓기 검사를 통과했는데, wrist_flex가 약 2.3도 넘치는 프레임이 2.7% 있다는 이유만으로 전량 탈락했습니다.

허용오차를 바꿨을 때의 후보 풀입니다. 조건은 품질 스크린 통과이면서 최대 위반이 허용오차 이하인 에피소드입니다.

| 허용오차 (rad) | 에피소드 | 프레임 | 소스 계열 |
| --- | --- | --- | --- |
| 0.00 (현재) | 76 | 34,525 | 6 |
| 0.02 | 105 | 51,953 | 7 |
| 0.05 | 188 | 119,680 | 9 |
| 0.10 | 208 | 130,119 | 9 |
| 0.20 | 229 | 141,872 | 9 |
| 0.40 | 285 | 174,930 | 9 |

이 결과가 과제 2의 옵션 C 기각 근거를 무너뜨립니다. "소스 10개가 전부 들어가도록 재생성 → raw 11 GB, 디스크 불가"라고 적혀 있지만, 재생성은 필요 없습니다. 퇴화한 demo_6을 뺀 9계열이 이미 디스크에 있습니다. 용량도 문제가 아닙니다. `lerobot_all`이 76개에 82 MB이므로 188개는 약 285 MB이고, 초기 프레임 재렌더 raw는 76개 1.3 GB에서 188개 약 3.2 GB입니다. `/data` 여유 16 GB 안에 들어갑니다.

세 번째 신호의 정의에도 문제가 있습니다. `obs/ee_frame_state`는 로봇 루트 기준 gripper 프레임입니다(`source/leisaac/leisaac/enhance/envs/mdp/observations.py:100-109`, target index 0이므로 jaw가 아닙니다). 반면 `states/rigid_object/cube/root_pose`는 env 원점 기준입니다. 계획의 "ee_frame_state를 cube root_pose로 나눈다"는 프레임이 다릅니다. 올바른 합성은 `states/articulation/robot/root_pose`를 끼운 3항 합성이고, 그 필드는 raw에 있으므로 계산 가능합니다. 쿼터니언은 나누기가 아니라 켤레 곱이라는 점도 스크립트에 명시하셔야 합니다.

더 중요한 것은 이 신호가 앞의 두 신호와 독립이 아니라는 점입니다. Mimic의 subtask 세그먼트는 object frame에서 소스 궤적을 강체 변환한 것이므로, object-frame eef 궤적이 소스와 일치하는 것은 알고리즘상 당연합니다. 독립 확증이 아니라 구현 검증에 가깝습니다. "세 신호가 독립"이라는 표현은 과합니다.

### (d) update_period 플래그

수용성은 소스로 확정됩니다. `update_period`는 `SensorBaseCfg`의 평범한 float 필드이고 기본값 0.0이 "매 스텝"입니다. `TiledCameraCfg`가 이를 상속합니다.

동치 여부도 소스로 확정됩니다. `sensor_base.py:186`이 다음과 같습니다.

```python
self._is_outdated |= self._timestamp - self._timestamp_last_update + 1e-6 >= self.cfg.update_period
```

`1e-6` 여유 항 때문에 dt = 1/60, update_period = 1/60이면 비교가 항상 참입니다. 따라서 `update_period = 0`과 `1/60`은 이 설정에서 동치입니다. 계획이 추정으로 분류한 항목을 확정으로 올리셔도 됩니다. `render_interval` 기본값은 1이고 `decimation = 1`이므로 렌더는 매 물리 스텝 일어납니다. 갱신 주기만 병목입니다.

생성기와 평가기를 둘 다 고쳐야 한다는 판단은 맞습니다. 평가기는 `configure_visual_options`를 쓰지 않고 `lerobot_act_so101.py:281-283`에서 자체적으로 width와 height만 덮어씁니다. 3파일 표는 정확합니다.

누락이 하나 있습니다. 기존 `evaluation.json`과 매니페스트에는 `camera_update_period` 필드가 없습니다. "기록된 값이 다르면 비교 도구가 예외를 낸다"를 그대로 구현하면 기존 산출물과의 비교가 전부 깨집니다. 필드 누락은 cfg 기본값 1/30으로 읽는 규칙을 같이 넣으셔야 합니다.

비용 추정에도 보완이 필요합니다. 카메라 버퍼 갱신이 2배가 되므로 시도당 시간이 달라질 수 있고, 3.3의 추정식은 성공 시도와 실패 시도의 벽시계 시간을 분리해야 합니다. 실패 시도는 horizon까지 가서 더 깁니다. 2/7 표본의 점추정이라는 점도 명시하십시오.

### (e) split 실험의 교란

가장 큰 문제는 두 모델을 서로 다른 검증 집합에서 평가한다는 점입니다. 계획 270행이 "각 모델에 대해 자기 검증 집합으로"라고 적고 있습니다.

홀드아웃 모델의 검증 집합은 학습에 없던 계열이고, 무작위 모델의 검증 집합은 학습에 있던 계열입니다. 이것은 분포 밖 일반화와 분포 내 성능의 비교이지, "학습 다양성이 닫힘 학습을 제약하는가"라는 질문의 답이 아닙니다. 홀드아웃 모델이 더 낮게 나오는 것은 거의 보장되며, 그 결과는 다양성 가설의 증거가 되지 못합니다.

부차 문제로 조작 강도가 약합니다. demo_9와 demo_4는 살아남은 6계열 중 가장 작은 둘이고 합쳐서 13/76, 즉 17%입니다.

---

## 2. 가장 강한 안티테제

**오프라인 프로브는 두 가설을 가르지 못합니다.**

프로브는 관측을 시연 것으로 고정합니다. 그런데 `observation.state`의 6번째 채널이 그리퍼 관절각이고 ACT는 단일 관측 스텝 조건부입니다. close 밴드 직전 프레임의 상태는 "그리퍼 열림, 팔은 파지 자세"인데, 이는 학습 집합에서 닫힘 직전에만 나타나는 거의 유일한 상태입니다. 따라서 프로브의 높은 점수는 "닫힘 신호를 배웠다"와 "학습 분포의 좁은 상태 이웃에서 다음 행동을 암기했다"를 구분하지 못합니다.

여기에 13.8의 관측이 겹칩니다. 고정 데이터 모델은 같은 장면에서 240스텝까지 명령 RMSE 0.118로 시연을 잘 따라가면서도 닫지 않습니다. 도달은 되는데 닫지 않는다는 조합은 "폐루프 표류"로 설명되지 않습니다. 프로브가 0.8을 넘겨 "배웠다"가 나오면 계획은 하이브리드 롤아웃으로 가는데, 두 관측이 이미 충돌하므로 프로브 결과 자체를 의심해야 하는 상태가 됩니다.

부수적으로, 그 RMSE 0.118은 닫힘 스텝 278 이전 창에서 잰 값입니다. 그리퍼 한 관절이 0.5 rad 틀려도 6관절 RMSE 기여는 약 0.20이므로, 240스텝 창에서는 닫힘 실패가 수치에 나타나지 않습니다. 이 수치를 "도달은 된다"의 근거로 쓸 때 창이 닫힘을 포함하지 않는다는 점을 명시하셔야 합니다.

안티테제를 무력화할 방법은 있고 비용이 거의 없습니다. 같은 청크 예측을 **폐루프 관측**에서도 돌리십시오. 1.2의 `--dump-policy-images-every`가 정책이 실제로 본 224×224 영상을 만들어 주고, trace에 상태가 있습니다. 시연 입력일 때와 롤아웃 입력일 때의 닫힘 예측 차이가 두 가설을 직접 가릅니다. 계획은 이 조합을 하지 않습니다. 가장 싼 결정적 실험이 빠져 있습니다.

---

## 3. 실재하는 상충 관계

**클리핑 허용오차 완화.** 한쪽은 재생성 없이 소스 9계열, 188 에피소드, 119,680 프레임을 얻는다는 점입니다. 다른 쪽은 그 행동들이 시뮬레이터가 실제로 실행하지 못한 목표각이라는 점입니다. 학습하면 정책이 도달 불가능한 목표를 배우고 실물에서는 한계 밖 명령이 됩니다. 13.8에서 고정 데이터 모델의 관절 제한 보정이 0/0/0으로 떨어진 것은 unclipped 선별의 성과일 수 있고, 그것을 되돌리는 셈입니다. 중간안은 허용오차 0.05에서 시작해 clip을 적용하되 위반 크기와 프레임 비율을 provenance에 남기는 것입니다.

**조작 강도 대 학습 규모 동등성.** demo_5는 27개로 36%를 차지하므로 이를 빼면 조작이 강해집니다. 대신 학습 규모가 49로 줄어 두 요소가 동시에 바뀝니다. demo_9와 demo_4를 빼면 규모는 63으로 맞지만 조작이 17%에 그칩니다. 현재 계획은 후자를 택했고 그 선택은 합리적이지만, 효과가 안 보였을 때 "다양성은 구속 조건이 아니다"로 읽는 것은 과잉 해석이 됩니다.

**프로브 입력 경로.** 디코딩 영상은 학습과 비교 가능하고 원본 배열은 서빙과 비교 가능합니다. 어느 하나를 주경로로 정하는 것보다 둘 다 close 구간에서 동등하게 돌리는 편이 낫습니다.

---

## 4. 종합 제안

순서를 이렇게 바꾸시기 바랍니다.

1. **신규 0단계, GPU 0분, 디스크 0.** 감사 게이트의 계열 편향을 근거로 확정합니다. `attribute_source_demo.py`의 범위를 76개가 아니라 raw 500개로 넓히고, 계열별 생성 수와 게이트 단계별 생존율, wrist_flex 위반 크기 분포를 근거 JSON에 넣습니다. 이 검토의 표가 그대로 재현 대상입니다. 과제 2의 전제가 여기서 정해집니다.
2. **과제 1.1 수정.** 두 판정 지표를 close 밴드로 제한하고, pre-close 오경보율을 추가하고, 귀무모형을 상수 예측기와 항상 닫힘 예측기 둘로 늘리고, device override를 명시합니다.
3. **과제 1.1에 세 번째 대상 추가.** 1.2의 롤아웃 관측 덤프를 프로브 입력으로 재사용해 폐루프 관측에서의 닫힘 예측을 냅니다. 이것이 안티테제를 정면으로 가릅니다.
4. **과제 2.1 유지.** 싸고 가설을 배제할 수 있습니다. 다만 이름 충돌을 해소하십시오. 계획 안에서 "demo_9"가 소스 시연과 생성 에피소드 `shard_009/demo_9`를 동시에 가리킵니다. 후자는 닫힘 278이라 소스 demo_5 계열입니다. 실행자가 틀린 에피소드를 고르기 쉽습니다.
5. **과제 2.2 재설계.** 두 안 중 하나를 고르십시오. 최소 수정안은 공통 테스트 집합을 계열 층화로 먼저 고정하고 두 학습 집합을 같은 크기로 짜서 같은 테스트 집합에서 평가하는 것입니다. 더 나은 안은 허용오차 0.05 후보 풀에서 같은 에피소드 수의 두 학습 집합을 짜되 하나는 6계열, 하나는 9계열로 두고 공통 테스트 집합에서 비교하는 것입니다. 후자가 다양성 가설의 직접 조작이고 재생성이 필요 없습니다.
6. **과제 3 유지.** `update_period = 0`과 `1/60`의 동치는 소스로 확정됐으므로 스모크의 목적을 "플래그가 실제로 먹는지"와 "비용"으로 좁히면 충분합니다.

---

## 5. 계획 자체의 원칙 위반

| 원칙 | 판정 | 근거 |
| --- | --- | --- |
| 1. 한 번에 한 요소 | 위반 | 2.2가 학습 계열 구성과 평가 집합 구성을 동시에 바꿉니다 |
| 2. 비용 오름차순 | 부분 위반 | GPU 0분에 디스크 0인 raw 500 게이트 편향 분석이 아예 없습니다. 가장 싼 단계가 빠졌습니다 |
| 3. 롤아웃 성공률을 판정 근거로 쓰지 않음 | 준수 | 2.1과 2.2 모두 성공률이 아님을 명시했습니다 |
| 4. 새 판정 기준은 귀무모형부터 | 위반 | 실제 판정 기준은 `chunk_close_hit`인데 귀무모형이 `null_gripper_rmse_by_band`에만 걸려 있습니다 |
| 5. 기본 동작 불변 | 부분 위반 | 3.1의 "기록값이 다르면 예외"가 필드 없는 기존 산출물을 깨뜨립니다 |

Decision Driver 1도 사실 오류를 담고 있습니다. "500개 재생성은 raw만 11 GB라 불가"는 맞지만, 이를 근거로 옵션 2C를 기각한 것은 틀렸습니다. 소스 9계열 확보에 재생성이 필요 없습니다.

### 그 밖의 구체 지적

- `prepare_mimic_act_split.py:29`에 `EXPECTED_EPISODES = 500` 상수가 있습니다. `--valid-episodes` 추가 시 이 상수와의 상호작용을 확인하십시오.
- `build_split_indices(76, seed=43, shard_size=25, valid_per_shard=5)`는 마지막 샤드가 1개라 `count = min(5, 0) = 0`이 되어 검증이 15개가 됩니다. 현재 구성과 일치합니다. 새 인자는 이 경로를 우회하므로 `split_provenance.json`에 어느 경로를 썼는지 기록해야 재현이 됩니다.
- `shard_010/demo_16`의 `target[0]` 오탐 대응은 맞습니다. 다만 close=1 계열 37개는 인덱스 1부터 세도 1이 나오는 진짜 퇴화 사례입니다. 오탐 방어와 퇴화 탐지를 코드에서 구분하십시오.
- 1.2의 완료 기준 중 초기 장면 해시 대조는 타당합니다. PNG 덤프가 루프에 들어가 벽시계 시간은 달라지지만 물리 상태에는 영향이 없습니다.

---

## 이 검토에서 새로 확정한 사실

- 소스 10개의 닫힘 스텝은 전부 구별되므로 귀속 매핑은 단사입니다. 계획의 확정 주장은 지지됩니다.
- Mimic은 소스 10개를 모두 사용했습니다. raw 500개 전수 귀속으로 확인했습니다.
- 6계열로의 축소는 생성이 아니라 감사 게이트, 그중에서도 wrist_flex 관절 한계 필터 한 곳에서 일어났습니다.
- demo_0의 35개는 다른 모든 검사를 통과한 채 약 2.3도 초과로 전량 탈락했습니다.
- 계획이 "디스크 때문에 불가능"이라고 판단한 소스 다양성 확보가 재생성 없이 가능합니다.

### 재현 방법

계열 귀속은 raw `obs/joint_pos_target[1:, 5] < 0.5`의 첫 인덱스에 1을 더한 값을 위 소스 닫힘 표와 대조해 구했습니다. 게이트 단계는 `outputs/evaluation/mimic_quality_20260919/episode_diagnostics.json`의 `screen_pass`와, contract `joint_limits`로 재계산한 클리핑 여부를 교차한 것입니다. 교차 결과의 합계가 `docs/evidence/mimic_quality_screen_20260919.json`의 294와 76에 정확히 일치하는 것으로 재구성의 타당성을 확인했습니다.

### 참조한 파일

- `/data/lim/leisaac/.omc/plans/remaining-causes-20260921.md`
- `/data/lim/leisaac/docs/HANDOFF_MIMIC_ACT_ROOT_CAUSE_20260921.md`
- `/data/lim/leisaac/docs/evidence/mimic_quality_screen_20260919.json`
- `/data/lim/leisaac/outputs/evaluation/mimic_quality_20260919/episode_diagnostics.json`
- `/data/lim/leisaac/outputs/mimic_target_unclipped_76_20260919/lerobot_all/action_contract.json`
- `/data/lim/leisaac/scripts/imitation_learning/convert_hdf5_to_lerobot.py`
- `/data/lim/leisaac/scripts/imitation_learning/serve_lerobot_act.py`
- `/data/lim/leisaac/scripts/imitation_learning/prepare_mimic_act_split.py`
- `/data/lim/leisaac/scripts/evaluation/lerobot_act_so101.py`
- `/data/lim/leisaac/scripts/mimic/runtime_options.py`
- `/data/lim/leisaac/source/leisaac/leisaac/enhance/envs/mdp/observations.py`
- `/data/lim/leisaac/source/leisaac/leisaac/tasks/template/single_arm_env_cfg.py`
- `$HOME/lerobot/src/lerobot/policies/act/modeling_act.py`
- `$HOME/lerobot/src/lerobot/processor/factory.py`
- `$HOME/lerobot/src/lerobot/processor/batch_processor.py`
- `/data/$USER/conda-envs/leisaac/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/sensors/sensor_base.py`
