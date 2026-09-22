# SO101 Mimic → ACT 남은 원인 3건 조사 인계 (세션 이관용)

> 2026-09-22 CPU 실행을 마쳤다. **2.0·1.1 완료, 1.1.7은 정책 입력 PNG 부재로 미결**이다.
> 아래 7절은 실행 전 기록이다. 이어서 작업할 때는 10절과 [원인 인계 14절](HANDOFF_MIMIC_ACT_ROOT_CAUSE_20260921.md#14-남은-원인-cpu-조사-2026-09-22)을 먼저 읽는다.

이 문서는 2026-09-21 세션의 결과를 다른 세션으로 옮기기 위한 정리다.
기준 커밋은 이 문서를 포함한 커밋이며, 직전 결과 커밋은 `58d9ac9`다.
이전 인계 문서 `docs/HANDOFF_MIMIC_ACT_ROOT_CAUSE_20260921.md`의 13절이 지금까지의 확정 사실이고,
이 문서는 **그다음 작업(남은 원인 3건)의 합의된 계획과 시작 지점**을 담는다.

## 1. 한 줄 요약

- 초기 카메라 영상 결함은 확정됐고 생성기 수정과 파생 데이터 학습 비교까지 끝났다. 고쳐도 집기는 0/3이다.
- 다음 조사 3건(파지 닫힘 신호 학습 여부, 소스 다양성, 카메라 갱신 주기)의 계획을 Planner·Architect·Critic 5회 반복으로 합의했다.
- 계획은 `docs/plans/remaining_causes_20260921/plan_r5.md`이며 **실행 전 필수 결함은 없다**(마지막 지적인 2.2 디스크 게이트 값은 반영함).
- **지금 실행을 막는 것은 디스크뿐이다.** `/data` 여유 4.4 GB(2026-09-22 새벽 기준)로 Isaac 단계와 2.2가 불가하다.

## 2. 환경과 권한 (변경 없음)

```bash
cd "/data/$USER/leisaac"
```

- 개인 원격 `mmporong` (`git@github.com:mmporong/leisaac.git`)에만 push한다. `origin`은 upstream이라 push하지 않는다.
- LeRobot Python: `$HOME/miniforge3/envs/lerobot/bin/python`. Isaac Python: `/data/$USER/conda-envs/leisaac/bin/python`.
- 실물 로봇 금지. 원본 데이터·모델 보존. 새 산출물은 새 디렉터리. strict replay gate·성공 판정 완화 금지.
- 다른 세션의 GPU·디스크 작업을 임의 종료하거나 삭제하지 않는다. 확인 후 보고한다.
- 작성 패스와 검증 패스를 분리한다(`verifier`/`code-reviewer` 위임 가능). 확정과 추정을 구분한다.

## 3. 2026-09-21에 끝낸 일 (전부 커밋·push됨)

| 커밋 | 내용 | 근거 |
| --- | --- | --- |
| `1a530e2` | reset 후 카메라 영상이 직전 렌더와 비트 동일함을 Isaac에서 재현, 76개 raw 전수 검사(짝수 쌍 17,293/17,293 중복), 저장 상태 기반 재렌더 정합, 동일 초기 장면 정책 비교(청크 30/5/1 모두 실패) | `docs/evidence/reset_camera_probe_20260921.json`, `initial_frame_sync_20260921.json`, 13.1~13.5절 |
| `4e7db70` | 생성기 `--reset-render-frames` 옵션(기본 0), 스모크로 손목 MAE(frame0, frame2) 54.8 → 3.4~3.7 확인 | `docs/evidence/generator_reset_refresh_smoke_20260921.json`, 13.7절 |
| `58d9ac9` | 76개 후보 frame 0·1만 재렌더한 파생 데이터로 10k 학습 vs 원본 대조군 10k: 첫 프레임 RMSE 0.145 → 0.106, 집기 0/3 그대로. demo_9 동일 장면에서 접근은 개선(240스텝 RMSE 0.498 → 0.118)되나 그리퍼를 닫지 않음. Mimic 실패 기록 등 29개 파일 10.4 GiB 정리 | `docs/evidence/reset_fixed_dataset_experiment_20260921.json`, `storage_cleanup_20260921.json`, 13.8절 |

새로 만든 도구(전부 커밋됨):

| 파일 | 용도 |
| --- | --- |
| `scripts/evaluation/inspect_initial_frames.py` | raw 프레임 중복·정합 검사 |
| `scripts/evaluation/probe_reset_camera_refresh.py` | reset 후 영상 갱신 Isaac 프로브 |
| `scripts/evaluation/compare_first_approach.py` | 정책 trace와 시연 정렬 비교 |
| `scripts/evaluation/lerobot_act_so101.py` | `--initial-state-hdf5/--initial-state-demo` opt-in 추가 |
| `scripts/mimic/runtime_options.py`, `generate_dataset.py`, `run_mimic_image_batch.py` | `--reset-render-frames` |
| `scripts/mimic/rerender_initial_frames.py` | 선택 매니페스트의 demo를 frame 0·1 재렌더 파생 shard로 복사 |

## 4. 계획 단계에서 새로 확정한 사실 (계획 파일 0절, 검토자 독립 재계산 일치)

1. **Mimic 생성기는 소스 시연 10개를 전부 썼다.** 76개 후보가 6계열로 줄어든 것은 감사 게이트의 **관절 클리핑 필터**(주로 `wrist_flex`) 때문이다. 품질 게이트 통과 후 클리핑으로 빠진 에피소드는 전체 218개이며, 그중 완전히 배제된 3계열(`src:demo_0/2/3`)이 110개다. 재생성 없이 기존 raw 500개에서 허용오차 0.05 풀 188개(9계열)를 만들 수 있다.
2. 생성 demo의 소스는 raw에 기록되지 않지만 **그리퍼 닫힘 스텝(소스 닫힘 + 6)** 으로 복원된다. 예외는 `src:demo_6`(닫힘 인덱스 1, 안정 놓기 0/37). 귀속은 pick 서브태스크 한정이다. 표기 규약: 소스는 `src:demo_Y`, 생성은 `shard_XXX/demo_Y`.
3. 청크 기반 닫힘 판정 지표는 닫힘 이후 구간이 양성의 90%를 차지해 **"c 이후만 닫힘" 퇴화 예측기가 0.9를 받는다.** 판정은 close 구간 `[c-30, c-1]` 한정, pre-close 오경보율과 짝, 퇴화 예측기 3종(학습 평균 상수·항상 닫힘·상태 복사)을 판정 지표 전부에 적용한다.
4. 변환기 `convert_hdf5_to_lerobot.py:134`의 `np.clip`은 목표각을 한계로 투영한다. close 구간의 클리핑 대상 프레임은 두 풀 모두 0(0/3,360, 0/2,280)이라 판정에 닿지 않는다. 2% 초과 시 중단을 사전 등록했다.
5. 실측 단가: 10k 학습 13분 12초, 롤아웃 1회 약 55초, 실험 1건 약 16.5분·440 MB, CPU 청크 예측 0.034초/프레임, LeRobot 디코딩 0.020초/프레임, 76개 재렌더 약 30분·1.3 GB.

## 5. 합의된 계획 (요약)

전문: `docs/plans/remaining_causes_20260921/plan_r5.md` (1,053줄). 검토 기록: 같은 디렉터리의 `architect_r1..r5.md`, `critic_r1..r5.md`.
5차 Critic 판정은 ITERATE였으나 남은 결함은 "2.2 디스크 진입 게이트 9.5 GB → 14,450 MiB" 하나였고 이 문서와 함께 계획에 반영했다.
과제 1·3·2.0·2.1은 4회 연속 착수 가능 판정이다.

**실행 순서**

```
2.0 소스 귀속·게이트 분해(CPU)
 → 1.1 티처포싱 오프라인 청크 프로브(CPU, 두 입력 경로: LeRobot 디코딩·원본 배열)
 → 1.2 닫힘 구간 입력 대조 + 정책 입력 영상 덤프 롤아웃(Isaac, stride 1, 구간 240~340)
 → 1.1.7 폐루프 관측 프로브(CPU)
 → 2.1 학습 장면 폐루프 재현 시험(Isaac, 5계열 대표 1개씩, 진단 전용)
 → 3.2 카메라 1/60 스모크(Isaac)와 중복 소거 검증
 → [4.1 게이트: 티처포싱×폐루프 2×2]
 → 2.2 계열 수 조작 실험: 원본 감사(허용오차 0.05) → 188개 단일 디렉터리 재렌더 → 파생본 재감사
       → arm별 108개(학습 84 + 공통 평가 24) 변환 3회 → split 3종 → 학습 3건 → [모호 분기] S′(7,502스텝)
```

**게이트(계획 4.1)**

| 티처포싱 `chunk_close_hit` | 폐루프 `chunk_close_hit` | 해석 | 다음 |
| --- | --- | --- | --- |
| ≥ 0.8 (4조건 충족) | ≥ 0.8 | 닫힘은 배웠고 도달이 문제 | 2.2 건너뜀, 하이브리드 롤아웃(옵션 1C) |
| ≥ 0.8 | ≤ 0.4 | 좁은 상태 이웃 암기 | 2.2 실행 |
| ≥ 0.8 | 0.4~0.8 또는 잔존 표본 < 15 | 부분·미결 | 2.2 실행 |
| < 0.2 | - | 닫힘 신호 미학습 | 2.2 실행 |

디스크 → 2.2: 진입 시 `/data` 여유 14,450 MiB 미만이면 시작하지 않고 회수 후보를 보고한다. 2.0→2.2: 귀속 마진 부족 또는 close 구간 클리핑 2% 초과면 2.4 축소판으로.

**만들 파일 / 고칠 파일**: 계획 "만들 파일과 고칠 파일" 절. 새 스크립트 3개(`probe_gripper_close_offline.py`, `compare_close_window_inputs.py`, `attribute_source_demo.py`), 테스트 4개, 근거 JSON 4개. 고칠 파일 8개는 전부 opt-in이고 기본 동작 불변을 테스트로 고정한다.

**자원**: GPU 약 154분(S′ 포함 167분), CPU 약 234분, 디스크 약 6.7 GB.

## 6. 지금 실행을 막는 것 (2026-09-22 00시 기준 실측)

| 항목 | 값 | 영향 |
| --- | --- | --- |
| `/data` 여유 | **4.4 GB** (97%) | 생성 스모크(`--min-free-gib 4`)·학습(`--min-free-gib 8`)·2.2 전부 불가 |
| `/home` 여유 | 4.1 GB | LeRobot 캐시 여유 적음 |
| 원인 | 다른 세션의 `/data/lim/robot-artifacts`가 4.1 GB → 19 GB로 증가(`pour_20260921`, `restaurant/*`) | 이 세션 소관 아님. 삭제하지 말고 사용자에게 확인 |
| GPU | 다른 세션 Isaac 작업(`tools/simulate_restaurant_mobile.py`)이 2.8 GB 점유했었음 | 각 Isaac 단계 직전 `nvidia-smi`로 확인, 점유 중이면 대기 |

**대용량 확보 없이 시작할 수 있는 단계**: 2.0과 1.1은 CPU로 가능하다. 2026-09-22 파일 확인 결과, 기존 trace에는 240~340스텝의 정책 입력 PNG가 없다. 1.1.7은 상태 거리 검사까지만 가능하며, ACT 폐루프 관측 프로브에는 1.2의 Isaac 영상 덤프가 선행해야 한다. MP4나 초기 PNG를 대신 넣어 판정하지 않는다.

**회수 후보(사용자 판단 필요)**: `outputs/mimic_reset_fixed_76_20260921/raw` 1.3 GB는 2.2의 188개 재렌더에 포함되므로 기능상 중복이지만 13.8의 근거라 삭제는 사용자 결정이다. `outputs/robomimic` 7.0 GB, `outputs/lerobot` 8.0 GB는 과거 실험이며 Mimic 무관이라 이전 정리에서 제외했다.

## 7. 다음 세션의 첫 명령

```bash
cd "/data/$USER/leisaac"
git status --short --branch
df -h /data /home
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
sed -n 1,60p docs/plans/remaining_causes_20260921/plan_r5.md      # 변경 요약과 표기 규약
sed -n 440,475p docs/plans/remaining_causes_20260921/plan_r5.md   # 2.0 구현 항목
sed -n 293,410p docs/plans/remaining_causes_20260921/plan_r5.md   # 과제 1 사전 등록
```

첫 구현은 2.0 `scripts/evaluation/attribute_source_demo.py`(500개 귀속·게이트 분해, CPU)이고 산출물은 `outputs/source_attribution_20260921/`·`docs/evidence/source_gate_decomposition_20260921.json`이다.
그다음 1.1 `scripts/evaluation/probe_gripper_close_offline.py`를 사전 등록(1.1.1~1.1.5)대로 만든다.
`preprocessor_overrides={"device_processor": {"device": "cpu"}}`를 반드시 준다. 기본값이 cuda라 GPU를 잡는다.

## 8. 피해야 할 해석 오류 (추가분)

- `obs/joint_pos_target[0]`은 이전 목표값이다. 닫힘 스텝 탐색은 인덱스 1부터 한다(`shard_010/demo_16`이 오탐 사례). `c == 1`은 `src:demo_6` 퇴화이며 오탐 방어와 다른 분기다.
- "소스 6개"를 데이터 다양성 결손으로 읽지 않는다. 게이트의 성질이다.
- 청크 닫힘 지표는 close 구간 한정 값만 판정에 쓴다. post-close는 보고만 한다.
- 13.8의 240스텝 RMSE 0.118 창은 닫힘(278)을 포함하지 않는다. 접근 개선의 근거이지 닫힘 근거가 아니다.
- 2.2 결론은 허용오차 0.05 체제에 한정된다. 엄격 76개 체제로 옮기지 않는다.

## 9. 미결·사용자 결정 사항

1. `/data` 확보 방법(다른 세션 산출물 정리 여부, 회수 후보 삭제 여부).
2. 2.2 진입 여부는 4.1 게이트 결과와 디스크로 자동 결정되지만, 회수가 필요하면 사용자 판단이 필요하다.
3. 카메라 `update_period` 최종 결정은 3.2 스모크 결과 후 조건부 권고(1/60 채택, 재생성은 미룸)로 기록한다.

## 10. CPU 실행 결과와 다음 시작점 (2026-09-22)

- **2.0 완료:** 500개 전수 귀속, 403/294/76 게이트, 허용오차 0.05 풀 188개·9계열을 재현했다. 실제 raw SHA와 diagnostics, strict 76 목록을 대조했다.
- **1.1 완료:** 두 모델 × raw/LeRobot 경로 × 검증 15개/학습 장면 1개의 8개 검사를 마쳤다. 수정 모델 close 적중률은 88.44%/87.33%, 기존 모델은 86.67%/81.78%로 네 검증 조합 모두 계획의 티처포싱 기준을 통과했다. 이 비율은 집기 성공률이 아니다.
- **입력 차이 확인:** raw/LeRobot 영상은 최대 화소 차 5/255 기준을 넘는다. 두 경로를 동등하다고 합치지 않는다. 기존 모델의 `src:demo_7` raw 입력은 74.67%로 계열별 기준 미달이다.
- **1.1.7 일부만 수행:** trace 상태 거리상 close 30개 중 17개가 남는다. 그러나 해당 스텝의 정책 PNG가 없어서 폐루프 모델 예측과 원인 2×2 판정은 미결이다.
- **보존:** 원본·파생 raw·모델 유지. Isaac·학습·실물 실행·파일 삭제 없음. CPU 산출물은 약 25 MiB이며 `/data` 여유는 실행 말미 약 4.1 GiB다.

근거 JSON:

1. [소스 귀속·게이트](evidence/source_gate_decomposition_20260921.json)
2. [닫힘 신호 8개 검사·폐루프 제약](evidence/grasp_close_signal_20260921.json)

다음은 **디스크 확보 후 1.2의 정책 영상 덤프 옵션 구현·검증 → 동일 장면 영상 수집 → 1.1.7 폐루프 프로브**다.
현재 evaluator에는 덤프 옵션이 아직 없다. 계획의 1.2 명령을 구현 전에 실행하지 않는다.
1.1 결과만 보고 원인을 도달 실패나 암기로 확정하지 않으며, 2.2 추가 학습은 판정 게이트와 14,450 MiB 디스크 게이트를 모두 통과한 뒤 진행한다.
