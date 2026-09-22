# SO101 Mimic → ACT 남은 원인 3건 조사 인계 (세션 이관용)

> 2026-09-22 **2.0·1.1 완료, 1.2 영상 수집 완료, 과거 실행과의 명령 재현은 실패**했다.
> 정책 PNG 202장을 확보했지만 close 잔존도 14/30으로 최소 15개보다 적어 **1.1.7 판정은 미결**이다.
> 같은 날 승인받은 중간 체크포인트 정리로 7.22 GiB를 회수했고 `/data` 여유는 약 15.1 GiB다.
> 아래 7절은 실행 전 기록이다. **13절에 팔·그리퍼 명령 분리 진단과 Diffusion 대안 준비를 추가했다.**
> 이어서 작업할 때는 12·13절과 [원인 인계 14절](HANDOFF_MIMIC_ACT_ROOT_CAUSE_20260921.md#14-남은-원인-cpu-조사-2026-09-22)을 먼저 읽는다.

이 문서는 2026-09-21 세션의 결과를 다른 세션으로 옮기기 위한 정리다.
기준 커밋은 이 문서를 포함한 커밋이며, 직전 결과 커밋은 `58d9ac9`다.
이전 인계 문서 `docs/HANDOFF_MIMIC_ACT_ROOT_CAUSE_20260921.md`의 13절이 지금까지의 확정 사실이고,
이 문서는 **그다음 작업(남은 원인 3건)의 합의된 계획과 시작 지점**을 담는다.

## 1. 한 줄 요약

- 초기 카메라 영상 결함은 확정됐고 생성기 수정과 파생 데이터 학습 비교까지 끝났다. 고쳐도 집기는 0/3이다.
- 다음 조사 3건(파지 닫힘 신호 학습 여부, 소스 다양성, 카메라 갱신 주기)의 계획을 Planner·Architect·Critic 5회 반복으로 합의했다.
- 계획은 `docs/plans/remaining_causes_20260921/plan_r5.md`이며 **실행 전 필수 결함은 없다**(마지막 지적인 2.2 디스크 게이트 값은 반영함).
- 인계 작성 당시에는 `/data` 여유가 4.4 GB(2026-09-22 새벽 기준)라 Isaac 단계와 2.2를 보류했다. 이후 용량 정리 결과는 11절, 정책 영상 부재 문제는 10절을 따른다.

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

1. `/data`는 승인받은 과거 중간 체크포인트 정리로 확보했다(11절). 다음 실행 직전에 여유 공간과 GPU 점유를 다시 확인한다.
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

## 11. 승인받은 중간 체크포인트 용량 정리 (2026-09-22)

사용자가 "중간 체크포인트까지 정리"를 승인한 뒤 삭제 목록을 별도 검토하고 실행했다.

| 대상 | 정리한 내용 | 회수 용량 |
| --- | --- | --- |
| `outputs/robomimic` | 최종·최저 validation loss 모델을 제외한 중간 `.pth` 60개 | 5.375 GiB |
| `outputs/lerobot` | 참조되지 않는 중간 체크포인트 9개에 속한 모델·학습 상태 파일 45개 | 1.848 GiB |
| 합계 | 파일 105개 | **7.223 GiB** |

보존한 것:

- 실험별 최종 모델, 최저 validation loss 모델, `last` 링크 대상, 문서·평가에서 참조하는 중간 모델.
- LeRobot의 평가 기록이 참조하는 중간 체크포인트 15개와 해당 학습 상태.
- 원본 시연·raw 500개·파생 raw 76개·현재 비교 모델·포트폴리오 영상·최근 진단 결과. 보호 목록 399개 파일의 크기·mtime은 전후 동일하다.
- config·로그·평가 JSON. 삭제한 체크포인트의 경로와 해시는 삭제 원장에 남겼다.

삭제 실행 직전 여유는 7.902 GiB, 직후는 **15.125 GiB**였다.
조사 시작의 4.1 GiB에서 삭제 직전까지 이미 용량이 변했으므로 그 증가분은 이번 정리 성과로 계산하지 않는다.
이번 삭제 파일 합계는 정확히 **7,755,505,990바이트**다. 다른 프로젝트·환경·캐시는 삭제하지 않았다.

현재 14,450 MiB 진입 기준은 충족하지만, 다른 작업이 공간을 사용할 수 있으므로 다음 실행 전에 다시 측정한다.
이번에는 용량 정리만 했으며 Isaac·추가 학습을 시작하지 않았다. 다음 순서는 1.2 영상 덤프 구현·수집 → 1.1.7 폐루프 판정이다.

**복구 제약:** 삭제한 중간 모델·optimizer 상태는 휴지통이나 Git에 남지 않는다. 정확한 복구에는 별도 백업이나 재학습이 필요하다.
남겨 둔 최종·최고 성능 모델의 해시는 삭제 전후 동일하다.

근거:

- [검증한 삭제 계획·보존 목록](evidence/storage_cleanup_20260922_plan.json)
- [삭제 결과·용량·보존 검사](evidence/storage_cleanup_20260922.json)

## 12. 정책 영상 수집과 재현 실패 분리 (2026-09-22 후속)

### 12.1 수행한 작업

1. evaluator에 opt-in PNG 저장을 추가했다. 기본값은 0이며 기존 제어 경로는 유지한다.
2. 계획 1.2 명령으로 1회 실행했다. 240~340스텝의 정책 입력 front/wrist **224×224 PNG 202장**, trace 420스텝, 실패 영상이 남았다.
3. 이전 실행과 초기 물리 상태·설정·명령을 비교했다. **초기 물리 상태 SHA와 설정 16개는 같지만 명령은 첫 스텝부터 다르다.** 420스텝의 최대 명령 차이는 0.073659 rad다.
4. 영상 저장을 끈 대조 실행도 수행했다. 이 실행 역시 이전과 초기 RGB가 다르고 최대 명령 차이는 0.061359 rad다. 차이는 PNG 저장 기능이 없어도 발생한다.
5. 세 실행의 초기 PNG·관절 상태를 수정 모델에 CPU로 다시 입력했다. 각 입력의 반복 예측은 비트 단위로 같고, **각 실행에 저장된 첫 30개 명령을 모두 정확히 재현**했다.

### 12.2 확인된 사실과 미결 사항

| 비교 | 초기 front 평균/최대 화소 차(0~255) | 초기 wrist 평균/최대 차 | 첫 30개 명령 최대 차 |
| --- | --- | --- | --- |
| 이전 → PNG 저장 실행 | 0.104964 / 6 | 0.022375 / 3 | 0.006778 rad |
| 이전 → 저장 끈 대조 | 0.078577 / 4 | 0.020468 / 3 | 0.003164 rad |
| PNG 저장 실행 → 대조 | 0.096354 / 6 | 0.018860 / 3 | 0.003614 rad |

첫 청크의 차이는 저장된 입력 RGB 차이와 결정적인 ACT 응답으로 재현된다. 이 검사에서 CPU 모델 계산의 비결정성은 관찰되지 않았다.
PNG 저장은 240스텝부터이므로 1스텝부터 발생한 차이를 저장 시점의 디스크 쓰기로 설명할 수 없다.
다만 초기 물리 상태 해시는 재질·조명·렌더 히스토리까지 보증하지 않는다. **RGB를 다르게 만든 구체적인 렌더 설정·히스토리·난수 요인은 아직 분리하지 못했다.**
이 차이가 전체 집기 실패의 원인이라고 단정하지 않는다.

현재 PNG 입력을 시연과 같은 스텝으로 맞추면 close 30개 중 관절 차이가 0.2 rad 이내인 것은 **14개**다. 이전 trace의 17개를 이번 실행의 표본 수로 가져오면 안 된다.
과거 명령 재현 실패와 최소 잔존 15개 미달이 겹쳐, **폐루프 적중률·암기 대 도달 실패의 2×2 판정은 하지 않았다.**
프로브 CLI도 strict gate에서 종료 코드 1로 중단했고, `closed_loop_fixedmodel_raw.json`은 생성하지 않았다.
표본 하한을 낮추거나 이전 reference를 이번 결과로 덮어쓰지 않았다. 새 학습도 시작하지 않았다.

### 12.3 산출물과 검증

저장소 루트 `/data/$USER/leisaac` 기준:

- PNG·trace·영상: `outputs/grasp_signal_probe_20260921/rollout_demo9_imagedump/`
- 영상 파일: `outputs/grasp_signal_probe_20260921/rollout_demo9_imagedump/rollout_001_failure.mp4`
- 저장 끈 대조: `outputs/grasp_signal_probe_20260921/rollout_demo9_nodump_control/`
- 입력 비교 상세: `outputs/grasp_signal_probe_20260921/close_window_input_comparison.json`
- Git에 보존한 근거: [폐루프 재현 진단](evidence/closed_loop_reproduction_20260922.json). 입력 해시·표본·세 실행 비교·CPU 재현·영상 파일 해시를 포함한다. 영상과 PNG 본체는 로컬 `outputs`에 있고 Git에는 넣지 않았다.

전체 단위 테스트 **189개 PASS**. 별도 리뷰에서 저장 시점/크기, 기본 off, gate 차단, 관측 정렬, 동일 표본 비교, 파일 변경 검출을 확인했다.
원본/파생 raw 81개 파일의 크기·mtime과 수정 모델 SHA `ebed7e54…`를 재확인했다. 추가 삭제·학습·실물 동작은 없다.
증거 생성 시 `/data` 여유는 약 **15.11 GiB**다.

### 12.4 다음 시작점

1. 동일 초기 상태에서 렌더 설정·히스토리·카메라 갱신 시점을 기록하고, 초기 RGB가 실행마다 바뀌는 요인을 분리한다. 같은 입력의 CPU 재현은 이미 통과했으므로 모델 학습부터 반복하지 않는다.
2. 과거 launch의 trace를 정확히 복원할 수 없는 경우 기존 근거는 보존한다. 새 기준선·반복 실행·관측 기반 재현 검사로 설계를 바꾸려면 계획에 변경 사유와 판정 범위를 먼저 기록한다. strict gate를 조용히 완화하지 않는다.
3. 계획의 2.1·3.2는 별도 조사로 남아 있다. 2.2 학습은 미해결 재현 전제와 4.1 분기를 정리한 뒤, 디스크 14,450 MiB 기준을 다시 확인하고 진행한다.

재검사 명령(새 출력 경로 사용):

```bash
cd "/data/$USER/leisaac"
CUDA_VISIBLE_DEVICES='' "$HOME/miniforge3/envs/lerobot/bin/python" \
  scripts/evaluation/compare_close_window_inputs.py \
  --rollout-dir outputs/grasp_signal_probe_20260921/rollout_demo9_imagedump \
  --output outputs/grasp_signal_probe_followup/close_window_recheck.json
```

현재 데이터에 대한 예상 결과는 **strict 재현 실패, exit 1, close 잔존 14개**다. 프로그램 장애나 성공 결과로 바꿔 해석하지 않는다.

## 13. 팔·그리퍼 명령 분리와 대안 모델 준비

### 13.1 이미 성공한 물리 제어 대조군

`outputs/evaluation/control_diagnosis_20260921/shard009_recorded_target/evaluation.json`을 재확인했다.
현재 진단과 같은 `shard_009/demo_9`, 초기 물리 상태 `375aaad2…`, velocity target 0, task effort, stable_release_v2 조건에서
시연의 `joint_pos_target[t+1]` 재생은 **642스텝에 안정 놓기를 달성**했다. 최대 큐브 상승은 약 0.11545m, clipping은 0회다.
따라서 이 장면에서 올바른 명령을 물리적으로 실행하는 것 자체가 불가능하지는 않다. 이 성공은 ACT 자율 성공이 아니다.

### 13.2 새 진단 구현과 실행 전제

[사전 등록](plans/remaining_causes_20260921/action_intervention_20260922.md)에 따라 같은 evaluator에서 네 조건을 비교한다.

| 조건 | 팔 5축 | 그리퍼 | 의미 |
| --- | --- | --- | --- |
| policy | ACT | ACT | 기준선 |
| teacher_gripper | ACT | 시연 | 그리퍼 명령을 교체 |
| teacher_arm | 시연 | ACT | 팔 명령을 교체 |
| teacher_all | 시연 | 시연 | 새 교체 경로의 양성 대조 |

각 seed(4101, 4102)에서 teacher_all의 안정 놓기 성공과 명령 감사를 먼저 통과해야 나머지 조건이 실행된다.
전체 675스텝 또는 성공 시 종료한 prefix의 원래 정책 명령·교체 후 명령·제한 적용 명령을 기록한다.
교체 채널은 raw target[t+1]와, 비교체 채널은 ACT 명령과 최대 차이 0인지 검사한다. 원본·모델·구현 파일 해시도 실행 전후 확인한다.
두 반복의 결과가 다르면 채널 원인을 확정하지 않는다. 초기 RGB의 launch 간 차이는 혼란변수로 남는다.

관련 구현:

- `scripts/evaluation/action_intervention.py`: 명령 정렬·채널 교체.
- `scripts/evaluation/lerobot_act_so101.py`: 기본 policy 경로 유지, opt-in 진단 옵션·trace 확장.
- `scripts/evaluation/run_action_intervention.py`: 양성 대조 gate, 순차 실행, 전구간 감사, 영상·로그·매니페스트 기록.

CPU 사전 검사와 전체 **197개 단위 테스트** 및 별도 코드 리뷰는 통과했다.
이 시점에는 다른 프로젝트의 `simulate_restaurant_mobile.py`가 GPU를 점유해 **새 진단 시뮬레이션을 시작하지 않았다.**
다른 프로세스를 종료하지 않았고, 자동 대기·예약 실행 프로세스도 만들지 않았다.
이 기록은 실행 준비 완료이지 새 집기 성공 결과가 아니다. [실행 준비 근거](evidence/action_intervention_readiness_20260922.json)를 참조한다.

GPU가 유휴 상태일 때 다음 명령으로 실행한다. `CUDA_VISIBLE_DEVICES=''`를 붙이면 하위 Isaac의 GPU까지 가려지므로 붙이지 않는다.

```bash
cd "/data/$USER/leisaac"
"$HOME/miniforge3/envs/lerobot/bin/python" scripts/evaluation/run_action_intervention.py
```

기본 산출물은 `outputs/action_intervention_20260922/`다. 기존 디렉터리가 있으면 덮어쓰지 않고 중단한다.
중단된 실행을 새 경로에서 다시 할 때는 이전 실패·부분 결과도 보존하고 함께 보고한다.

### 13.3 ACT는 최종 선택으로 고정하지 않음

사용자가 ACT 외 학습 방법도 비교할 것을 제안했다. 우선 대안은 **Diffusion Policy**다.
현재 LeRobot 환경에 Diffusion Policy와 diffusers가 설치돼 있고, 수정 76개 LeRobot 데이터·학습/검증 분할을 재사용할 수 있다.
Mimic 증강 데이터가 ACT만을 위한 데이터인 것은 아니다.

다만 현재 학습 runner·서버·오프라인 평가·checkpoint 검증의 ACT 고정 가정을 정책별로 분리해야 한다.
Diffusion은 state/action MIN_MAX 정규화, 관측/행동 시간창, 확률적 추론을 별도로 다뤄야 한다.
구현 존재와 데이터 호환 가능성을 확인한 것이며 **GPU 학습 가능 배치 크기·자율 성능·ACT 대비 우위는 아직 검증하지 않았다.**
공식 근거는 [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)와
[LeRobot 학습 예제](https://github.com/huggingface/lerobot/blob/main/examples/tutorial/diffusion/diffusion_training_example.py)다.
