# SO101 자율 집기 최소 성공 계획과 진행 기록

## 최종 성능 결과

2026-09-28: **ACT6축 자율 집기·들어 올리기10/10, 박스 안 안정 놓기8/10.**
선별과 분리한 새 launch10회 전체를 집계해 최소 고정 장면 성능 목표를 달성했다.
단일 증강 시연 맞춤 학습이며, 위치 일반화·4cm 실물·실물 실행·강화학습 성공은 아니다.
원본·모델·성공/실패 영상은 보존했다. 독립 산출물 검증은 PASS다. 코드 리뷰의 검증 guard3개를 보완했고, 추가 정적 진단도 통과했다. 최신 보완 코드의 독립 재승인은 아직 미확보이며 성능 달성과 전체 작업 완료를 구분한다.

## 목표

ACT가 팔 5축과 그리퍼를 모두 제어하여 같은 학습 장면에서 집고 박스 안에 안정적으로 놓는다.
최종 모델을 고정한 뒤 새 프로세스 10회 중 `stable_release_v2` 성공 8회 이상을 목표로 한다.
교사 명령 재생은 자율 성공에 포함하지 않는다. 위치 일반화·4cm 실물 큐브·실물 실행·강화학습 성공을 뜻하지 않는다.

## 왜 이 순서인가

[이전 접촉 진단](ARM_HANDOFF_CONTACT_DIAGNOSIS_20260928.md)에서 같은 적용 명령을 재생하면 성공과 실패의 물리 궤적이 각각 재현됐다.
성공 데이터가 있다는 것과 그 데이터를 학습한 정책이 성공한다는 것은 다르다.
많은 시연을 다시 늘리기 전에 성공 시연 하나를 정책이 재현할 수 있는지 확인한다.
이는 의도적인 단일 시연 맞춤 학습이다. 과적합 방지나 데이터 증강의 효과 검증이 아니다.

## 실행 순서와 분기

| 단계 | 수행 | 통과 조건 |
|---|---|---|
| 1. 입력 계약 | train episode20만 선택, 675frame, state[t]와 target[t+1], sampler·정규화·영상 SHA 검사 | 원본 정렬 오차0, 675개 샘플만 사용 |
| 2. 학습·선별 | 아래 후보를 순서대로 새 학습, 후보별 새 launch 3회 평가 | 첫 3/3이면 후보 탐색 종료 |
| 3. 최종 평가 | 모델 SHA와 queue 잠금, 선별에 쓰지 않은 새 launch 10회 | ACT 6축 자율 안정 놓기 ≥8/10 |
| 4. 검증·기록 | 원본 SHA 전후 대조, 영상·trace 감사, 독립 리뷰, 결과·코드 커밋 | 성능 조건과 최종 품질 gate 모두 충족 |

후보는 다음 순서이며 조건을 실행 중 몰래 바꾸지 않는다.

1. `p3k`: VAE 사용·dropout0.1, 3,000step.
2. `p10k`: 같은 조건으로 새 10,000step.
3. `d10k`: VAE 미사용·dropout0, 새 10,000step.

공통: ACT dim256, feedforward1024, chunk30, batch8, seed43, lr1e-4/backbone1e-5.
각 후보는 queue30으로 평가하고 3/3이 아니면 queue1을 별도 평가한다.
전체 후보를 평가해도 3/3이 없지만 성공이 있으면 성공수→lift수→파지 구간 오차→사전 순서로 하나를 선택한다.
전부 성공0이면 최종10회를 반복하지 않는다. 파지 구간 샘플링 또는 실제 재실행으로 얻은 복구 관측·행동 등 새 학습 개입을 검토하고 계획 revision을 분리한다.
최종이 8/10 미만이면 목표 미달이다. 결과를 보존하고 새 개입·평가 root를 사용한다.

## 고정 조건과 판정

- 초기 장면: 원본 `shard_009.hdf5/demo_9`; train 매핑은 aggregate24→episode20.
- action은 기록된 다음 목표값이다. 관측된 다음 관절 위치로 대체하지 않는다.
- 정규화 통계는 기존 train61, 영상 통계는 ImageNet이다. 단일 시연만으로 만든 통계가 아니다.
- 60Hz, 최대675step, reset render4, front/wrist 입력224, 저장 영상1280×480/60fps.
- 원래 관절 한계·물리·fixed gripper effort0.06666666269302368·성공 기준을 유지한다.
- 시뮬 큐브 시각 크기는3cm이며, 이 결과를 사용자의4cm 실물 조건과 같다고 하지 않는다.
- 최초2cm lift와 박스 안 안정 놓기를 별도로 기록한다. loss·offline RMSE·그리퍼 닫힘만으로 성공을 판단하지 않는다.
- screening seed4201..4203, 최종 seed4301..4310. 고정 장면의 새 launch seed이며 위치 holdout은 아니다.

## 초기 검증 기록

OMX `ralplan`의 Architect→Critic 검토에서 r2 계획·명세를 승인받고 `ultragoal`로 실행 중이다.
실제 episode20의675frame 선택과 원본 t+1 정렬을 검증했다. 전체 imitation_learning unittest234개가 통과했다.
출력 보호·완료 stage 감사·선별 종료 조건 보완 후 독립 구조 재검토는 CLEAR다.
독립 code-reviewer 생성은 native thread limit으로 실패했다. 구조 검토와 테스트로 전체 코드 승인을 대신하지 않으며, 최종 품질 gate는 미완료다.

v1의3,000step 학습은 완료했다. v1·v2는 계약 보완 중 만든 부분 실행으로 보존하며 STOP 경계에서 후속 평가를 하지 않는다.
최종 보완 실행기는 v3 root를 사용한다. 아래 기록은 준비 시점부터 순서대로 남겼으며 최신 결과는 첫 절과 최종10회 표를 따른다.

### 2026-09-28 13:30 KST 추가 결과

`p3k` 학습은 완료했고 원본·통계 계약을 통과했다. 모델 SHA는 `25dca5d17fffb32a63d958957dceab99b82d13f9dd0aceb1d382e9cf9a8fdc0a`다.
저장 관측 self-fit RMSE는 전체0.0439548rad, 파지 구간0.0356358rad다. 자율 성공을 뜻하지 않는다.

| 후보 | queue | 선별 성공 | lift | 감사 |
|---|---|---|---|---|
| p3k | 30 | 0/3 | 0/3 | 3회 모두 PASS |
| p3k | 1 | 0/3 | 0/3 | 3회 모두 PASS |

첫 q30 실패의 닫힘 기준0.2rad 도달은 시연285step 대비 정책510step이었다. 초기 팔 경로 오차도 있었다.
이는 관측된 지연·이탈이며 단일 원인으로 단정하지 않는다. `p10k` 학습을 시작했다. 최종10회 평가는 아직 시작하지 않았다.

독립 Critic은 모든 사전 후보가0일 때의 조건부 후속 개입을 검토했다.
phase-weighting만으로는 이탈한 관측을 추가하지 못하고, 현 ACT는 generic sample-weighting의 `reduction="none"` API를 지원하지 않는다는 반론이 있다.
후속 분기가 필요하면 phase/trajectory ambiguity를 먼저 측정하고, exact-evaluator의 실제 성공 재실행 관측·6축 행동 또는 검증된 복구 궤적을 우선 검토한다.
기존 관측·행동의 중복이나 success attr만 있는 데이터는 채택하지 않는다. 기존 recovery merger는 next-state action 계약이므로 그대로 쓰지 않는다.
이 내용은 조건부 제안이며 아직 새 계획 승인·데이터 생성·학습 개입을 하지 않았다.

### 2026-09-28 13:43 KST 선별 완료

`p10k/queue30`은 seed4201·4202·4203에서 안정 놓기 **3/3**에 성공했다. 교사 source가 없는 ACT6축 자율 평가이며 모두 명령·물리·effort·trace·입력 PNG·영상 감사를 통과했다.
성공 step은597·621·632, 최대 lift는0.1119843·0.1114252·0.1106041m다.
`p10k` self-fit RMSE는 전체0.0292319rad, 파지0.0240350rad로 줄었다. 이것만으로 성공을 판정하지 않았다.

모델 SHA `9d2a136f3aabf49fdd5761d31b469f2a1961ef65f8e0c5843e5d5bd489c24dce`, selection SHA `e5fa585cfc632b13f4eb38ecee4c8056400782144d0542267c0aaaf9829a9a6d`로 고정했다.
사전 중단 규칙에 따라 p10k/queue1·d10k는 실행하지 않았다. 최종 seed4301..4310 평가를 시작했으며, 선별3/3을 최종 성공률로 대체하지 않는다.

추가 CPU 반례 검사: 30frame 이상 떨어진 시연 구간에서 관절 최대거리≤0.02rad인 쌍을 검사했으나 쌍0개였다. 미래30step chunk를 비교할 수 있는646frame 기준이다.
RGB 유사성은 측정하지 않았으며 full-observation phase ambiguity의 정상·비정상 증명이나 검증된 실패 원인으로 해석하지 않는다.

### 최종 10회 결과

| seed | 안정 놓기 | 최초 lift step | 종료 step | 결과 |
|---|---|---|---|---|
| 4301 | 실패 | 443 | 675 | low_after_lift |
| 4302 | 성공 | 369 | 668 | success |
| 4303 | 실패 | 492 | 675 | lifted_not_completed |
| 4304 | 성공 | 349 | 595 | success |
| 4305 | 성공 | 360 | 607 | success |
| 4306 | 성공 | 380 | 624 | success |
| 4307 | 성공 | 353 | 615 | success |
| 4308 | 성공 | 375 | 671 | success |
| 4309 | 성공 | 362 | 608 | success |
| 4310 | 성공 | 351 | 607 | success |

성공8/10·lift10/10. 각 값은 최종 manifest의 감사 결과와 대조했다.
모든10회는 새 프로세스, 같은 잠금 모델·queue30, ACTpolicy source만 사용했다. 교사 행동·추가 힘·변경된 성공 기준은 없다. clip0, fixed effort0.06666666269302368다.
완료 stage를 재사용하는 별도 감사 패스를 다시 수행해 모델·선별·명령·로그·실제 trace/입력 PNG/영상·원본 SHA가 동일함을 확인했다. 추가 시뮬레이션 실행으로 성적을 바꾸지 않았다.

3,000step 모델은0/6, 10,000step 후보는 선별3/3·최종8/10이었다. 이 고정 단일 시연 시험에서는 학습량 증가 후 실제 자율 성능이 개선됐다.
이를 데이터 수 증가·데이터 증강·강화학습의 효과나 전체 작업의 근본 원인이 하나로 밝혀졌다는 주장으로 확대하지 않는다.

후속 작업은 실패2회의 놓기 지연·경로 분석과 새로운 위치에서의 별도 평가다. 단일 시연 성공을 복수 위치 정책으로 확장하는 과정이 남아 있다.

### 검증 guard 보완과 실행 버전 분리

성공 실험의 실행 소스는 로컬 커밋 `4841bdad798cc5245f52d0a580ac72936a719532`에 보존했다.
독립 code-reviewer는 해당 결과의78개 artifact SHA·6,345개 trace step·10개 영상을 검사해 성능 근거를 확인했다.
동시에 향후 실행의 원본 절대 SHA·실패675step 강제·전체 학습 설정/유한 loss 검사3개 보완을 요청했다.

보완한 실행기는 RAW와 선택 입력8개의 사전 SHA를 고정하고, RAW와 reference를 함께 바꿔도 거부한다.
autonomous audit에서 실패는675step을 모두 수행해야 한다. 기존 historical audit API는 유지했다.
KL10을 명령에 명시하고 seed·lr·KL·구조·subset·외부 전송 설정 등 사전 등록 config를 검증한다. loss 로그 행 수와 유한성도 stage result에 기록한다.
기존 p3k·p10k의 실제 config와 로그를 CPU로 검증했으며 각각30개·100개 loss 행 모두 유한했고 KL10 등 설정도 일치했다.
새 회귀 테스트를 포함한 전체238개가 통과했다. 이 보완을 과거 시뮬레이션 실행에 소급 적용했다고 주장하지 않는다.

보완 실행기는 새 `outputs/act_single_demo_checked_20260928` root를 기본값으로 사용한다.
완료된 v3 root는 source hash가 다른 최신 코드로 resume하지 않는다. 새 guard를 쓰는 실험은 새 root에서 준비한다. 이번 검증 때문에 모델을 다시 학습하거나 최종 평가를 반복하지 않았다.

### 독립 검증과 정적 진단 추가 보완

독립 verifier는 최종10회 전체 분모·6,345step trace·영상10개·정책 입력PNG1,800개·초기PNG20개·보호 원본·실행 소스6개 SHA를 대조해 PASS를 반환했다.
정책 source만 사용하고 teacher source가 없으며, 모든 step의 requested/policy/applied action이 같고 clip0·fixed effort·물리 계약이 유지됨을 확인했다. 영상 내용의 사람 판독이나 신규 실물 검증은 수행하지 않았다.

독립 code-reviewer는 앞의 guard3개 해소와238개 테스트를 확인했지만 diagnostics 도구 미노출 때문에 COMMENT로 승인을 보류했다. 이를 APPROVE로 바꾸지 않고 OMX G005 보완 단계로 기록했다.
기존에 설치된 Pyright1.1.414와 Pylance typeshed를 찾아 Python4파일을 검사했다. HDF5 Group/Dataset·LeRobot 단일 sample·metadata/후보 Optional 경계를 명시 검사하고, 중복 후보 선택 호출을 제거했다. 진단 무시·Any cast·새 dependency는 추가하지 않았다.
추가 반례2개를 포함한 전체240개 unittest, 실제 prepare/check-only, Pyright4파일 error0/warning0, diff-check가 통과했다.
OMX의 설치된 `lsp_diagnostics` handler도6파일에서 실행했으나 TypeScript 전용이어서 no-tsconfig로 skip됐다. 이 skip를 Python 검사 통과 근거로 사용하지 않으며, Python은 별도의 Pyright 결과로 확인했다.

현재 최신 보완의 native code-reviewer/architect 재검토 호출은 `agent thread limit reached`로 실패했다. verifier의 성능 PASS와 이전 구조 CLEAR를 최신 코드 승인으로 대체하지 않는다. 독립 최종 승인·원격 push·aggregate goal 완료는 보류한다.

## 파일과 실행

아래는 향후 새 실험을 재현하는 명령이다. 완료된 v3 성능 결과를 보려면 영상·manifest 경로를 열며 다시 실행하지 않는다.
원본 데이터·모델은 읽기 전용으로 사용하고 기존 출력은 덮어쓰지 않는다.

```bash
cd "/data/$USER/leisaac"
"$HOME/miniforge3/envs/lerobot/bin/python" scripts/imitation_learning/run_single_demo_act.py --stage prepare --check-only
"$HOME/miniforge3/envs/lerobot/bin/python" scripts/imitation_learning/run_single_demo_act.py --stage prepare
"$HOME/miniforge3/envs/lerobot/bin/python" scripts/imitation_learning/run_single_demo_act.py --stage screen --resume
# selection 잠금과 screening 완료 후에만 실행
"$HOME/miniforge3/envs/lerobot/bin/python" scripts/imitation_learning/run_single_demo_act.py --stage final --resume
```

- 실행기: `scripts/imitation_learning/run_single_demo_act.py`
- 결과: `outputs/act_single_demo_success_v3_20260928/manifest.json`
- 단계 로그: 동일 root의 `logs/`
- 모델: 동일 root의 `<candidate>/model/checkpoints/<step>/pretrained_model/`
- 영상·전step trace·정책 입력 PNG: 동일 root의 `<candidate>/screen_q<queue>_seed<seed>/` 및 `final_q<queue>_seed<seed>/`
- 계획·명세: `.omx/plans/so101-act-success-r2.md`, `.omx/specs/so101-act-success-test-r2.md`
- 내구성 진행 기록: `.omx/ultragoal/goals.json`, `.omx/ultragoal/ledger.jsonl`
- 추적된 성능/계약 근거: [evidence/single_demo_act_20260928.json](evidence/single_demo_act_20260928.json)

신규 단계는 /data8GiB·home1GiB 여유 및 GPU idle을 확인한다. 실험 output상한4GiB다.
중단은 해당 root에 STOP 파일을 두면 단계 경계에서 이뤄진다. 진행 중 결과는 보존하며 다른 GPU 작업을 종료하지 않는다.
실물 팔과 리더는 이 단계에서 사용하지 않는다.
