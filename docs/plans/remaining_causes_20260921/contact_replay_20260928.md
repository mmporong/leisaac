# 양성 대조 파지 재현성 검사 (2026-09-28)

## 전환한 이유

팔 prefix 진단의 첫 teacher_all은 성공했지만 teacher_arm은 no_lift여서 후속 조건을 중단했다.
이전 성공 두 실행과 새 실패의 팔 목표각은 같고, 그리퍼 명령이 0.2rad 아래로 처음 내려간 step도 283으로 같다.
그리퍼 명령은 약간 다르며 접촉 이후 큐브 위치가 벌어졌다. 이미지 차이만의 원인이나 물리 오류로 단정할 수 없다.

## 질문과 고정 조건

실제 적용 명령을 고정해도 집기 결과가 바뀌는가?
기존 두 성공 trace와 새 실패 trace의 applied_action을 각각 두 seed에서 재생한다(3×2=6회).
모든 실패·부분 결과를 보존하며 성공할 때까지 반복하지 않는다.
각 source의 실제 trace 길이 639/642/675를 horizon으로 쓴다. 명령 연장·마지막값 반복·추가 settling은 없다.
source trace를 끝까지 보유한 것이므로 미래 명령을 사용하는 비자율 진단이다.

초기 raw·physical SHA·stable_release_v2·joint limit·dt·fixed effort는 앞선 진단과 같다.
임시 HDF5나 성공 label 변경 없이 원본 초기 상태를 그대로 사용한다.
원본 모델은 보존하지만 이 실행에서 모델을 로드하거나 ACT를 추론하지 않는다.
전용 기록 명령 서버를 `--server-script`로 연결하고 `--checkpoint`에는 모델이 아닌 진단 manifest 경로를 준다.
기존 evaluator의 teacher_arm 모드에서 팔은 raw target[t+1], 그리퍼는 기록 서버 명령이다.
기록한 팔 명령이 raw와 정확히 같은지 사전 검사하고, 실행된 6축 전체 applied_action이 source와 같은지 감사한다.
evaluator가 반환하는 policy_action은 이 조건에서는 학습 정책이 아니라 기록 서버 응답이다. 결과·manifest·로그에 이를 명시한다.

## gate·검증·해석

- source의 evaluation·trace SHA·clip 0·fixed effort·initial physics·성공 판정과 길이를 검사한다.
- 부모 manifest의 고정 SHA·run identity와 부모 audit에 저장된 trace/evaluation SHA·길이·성공 여부도 대조한다.
  사전 등록 기대값은 [source pins](../../evidence/contact_replay_sources_20260928.json)에 고정한다.
  현재 파일의 SHA를 새로 계산하는 것만으로 역사 근거를 승인하지 않는다.
- source raw와 actual applied arm[t]의 target[t+1] 정렬을 전 step 확인한다.
- 실행 후 같은 물리 초기 상태·criteria·dt·effort·joint limit·모든 적용 명령·원본·모델·코드 SHA를 감사한다.
- 각 실행의 입력 PNG/영상도 앞선 prefix 진단과 같은 규약으로 검사한다.
- source보다 조기 성공하면 성공까지 실행한 prefix만 감사한다. horizon에서 실패하면 lift와 안정 놓기를 구분해 보고한다.
- 같은 명령에서도 두 seed 결과가 다르면 접촉/실행 재현성 영향의 근거다. 정확한 물리 원인은 아직 별도 확인이 필요하다.
- 성공 명령은 일관되게 집고 실패 명령은 일관되게 못 집으면, 이 작은 명령 차이에 따른 파지 민감도를 조사한다. 관측만의 단일 원인이나 일반화 성공률로 해석하지 않는다.
- 원본 학습 데이터·성공 label·모델은 수정하지 않는다. 이 진단에서 새 학습은 하지 않는다.
