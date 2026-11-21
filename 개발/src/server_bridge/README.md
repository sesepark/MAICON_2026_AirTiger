# server_bridge

## 미션별 server_bridge가 담당해야 하는 부분

아래는 박세환이 생각했을 때, 현재 5개 노드 구조에서 이 노드가 반드시 맡아야 하는 기능이다. 다른 노드와 담당이 겹치거나 판단이 필요한 부분은 여기에 적지 않았다. 그런 부분은 대회 중 회의·토론을 통해 정한다.

### 미션 1 — 전술명령 해석

- 경기 서버에서 전술명령 원문을 받아 supervisor에 전달한다.
- 요청받은 해석 결과를 공식 제출 형식으로 변환해 경기 서버에 보고한다.

### 미션 3 — 전장객체 인식

- 요청받은 섹터별 객체 보고를 공식 제출 형식으로 변환해 경기 서버에 전송한다.

### 미션 5 — 목표거점 진입

- 경기 서버에서 실제 M5 목표거점을 받아 supervisor에 전달한다.

### 미션 6 — 전장 위험도 분석

- 요청받은 위험도 순위를 공식 제출 형식으로 변환해 경기 서버에 보고한다.

### 미션 공통

- 준비완료 요청의 전송과 경기 서버의 상태 조회를 담당한다.
- 경기 서버의 실제 응답·오류·통신 결과를 전달한다. 보고 접수와 채점 성공을 구분한다.
- M2·M4 타격 및 M5 점령의 판정 보고를 임의로 만들어 보내지 않는다.

## 역할 및 구현 상태

준비완료·명령·상공 영상·보고의 HTTP 통신과 ROS 변환을 담당한다.

초기 생성 시점(2026-11-15)에는 패키지 껍데기만 작성했다. 최신 구현 상태는 YAML의 해당 노드 항목과 실제 코드로 확인한다.

## 2026-11-18 구현 상태

실행 프로그램 `server_bridge_node`가 있다 (`ros2 run server_bridge server_bridge_node`). 팀장 결정으로 supervisor 쪽에서 만들었다.
경기 서버를 흉내 낸 프로그램(`임시_테스트_파일/supervisor_가짜상대/fake_world.py --http`)과 한 경기를 돌려 봤고, 실제 서버에는 2026-11-18에 `GET /me`만 보내 봤다 (본 경기 포트 · 시험 포트 모두 200, 우리 팀 단계 `idle`). **실제 서버로 한 경기(준비완료 · 임무 받기 · 보고)를 돌려 보지는 않았다.** API는 `공식_자료/텍스트/경기서버연동API.md`(2026-11-04) 기준이고, 바뀌면 `constants.py`만 고친다.

**포트** (참가팀 포털에서 확인, [포털에서_확인한_것.md](../supervisor/docs/포털에서_확인한_것.md)): 본 경기는 서버의 8000, 팀 테스트는 같은 주소의 시험 포트 8100이다. `port_mode`가 `auto`(기본)면 두 포트의 `GET /me`를 보고 우리 팀의 경기가 열린 쪽(단계가 `waiting` · `ready` · `running` · `reporting`)을 스스로 고른다. 둘 다 열려 있으면 본 경기 쪽. 한 번 `ready` 이후를 보면 그 실행 동안은 바꾸지 않는다. 예전의 `/api/rehearsal/`은 서버에서 없어져 뺐다.

- `server_bridge/node.py`: ROS 연결. 아래 다섯 가지를 한다
  - `GET /me`를 계속 물어 `/team/server/state`로 내보낸다 (1초마다, 준비 · 보고 단계에서는 0.3초마다). 세 번 연달아 실패하면 `connected = false`
  - `/team/server/set_ready` → `POST /ready` 한 번. 서버가 이미 준비완료로 알고 있으면(`GET /me`) 다시 보내지 않고 `ACCEPTED`
  - `GET /order` · `GET /target`을 서버가 줄 때까지 걸어 두고(`wait`, 기본 60초), 받는 즉시 `/team/server/command`로. `command_id`는 내용으로 만든 값이라 다시 받아도 같다
  - `/team/server/submit_report` → M1 · M3 · M6를 한 번 보내고 서버의 답 그대로 (`ACCEPTED` / `REJECTED` / `UNKNOWN`). 보고값이 아닌 항목(모르는 구역 · 종류 · 명칭)은 빼고 보낸다
  - 진행 중 `GET /drone`을 0.5초마다 물어 새 장만 `/team/server/uav_image`로
- `server_bridge/client.py`: HTTP 한 번을 보내고 상태 코드 · 시한 · 예외를 처리한다. 시한은 창구마다 따로, 넘기면 두 배로 늘린다 (기본 4초, 상한 12초)
- `server_bridge/reports.py`: ROS 메시지의 내용 ↔ 서버 JSON. `server_bridge/constants.py`: 경로 · 단계 · 보고값 이름

계약과 다르게 한 것 (까닭)

- 보고 응답을 2초에 끊지 않고 서버의 실제 답을 기다린다 (창구 시한까지, 최대 12초). 2초에 끊으면 답이 2초보다 늦는 망에서 보고가 하나도 성공으로 확인되지 않는다
- 요청의 `run_id`가 지금 것과 달라도 보낸다 (보고를 잃는 쪽이 더 나쁘다). 기록에만 남긴다

### 설정

로봇의 `~/team/secrets.json` (저장소에 올리지 않는다. 형식은 `개발/robot/secrets.example.json`)

- `server_url`: 예 `http://<서버 IP>:8000` (환경변수 `MAICON_SERVER_URL`로도 준다)
- `mission_code`: 임무코드 (환경변수 `MAICON_MISSION_CODE`). 기록 · 출력 어디에도 찍지 않는다

`~/team/server_bridge.json` (없어도 된다)

- `port_mode`: `auto`(기본 — 본 경기 포트와 시험 포트 중 우리 팀의 경기가 열린 쪽) · `match`(`server_url` 그대로) · `test`(시험 포트만). `test_port`: 기본 8100
- `drone`: 드론 사진을 받을지 (기본 `true`). `drone_camera`: 경기 때는 비운다. 연습 기간 시험은 `main` · `test1` · `test2` — 적어 두면 경기 단계와 상관없이 그 카메라의 사진을 받는다 (API 문서 4단계: 테스트 모드가 꺼진 경기에서는 서버가 이 값을 무시한다)
- `wait_s`(60) · `me_period_s`(1.0) · `timeout_s`(4.0) · `drone_poll_s`(0.5)
- `record`(true): 서버와 주고받은 것을 `<기록 폴더>/server_bridge/http.jsonl`에 남긴다. `drone_save_s`(10): 드론 사진을 이 간격으로 한 장씩 `drone/`에 남긴다

### 시험

```bash
cd 개발/src/server_bridge && PYTHONPATH=. python3 -m unittest discover -s test -p 'test_*.py'     # 보고 JSON 의 모양 (ROS 없이)
임시_테스트_파일/supervisor_가짜상대/run_team_node.sh place 300 240 B-1                           # 컨테이너에서 한 경기
```

## 필수 작업 지침

- 작업 전 [공통 규칙](../../AGENTS.md), [통신 계약](../../../노드_관리/interfaces.yaml), [타입 정의](../team_interfaces/)를 읽는다. 노드 id는 `server_bridge`이다.
- 미션 판단·목표 선택·물리 제어는 supervisor 등 담당 노드에 맡긴다.
- 응답은 접수 대기가 아닌 실제 서버 결과로 반환한다. 타임아웃은 성공 여부 미상으로 구분한다.
- 명령 대기·영상 수신·보고가 서로의 마감을 막지 않도록 한다. 재시도·중복 처리·전체 보고 교체 의미는 YAML을 따른다.
- 인터페이스 변경은 사용자 승인 후 관련 정의·코드·YAML·changes를 함께 갱신한다. 이미 승인된 범위는 재확인하지 않는다.
- 다른 노드나 공통 타입 수정은 사전 승인을 받고, 연결된 노드의 영향을 확인한다.
- 현재 파일 존재만으로 구현 완료 처리하지 않는다. 구현 상태와 실제 소스 경로를 YAML에 맞춘다.
