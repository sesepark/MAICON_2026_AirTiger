# 로그_파일

연습 경기장과 시험에서 돌린 기록을 모으는 곳. 기록은 로봇의 `~/team/log/<날짜_시각>/`에 실행할 때마다 하나씩 생긴다 (`개발/robot/team_node.py`가 만든다).

## 기록이 있는 곳 (사람 · 에이전트 공통)

로봇은 경기(포털의 팀 테스트 포함)마다 기록 폴더를 하나 만든다. 폴더 이름은 `<날짜_시각>` (예: `20261118_222257`).

| 어디 | 경로 | 무엇이 있나 |
|---|---|---|
| 로봇 | `/home/jetson/team/log/<날짜_시각>/` (컨테이너 안에서는 `/home/ws/team/log/<날짜_시각>/`) | 전부 — 글 기록, 로봇 카메라 사진(`supervisor/photos/`), 드론 사진(`server_bridge/drone/`, `perception/drone_used.jpg`) |
| **연습경기 보기** | https://maicon-match.tlate.ai | 진행 중인 경기를 실시간으로, 지난 경기를 다시 보기로 (경기장 그림 · 임무별 상태 · 일어난 일 · 사진과 그 위의 판독 네모 · 드론 사진). 글 기록 원본도 받을 수 있다 |
| 이 폴더 | `로그_파일/<날짜_무엇>/` | 골라 올린 글 기록과 `상황.md` (그 판에서 본 것 · 원인 분석) |

에이전트가 읽을 때:

- 목록: `curl -s https://maicon-match.tlate.ai/api/matches` (새 것부터. `live` 가 true 면 지금 진행 중)
- 한 경기를 한 덩어리로: `curl -s https://maicon-match.tlate.ai/api/match/<날짜_시각>` (지도 · 0.5초마다의 위치 · 일어난 일 · 사진과 판독 · 임무별 결과 · 파일 목록)
- 원본 파일: `curl -s https://maicon-match.tlate.ai/files/<날짜_시각>/supervisor/events.jsonl` 처럼
- 파일마다 무엇이 들었는지와 칸의 뜻: https://maicon-match.tlate.ai/agents.md
- 사진(로봇 카메라 · 드론)은 경기장과 사람이 찍혀 있어, **팀 링크**(주소 뒤에 `?k=<열쇠>` 가 붙은 것 — 팀장에게 받는다)로 한 번 들어온 기기에서만 보인다. 비밀번호를 치는 일은 없다.
  에이전트는 사진 주소 뒤에 같은 열쇠를 붙인다: `/photo/<날짜_시각>/<파일 이름>?k=<열쇠>` · `/drone/<날짜_시각>/<파일 이름>?k=<열쇠>`. **열쇠는 저장소에 적지 않는다**

그 사이트는 팀장의 연구실 PC 에서 돌고, 로봇의 기록 폴더를 2초마다 받아 온다 (로봇에 닿을 때만). 본 경기 때는 로봇이 잠겨 실시간으로는 보이지 않는다.

## 넣는 방법

1. 로봇에서 묶는다 (Jupyter 셀이면 앞에 `!`)

   ```bash
   cd ~/team && python3 team_node.py --pack-logs          # 아직 묶지 않은 실행만. 전부는 --all, 사진을 빼려면 --no-photos
   ```

2. 만들어진 `~/team/team_logs_<날짜_시각>.tgz`를 내려받아 이 폴더 아래에 넣는다. 폴더 이름은 `날짜_무엇을 했는지` (예: `1119_연습경기장_정찰비교/`)
3. 같은 폴더에 `상황.md`를 적는다: 언제 · 어디서(연습 경기장 몇 번) · 상자 배치(사진이나 그림) · 출발 자리 · 무엇을 보려 했는지 · **눈으로 본 결과**(표적의 빨간 불이 켜졌는가, 거점에 들어갔는가, 어디서 막혔는가). 기록에는 로봇이 안 것만 남으므로, 사람이 본 것이 있어야 견줄 수 있다
4. 커밋해 올린다

비밀 파일(`secrets.json`)은 묶음에 들어가지 않고, 임무코드 · LLM 키는 기록에 찍히지 않는다.

## 묶음 안에 든 것

```
log/<날짜_시각>/
  version.json          그때 돌린 코드의 지문 (패키지별). 어느 커밋이었는지는 `python3 개발/robot/team_node.py --fingerprint 개발/src` 의 값과 견줘 찾는다
  team_node.log         시작 파일의 출력 (무엇을 켰고, 무엇이 죽어 다시 켰는지)
  <프로그램>.log         자식 프로그램의 출력 (navigation, perception, server_bridge, llm_worker …)
  ros/                  각 프로그램의 ROS 기록 파일 (supervisor 의 글 기록도 여기)
  supervisor/events.jsonl   경기 신호 · 0.5초마다의 위치와 하는 일 · 이동 목표와 도착 · 팬틸트 · 발사와 답 · 판독 요청과 답 · 보고와 서버의 답
  supervisor/photos.jsonl   사진마다 「찍혀야 할 면 · 판독 답 · 짝지은 결과」
  supervisor/photos/        판독에 보낸 사진
  server_bridge/http.jsonl  서버와 주고받은 것 (창구 · 상태 코드 · 걸린 시간 · 본문)
  server_bridge/drone/      드론 사진 (10초에 한 장)
  llm_worker/llm.jsonl      LLM 에 물은 것과 받은 것
  navigation/debug.jsonl    주행 노드 내부 상태 — 목표 · 단계 바뀜 · 결과 · 경로 · 지도에 없는 라이다 점 · 위치 상태 (2026-11-19, 개발/src/navigation/README.md 「경기 기록」)
  navigation_slam/debug.jsonl  SLAM 노드 내부 상태 — 받은 지도 · 출발 칸 탐색 · 정합 결과 · 위치
mil_log/team_node.log   로봇이 team_node.py 의 출력을 받아 적은 파일의 끝부분 (있을 때)
```

## 한눈에 보기

```bash
python3 로그_파일/summarize.py 로그_파일/<폴더>/team_logs_<날짜_시각>.tgz      # 풀어서 실행마다 요약을 찍는다. 풀린 폴더를 줘도 된다
```

경기 신호, 하는 일이 바뀐 시각, 간 거리, M1 해석, 이동 결과, 사진 · 판독 · 짝짓기 수, 발사, 보고와 서버의 답, 목표거점, 서버 응답 시간, 오류로 보이는 줄을 찍는다.

로봇의 디스크가 1.5 GB보다 적게 남으면 사진 · 드론 사진은 남기지 않고, 200 MB보다 적으면 기록을 아예 남기지 않는다. 다 옮긴 기록은 로봇에서 `rm -rf ~/team/log/<날짜_시각>`으로 지운다.
