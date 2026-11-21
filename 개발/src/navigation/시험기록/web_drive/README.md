# 웹 주행 시험기 (`web_drive`)

경기와 같은 순서로 navigation 두 노드(SLAM · 주행)를 시험하는 브라우저 도구다.

| 단계 | 경기에서 하는 쪽 | 시험기에서 |
|---|---|---|
| ① 지도 | perception 노드(`map_maker.py`)가 경기 시작 뒤 드론 사진으로 계산 → `/team/perception/field_map` | **드론 사진**을 받아 perception 의 같은 함수로 지도를 만든다. 사진이 없으면 라이다 · 직접 편집 |
| ② 처음 위치 | server_bridge 단계(ready → running) + 지도 → SLAM 이 출발 칸 탐색 | 출발 칸 버튼 하나로 같은 순서를 낸다 |
| ③ 주행 | supervisor 가 목표 · 주행 허용 | 지도를 클릭하면 그 좌표로 간다 |

supervisor · server_bridge · perception 이 하는 일을 사람이 대신하는 **시험 전용 도구**다. 경기에서는 켜지 않는다.

- 2026-11-18 UGV-08 모의 경기장: 「라이다」 지도 → 목표 9개 연속 `REACHED` (끝난 자리와 목표의 차이 5 cm 안, SLAM 기준), 정합 1회 약 35 ms.
- 2026-11-18 가상 경기장(PC): 연습 경기장 샘플 사진 → 지도(상자 + 기둥 10개) → B-2 출발 칸 탐색이 정답 자리와 같음 → `REACHED`.
- 2026-11-18 UGV-08 컨테이너: 샘플 사진 한 장 지도 계산 18.4 s (따로 띄운 프로세스), SLAM 이 그 지도를 받음. 로봇을 실제 드론 지도 배치에 놓고 출발 칸 → 주행까지 해 본 적은 아직 없다.

## 로봇에서 쓰기

PC(Git Bash, 팀 와이파이)에서. SSH 비밀번호는 물을 때마다 넣는다.

```bash
cd 개발/src/navigation/시험기록/web_drive
bash robot.sh deploy <로봇 IP>     # 지금 PC 의 소스(올리지 않은 수정 포함) + perception 드론 코드를 올리고 컨테이너 안에서 빌드
bash robot.sh start  <로봇 IP>     # 라이다(꺼져 있으면) · SLAM · 주행 · 웹 시험기 · rosbag 을 켠다
#   → 브라우저에서 http://<로봇 IP>:5960
bash robot.sh status <로봇 IP>
bash robot.sh stop   <로봇 IP>
bash robot.sh logs   <로봇 IP> ./받을_폴더     # 마지막 실행의 로그 · rosbag
MAX_V=0.35 DRIVE_ARGS="-p min_turn_w:=1.0" bash robot.sh start <로봇 IP>   # 속도 제한(기본 0.25 m/s) · 주행 파라미터
```

로봇의 작업공간은 `/home/jetson/team/nav_web` (컨테이너 안 `/home/ws/team/nav_web`) 이고 경기용 `~/team/ws` 는 건드리지 않는다.
`start` 는 `/scan` 발행자가 없으면 LD19 드라이버(`ros2 launch ldlidar ld19.launch.py`)를 켠다 — 로봇 기본 기동은 라이다를 켜지 않는다.

## 쓰는 순서

1. **① 지도 — 드론 사진**: 사진 출처를 고르고 「사진 받아 지도 만들기」.
   - 서버 카메라 `test1` · `test2` (연습 경기장) · `main`: 서버 주소 · 임무코드가 있어야 한다 → 로봇의 `nav_web/secrets.json` 에 `{"server_url": "http://<서버>:8100", "mission_code": "…"}` (server_bridge 와 같은 칸, 저장소에 올리지 않는다) 또는 환경변수 `MAICON_SERVER_URL` · `MAICON_MISSION_CODE`. 주소가 없으면 연습 서버 `http://192.168.110.50:8100`.
   - server_bridge 가 받은 사진 (`/team/server/uav_image`): server_bridge 를 따로 켜 둔 경우.
   - 샘플 (연습 경기장 사진 2장, 11-18) · 파일 올리기.
   - 계산은 따로 띄운 파이썬에서 한다 (PC 10 ~ 35 s). 끝나면 사진 위에 찾은 상자, 지도 위에 **보라 점선 = 후보 상자**와 위에서 본 바닥 그림이 깔린다. 맞으면 「이 지도 쓰기」.
   - 「고정 기둥 10개 더하기」(기본 켬): perception 은 벽 쪽 끝 띠 안 물체를 내지 않아 기둥이 빠진다.
2. **② 처음 위치**: 로봇을 출발 칸에 세운 채 그 칸(`B-1` … `R-3`)을 누른다 → ready 1.2 s → running → 지도 → SLAM 이 그 칸 안을 라이다로 찾는다. 진행이 체크 목록으로 보인다. 출발 칸이 아닌 곳이면 「다른 방법」의 주변에서 찾기 · 자세 직접 주기.
3. **③ 주행**: 지도 클릭 = 목표, 끌면 도착 방향. **Space · Esc = 정지** (목표 취소 + 주행 허용 끔). 화면이 3초 넘게 끊겨도 허용이 꺼진다.

드론 사진이 없는 모의 배치에서는 ① 의 「라이다」 탭: 로봇 자리를 누른 채 보는 방향으로 끌면(±30 cm · ±30° 면 충분) 자세를 벽에 맞추고 라이다에 보이는 상자로 지도를 만든 뒤 그 자세에서 바로 시작한다 (② 를 건너뜀).

그림: 회색 = SLAM 이 내보낸 지도의 상자, 보라 점선 = 드론 사진 후보, 붉은 × = perception 이 버린 후보, 청록 점 = 라이다(SLAM 위치에 놓음 — 벽 · 상자 윤곽과 겹쳐야 위치가 맞는 것),
파랑 = SLAM 위치, 초록 점선 = 주행 노드가 쓰는 자세, 초록 선 = 경로, 붉은 칸 = 임시 장애물, 주황 점 = 지도에 없는 라이다 점. 오른쪽 위에서 층을 켜고 끈다.

## PC 에서 로봇 없이 써 보기

```bash
python web_drive.py --sim          # http://localhost:5960  (numpy 필요, 드론 사진은 OpenCV 도)
python web_drive.py --sim --sim-robot 이상   # 예전처럼 반응이 빠른 가상 로봇
```

가상 경기장(`sim_world.py`)에서 **실제 노드 코드**(SLAM · 주행)를 돌린다. 드론 사진 지도를 쓰면 가상 세계의 상자도 그 지도로 바뀐다.
「지도에 없는 상자 놓기」로 회피를 볼 수 있다. 같은 가상 경기장으로 `test/test_closed_loop.py` 가 두 노드를 묶은 닫힌 루프 시험을 한다.

가상 로봇 (`--sim-robot`, 기본 `실차`) — 2026-11-18 팀 테스트를 재현하려고 넣었다 ([정리](../2026-11-18_팀테스트_재현/README.md)):

- `실차` (`sim_world.REAL_ROBOT` [추정]): 명령 반응 지연 0.30 s · 1차 지연 0.20 s, **서서 도는 명령은 0.45 s 이어져야 돌기 시작**한다 (팀 테스트에서 0.3 s 씩 끊긴 회전 명령으로는 돌지 않았다). 지금의 주행 노드는 도착 방향을 조금만 더 돌아야 할 때 이 로봇에서 멈춘다 — 실차와 같다
- `이상`: 지연 0.1 s · 0.12 s, 회전 마찰 없음 (예전 동작)
- 부딪힘은 차체 사각형(250 × 230 mm)으로 센다. `World(vehicles=[(x, y)])` 로 다른 UGV(몸 250 × 230, 라이다에는 가운데 `vehicle_seen` 만 보임)를 놓을 수 있다 (화면에서는 아직 못 놓는다)
- 같은 입력이면 결과가 같다 (목표 실행 스레드를 기다리는 동안 가상 시계를 멈춘다)

## 파일

| 파일 | 내용 |
|---|---|
| `web_drive.py` | HTTP 서버 + 시험기 로직. `RosBus`(로봇) · `SimBus`(가상 경기장) |
| `page.html` | 화면 (캔버스, 0.2초마다 `/state` 를 읽는다) |
| `drone_map.py` | 드론 사진 → 지도. perception 의 `drone_boundary` · `drone_boxes` · `drone_field_map` 을 고치지 않고 불러 쓴다. 서버 카메라 받기 |
| `lidar_map.py` | 라이다 한 번으로 자세 다듬기(`fit_walls`) · 상자 찾기(`suggest_boxes`) |
| `sim_world.py` | 가상 경기장 · 가상 로봇 · 두 노드를 묶는 `ClosedLoop` |
| `arena_default.py` | 경기장 기본값 (실측 612 × 500 cm, 출발 칸 40 × 120 cm, 고정 기둥 10개 [추정]) |
| `robot.sh` · `run.sh` | PC 에서 올리기 · 켜기 / 로봇 컨테이너 안에서 빌드 · 켜기 · 끄기 |

perception 드론 코드는 `개발/src/perception` (main 에 합쳐진 뒤) → 없으면 git 의 `origin/perception` 에서 꺼내 `_perception/` 에 둔다 (git 에 올리지 않음, 브랜치가 바뀌면 이 폴더를 지우면 다시 꺼낸다). `PERCEPTION_SRC` 로 직접 줄 수도 있다.

## 알아 둘 것

- 시험기가 내는 토픽: `/team/perception/field_map` · `/team/server/state` · `/team/control/state` · `/team/mission/state`. 진짜 supervisor · server_bridge · perception 과 **같이 켜면 안 된다** — 같이 켜져 있으면 화면 맨 위에 붉은 경고가 뜬다 (perception 노드는 이제 지도를 직접 낸다). server_bridge 의 사진 토픽(`/team/server/uav_image`)은 받기만 한다.
- 경기장 크기는 실측 6.12 × 5.00 m 다 (2026-11-18 팀 결정 — 포털의 600 × 480 은 서버의 그림 모형. supervisor · perception 과 같은 값). 드론 사진 지도는 늘 이 크기로 계산하고, 쓰면 지금 지도도 이 크기가 된다. 600 × 480 으로 계산하면 같은 사진에서 상자가 4 % 작게 · 덜 잡힌다 (test2: 14개 → 13개).
- 지도의 출발 칸은 6개 모두 이름(`B-1` … `R-3`)으로 넣는다 — perception 노드(`map_base_naming` 기본 `slot`)와 같고, 크기도 같은 도면값(벽에서 40 cm × 줄 가운데 ±60 cm)이다.
- 앞 시험기가 저장한 `map.json`(경기장 600 × 480 시절, `source` 칸 없음)은 읽지 않고 새 기본값으로 시작한다.
- 「라이다」 지도는 한 자리에서 보이는 면만 쓴다. 가려진 상자는 빠지고(가까이 가면 주행 노드가 임시 장애물로 피한다), 면이 조금만 보인 상자는 몇 cm 어긋날 수 있다.
- 경기장은 좌우 · 위아래가 대칭이라 라이다만으로는 어느 쪽 진영인지 가릴 수 없다. 그래서 출발 칸(경기에서는 서버가 알려 주는 slot)을 사람이 고른다.
- 지도 설정은 로봇의 `nav_web/map.json` 에 저장된다.
