# 로봇에 올리는 것 (`~/team/`)

로봇은 경기 '시작 대기' 때 `python3 ~/team/team_node.py` 하나만 실행한다 (공식 뼈대). 이 폴더의 `team_node.py`가 그 파일이다.
supervisor를 자기 안에서 돌리고, `programs.json`에 적힌 나머지 프로그램을 자식으로 켰다가 종료 신호를 받으면 함께 끈다.

**2026-11-18 기준: 컨테이너에서 가짜 상대와만 돌려 봤다. 실제 로봇에서는 아직 실행해 보지 않았다.** 아래 경로 중 「확인 필요」는 로봇에서 맞는지 봐야 한다.

## 로봇의 폴더

```
~/team/
  team_node.py         이 폴더의 파일 그대로 (이름 · 위치 고정)
  programs.json        켤 프로그램 목록 — 각 담당이 자기 줄을 고친다
  secrets.json         서버 주소 · 임무코드 · LLM 키. 저장소에 없다 — secrets.example.json 을 복사해 로봇에서 직접 채운다
  supervisor.json      supervisor 설정 (없어도 됨 — 개발/src/supervisor/README.md 「설정」)
  server_bridge.json   server_bridge 설정 (없어도 됨 — 개발/src/server_bridge/README.md 「설정」)
  llm_worker.json      llm_worker 설정 (없어도 됨 — 개발/src/llm_worker/README.md 「설정」)
  ws/src/              저장소의 개발/src/ 아래 패키지들을 그대로 복사
  ws/install/          로봇 안에서 빌드하면 생긴다
  log/<날짜_시각>/      실행할 때마다 하나씩 생기는 기록 폴더 (team_node.py 가 만든다 — 아래 「기록」)
```

`~/team`은 로봇 사용자 `jetson`의 `/home/jetson/team`이고, 컨테이너 안에서는 `/home/ws/team`으로 보인다 (구조 담당의 로봇 조사 기록 기준 — 확인 필요).

## 로봇이 우리 프로그램을 켜고 끄는 방식 (2026-11-18 로봇에서 확인)

로봇의 경기 프로그램(`mil_match_node`, 원본 `/home/ws/ugv_ws/src/mil/mil_base/mil_base/match_node.py`)을 읽고, 포털의 팀 테스트를 한 번 열어 확인한 내용이다.

- 경기 신호가 **시작 대기(standby)**가 되면 로봇이 `python3 /home/ws/team/team_node.py`를 **직접 켠다.** 본 경기뿐 아니라 **포털의 팀 테스트에서도 그렇다** (포털 안내문의 「팀 노드를 직접 실행」과 다르다). 그러므로 우리 것은 반드시 `~/team/team_node.py` 자리에 있어야 한다
- 그때부터 경기가 끝날 때까지, 로봇은 **자기가 켠 `team_node.py`와 그 자식 말고는** 팀 폴더(`~/team`)와 관련된 python · ros2 프로그램을 2초쯤마다 찾아 끈다. 손으로 켜 둔 것은 모두 꺼진다 — 실제로 조종 화면(`ugv_console.py`), 실시간 인식, **손으로 켠 라이다**가 시작 대기에 들어가자마자 꺼졌다
- 그래서 **라이다는 `team_node.py`가 자식으로 켠다** (`lidar.sh`, `programs.json`의 `lidar` 줄). `/scan`을 내보내는 프로그램이 이미 있으면 켜지 않는다. **[확인 필요]** 본 경기에서 팀이 라이다를 켜도 되는지는 운영사무국에 물어야 한다 (로봇 쪽 주석에 「참가자 API에서 빠져 기본 꺼짐」이라고 적혀 있다)
- 로봇이 켠 프로그램의 출력은 `/home/ws/mil/log/team_node.log`에 남는다 (우리 기록 폴더 `~/team/log/<시각>/`과는 별개). 로봇은 우리 프로그램을 낮은 CPU 우선순위로 켠다
- 끝낼 때: stop 신호 뒤 우리 프로그램이 스스로 끝나기를 기다리고, 안 끝나면 종료 신호 → 강제 종료 순서로 자식까지 모두 끈다
- 경기 중에 로봇 안에서 `ros2 …` 명령으로 들여다보는 것도 꺼질 수 있다 (팀 폴더에서 실행하면 떠돌이 프로그램으로 본다). 경기 중 상태는 기록 파일로 본다

## 올리고 빌드하기

한 번에 하는 방법 (권장 — 어느 커밋의 코드인지가 `VERSION` 파일로 같이 들어가 기록에 남는다):

```bash
bash 개발/robot/make_bundle.sh                 # 저장소가 있는 PC 에서 (Windows 는 Git Bash). team_bundle.tgz 가 생긴다
# team_bundle.tgz 를 로봇으로 옮긴 뒤, 로봇 컨테이너 안에서
mkdir -p ~/team && tar xzf team_bundle.tgz -C ~/team && cd ~/team/ws && colcon build
```

묶음에는 `team_node.py` · `programs.json` · `secrets.example.json` · `ws/src/`의 패키지들이 들어간다. `secrets.json`과 각 프로그램의 설정 파일(`supervisor.json` 등)은 들어가지 않으므로 로봇에 있는 것이 그대로 남는다.

손으로 하는 방법:

1. 저장소의 `개발/robot/team_node.py`, `개발/robot/programs.json`을 로봇의 `~/team/`에 복사한다. `secrets.example.json`은 `~/team/secrets.json`으로 복사해 값을 채운다 (한 번만. 임무코드 · 키가 든 이 파일은 저장소에 올리지 않는다)
2. 저장소의 `개발/src/` 아래 폴더들(`team_interfaces`, `supervisor`, `navigation`, …)을 로봇의 `~/team/ws/src/`에 복사한다. `임시_테스트_파일/`의 `mil_msgs`는 **복사하지 않는다** (시험용 대역이다. 로봇에는 진짜가 있다)
3. 로봇 컨테이너 안에서 빌드한다 (컨테이너에 들어가는 명령은 조사 기록에 `sudo ugv-shell`)

   ```bash
   cd ~/team/ws
   colcon build
   ```

   코드를 고쳐 다시 복사했으면 다시 빌드한다. 메시지 형식(`team_interfaces`)이 바뀌었으면 `rm -rf build install log` 뒤에 빌드한다.

## 돌려 보기

- 연습 때 손으로: Jupyter의 `Python 3 (ROS 2)` 커널 셀에서 `!cd /home/jetson/team && python3 team_node.py` (공식 뼈대의 안내). 셀을 멈추면 종료 신호가 간다
- 경기 신호 없이 흐름을 보려면 `~/team/supervisor.json`에 `{"force_start": true, "slot": "B-1"}` — 켜지자마자 시작한 것으로 친다
- 출력: 로봇이 실행했을 때는 `~/mil/log/team_node.log`, 자식 프로그램은 `~/team/log/<날짜_시각>/<이름>.log`

## 기록

기록을 보는 가장 쉬운 방법은 https://maicon-match.tlate.ai 다 (실시간 · 다시 보기 · 원본 파일). 어디에 무엇이 있는지는 [로그_파일/README.md](../../로그_파일/README.md) 맨 위에 있다.

`team_node.py`는 켜질 때마다 `~/team/log/<날짜_시각>/` 폴더를 만들고, 그 경로를 환경변수 `TEAM_LOG_DIR`로 자식 프로그램에 알린다 (ROS 기록 파일의 위치 `ROS_LOG_DIR`도 그 아래 `ros/`로 맞춘다). 프로그램들은 그 아래 자기 이름의 폴더에 쓴다.
다른 프로그램도 남기고 싶은 것이 있으면 `os.environ.get("TEAM_LOG_DIR")` 아래 자기 이름의 폴더에 쓰면 같이 묶인다.

```
log/<날짜_시각>/
  team_node.log · <프로그램>.log    시작 파일과 자식 프로그램의 출력
  version.json                      그때 돌린 코드의 지문 (패키지별)
  supervisor/                       events.jsonl · photos.jsonl · photos/
  server_bridge/                    http.jsonl · drone/
  llm_worker/                       llm.jsonl
  ros/                              ROS 기록 파일
```

연습이 끝나면 묶어서 저장소의 `로그_파일/`에 넣는다 ([로그_파일/README.md](../../로그_파일/README.md)).

```bash
cd ~/team && python3 team_node.py --pack-logs        # team_logs_<날짜_시각>.tgz 를 만든다. 사진을 빼려면 --no-photos
```

디스크가 1.5 GB보다 적게 남으면 사진 · 드론 사진은 남기지 않고(`TEAM_LOG_LIGHT=1`), 200 MB보다 적으면 기록을 남기지 않는다(`TEAM_LOG_OFF=1`).

`python3 team_node.py --fingerprint [폴더]`는 그 폴더(기본 `ws/src`) 아래 패키지들의 코드 지문만 찍는다. 기록의 `version.json`과 견줘 어느 커밋의 코드였는지 찾을 때 쓴다 (저장소에서는 `python3 개발/robot/team_node.py --fingerprint 개발/src`).

## `programs.json`

```json
{"name": "navigation", "package": "navigation", "executable": "navigation_drive", "enabled": true, "args": ["--ros-args", "-p", "이름:=값"]}
```

- `package` · `executable`: `setup.py`의 `console_scripts`에 적은 이름. `ws/install` 안의 실행 파일을 바로 부른다
- `enabled`: `false`면 켜지 않는다. 2026-11-18부터 다섯 프로그램을 모두 켠다 (perception은 로봇의 `~/team/models/pictures.engine`을 읽는다)
- `lidar` 줄: `cmd`로 `lidar.sh`를 부른다 (`package`/`executable` 대신 `cmd`를 쓰면 그 명령을 그대로 실행한다). 라이다를 우리가 켜면 안 되는 것으로 확인되면 `enabled`를 `false`로
- `programs.json`이 없거나 깨져 있으면 `team_node.py` 안에 적힌 기본 목록(다섯 프로그램 모두)으로 켠다
- `args`: 실행 파일 뒤에 그대로 붙는다. `env`: 환경 변수 (값에 `${PYTHONPATH}`처럼 쓸 수 있다)
- `restart`: 죽으면 다시 켤지 (기본 `true`, 프로그램마다 3번까지)
- 각 프로그램은 종료 신호(SIGINT)를 받으면 **0.8초 안에** 끝나야 한다. 안 끝나면 강제로 끈다 (로봇이 stop 뒤 1.5초에 `team_node.py`에 종료 신호를 보내기 때문)

## 확인한 것

`임시_테스트_파일/supervisor_가짜상대/run_team_node.sh`: 컨테이너에서 작업공간을 빌드하고 `team_node.py`로 server_bridge · llm_worker와 가짜 상대(로봇 · navigation · perception · 경기 서버 흉내)를 자식으로 켜서 한 경기를 돌린다.
2026-11-18: 자식이 켜지고, supervisor가 경기를 끝까지 돌리고, stop 신호 뒤 0.2초에 종료 코드 0으로 끝났다. 기록 폴더에 위의 파일들이 생겼고 임무코드 · LLM 키는 어디에도 찍히지 않았다.
