# SLAM 실차 검증

팀 와이파이에 연결한 PC의 Bash에서 이 디렉터리로 이동합니다. 아래
`<UGV_IP>`에는 로봇의 IPv4 주소를 넣습니다. 앞에 0이 붙은 각 자리는
스크립트에서 십진수로 정규화합니다. SSH 로그인은 기존 팀 인증을 사용합니다.

```bash
bash robot.sh stage <UGV_IP> trial01
bash robot.sh shell <UGV_IP>
```

컨테이너 셸에서 아래를 실행합니다. `trial01`은 배포 ID이고 `baseline`은
기록 ID입니다. 기록마다 새 ID를 사용합니다.

```bash
cd /home/ws/team/slam_trial_trial01
tar -xOzf source.tar.gz navigation/시험기록/slam_replay/trial.sh | bash -s -- prepare
tools=src/navigation/시험기록/slam_replay
bash "$tools/trial.sh" probe baseline
bash "$tools/trial.sh" record baseline
```

`prepare`는 navigation과 team_interfaces를 전용 워크스페이스에 빌드합니다.
`probe`는 디스크 여유, ROS 노드, 센서·지도·서버 상태 토픽의 타입과 QoS,
라이다 TF를 `runs/baseline.probe.log`에 남깁니다. 특히 `/scan` 발행자,
`/team/perception/field_map` 발행자, `/team/server/state` 발행자를 확인합니다.
후자의 두 토픽은 기존 perception 및 server_bridge 인스턴스에서 공급해야 합니다.

`record`는 실험 SLAM과 rosbag을 띄웁니다. 실험 출력은 `/slam_trial/map`과
`/slam_trial/pose`로 나갑니다. 이동은 기존 승인된 조종 수단으로 수행합니다.
같은 컨테이너의 두 번째 셸에서 구간 시작과 끝을 같은 기록 ID에 표시합니다.

```bash
cd /home/ws/team/slam_trial_trial01
tools=src/navigation/시험기록/slam_replay
bash "$tools/trial.sh" mark baseline ready_start
bash "$tools/trial.sh" mark baseline stationary_end
# 이후 구간도 같은 방식으로 시작과 종료를 표시
```

기록을 Ctrl+C로 종료한 뒤 PC에서 가져옵니다. 원본 센서, 양쪽 카메라 이미지,
서버 상태, TF, 이동 명령 관측, SLAM 입출력이 bag에 묶입니다. 배포한 소스
아카이브와 빌드 로그도 함께 가져오므로 미커밋 코드로 실행해도 재현할 수 있습니다.

```bash
bash robot.sh collect <UGV_IP> trial01 ./trial01-results
```

## 실험 순서

| 구간 마커 | 수행과 추가 기록 | 확인 항목 |
|---|---|---|
| `ready_start/end` | 준비 상태에서 정지하고 배정 슬롯과 실측 시작 위치·방향을 기록합니다. | 서버 배정과 지도 슬롯명 일치, 항공 관측과 스캔의 시각 차이, 자기차 자세가 처음 발행된 시점. |
| `stationary_start/end` | 약 10초간 정지합니다. | 자세 드리프트, 라이다 보정이 일으킨 점프, 입력 누락과 수신 간격. |
| `straight_start/end` | 알려진 거리를 저속으로 직진하고 출발점·정지점을 실측합니다. | `/odom`과 SLAM의 이동 거리, 지도 축과 부호, 정지 후 드리프트. |
| `turn_start/end` | 실측 각도를 아는 제자리 회전을 수행합니다. | yaw 부호와 각도 오차, 회전 중 추정 위치 이동, 라이다 TF 오류 징후. |
| `aerial_gap_start/end` | 처음에는 정지 상태로 항공 입력 공백을 관측하고, 회복 프레임 도착 시각을 표시합니다. 제어 가능한 공백이 필요하면 복제 bag에서 항공 입력만 제외한 구간을 만듭니다. | 항공 입력이 없는 동안 스캔이 있을 때 자세 갱신이 이어지는지, 보정이 끊기면 마지막 유효 기준에서 2초 이후 새 예측이 멈추는지, 회복 후 자세가 되감기거나 튀는지. |
| `ambiguous_start/end` | 시야가 가려지거나 반복되는 배치의 자연 발생 구간을 기록합니다. | 잘못된 정합으로 자세가 점프하는지, 스캔 점 수와 고정 박스 배치가 판단에 충분했는지. |

`marks.tsv`는 로봇 시스템 시각의 나노초를 기록합니다. `receive_timing.json`은
토픽별 메시지 수, 평균 수신률, 최대 수신 공백을 보여 줍니다. 수신 시각과
메시지 `header.stamp`는 비교 목적이 다릅니다. 위치 오차와 센서 정렬은
bag의 헤더 시각과 실측값으로 비교합니다. 토픽이 JSON에서 빠지면
`bag_info.txt`, `recorder.log`, probe 로그를 대조해 발행자 부재와 QoS 문제를
구분합니다. 현재 정합 기준은 축 정렬된 박스와 시험 기록의 라이다 장착 변환을
가정합니다. 정합 실패 판별에 필요한 원본 스캔과 지도는 모두 bag에 남습니다.

## 재생과 테스트 케이스

실차와 분리된 ROS domain에서 워크스페이스를 source합니다. 첫 셸에서
SLAM을 실행하고, 같은 domain의 두 번째 셸에서 입력만 재생합니다.

```bash
ros2 run navigation navigation_slam --ros-args -p use_sim_time:=true
ros2 bag play /path/to/run/bag --clock --topics \
  /team/perception/field_map /odom /odom/odom_raw /imu/data /scan /team/server/state
```

`/odom/odom_raw`(바퀴 누적 거리)에는 header가 없어 노드가 받은 시각을 씁니다. 그래서 재생 시계(`--clock`)와
`use_sim_time:=true`를 같이 써야 `/odom` 기록 시각과 맞습니다. odom_raw가 없는 옛 기록은
`-p yaw_source:=imu`로 돌립니다. `replay_tracking.py`는 이것을 스스로 고릅니다.

원래 출력 `/slam_trial/pose`와 새 출력 `/team/navigation/pose`를 헤더 시각으로
비교합니다. 테스트 케이스를 만들 때는 bag, 마커 범위, 시작·종료 실측 자세,
기대 오차와 허용 편차를 함께 보존합니다. 위치 초기화가 필요한 구간은 준비
단계의 초기 입력부터 재생합니다. 항공 공백 시험에서는 초기화를 위한 프레임을
남기고 공백 구간의 항공 지도 메시지만 제외합니다.

## 실차에서 /scan 발행자가 없을 때

`ros2 topic info /scan`의 Publisher count를 확인합니다. 토픽 목록에
`/scan`이 있어도 웹 콘솔 구독자만 연결되어 있을 수 있습니다. 2026-11-18
실차에서는 발행자가 없고 시리얼 포트가 비어 있는 상태에서 아래처럼 LD19
드라이버만 잠시 실행해 스캔을 기록했습니다. 재실행 시 기존 발행자와
`/dev/ttyACM0` 사용 여부를 먼저 확인합니다. 기본 로봇 서비스는 계속 실행됩니다.

```bash
source /opt/ros/humble/setup.bash
source /home/ws/ugv_ws/install/local_setup.bash
export ROS_LOCALHOST_ONLY=1
ros2 topic info /scan
fuser /dev/ttyACM0
timeout -s INT 60s ros2 launch ldlidar ld19.launch.py
```

드라이버가 실행되는 동안 두 번째 컨테이너 셸에서 새 ID로 `record`를
실행합니다. 이 시험으로 스캔 수집과 주기는 확인할 수 있습니다. 실제 지도
정합을 평가하려면 같은 시간대의 항공 지도와 초기화 입력도 기록해야 합니다.

ROS 2와 워크스페이스 환경을 source한 셸에서는 메시지 헤더와 수신 시각을
별도로 점검할 수 있습니다. 프레임, 누락 입력, 0 또는 역순 stamp, 스캔별
점 수 변화와 유효 점 수를 JSON으로 남깁니다.

```bash
python3 inspect_inputs.py /path/to/run/bag > input-diagnostics.json
```

재생에서 항공 입력 공백을 재현하려면 `marks.tsv`와 `receive_timing.json`의
나노초 시각을 사용해 새 bag을 만듭니다. 공백 앞에 초기화 프레임이 남도록
범위를 잡습니다. 시작 시각은 포함하고 종료 시각은 제외합니다.

```bash
python3 make_gap_bag.py /path/to/run/bag /path/to/new-gap-bag START_NS END_NS
ros2 bag info /path/to/new-gap-bag
```

스크립트는 원본을 읽기 전용으로 열고 SLAM 입력만 복사합니다. 기본값은
항공 지도 공백이며, 라이다 끊김을 시험할 때는 `--topic /scan`을 지정합니다.
