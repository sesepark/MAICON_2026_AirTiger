# 2026-11-18 주행 시험 정리 — UGV-08 `/cmd_vel` 개방 루프

navigation 담당(주행)의 첫 실물 시험 기록이다. 공식 규격이 아니라 **UGV-08 한 대에서 측정한 결과**이며, 가설과 확인된 사실을 구분해 적었다. 이 폴더의 코드는 시험 도구이고 `navigation` 패키지의 노드가 아니다. 새 노드·토픽·서비스를 만들지 않았다(시험 중에만 `tmp_teleop` 노드가 기존 `/cmd_vel`에 발행, `/odom`·`/mil/match_state` 구독).

## 한눈에

1. **직진**: 0.1 m/s로 약 1 s 명령하면 사용자 관찰로 "대충 비슷한 cm"만큼 간다(줄자 실측 값은 입력하지 않음).
2. **회전**: 0.3 rad/s 개방 루프로는 각도를 못 맞춘다. 우 90° 명령 ≈ 실제 약 10°, 좌 180° 명령 ≈ 실제 약 90°.
3. **`/odom`은 회전을 알리지 않는다**: 발행 orientation이 상수(단위 사원수)다. 그러나 `ugv_base_node` 내부에는 휠 기반 방향이 있고, 이것이 실제 회전을 따라간다(근거: [가설_검토.md](가설_검토.md)). 따라서 `/odom/odom_raw`(좌우 바퀴 누적 거리)로 방향을 계산하면 회전을 잴 수 있을 가능성이 높다.
4. **라이다 `/scan`은 기본 꺼져 있다**(운영 측이 `lidar:=true`로 켜는 구조). 손으로 켜면 정상 동작하고, `slam_toolbox`+Nav2(`mil_nav.launch.py`)도 떠서 `map` 프레임과 `/map`이 생기는 것까지 확인했다. 목표점 주행은 시험하지 않았다.

## 문서

| 문서 | 내용 |
|---|---|
| [가설_검토.md](가설_검토.md) | 소스·실시간 조회로 가설 8개 검토, `/odom` 방향의 정체 |
| [로그_분석.md](로그_분석.md) | 16회 이동의 수치(경로, 지연, 정지 후 미끄러짐, 샘플 통계) |
| [도구/사용법.md](도구/사용법.md) | 웹 조종 시험 도구(`teleop_server.py`) 실행·측정·로그 형식·정리 |
| `도구/analyze_logs.ps1`, `도구/heading_analysis.ps1` | 로그 분석 스크립트(PowerShell) |
| `logs/` | 원본 로그(jsonl). `teleop_20261118_150008.jsonl`이 시험 본체, 나머지는 서버 시작·시작 실패 기록 |

## 로봇 환경 확인 결과 (UGV-08, 읽기 전용 확인)

- **구성**: Jetson Orin, Ubuntu 22.04, ROS 2 Humble이 도커 컨테이너(`ugv_jetson_ros_humble`) 안에서 돈다. 팀 계정은 도커 권한이 없고 `sudo ugv-shell`(비밀번호 불필요)로 컨테이너 셸에 들어간다. `ROS_LOCALHOST_ONLY=1`.
- **자동 기동**: `mil-stack.service`가 `mil_bringup.launch.py`를 띄운다. 인자는 `~/mil/stack.args`(현재 `camera:=true`)이고 팀 계정은 수정할 수 없다.
- **실행 노드**: `mil_base_node`(ESP32 브리지), `ugv_base_node`(오도메트리), `imu_filter_madgwick`, `usb_cam`, `mil_match_node`, `robot_state_publisher`, `joint_state_publisher`.
- **`/cmd_vel`**: `mil_base_node`가 구독해 ESP로 `{"T":13,"X":lin,"Z":ang}`를 보낸다. 상한 `max_linear` 0.6 m/s, `max_angular` 3.0 rad/s, `cmd_vel_timeout` 0.5 s(끊기면 자동 정지), 0이 아닌 명령은 최대 20 Hz. 평상시 `/cmd_vel` 발행자는 `mil_match_node`(매치 종료 시 0 속도)이고, 매치 중 손으로 띄운 팀 스크립트는 이 노드가 2초 간격으로 찾아 정지시킨다.
- **오도메트리**: `ugv_base_node`가 `/odom/odom_raw`(좌·우 바퀴 누적 거리 m, 1 cm 해상도, 약 19.7 Hz)를 받아 계산한다. 바퀴 간격 0.175 m 가정, `/odom` 10 Hz 발행, `use_imu_yaw:=false`. 위 "한눈에" 3 참고.
- **IMU**: `/imu/data`(madgwick, 약 19.6 Hz)가 나오지만 오도메트리에는 연결돼 있지 않다.
- **라이다**: LD19, USB `1a86:55d3` → `/dev/ttyACM0`(by-id `usb-1a86_USB_Single_Serial_5B79024130-if00`), 230400 baud. launch 주석(2026-10-16): 참가자 API에서 빠져 기본 꺼짐. 그러나 운영 제공 `team_node.py`(서브게임)는 `/scan`을 구독한다. 경기 중 켜지는지는 운영사무국에 확인이 필요하다.
- **`/scan`(손으로 켠 경우)**: `sensor_msgs/LaserScan`, 10 Hz 고정, `frame_id: base_lidar_link`, 0~2π rad, 증분 0.0125 rad, 점 약 505개, `range_max` 25 m, 반시계. launch가 각도 225~315°를 가려(`angle_crop`) 한 바퀴 505점 중 유효 약 313점이다(실측에서 비는 구간은 `/scan` 각도 약 45~135°이고 로봇 뒤쪽으로 추정). 구독 QoS는 `qos_profile_sensor_data`.
- **TF**: `base_link → base_lidar_link` = x 0.0398 m, z 0.04 m, z축 +90° 회전. `odom → base_footprint → base_link`는 나온다(회전은 위 상수 orientation).
- **SLAM/Nav2**: `slam_toolbox`, Nav2, `rf2o_laser_odometry`, `gmapping`, `cartographer`, `explore_lite`, `emcl2`, `teb_local_planner`가 컨테이너에 설치돼 있다. `mil_nav.launch.py`로 `slam_toolbox`+Nav2를 띄웠을 때 `map` 프레임과 `/map`(0.05 m/칸)이 생기고 Nav2 서버(controller/planner/bt_navigator/behavior)가 active가 됐다. 처음 약 30 s는 `map` 프레임 없음 경고가 나온다.

## 저장소 규칙과 부딪히는 점

- **`/cmd_vel` 단독 발행자**: `navigation/AGENTS.md` 규칙은 팀 코드의 `/cmd_vel` 발행자를 navigation 노드 하나로 정한다. Nav2의 `velocity_smoother`·`behavior_server`·`mil_match_node`도 `/cmd_vel`을 발행하므로, Nav2를 쓸 경우 Nav2 출력을 `cmd_vel_nav`로 받아 navigation 노드가 `/cmd_vel`로 내보내는 구조가 필요하다. 노드·토픽 변경이라 사용자 승인과 `노드_관리/interfaces.yaml` 갱신이 필요하다(이번에는 하지 않았다).
- **`/odom/odom_raw` 구독**: navigation 노드가 방향 계산에 쓰려면 기존 토픽의 구독 추가이므로 같은 승인 절차가 필요하다.
- **라이다·TF 계약**: 지침상 "공식 규격 확인 후 계약에 추가". 위 `/scan`·TF 값은 시험 기록일 뿐 계약에는 아직 넣지 않았다.

## 다음 단계

1. 웹 조종 서버에 `odom_raw` 기반 방향 기록·표시 추가, 손으로 돌려 보는 검증(이동 없음).
2. 같은 조건 반복(좌·우 90° 각 5회)과 회전 속도 상승(0.8, 1.5 rad/s)으로 필요한 최소 속도 확인. 서버 상한 `MAX_ANG`(0.8)를 올려야 한다.
3. 줄자·각도기 실측을 웹 화면 실측 칸에 입력해 오차 표 완성.
4. 라이다를 켠 상태에서 `slam_toolbox`/`rf2o`의 방향이 실제 회전을 따라가는지 확인하고, 닫힌 루프 회전(목표 yaw까지 돌고 정지)의 방향 소스를 정한다. 지도 담당 팀원과 맞출 점: SLAM이 주는 `map → odom` 변환과 방향을 주행 쪽이 그대로 받아 쓸 수 있는지.
5. 위 "저장소 규칙" 항목의 구조 승인.

## 시험 조건 참고

- 일시: 2026-11-18 15:03~15:09(웹 조종 시험), 매치 상태 내내 `practice/idle`. 한 번에 한 이동, 직진 0.1 m/s·회전 0.3 rad/s, 시간 기준(개방 루프)만 시험했다.
- 로그에는 #1~#16 이후(15:10~15:11)에 좌 180°(#17), 좌 360°(#18) 이동이 2건 더 있다(0.3 rad/s, 시간 기준). 둘 다 `/odom` 방향 변화 0.0°, 경로 0.003~0.006 m였고, 사용자가 본 실제 각도는 따로 받지 못했다. 아래 분석 수치는 #1~#16 기준이다.
- 실측 입력(`measured`)은 로그에 없다. 사람이 본 값은 사용자가 말로 알려 준 관찰이다.
