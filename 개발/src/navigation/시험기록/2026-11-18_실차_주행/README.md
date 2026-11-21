# 2026-11-18 navigation 실차 기록

UGV-08의 ROS 2 Humble 환경에서 정지 기록 두 건과 짧은 직진 기록 한 건을
수집했다. 원본은 이 저장소의 Git 제외 경로 `runs/slam_trial01/runs/`와
로봇의 `/home/jetson/team/slam_trial_trial01/runs/`에 있다.

| 기록 | 공유 자료 |
|---|---|
| `baseline` | 기본 기동 상태에서 수집한 센서 bag, 수신 간격, 토픽별 집계 |
| `lidar_stationary` | LD19를 별도 실행한 정지 bag, 스캔과 odom 시간차 진단 |
| `straight04` | mock 항공 지도와 목표를 사용한 직진 bag, 명령·자세 CSV, 노드 로그 |

각 `sensors/`는 ROS 2 SQLite bag이다. 원본에서 카메라 영상 메시지를
제외했으며, 나머지 메시지의 원본 직렬화 값과 수신 시각을 대조했다.
`motion.csv`는 odom·IMU·SLAM 자세·속도 명령의 디코딩 결과다. 원래
전체 기록의 메시지 수와 용량은 각 `bag_info.txt`에서 확인할 수 있다.
재생은 실차와 분리된 ROS 도메인에서 입력 토픽을 선택해 실행한다.
`straight04`의 기록에는 실차 `/cmd_vel`도 들어 있다.

직진 시험은 `settling`에 진입했지만 시험 클라이언트가 제어 허용을 먼저
철회해 `CONTROL_LOST`로 끝났다. odom 위치는 약 12 cm 변했고 odom yaw는
모든 샘플에서 0이었다. 정지 명령은 bag에서 확인했다. 실제 횡변위 실측값은
없으며, 당시 mock 지도에는 라이다 정합에 쓸 박스가 없었다.
