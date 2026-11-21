#!/usr/bin/env bash
# LiDAR(LD19) 드라이버를 켠다 — /scan 을 내보내는 프로그램이 이미 있으면 켜지 않는다.
# team_node.py 가 자식으로 켠다 (programs.json 의 lidar 줄).
#
# 왜 여기서 켜는가 (2026-11-18 팀 테스트에서 확인):
#   로봇의 기본 기동은 라이다를 켜지 않는다. 그리고 로봇은 경기 중(시작 대기 ~ 종료)에 「로봇이 켠 team_node.py 와 그 자식」 말고는
#   팀 폴더에서 돈 프로그램을 2초쯤마다 찾아 끈다 — 손으로 켜 둔 라이다도 시작 대기에 들어가자마자 꺼졌다.
#   그러므로 운영 측이 켜 주지 않는 한, 우리 시작 파일이 켜야 /scan 이 나온다 (navigation 이 위치를 잡으려면 /scan 이 있어야 한다).
# [확인 필요] 본 경기에서 팀이 라이다를 켜도 되는지는 운영사무국에 물어야 한다 (로봇 쪽 주석: 「참가자 API 에서 빠져 기본 꺼짐」).
#            안 된다고 하면 programs.json 에서 이 줄의 enabled 를 false 로 바꾼다.
if timeout 8 ros2 topic info /scan 2>/dev/null | grep -q 'Publisher count: [1-9]'; then
  echo "라이다: 이미 켜져 있음 (/scan 을 내보내는 프로그램이 있다) — 켜지 않는다"
  exec sleep infinity
fi
if ! ros2 pkg prefix ldlidar >/dev/null 2>&1; then
  echo "라이다: ldlidar 꾸러미가 없다 (로봇이 아닌 곳) — 켜지 않는다"
  exec sleep infinity
fi
echo "라이다: 켠다 (ros2 launch ldlidar ld19.launch.py)"
exec ros2 launch ldlidar ld19.launch.py
