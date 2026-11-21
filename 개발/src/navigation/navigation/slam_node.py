import json
import time
from collections import deque
from copy import deepcopy
from math import atan2, cos, hypot, isfinite, pi, sin, sqrt

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Float32MultiArray, String
from team_interfaces.msg import FieldMap, FieldVehicle, ServerState

from navigation.map_fusion import FusionParams, MapFusion   # 라이다로 드론 지도 상자 고치기 (map_fusion:=false 로 끔)

from navigation.drive.params import DriveParams   # 드론 UGV 궤적의 값들 (주행 노드와 같은 기본값)
from navigation.drive.ugvs import UgvTracks, without_points_near   # 드론이 본 UGV 궤적 — 주행 노드와 같은 것을 쓴다 (우리 차 고르기 · 믿는 조건이 한 곳에 있다)
from navigation.drive.fieldmap import fixed_pillars, pillar_side_faces   # 고정 기둥의 기본 배치 (주행 노드와 같은 자리) · 그 옆면
from navigation.run_log import RunLog   # 경기 기록: TEAM_LOG_DIR 이 있으면 navigation_slam/debug.jsonl (ROS 와 무관)

# /odom 사용법 (2026-11-18 실측, 시험기록/2026-11-18_cmd_vel_개방루프):
# - 발행 orientation 은 항상 (0,0,0,1) 이다. 방향으로 쓰면 안 된다.
# - x·y 는 바퀴 기준 내부 방향으로 적분된다. 그래서 odom 좌표축은 경기장 좌표축과 어떤 각도(delta)만큼 돌아 있다.
# 그래서 이동량은 /odom x·y 를 delta 만큼 돌려 쓰고, delta 는 위치 관측(라이다 정합·항공)과 /odom 의 쌍으로 맞춘다.
# 관측 사이의 방향 변화는 기본적으로 /odom/odom_raw 바퀴 거리 차이로 잰다 (주행 노드와 같은 방향 소스,
# 로봇 odom 이 x·y 를 적분하는 내부 방향과 같은 식). yaw_source=imu 이면 /imu/data yaw.
DEFAULT_WHEEL_BASE_M = 0.175   # base_node.cpp 하드코딩 값. 스키드 조향이라 실제 유효 간격은 더 클 수 있다 → 각도기로 보정
SAMPLE_MATCH_NS = 100_000_000
PROPAGATE_NS = 2_000_000_000
UNCALIBRATED_MOVE_M = 0.05      # delta 를 모를 때 위치를 그대로 둔 채 넘길 수 있는 이동량
CALIBRATION_WINDOW_NS = 4_000_000_000
CALIBRATION_MIN_POINTS = 4
CALIBRATION_MIN_SPREAD_M = 0.10
CALIBRATION_MAX_RMS_M = 0.06
# 라이다 정합에 경기장 외벽(0 ≤ x ≤ width, 0 ≤ y ≤ height)도 쓴다. 스캔 점 대부분이 벽에 맞는다.
USE_WALLS = True
# 벽에 붙은 고정 기둥 10개(노랑 · 검정 블록)도 정합에 쓴다 (2026-11-19). 지도 메시지에는 없어 규칙의 기본 배치를 더한다
# (drive/fieldmap.py fixed_pillars — 주행 노드와 같은 자리). 거점 안에서는 기둥 면이 양옆 0.5 m 에 있어 벽을 따라 미끄러지지 않게 잡아 준다
# (가상: 거점 안 위치 오차가 기둥 없이는 0.20 m 까지, 넣으면 0.02 m 안). 노드 파라미터 fixed_pillars:=false 로 끈다
USE_PILLARS = True
# 고정 기둥의 옆면(붙은 벽에 수직인 면)을 외벽처럼 「믿는 구조」 로 쓴다 (2026-11-19 밤): 벽으로 다시 보기(wall_candidate)의 벽 점수와
# 지도 고치기의 믿는 점(map_fusion trust_pillar_faces)에 넣는다. 외벽은 벽에 수직인 방향만 잡아 주는데, 옆면은 벽을 따라가는 방향을 잡아 준다.
# 본선 경기장 9판의 스캔: 기둥이 덮은 벽 구간에 찍힌 점 4개(= 늘 있다), 옆면은 가정한 자리에서 +0.2 ~ +2.3 cm (drive/fieldmap.pillar_side_faces).
# 전에 기둥 전체를 믿었다가 기둥이 없는 모의 경기장 기록(1118_202820)에서 튄 위치가 35 → 133번이 되었다 — 기둥이 없는 경기장에서는
# 노드 파라미터 trust_pillars:=false 로 끈다 (fixed_pillars:=false 면 함께 꺼진다)
TRUST_PILLARS = True
# 출발 칸 초기화 (2026-11-18). 드론 사진(→ 지도)은 경기 진행(running) 중에만 오므로 ready 단계에서는 초기화할 수 없다.
# 로봇은 waiting·ready 동안 잠겨 있고, 위치를 얻기 전에는 주행 노드가 움직이지 않는다(/cmd_vel 은 주행 노드만 낸다).
# 그래서 잠긴 동안 마지막 /odom x·y 와 방향을 기억해 두고, 그 뒤로 거의 안 움직였으면 단계와 상관없이 출발 칸에서 찾는다.
LOCKED_PHASES = ('waiting', 'ready')
SLOT_STILL_M = 0.05
SLOT_STILL_RAD = 0.10
SLOT_RETRY_NS = 2_000_000_000   # 출발 칸 탐색이 실패하면 이만큼 뒤에 다시 (탐색이 무겁다)
SLOT_STEP_M = 0.10
SLOT_HEADINGS = 48              # 7.5°. 15° 로는 자기 벽을 가까이 볼 때 벽을 따라 미끄러진 답이 나왔다 (가상 경기장 시험)
SLOT_SEEDS = 6
OUTSIDE_M = 0.15                # 외벽 밖으로 이만큼 넘게 나간 라이다 점은 「있을 수 없는 점」 으로 센다
# 드론이 본 UGV (/team/perception/ugvs) — 지도 고치기에서 그 UGV 에 맞은 라이다 점을 빼고(_without_ugv_points), 잃은 위치를 다시 찾을 자리의 후보로 쓴다.
# 우리 차 고르기 · 믿는 조건 · 값은 주행 노드와 같은 것(drive/ugvs.UgvTracks · DriveParams 의 ugv_*)을 쓴다 (2026-11-20 — 전에는 같은 규칙을 여기 따로 적어 두었다)
UGV_MASK_AGE_NS = 6_000_000_000  # 잃은 위치 다시 찾기: 드론 자리가 이보다 오래됐으면 후보로 쓰지 않는다 (사진 간격은 1 · 2 · 5 s)
SELF_MATCH_M = 0.35             # 지도가 우리 차를 UNKNOWN 으로만 잡았을 때 SELF 로 바꿔 줄 거리
SELF_HALF_LENGTH = 0.155        # 차체 반 길이 0.125 + 0.03 — 이 사각형 안의 스캔 점은 우리 몸 (주행 노드 scan_self_margin_m 과 같은 값)
SELF_HALF_WIDTH = 0.145         # 차체 반 폭 0.115 + 0.03
LIDAR_X = 0.0398                # base_link → base_lidar_link (실측). scan_points 의 차체 좌표 점은 이 점에서 나온 광선이다
# 위치 검증 (2026-11-19 본선 경기장 8판 기록): 정합이 「성공」 이라고 하면서 위치가 0.1 ~ 0.5 m 틀린 구간이 5판에 있었다 (18:06 판은 99초에
# 0.5초 만에 0.5 m 를 뛰어 30초, 171초부터 끝까지 0.12 m). 틀린 자세는 드론 상자 · 라이다로 더한 상자에 걸려 통과 기준(맞은 점 25 %)을 넘었지만
# 외벽에 맞는 점은 진짜 자리의 1/3 ~ 1/4 뿐이었다. 외벽은 지도에서 유일하게 틀릴 수 없는 구조다. 그래서
#   1. 뜀 거르기   정합 결과가 지난 JUMP_WINDOW_NS 안의 정합 자리에서 「그 사이 /odom 이 간 거리 x JUMP_GAIN + JUMP_SLACK_M」 보다 멀면 버린다
#                 (바퀴가 구른 것보다 멀리 갈 수는 없다. /odom 의 축이 돌아 있어도 간 거리는 같다). JUMP_MAX_REJECTS 번 이어 버리면 받아들인다
#                 (정말 틀려 있던 위치를 고치는 길을 막지 않게 — 그 뒤는 2 가 본다).
#                 다만 외벽(과 기둥 옆면)이 새 자리를 뚜렷이 편들면 버리지 않는다 (walls_prefer, 2026-11-19 밤). 22:39 판 128 ~ 132 s: 왼쪽 벽 곁에
#                 선 동안 정합이 x = 0.99 와 0.86 을 오갔고 0.86 쪽이 「0.01 m 더 뜀」 으로 13번 버려졌다 (서 있을 때의 한도 0.12 m, 두 자리 차이 0.13 m).
#                 그 스캔의 벽 점수는 0.99 에서 49, 0.86 에서 92 — 지키던 자리가 틀린 쪽이었다. 틀린 답과 맞는 답이 번갈아 나와 「이어서 8번」 도 차지 않았고,
#                 그 뒤 46 s 동안 그 둘레에서 BLOCKED 4번. 18:06 판 99 s 의 틀린 쪽 뜀은 벽 점수 2 → 23 (점의 20 % 에 못 미침)이라 전처럼 버린다.
#   2. 벽으로 다시 보기   스캔마다 정합 자리 둘레 ±WALL_REACH_M 를 외벽만으로, 그리고 외벽 + 기둥 옆면으로 훑는다 (wall_search · TRUST_PILLARS). 다른 자리가 외벽에 훨씬 잘 맞으면 그 자리에서
#                 드론 지도만으로(라이다로 더한 상자 빼고) 다시 정합해 보고, WALL_STREAK 번 이어 같은 쪽을 가리키면 그 자리로 옮긴다.
#                 의심하는 동안과 옮긴 뒤에는 그 사이에 고친 지도를 되돌린다 (map_fusion.distrust).
JUMP_WINDOW_NS = 1_000_000_000
JUMP_GAIN = 1.3
JUMP_SLACK_M = 0.12
JUMP_MAX_REJECTS = 8
WALL_REACH_M = 0.60
WALL_STEP_M = 0.05
WALL_YAWS = (-0.10, -0.05, 0.0, 0.05, 0.10)
WALL_HIT_M = 0.05               # 외벽에서 이 안의 점 = 벽에 맞은 점 (본선 경기장 가로는 지도보다 약 6 cm 좁게 보였다 — 양쪽 3 cm 씩)
WALL_MIN_SHIFT_M = 0.10         # 이보다 가까운 자리는 정합이 알아서 고친다
WALL_MIN_SHARE = 0.20           # 옮길 자리에서는 스캔 점의 이만큼이 외벽에 맞아야 한다
WALL_GAIN = 1.5                 # 옮길 자리의 벽 점수 ≥ 지금 자리의 벽 점수 x 이 값 + 점 수 x WALL_MARGIN_SHARE
WALL_MARGIN_SHARE = 0.06
WALL_STREAK = 5                 # 이어서 이만큼(0.5 s) 같은 쪽을 가리켜야 옮긴다
WALL_MAX_TURN = 0.15            # 옮길 자리의 방향이 지금과 이보다 다르면 받지 않는다 (rad) — 도는 중에 번진 스캔
WALL_MAX_YAW_RATE = 0.6         # 이보다 빠르게 도는 동안에는 보지 않는다 (rad/s, map_fusion 과 같은 값)
WALL_AGREE_M = 0.10             # 이어진 판정끼리 옮길 양이 이 안에서 같아야 한다
WALL_DISTRUST_NS = 5_000_000_000  # 옮기면 의심이 시작되기 이만큼 전부터 고친 지도를 되돌린다
# 잃은 위치 다시 찾기 (2026-11-20, relocate_lost:=false 로 끔). 정합이 PROPAGATE_NS(2 s) 동안 이어 실패하면 씨앗이 없어지고, 출발 칸 탐색은 이미 썼거나
# (출발 칸을 떠났다) 쓸 수 없어 — 그 뒤로는 위치를 영영 내지 못했다 (주행은 목표마다 LOCALIZATION_LOST). 2026-11-19 21:53 · 22:39 판: 「정합 실패」 2 s 뒤
# 「처음 위치 찾기 실패」 만 끝까지 (둘 다 경기가 끝난 직후였다 — 경기 중에 난 적은 아직 없다). 처음 설계는 이때 지도의 SELF(드론이 본 우리 차)로
# 다시 잡는 것이었지만 perception 의 지도에는 차가 없다. 그래서 씨앗이 없으면 아래 자리들의 둘레를 출발 칸 탐색과 같은 방법(slot_search)으로 훑는다
#   1. 마지막으로 믿던 자리를 /odom 으로 옮긴 것 둘레 RELOCATE_NEAR_M (방향을 이어 알면 그 둘레 RELOCATE_YAW 만)
#   2. 드론이 본 UGV 자리들(/team/perception/ugvs — 우리 차도 그 가운데 있다) 둘레 RELOCATE_UGV_M
#      (1 이 있으면 가까운 것부터 RELOCATE_MAX_UGVS 곳, 없으면 모두 — 다 봐야 「한 곳에서만 맞는다」 를 말할 수 있다)
# 정확히 한 곳에서만 맞아야 한다 (둘 이상이면 버린다 — 경기장이 맞보는 꼴이라 틀린 자리로 잡느니 잃은 채로 둔다).
# 1 에서 찾았고 (방향을 이어 알았거나 드론의 한 자리가 RELOCATE_AGREE_M 안이면) 바로 받는다. 그 밖에는 이어진 두 번이 같은 자리일 때 받는다.
RELOCATE_RETRY_NS = 2_000_000_000   # 이 간격으로 다시 해 본다 (훑는 데 걸린 시간의 세 배가 이보다 길면 그만큼)
RELOCATE_NEAR_M = 0.40
RELOCATE_UGV_M = 0.25
RELOCATE_MAX_UGVS = 3
RELOCATE_YAW = 0.6
RELOCATE_AGREE_M = 0.40
RELOCATE_SAME_M = 0.15              # 이어진 두 번의 답이 이 안(방향 0.2 rad)이면 같은 자리
RELOCATE_CLEAR = 0.7                # 여러 곳에서 맞았으면, 둘째의 맞은 점 수가 첫째의 이 비율 아래일 때만 첫째를 고른다 (아니면 버린다)
# 2026-11-20 06:17 판 (다른 로봇 5대): 50 s 에 서서 도는 중에 지나가던 로봇에 0.3 m 밀렸다 (드론 · 스캔 다시 맞춤이 같은 자리). 바퀴는 2.4 rad/s 로 돈다고 셌고
# 실제는 1 rad/s 쯤 — 2 s 뒤 위치를 잃었을 때 「바퀴로 이어 온 방향」 은 거의 반대였다. 그때의 다시 찾기는 「/odom 으로 옮긴 자리 둘레 · 그 방향 ±RELOCATE_YAW」 만
# 훑어 진짜 자리(방향 0.5 rad)를 볼 수 없었고, 닮은 자리 (3.54, 3.74, −153°) 를 받아 232 s 까지 3분을 틀린 채 다녔다 (M3 0 / 30).
# 믿을 것의 순서를 바꾼다 — 바퀴가 센 방향은 가장 못 믿고(정상 판에서도 돌 때마다 0.3 ~ 3 rad 틀린다), 마지막으로 정합한 자리가 가장 믿을 만하다 (2 s 에 멀리 못 간다):
#   - 훑는 곳: 마지막 정합 자리(anchor) 둘레, 그리고 /odom 으로 옮긴 자리(last)가 거기서 RELOCATE_SPLIT_M 넘게 떨어져 있으면 그 둘레도. 드론 UGV 자리는 전과 같다
#   - 방향: 모두 (RELOCATE_ALL_HEADINGS). 지도: 드론 지도만 (라이다로 고친 상자는 쓰지 않는다 — 틀린 자리에서 고친 지도가 틀린 자리를 편든다)
#   - 받기: anchor · last 에서 나온 답이 스캔 점의 RELOCATE_SURE_SHARE 이상에 맞으면 바로, 아니면 이어진 두 번이 같을 때 (드론이 그 자리에 있으면 바로 — 전과 같다)
# 기록 재생(스캔 1 Hz, 드론 지도만): 06:17 판의 잃은 구간 스캔 12장 모두 진짜 자리 (54 s 뒤 (4.22, 3.36), 드론 (4.20, 3.31)) — 맞은 점 60 ~ 69 / 68, 닮은 자리는 30 ~ 37.
# 정상 판 셋(02:39 · 03:28 · 04:48)의 스캔 288장: 기록된 자리와 같은 답 286 · 답 없음 2 · 다른 답 0.
RELOCATE_ALL_HEADINGS = True
RELOCATE_SPLIT_M = 0.20
RELOCATE_SURE_SHARE = 0.75
# 뜀 거르기가 버린 정합의 **방향** 은 다음 정합의 씨앗으로 쓴다 (2026-11-20). 버리는 동안 자세를 바퀴로만 이어 가면 방향이 스캔마다 벌어져
# (06:17 판: 스캔마다 0.14 rad) 정합이 찾는 범위(±0.2 rad)를 벗어나고, 그 뒤로는 「정합 실패」 다. 거리는 의심해도 방향은 라이다 것이 낫다.
# 버린 방향이 바퀴 씨앗과 이보다 다르면 쓰지 않는다 (rad)
JUMP_YAW_HINT_MAX = 1.0
# 위치가 틀렸을 수 있는 동안에는 지도를 고치지 않는다 (2026-11-20). 06:17 판: 틀린 자리를 받은 지 1 s 만에 상자 둘을 더했고 221 s 까지 드론 상자 14 중 10개를 뺐다 —
# 고친 지도가 틀린 자리에 맞아 정합은 계속 「성공」 이었다. 두 가지로 막는다 (둘 다 라이다만 본다):
#   - 위치를 다시 잡은(잃은 위치 다시 찾기 · 외벽으로 옮김) 뒤 FUSION_HOLD_NS 동안
#   - 이 스캔이 **드론 지도**(고치기 전의 것)에 맞는 비율이 FUSION_MIN_DRONE_SHARE 아래일 때. 실차 정상 판 넷(02:39 · 03:28 · 04:48 · 06:32)의 스캔 1102장:
#     가운데 91 ~ 94 %, 40 % 아래는 14장이고 이어진 적이 없다. 06:17 판의 틀린 구간 142장: 가운데 27 %, 40 % 아래 115장
FUSION_HOLD_NS = 10_000_000_000
FUSION_MIN_DRONE_SHARE = 0.40

# 드론이 본 우리 자리로 위치를 의심하기 (2026-11-20, drone_pose_check:=false 로 끔 — 꺼도 기록은 남는다).
# 드론 사진(/team/perception/ugvs)에는 우리 차도 들어 있다. 실차 다섯 판(00:00 ~ 02:23)에서 서 있을 때 드론 자리 − SLAM 자리는 가운데 0.04 ~ 0.06 m,
# 90 % 가 0.15 ~ 0.22 m 였다 — 0.1 m 급의 어긋남은 드론으로 못 가리지만 0.4 m 넘는 것은 가린다. 그리고 우리 차가 안 잡히는 사진도 흔하다
# (화면 위아래 끝 · 오른쪽 벽 거점 쪽: 02:23 판은 198장 중 48장). 그래서 드론은 「의심」 만 하고, 옮길지는 라이다가 정한다 (사용자 방향: 라이다가 진실):
#   1. 사진마다 그 무렵 우리가 있던 자리에서 0.40 m(ugv_self_match_m) 안의 것을 우리 차로 본다. 보이면 그 차이를 적고(pose_check.drone_seen · state.ugvs.self_off),
#      같은 사진의 나머지는 「다른 UGV」 로 DRONE_OTHERS_KEEP_NS 동안 기억한다 (그 뒤의 사진에서 그 곁에 보이는 것은 같은 UGV 로 이어 따라간다 —
#      움직이는 UGV 도 한 번 우리와 함께 보였으면 「아는 것」 이다).
#   2. 우리 차가 안 보이는데, 기억한 다른 UGV 가 아닌 것이 우리 자리에서 DRONE_STRAY_REACH_M 안에 딱 하나 있고, 그것이 DRONE_DOUBT_FRAMES 장 이어
#      우리와 같은 간격(DRONE_STRAY_SAME_M 안)으로 따라오면 — 그것이 우리 차이고 SLAM 자리가 그만큼 틀렸을 수 있다 (의심).
#   3. 의심하는 동안 DRONE_RECHECK_NS 마다 그 자리 둘레 RELOCATE_UGV_M 를 출발 칸 탐색과 같은 방법(slot_search, 드론 지도만으로)으로 훑는다.
#      거기서 스캔이 지금 자리보다 뚜렷이 잘 맞고(지금 자리의 맞은 점 수 ≤ RELOCATE_CLEAR x 그 자리) 이어진 두 번이 같은 자리면 옮긴다.
#      지금 자리가 맞으면 아무 일도 없다 (다른 UGV 가 따라오는 것이었다).
DRONE_OTHERS_KEEP_NS = 120_000_000_000
DRONE_OTHER_M = 0.20                # 기억한 다른 UGV 자리에서 이 안의 것은 그 UGV 다 (사진 사이에 간 만큼은 UgvTracks 의 문턱으로 넓힌다)
DRONE_STRAY_REACH_M = 1.20
DRONE_STRAY_SAME_M = 0.25
DRONE_DOUBT_FRAMES = 3
DRONE_RECHECK_NS = 2_000_000_000
DRONE_MIN_SHIFT_M = 0.25            # 이보다 가까운 자리는 정합이 알아서 고친다 (드론 오차 안이기도 하다)


def stamp_ns(header):
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def wrap(angle):
    return atan2(sin(angle), cos(angle))


def planar_yaw(q):
    """단위 쿼터니언의 yaw. 값이 이상하면 None."""
    values = (q.x, q.y, q.z, q.w)
    if not all(map(isfinite, values)) or abs(sum(v*v for v in values) - 1) > 0.01:
        return None
    return atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))


def planar_pose(pose):
    position = (pose.position.x, pose.position.y)
    yaw = planar_yaw(pose.orientation)
    if yaw is None or not all(map(isfinite, position)):
        return None
    return (*position, yaw)


def rigid_fit(source, target):
    """target ≈ R(theta)·source + (tx, ty) 를 최소제곱으로. (tx, ty, theta, 잔차 RMS), 점이 2개 미만이면 None."""
    n = len(source)
    if n < 2 or n != len(target):
        return None
    sx = sum(p[0] for p in source) / n
    sy = sum(p[1] for p in source) / n
    mx = sum(q[0] for q in target) / n
    my = sum(q[1] for q in target) / n
    dot = cross = 0.0
    for (px, py), (qx, qy) in zip(source, target):
        px, py, qx, qy = px - sx, py - sy, qx - mx, qy - my
        dot += px*qx + py*qy
        cross += px*qy - py*qx
    theta = atan2(cross, dot)
    c, s = cos(theta), sin(theta)
    tx, ty = mx - (c*sx - s*sy), my - (s*sx + c*sy)
    error = sum((c*px - s*py + tx - qx)**2 + (s*px + c*py + ty - qy)**2
                for (px, py), (qx, qy) in zip(source, target))
    return tx, ty, theta, sqrt(error / n)


def spread(points):
    """중심에서의 RMS 거리."""
    n = len(points)
    if n == 0:
        return 0.0
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    return sqrt(sum((p[0]-cx)**2 + (p[1]-cy)**2 for p in points) / n)


def propagate(anchor, start, current, delta, turn):
    """관측 자세 anchor 를 그 뒤 시각으로 옮긴다.

    start · current = 두 시각의 /odom (x, y, ...). delta = odom 축 → 경기장 축 각도(모르면 None).
    turn = 두 시각 사이 방향 변화(IMU). delta 를 모르면 UNCALIBRATED_MOVE_M 안에서만 위치를 그대로 두고, 넘으면 None.
    """
    x, y, yaw = anchor
    dx = current[0] - start[0]
    dy = current[1] - start[1]
    if delta is None:
        if hypot(dx, dy) > UNCALIBRATED_MOVE_M:
            return None
        dx = dy = 0.0
    else:
        c, s = cos(delta), sin(delta)
        dx, dy = c*dx - s*dy, s*dx + c*dy
    return x + dx, y + dy, wrap(yaw + turn)


def nearest_structure(world, boxes, half, width, height, with_index=False):
    """점들 world (P, 2) 에서 가장 가까운 상자 면·외벽까지: (거리 (P,), 그 면·벽 위의 점 (P, 2)).

    with_index 이면 세 번째로 그 구조의 번호 (P,) 도: 0 … B-1 = 상자, B … B+3 = 외벽 (벽을 안 쓰면 없음).

    numpy 로 한 번에 계산한다 (2026-11-18: 점마다 파이썬으로 돌리면 로봇 CPU 에서 스캔 한 번 정합에 114 ms 가 걸려
    10 Hz 스캔을 따라가지 못했다). 상자가 여럿 같은 거리면 먼저 적힌 것, 그다음 벽 순서로 고른다.
    """
    world = np.asarray(world, dtype=float).reshape(-1, 2)
    wx, wy = world[:, 0], world[:, 1]
    distances, xs, ys = [], [], []
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 2)
    if len(boxes):
        dx = wx[:, None] - boxes[None, :, 0]
        dy = wy[:, None] - boxes[None, :, 1]
        # 상자 밖: 상자 위의 가장 가까운 점
        qx = boxes[None, :, 0] + np.clip(dx, -half, half)
        qy = boxes[None, :, 1] + np.clip(dy, -half, half)
        distance = np.hypot(wx[:, None] - qx, wy[:, None] - qy)
        # 상자 안: 가까운 쪽 면
        gap_x, gap_y = half - np.abs(dx), half - np.abs(dy)
        inside = (gap_x >= 0) & (gap_y >= 0)
        if inside.any():
            by_x = gap_x < gap_y
            face_x = boxes[None, :, 0] + np.where(dx >= 0, half, -half)
            face_y = boxes[None, :, 1] + np.where(dy >= 0, half, -half)
            qx = np.where(inside, np.where(by_x, face_x, wx[:, None]), qx)
            qy = np.where(inside, np.where(by_x, wy[:, None], face_y), qy)
            distance = np.where(inside, np.where(by_x, gap_x, gap_y), distance)
        distances.append(distance)
        xs.append(qx)
        ys.append(qy)
    if width > 0 and height > 0:
        zero = np.zeros_like(wx)
        distances.append(np.stack((np.abs(wx), np.abs(width - wx), np.abs(wy), np.abs(height - wy)), axis=1))
        xs.append(np.stack((zero, zero + width, wx, wx), axis=1))
        ys.append(np.stack((wy, wy, zero, zero + height), axis=1))
    if not distances:
        empty = (np.full(len(world), np.inf), world.copy())
        return (*empty, np.full(len(world), -1)) if with_index else empty
    distance = np.concatenate(distances, axis=1)
    index = np.argmin(distance, axis=1)
    rows = np.arange(len(world))
    target = np.stack((np.concatenate(xs, axis=1)[rows, index], np.concatenate(ys, axis=1)[rows, index]), axis=1)
    if with_index:
        return distance[rows, index], target, index
    return distance[rows, index], target


def to_world(points, pose):
    """차체 좌표 점 (P, 2) 를 자세 pose 의 경기장 좌표로."""
    x, y, yaw = pose
    c, s = cos(yaw), sin(yaw)
    return np.stack((x + c*points[:, 0] - s*points[:, 1], y + s*points[:, 0] + c*points[:, 1]), axis=1)


def refine(points, pose, boxes, half, width, height, iterations=8):
    """점 → 가장 가까운 상자 면·벽 대응으로 반복 최소제곱(ICP). 대응이 모자라면 None."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    x, y, yaw = pose
    for iteration in range(iterations):
        gate = 0.10 if iteration < 3 else 0.06
        distance, target = nearest_structure(to_world(points, (x, y, yaw)), boxes, half, width, height)
        keep = distance < gate
        if int(keep.sum()) < 8:
            return None
        source, target = points[keep], target[keep]
        # target ≈ R(theta)·source + t (rigid_fit 과 같은 식)
        source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
        p, q = source - source_mean, target - target_mean
        nyaw = atan2(float((p[:, 0]*q[:, 1] - p[:, 1]*q[:, 0]).sum()), float((p*q).sum()))
        c, s = cos(nyaw), sin(nyaw)
        nx = float(target_mean[0] - (c*source_mean[0] - s*source_mean[1]))
        ny = float(target_mean[1] - (s*source_mean[0] + c*source_mean[1]))
        step = hypot(nx - x, ny - y) + abs(wrap(nyaw - yaw)) * 0.2
        x, y, yaw = nx, ny, nyaw
        if step < 0.001:
            break
    return x, y, wrap(yaw)


def scan_points(scan, step=5):
    """/scan 을 차체(base_link) 좌표 점으로 (step 개에 1개, 정합은 5). frame 이 다르면 None."""
    if scan.header.frame_id != 'base_lidar_link':
        return None
    points = []
    for index in range(0, len(scan.ranges), step):
        distance = scan.ranges[index]
        if isfinite(distance) and scan.range_min <= distance < scan.range_max:
            angle = scan.angle_min + index * scan.angle_increment
            # Measured base_link to base_lidar_link mounting transform.
            px, py = LIDAR_X - distance*sin(angle), distance*cos(angle)
            if abs(px) <= SELF_HALF_LENGTH and abs(py) <= SELF_HALF_WIDTH:
                continue  # 우리 몸에 달린 것 (2026-11-19 16:55 실차: 뒤쪽 안 보이는 구간의 양 끝에 매 스캔 찍혔다)
            points.append((px, py))
    return points


def structure(field, use_walls=USE_WALLS, boxes=None, use_pillars=None):
    """(상자 중심들 (B, 2) 배열, 상자 반 크기, 벽 width, 벽 height). 벽을 안 쓰면 width·height = 0.

    boxes 를 주면 지도의 상자 대신 그것을 쓴다 (라이다로 고친 상자 — map_fusion).
    상자 목록 끝에 고정 기둥을 더한다 (use_pillars 를 안 주면 USE_PILLARS. 이미 그 자리에 상자가 있으면 뺀다).
    """
    if boxes is None:
        boxes = [(box.x, box.y) for box in field.boxes]
    centres = [(float(x), float(y)) for x, y in np.asarray(boxes, dtype=float).reshape(-1, 2)]
    full_width = float(getattr(field, 'width', 0.0) or 0.0)
    full_height = float(getattr(field, 'height', 0.0) or 0.0)
    if USE_PILLARS if use_pillars is None else use_pillars:
        half = field.box_size / 2
        centres += [p for p in fixed_pillars(full_width, full_height, field.box_size)
                    if not any(abs(p[0] - x) < half and abs(p[1] - y) < half for x, y in centres)]
    boxes = np.array(centres, dtype=float).reshape(-1, 2)
    width = full_width if use_walls else 0.0
    height = full_height if use_walls else 0.0
    return boxes, field.box_size / 2, width, height


def trusted_faces(width, height):
    """벽으로 다시 보기에 외벽과 함께 쓸 기둥 옆면 (drive/fieldmap.pillar_side_faces). 끄면 (TRUST_PILLARS · USE_PILLARS) 빈 것."""
    if not (TRUST_PILLARS and USE_PILLARS) or not (width > 0 and height > 0):
        return ()
    return pillar_side_faces(width, height, fixed_pillars(width, height))


def wall_search(points, pose, width, height, reach=WALL_REACH_M, step=WALL_STEP_M, yaws=WALL_YAWS, faces=None):
    """외벽(과 기둥 옆면)만으로 본 위치를 한 번 훑어 두 가지로: [외벽만, 외벽 + 기둥 옆면] 각각 (지금 자리 점수, 가장 잘 맞는 자리 점수, 그 자리).

    점수 = 외벽 WALL_HIT_M 안의 점 수 (+ 기둥 옆면 WALL_HIT_M 안의 점 수) − 2 x 벽 밖 OUTSIDE_M 넘게 나간 점 수.
    pose 둘레 ±reach 를 step 간격으로, 방향은 yaws 만큼 틀어 본다. faces = 기둥 옆면 (None 이면 trusted_faces, () 면 둘째도 외벽만).
    x · y 를 따로 옮긴 표 두 장의 곱으로 센다 (자리 625 x 방향 5 개가 이 PC 에서 1 ms 안쪽). 옆면도 「선 가까이(한 축) x 선분 안(다른 축)」 의 곱이다.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    faces = trusted_faces(width, height) if faces is None else faces
    x, y, yaw = pose
    offsets = np.arange(-reach, reach + 1e-9, step)
    centre = len(offsets) // 2
    best = [(-np.inf, pose), (-np.inf, pose)]
    here = [0.0, 0.0]
    for turn in yaws:
        c, s = cos(yaw + turn), sin(yaw + turn)
        wx = (x + c*points[:, 0] - s*points[:, 1])[:, None] + offsets[None, :]      # (P, N) — x 를 옮긴 것
        wy = (y + s*points[:, 0] + c*points[:, 1])[:, None] + offsets[None, :]      # (P, N) — y 를 옮긴 것
        near_x = ((np.abs(wx) < WALL_HIT_M) | (np.abs(wx - width) < WALL_HIT_M)).astype(float)
        near_y = ((np.abs(wy) < WALL_HIT_M) | (np.abs(wy - height) < WALL_HIT_M)).astype(float)
        in_x = ((wx > -OUTSIDE_M) & (wx < width + OUTSIDE_M)).astype(float)
        in_y = ((wy > -OUTSIDE_M) & (wy < height + OUTSIDE_M)).astype(float)
        # [x 옮김, y 옮김] 표: 왼 · 오른 벽에 맞은 점 + 위 · 아래 벽에 맞은 점 − 모퉁이에서 두 번 센 점 − 2 x 벽 밖의 점
        wall = near_x.T @ in_y + in_x.T @ near_y - near_x.T @ near_y - 2.0 * (len(points) - in_x.T @ in_y)
        both = wall.copy()
        for axis, at, lo, hi in faces:  # 기둥 옆면: 선분은 붙은 벽에서 WALL_HIT_M 넘게 떨어진 곳부터라 외벽 점과 겹쳐 세지 않는다
            # 옮겨 보는 범위(±reach) 안에서 그 면에 닿을 수 있는 점만 (계산량 — 면 12개 x 방향 5개)
            ax, ay = (wx, wy) if axis == 'y' else (wy, wx)   # ax = 면을 따라가는 축, ay = 면에 수직인 축
            near = (np.abs(ay[:, centre] - at) < reach + WALL_HIT_M) & (ax[:, centre] > lo - reach) & (ax[:, centre] < hi + reach)
            if not near.any():
                continue
            band = ((ax[near] > lo) & (ax[near] < hi)).astype(float)
            hit = (np.abs(ay[near] - at) < WALL_HIT_M).astype(float)
            both += band.T @ hit if axis == 'y' else hit.T @ band
        for k, score in enumerate((wall, both)):
            if turn == 0.0:
                here[k] = float(score[centre, centre])
            i, j = np.unravel_index(int(np.argmax(score)), score.shape)
            if score[i, j] > best[k][0]:
                best[k] = (float(score[i, j]), (x + float(offsets[i]), y + float(offsets[j]), wrap(yaw + turn)))
    return [(here[k], best[k][0], best[k][1]) for k in (0, 1)]


def wall_candidate(points, pose, width, height, reach=WALL_REACH_M, step=WALL_STEP_M, yaws=WALL_YAWS, faces=None):
    """외벽 + 기둥 옆면(faces, 기본 trusted_faces)으로 본 위치: (지금 자리의 벽 점수, 가장 잘 맞는 자리의 벽 점수, 그 자리 (x, y, yaw))."""
    return wall_search(points, pose, width, height, reach, step, yaws, faces)[1]


def _wall_gate(here, there, count):
    """옮길 자리의 벽 점수가 「훨씬」 나은가: 점의 WALL_MIN_SHARE 이상이고 지금의 WALL_GAIN 배 + 점의 WALL_MARGIN_SHARE 를 넘는다."""
    return there >= WALL_MIN_SHARE * count and there >= WALL_GAIN * max(here, 0.0) + WALL_MARGIN_SHARE * count


def wall_suspect(points, pose, width, height):
    """지금 자리보다 외벽에 훨씬 잘 맞는 다른 자리가 있으면 (지금 벽 점수, 그 자리 벽 점수, 그 자리), 없으면 None.

    「훨씬」 = WALL_MIN_SHIFT_M 넘게 떨어져 있고 _wall_gate 를 넘는다. 외벽 + 기둥 옆면으로 먼저 보고, 아니면 외벽만으로(2026-11-19 처음 만든 대로) 본다.
    - 옆면을 먼저: 둘 다 넘으면 옆면까지 본 자리가 벽을 따라가는 방향도 맞다 (16:55 판 290 s: 외벽만으로는 y 를 +0.09 로 잘못 가리켜 이어진 판정이 끊겼다).
    - 외벽만도 본다: 합친 점수만 보면 틀린 자리에서도 맞는 옆면(그 옆면을 따라 미끄러진 경우) 점수가 양쪽에 더해져 1.5 배 기준이 묽어진다 —
      18:06 판의 「y 가 0.12 m 틀림」 스캔이 그렇게 빠졌다. 그래서 전에 잡던 것은 그대로 잡고, 옆면으로 새로 잡히는 것만 더한다.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) < 40 or not (width > 0 and height > 0):
        return None
    for here, there, where in reversed(wall_search(points, pose, width, height)):
        if hypot(where[0] - pose[0], where[1] - pose[1]) >= WALL_MIN_SHIFT_M and _wall_gate(here, there, len(points)):
            return here, there, where
    return None


def score_pose(points, pose, boxes, half, width, height, weights=None):
    """(상자 면·벽에 8 cm 안으로 맞은 점 수 − 외벽 밖으로 나간 점 수, 12 cm 로 자른 거리 합).

    라이다는 외벽 너머를 볼 수 없다. 그래서 점이 벽 밖 OUTSIDE_M 넘게 나가는 자세는 그만큼 점수를 깎는다
    (2026-11-18 실차: 지도에 상자가 빠져 있을 때 점들이 벽 밖으로 나간 틀린 자세가 정합에 통과했다).
    weights (B,) 를 주면 상자에 맞은 점은 그 상자의 무게로 센다 (벽 = 1). 라이다로 확인 못 한 드론 상자를 덜 믿는다.
    무게를 준 점수는 후보 자세끼리 견줄 때만 쓴다. 「맞은 점이 충분한가」(통과 기준)는 enough_hits 로 무게 없이 센다.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    world = to_world(points, pose)
    if weights is None:
        distance, _ = nearest_structure(world, boxes, half, width, height)
        residual = np.minimum(0.12, distance)
        hits = int((residual < 0.08).sum())
    else:
        distance, _, index = nearest_structure(world, boxes, half, width, height, with_index=True)
        residual = np.minimum(0.12, distance)
        weight = np.concatenate((np.asarray(weights, dtype=float), np.ones(4)))[index]
        hits = float(weight[residual < 0.08].sum())
    if width > 0 and height > 0:
        hits -= int(((world[:, 0] < -OUTSIDE_M) | (world[:, 0] > width + OUTSIDE_M)
                     | (world[:, 1] < -OUTSIDE_M) | (world[:, 1] > height + OUTSIDE_M)).sum())
    return hits, float(residual.sum())


def _pad_weights(weights, count):
    """상자 무게(map_fusion) 뒤에, structure() 가 끝에 더한 고정 기둥의 무게 1 을 붙인다. 무게가 없으면 None."""
    if weights is None:
        return None
    weights = np.asarray(weights, dtype=float)
    return np.concatenate((weights, np.ones(max(0, count - len(weights)))))


def enough_hits(points, pose, boxes, half, width, height):
    """이 자세가 정합으로 통과할 만큼 맞았나: 무게 없이 센 맞은 점 수가 max(12, 점의 25 %) 이상.

    무게(확인 못 한 드론 상자 0.7)를 준 점수를 이 기준에 견주면 안 된다. 경기 시작 때는 모든 드론 상자가 확인 전이라
    같은 스캔 · 같은 자세가 무게만으로 기준 아래로 떨어지고, 정합이 안 되니 상자를 확인할 기회도 없다
    (2026-11-19 실차 기록 1118_201710: 무게 없이 19점 / 무게 주면 13.9점 / 기준 17점 → 출발 칸 탐색이 105 s 실패).
    """
    return score_pose(points, pose, boxes, half, width, height)[0] >= max(12, len(points)*0.25)


def _no_structure(boxes, width, height):
    """맞출 것이 없다. 상자가 하나도 없어도 외벽이 있으면 벽만으로 맞춘다 (드론이 상자를 못 찾은 지도)."""
    return not len(boxes) and not (width > 0 and height > 0)


def scan_match(field, scan, seed, initializing, use_walls=USE_WALLS, boxes=None, weights=None):
    """Match returns against fixed box faces (and arena walls) in the field frame.

    boxes · weights: 지도 상자 대신 쓸 상자와 그 무게 (map_fusion 의 structure()). 없으면 지도 그대로, 무게 1.
    """
    points = scan_points(scan)
    if points is None or not 0 < field.box_size:
        return None
    boxes, half, width, height = structure(field, use_walls, boxes)
    weights = _pad_weights(weights, len(boxes))
    if len(points) < 12 or _no_structure(boxes, width, height):
        return None
    points = np.array(points, dtype=float)

    def score(x, y, yaw):
        return score_pose(points, (x, y, yaw), boxes, half, width, height, weights)

    # 1단계: 넓게 (10 cm · 0.1 rad 간격, 초기화는 15° 간격 전 방향)
    x, y, yaw = seed
    headings = [step*pi/12 for step in range(24)] if initializing else [
        yaw + step*0.10 for step in range(-2, 3)
    ]
    candidates = []
    for heading in headings:
        for dx in (-0.10, 0, 0.10):
            for dy in (-0.10, 0, 0.10):
                pose = (x + dx, y + dy, heading)
                hits, error = score(*pose)
                candidates.append((hits, error, pose))
    candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))
    hits, error, best = candidates[0]
    if not enough_hits(points, best, boxes, half, width, height):
        return None
    for other_hits, other_error, pose in candidates[1:]:
        angle = pose[2] - best[2]
        if abs(atan2(sin(angle), cos(angle))) > 0.3 and (
            other_hits >= hits - 2 and other_error <= error + 0.15
        ):
            return None

    # 2단계: 좁게 (ICP). 1단계 결과에서 멀리 가거나 맞음이 나빠지면 1단계 결과를 쓴다
    refined = refine(points, best, boxes, half, width, height)
    if refined is not None and hypot(refined[0] - best[0], refined[1] - best[1]) <= 0.15 \
            and abs(wrap(refined[2] - best[2])) <= 0.25:
        _, refined_error = score(*refined)
        if enough_hits(points, refined, boxes, half, width, height) and refined_error <= error:
            return refined
    return (best[0], best[1], wrap(best[2]))


def wall_score(points, pose, width, height):
    """이 자세의 벽 점수 (외벽 + 기둥 옆면 — wall_candidate 와 같은 식, 한 자리만)."""
    return wall_candidate(points, pose, width, height, reach=0.0, yaws=(0.0,))[0]


def wall_scores(points, pose, width, height):
    """이 자세의 벽 점수 둘: (외벽만, 외벽 + 기둥 옆면)."""
    return tuple(r[0] for r in wall_search(points, pose, width, height, reach=0.0, yaws=(0.0,)))


def walls_prefer(points, old, new, width, height):
    """외벽(과 기둥 옆면)이 자세 new 를 old 보다 뚜렷이 편드는가 — 벽으로 다시 보기와 같은 기준(_wall_gate: 점의 WALL_MIN_SHARE 이상이고
    old 의 WALL_GAIN 배 + 점의 WALL_MARGIN_SHARE 를 넘는다), 외벽만 · 외벽 + 기둥 옆면 중 하나라도. points = 촘촘한 스캔 점 (scan_points(scan, 2))."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) < 40 or not (width > 0 and height > 0):
        return False
    here, there = wall_scores(points, old, width, height), wall_scores(points, new, width, height)
    return any(_wall_gate(a, b, len(points)) for a, b in zip(here, there))


def confirm_wall_candidate(field, scan, pose, where, boxes=None, weights=None):
    """외벽이 가리킨 자리 where 에서 다시 정합해 본다. 받을 만하면 그 자세, 아니면 None.

    boxes · weights 에는 라이다로 더한 상자를 뺀 것(드론 지도 상자)을 준다 — 더한 상자는 틀린 자세에서 생겼을 수 있어 판정에 쓰지 않는다.
    받는 조건: (1) 정합이 통과하고 where 에서 0.15 m 안에 머문다 (2) 지금 자리 pose 에서 WALL_MIN_SHIFT_M 넘게 떨어져 있다
    (방향은 WALL_MAX_TURN 안)
    (3) 다듬은 자리에서도 벽 점수가 wall_suspect 의 기준을 넘는다 (외벽만 또는 외벽 + 기둥 옆면) (4) 벽 · 상자에 맞은 점 전체(무게 없이)가 지금 자리보다 적지 않다.
    """
    second = scan_match(field, scan, where, False, boxes=boxes, weights=weights)
    if second is None or hypot(second[0] - where[0], second[1] - where[1]) > 0.15:
        return None
    if hypot(second[0] - pose[0], second[1] - pose[1]) < WALL_MIN_SHIFT_M or abs(wrap(second[2] - pose[2])) > WALL_MAX_TURN:
        return None
    dense = np.array(scan_points(scan, 2), dtype=float)
    width, height = float(field.width), float(field.height)
    here, there = wall_scores(dense, pose, width, height), wall_scores(dense, second, width, height)
    if not any(_wall_gate(a, b, len(dense)) for a, b in zip(here, there)):   # 외벽만 · 외벽 + 기둥 옆면 중 하나라도 (wall_suspect 와 같은 까닭)
        return None
    points = np.array(scan_points(scan), dtype=float)
    solid, half, width, height = structure(field, boxes=boxes)
    if score_pose(points, second, solid, half, width, height)[0] < score_pose(points, pose, solid, half, width, height)[0]:
        return None
    return second


def _steps(low, high, step):
    count = max(1, int(round((high - low) / step)))
    return [low + (high - low) * i / count for i in range(count + 1)]


def slot_search(field, scan, region, use_walls=USE_WALLS, boxes=None, weights=None, headings=None):
    """출발 칸(region = min_x, min_y, max_x, max_y) 전체에서 라이다만으로 자세를 찾는다.
    headings 를 주면 그 방향들만 본다 (잃은 위치 다시 찾기 — 방향을 대강 아는 때). 없으면 전 방향.

    항공 지도에서 우리 차를 찾지 않아도 된다. 칸 안을 SLOT_STEP_M · 360°/SLOT_HEADINGS 간격으로
    훑고, 서로 떨어진 상위 SLOT_SEEDS 개를 ICP 로 다듬어 가장 잘 맞는 것을 고른다.
    맞음이 모자라거나 비슷하게 맞는 다른 자세가 있으면 None.
    """
    points = scan_points(scan)
    if points is None or not 0 < field.box_size:
        return None
    boxes, half, width, height = structure(field, use_walls, boxes)
    weights = _pad_weights(weights, len(boxes))
    if len(points) < 12 or _no_structure(boxes, width, height):
        return None
    points = np.array(points, dtype=float)
    x0, y0, x1, y1 = region
    candidates = []
    for heading in (headings if headings is not None else [2*pi*step/SLOT_HEADINGS for step in range(SLOT_HEADINGS)]):
        for x in _steps(x0, x1, SLOT_STEP_M):
            for y in _steps(y0, y1, SLOT_STEP_M):
                pose = (x, y, heading)
                candidates.append((*score_pose(points, pose, boxes, half, width, height, weights), pose))
    candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))
    seeds = []
    for _, _, pose in candidates:
        if all(hypot(pose[0] - other[0], pose[1] - other[1]) > 0.15
               or abs(wrap(pose[2] - other[2])) > 0.2 for other in seeds):
            seeds.append(pose)
            if len(seeds) == SLOT_SEEDS:
                break
    results = []
    for pose in seeds:
        for _ in range(3):   # ICP 한 번은 1단계에서 0.15 m · 0.25 rad 안쪽만 믿으므로 몇 번 잇는다
            refined = refine(points, pose, boxes, half, width, height)
            if refined is None:
                break
            pose = refined
        if x0 - 0.10 <= pose[0] <= x1 + 0.10 and y0 - 0.10 <= pose[1] <= y1 + 0.10:
            results.append((*score_pose(points, pose, boxes, half, width, height, weights), pose))
    results.sort(key=lambda result: (-result[0], result[1]))
    if not results or not enough_hits(points, results[0][2], boxes, half, width, height):
        return None
    hits, error, best = results[0]
    if weights is not None:
        # 「비슷하게 맞는 다른 자세가 있는가」 도 통과 기준처럼 무게 없이 센다 (2026-11-19, supervisor 쪽에서 고침 — navigation 담당 확인 필요).
        # 무게 0.7 로 센 점수에 「맞은 점 2개 차이」 를 그대로 견주면 2.9개 차이가 되어, 모든 상자가 확인 전인 경기 시작 때 멀쩡한 자리를
        # 헷갈린다고 버렸다 (test_match_threshold 의 실차 스캔 scans[2]: 19 · 18 · 16점 → 13.6 · 12.9 · 11.8 로 세어 None)
        results = [(*score_pose(points, pose, boxes, half, width, height), pose) for _, _, pose in results]
        hits, error, _ = results[0]
    for other_hits, other_error, pose in results[1:]:
        far = hypot(pose[0] - best[0], pose[1] - best[1]) > 0.10 or abs(wrap(pose[2] - best[2])) > 0.15
        if far and other_hits >= hits - 2 and other_error <= error + 0.15:
            return None
    return (best[0], best[1], wrap(best[2]))


def fused_map(field, kept):
    """드론 지도 field 의 복사본에서 라이다로 뺀 상자를 뺀다. kept = 남길 상자 [(x, y, 드론 지도 번호)].

    - 상자는 kept 순서. 뺀 상자의 면은 빠지고, 남은 상자의 면은 새 번호(faces[].box)를 쓴다 (sector · image_ref 는 그대로).
    - 뺀 상자와 맞닿아 있어 원래 없던 면은 되살리지 않는다 (supervisor 는 면을 상자에서 다시 만든다).
    - 라이다로 더한 상자는 여기에 오지 않는다 (정합에만 쓴다 — map_fusion.MapFusion.output).
    받은 field 는 건드리지 않는다.
    """
    output = deepcopy(field)
    renumber = {source: index for index, (_, _, source) in enumerate(kept)}
    output.boxes = [Point(x=float(x), y=float(y), z=0.0) for x, y, _ in kept]
    faces = []
    for face in getattr(output, 'faces', []):
        if face.box in renumber:
            face.box = renumber[face.box]
            faces.append(face)
    output.faces = faces
    return output


def lidar_boxes(fusion):
    """라이다로 찾아 정합에만 쓰는 상자의 가운데 [(x, y)] (mm 까지) — /team/navigation/lidar_boxes 로 낼 목록.

    지도 고치기를 껐으면(PassThroughMap — 더한 상자가 없다) None: 그때는 토픽을 내지 않는다.
    """
    added = getattr(fusion, 'added', None)
    if added is None:
        return None
    return [(round(float(x), 3), round(float(y), 3)) for x, y in added()]


class PassThroughMap:
    """map_fusion:=false 일 때 MapFusion 대신 들어간다: 드론 지도 상자를 그대로 쓰고 아무것도 고치지 않는다."""

    version = 0

    def __init__(self):
        self.reset()

    def reset(self):
        self._boxes = np.zeros((0, 2))

    def set_map(self, boxes, box_size, width, height, fixed=()):
        self._boxes = np.asarray(boxes, dtype=float).reshape(-1, 2)

    def structure(self, added=True):
        return self._boxes, None

    def distrust(self, since_ns):
        return False

    def wants_scan(self):
        return False

    def observe(self, *args, **kwargs):
        return False

    def changed(self):
        return False

    def output(self):
        return [(float(x), float(y), index) for index, (x, y) in enumerate(self._boxes)]

    def summary(self):
        return None


# 늦게 뜨거나 다시 뜬 주행 노드도 마지막 지도를 바로 받게 한다 (2026-11-18 사용자 결정).
# 구독하는 주행 노드(driving_node.py)도 같은 설정이어야 연결된다.
MAP_QOS = QoSProfile(
    depth=1,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class SlamNode(Node):
    """Maintain the field map and propagate the observed ego pose with odometry and IMU yaw."""

    def __init__(self):
        super().__init__('navigation_slam')
        self.yaw_source = str(self.declare_parameter('yaw_source', 'wheel').value)
        self.wheel_base_m = float(self.declare_parameter('wheel_base_m', DEFAULT_WHEEL_BASE_M).value)
        globals()['USE_PILLARS'] = bool(self.declare_parameter('fixed_pillars', USE_PILLARS).value)
        globals()['TRUST_PILLARS'] = bool(self.declare_parameter('trust_pillars', TRUST_PILLARS).value)   # 기둥 옆면을 믿는 구조로 (위 TRUST_PILLARS)
        self.latest_perception_map = None
        self._odometry = deque(maxlen=200)
        self._odom_frames = None
        self._imu = deque(maxlen=400)      # (시각, 이어 붙인 yaw) — yaw_source 가 wheel 이면 바퀴로 쌓은 yaw
        self._wheels = None                # 마지막 (왼쪽, 오른쪽) 누적 거리
        self._wheel_yaw = 0.0
        self._anchors = deque(maxlen=80)   # (시각, 출처, 경기장 x·y, odom x·y) — delta 맞추기용
        self._delta = None                 # odom 축 → 경기장 축 각도
        self._last_output_stamp = -1
        self._assignment = None
        self._scan_anchor = None
        self._locked = None        # 잠긴 동안(waiting·ready) 마지막 (odom x, odom y, 방향 yaw 또는 None)
        self._slot_pose = None     # 출발 칸에서 찾은 경기장 자세 (x, y, yaw)
        self._slot_attempt = None  # 마지막 출발 칸 탐색의 스캔 시각
        self._slot_note = ''       # 출발 칸 탐색을 왜 했는지/안 했는지 (기록 · 시험 화면용)
        self._scan_note = {}       # 마지막 스캔 처리 결과 (기록 · 시험 화면용)
        self._published = 0        # 내보낸 위치 수
        # 위치 검증 (파일 위의 「위치 검증」 설명). pose_check:=false 로 끈다
        self.pose_check = bool(self.declare_parameter('pose_check', True).value)
        self._path = deque(maxlen=200)     # (/odom 시각, 그때까지 /odom 이 간 거리 m)
        self._accepted = deque(maxlen=40)  # 받아들인 정합 (스캔 시각, x, y, 그때까지 간 거리)
        self._jump_rejects = 0             # 이어서 버린 정합 수
        self._wall_streak = []             # 이어진 「벽이 다른 자리를 가리킴」 판정 [(스캔 시각, 옮길 x, 옮길 y)]
        self._yaw_hint = None              # 뜀 거르기가 버린 정합의 (시각, 방향) — 다음 정합의 씨앗 방향 (JUMP_YAW_HINT_MAX)
        self._fusion_hold = 0              # 이 시각(ns)까지 지도를 고치지 않는다 (위치를 다시 잡은 직후 — FUSION_HOLD_NS)
        self._wall_doubt = False           # 이번 스캔에서 외벽이 다른 자리를 가리켰다 (다시 정합이 받지 않았어도) → 이 스캔으로는 지도를 고치지 않는다
        self._pose_checks = {'jump_rejected': 0, 'jump_forced': 0, 'jump_wall_ok': 0, 'wall_suspect': 0, 'relocated': 0, 'fusion_skipped': 0, 'fusion_held': 0, 'yaw_hints': 0,
                             'lost_tries': 0, 'lost_found': 0,
                             'drone_seen': 0, 'drone_missed': 0, 'drone_doubt': 0, 'drone_tries': 0, 'drone_moved': 0}
        # 드론이 본 우리 자리로 위치를 의심하기 (파일 위의 설명). drone_pose_check:=false 로 끈다 (기록은 남는다)
        self.drone_pose_check = bool(self.declare_parameter('drone_pose_check', True).value)
        self._drone_others = []            # 우리 차가 보인 사진의 나머지 = 다른 UGV [[마지막으로 본 사진 시각 ns, x, y]] (그 뒤로 이어 따라간다)
        self._drone_stray = None           # 우리 차가 안 보일 때 우리를 따라오는 「모르는 것」: (사진 시각, 우리 자리와의 차 x, y, 이어진 장 수, 처음 본 시각)
        self._drone_off = None             # 마지막으로 우리 차가 보였을 때 드론 자리 − SLAM 자리 (x, y)
        self._drone_next = None            # 이 스캔 시각(ns)부터 다시 훑는다
        self._drone_pending = None         # 한 번 뚜렷이 나았던 자세 (다음에도 같으면 옮긴다)
        # 잃은 위치 다시 찾기 (파일 위의 설명). relocate_lost:=false 로 끈다
        self.relocate_lost = bool(self.declare_parameter('relocate_lost', True).value)
        self._relocate_next = None         # 이 스캔 시각(ns)부터 다시 해 본다
        self._relocate_pending = None      # 한 번 찾았지만 아직 받지 않은 자세 (다음에도 같으면 받는다)
        self._relocate_note = ''           # 마지막으로 해 본 결과 (기록 · 시험 화면용)
        # 라이다로 드론 지도 상자 고치기 (navigation/map_fusion.py). false 면 드론 지도를 그대로 쓴다
        self._fusion = MapFusion(FusionParams(trust_pillar_faces=TRUST_PILLARS)) if bool(self.declare_parameter('map_fusion', True).value) else PassThroughMap()
        self._fused_cache = None   # (지도 객체, 고친 횟수, 고친 지도)
        self.perception_map_subscription = self.create_subscription(
            FieldMap, '/team/perception/field_map', self._receive_perception_map, 1
        )
        self.map_publisher = self.create_publisher(
            FieldMap, '/team/navigation/map', MAP_QOS
        )

        self.pose_publisher = self.create_publisher(PoseStamped, '/team/navigation/pose', 5)
        # 시험 화면(시험기록/web_drive)용 상태. debug:=true 일 때만 2 Hz 로 낸다
        self.debug_publisher = self.create_publisher(String, '/team/navigation/debug', 5)
        if bool(self.declare_parameter('debug', False).value):
            self.debug_timer = self.create_timer(0.5, self._publish_debug)
        # 라이다로 찾은 상자(map_fusion 이 정합에만 쓰는 것)를 따로 알린다 (2026-11-19). 지도(/team/navigation/map)에는 전처럼 넣지 않는다 —
        # 주행 노드가 지도의 상자를 안 움직이는 벽으로 다루기 때문이다. supervisor 가 이 목록으로 지도에 없는 상자의 면을 찍으러 간다
        # (드론 사진이 위 · 아래 벽 쪽에서 잘려, 벽에 붙은 상자가 지도에 없다). 1초마다, 그리고 목록이 바뀐 때 낸다. map_fusion:=false 면 내지 않는다
        self.lidar_boxes_publisher = self.create_publisher(PoseArray, '/team/navigation/lidar_boxes', 1)
        self._lidar_boxes_sent = None      # 마지막으로 낸 목록 [(x, y)]
        if lidar_boxes(self._fusion) is not None:
            self.lidar_boxes_timer = self.create_timer(1.0, self._publish_lidar_boxes)
        # 라이다 스캔 원본을 이 간격(초)으로 기록 파일에 남긴다 (event scan_raw). 0 이면 남기지 않는다. 기록 폴더(TEAM_LOG_DIR)가 있을 때만
        self.log_raw_scan_s = float(self.declare_parameter('log_raw_scan_s', 1.0).value)
        self._raw_scan_at = None           # 마지막으로 스캔 원본을 남긴 시각 (노드 시계, ns)
        # 같은 상태를 이번 실행의 기록 폴더에도 (TEAM_LOG_DIR 이 있을 때만 — team_node.py 가 켰을 때). 경기 중에는 ros2 로 못 봐서 (2026-11-19)
        self.run_log = RunLog('navigation_slam')
        log_hz = float(self.declare_parameter('log_hz', 2.0).value)
        if self.run_log.enabled:
            self.run_log.write({'event': 'start', 't': round(self.get_clock().now().nanoseconds / 1e9, 3), 'yaw_source': self.yaw_source,
                                'fixed_pillars': USE_PILLARS, 'trust_pillars': TRUST_PILLARS})
            if log_hz > 0.0:
                self.log_timer = self.create_timer(1.0 / log_hz, self._write_log)
            self.get_logger().info('내부 상태 기록: %s' % self.run_log.path)

        self.odom_subscription = self.create_subscription(
            Odometry, '/odom', self._receive_odometry, qos_profile_sensor_data
        )
        self.imu_subscription = self.create_subscription(
            Imu, '/imu/data', self._receive_imu, qos_profile_sensor_data
        )
        self.wheel_subscription = self.create_subscription(
            Float32MultiArray, '/odom/odom_raw', self._receive_wheels, qos_profile_sensor_data
        )

        self.scan_subscription = self.create_subscription(
            LaserScan, '/scan', self._receive_scan, qos_profile_sensor_data
        )
        self.server_subscription = self.create_subscription(
            ServerState, '/team/server/state', self._receive_server_state, 1
        )
        # 드론이 본 UGV 자리 (perception/ugv_watch.py, 약 1 Hz, 우리 차 포함). 지도 고치기(map_fusion)에서 다른 UGV 에 맞은 라이다 점을 뺀다 —
        # 서 있는 UGV 가 상자 후보가 되거나 뺀 상자를 되살리지 않게 (2026-11-19). drone_ugvs:=false 면 받지 않는다
        self._ugv_tracks = UgvTracks(DriveParams())   # 다른 UGV 들의 궤적 (우리 차는 골라낸다) — 주행 노드와 같은 규칙
        self._ugv_seen = None              # (사진 시각 ns, 드론이 본 UGV 모두 [(x, y)] — 우리 차 포함). 위치를 잃었을 때 다시 찾을 자리의 후보
        self._pose_trail = deque(maxlen=40)  # 받아들인 자세 (스캔 시각 ns, x, y) — 드론 사진에서 우리 차를 골라내는 데 쓴다
        self._ugv_masked = 0               # 지도 고치기에서 뺀 점 수 (기록용)
        if bool(self.declare_parameter('drone_ugvs', True).value):
            self.ugvs_subscription = self.create_subscription(PoseArray, '/team/perception/ugvs', self._receive_ugvs, qos_profile_sensor_data)

    def _reset_calibration(self):
        self._anchors.clear()
        self._delta = None

    def _receive_server_state(self, message):
        if self._assignment is not None and message.run_id != self._assignment.run_id:
            self._scan_anchor = None
            self._forget_checks()
            self._reset_slot()
            self.latest_perception_map = None
            self._fusion.reset()
            self._reset_calibration()
            self._ugv_tracks.clear()
            self._ugv_seen = None
            self._drone_others, self._drone_stray, self._drone_off = [], None, None
            self._drone_next = self._drone_pending = None
        self._assignment = message

    def _receive_ugvs(self, message):
        """드론이 본 UGV 들을 궤적(UgvTracks)에 넣는다 — 우리 차는 그 사진 무렵 우리가 있던 자리로 골라낸다. 그때의 자세를 모르면 그 사진은 버린다."""
        stamp = stamp_ns(message.header)
        if message.header.frame_id != 'arena':
            return
        points = [(float(p.position.x), float(p.position.y)) for p in message.poses]
        self._ugv_seen = (stamp, points)   # 자세를 몰라도 들고 있는다 (_relocate)
        window = int(self._ugv_tracks.p.ugv_self_window_s * 1e9)   # 사진은 header.stamp 보다 앞서 찍혔다 — 이만큼 앞까지의 우리 자리와 견준다
        recent = [(x, y) for at, x, y in self._pose_trail if -window <= at - stamp <= 200_000_000]
        self._ugv_tracks.on_frame(stamp / 1e9, points, recent)
        self._drone_self(stamp, points, recent)

    def _drone_self(self, stamp, points, recent):
        """드론 사진에서 우리 차를 골라 SLAM 자리와 견준다 (파일 위 「드론이 본 우리 자리로 위치를 의심하기」 의 1 · 2). recent = 그 무렵 우리가 있던 자리들."""
        if not recent:
            return
        gate = self._ugv_tracks.p.ugv_self_match_m

        def nearest(q):   # (우리 자리까지의 거리, 그 자리 x, y)
            return min((hypot(q[0] - x, q[1] - y), x, y) for x, y in recent)

        seen = [(nearest(q), q) for q in points]
        me = min((item for item in seen if item[0][0] <= gate), key=lambda item: item[0][0], default=None)
        self._drone_others = [o for o in self._drone_others if stamp - o[0] <= DRONE_OTHERS_KEEP_NS]
        tracks = self._ugv_tracks
        follow = max(DRONE_OTHER_M, min(1.5, tracks.p.ugv_track_gate_m * max(1.0, tracks.interval)))   # 사진 사이에 다른 UGV 가 갈 수 있는 거리

        def known(q, add):
            """q 가 기억한 다른 UGV 인가 — 맞으면 그 자리를 q 로 옮겨 둔다. add = 모르는 것이면 새로 기억한다."""
            best = min(self._drone_others, key=lambda o: hypot(q[0] - o[1], q[1] - o[2]), default=None)
            if best is not None and hypot(q[0] - best[1], q[1] - best[2]) <= follow:
                best[0], best[1], best[2] = stamp, q[0], q[1]
                return True
            if add:
                self._drone_others.append([stamp, q[0], q[1]])
            return False

        if me is not None:
            (_, x, y), q = me
            self._drone_off = (q[0] - x, q[1] - y)
            self._pose_checks['drone_seen'] += 1
            self._drone_stray = self._drone_pending = None
            for _near, other in seen:
                if other is not q:
                    known(other, add=True)
            return
        self._pose_checks['drone_missed'] += 1
        strays = [(near, q) for near, q in seen if not known(q, add=False) and near[0] <= DRONE_STRAY_REACH_M]
        if len(strays) != 1:
            self._drone_stray = self._drone_pending = None
            return
        (_, x, y), q = strays[0]
        off = (q[0] - x, q[1] - y)
        before = self._drone_stray
        same = before is not None and hypot(off[0] - before[1], off[1] - before[2]) <= DRONE_STRAY_SAME_M
        count = before[3] + 1 if same else 1
        self._drone_stray = (stamp, off[0], off[1], count, before[4] if same else stamp)
        if count == DRONE_DOUBT_FRAMES:
            self._pose_checks['drone_doubt'] += 1
            self._log_event('drone_doubt', stray=[round(q[0], 3), round(q[1], 3)], off=[round(off[0], 3), round(off[1], 3)], frames=count)

    def _drone_recheck(self, field, scan, result, timestamp):
        """드론으로 위치를 의심하는 동안: 우리를 따라오는 「모르는 것」 의 자리 둘레를 훑어, 스캔이 지금 자리 result 보다 뚜렷이 잘 맞으면 그 자세
        (이어진 두 번이 같을 때), 아니면 None (파일 위 설명의 3)."""
        stray = self._drone_stray
        if (stray is None or stray[3] < DRONE_DOUBT_FRAMES or not 0 <= timestamp - stray[0] <= UGV_MASK_AGE_NS
                or (self._drone_next is not None and timestamp < self._drone_next)):
            return None
        if abs(self._turn(timestamp - 200_000_000, timestamp) / 0.2) > WALL_MAX_YAW_RATE:
            return None      # 도는 중의 스캔은 번진다
        started = time.perf_counter()
        self._pose_checks['drone_tries'] += 1
        width, height = float(field.width), float(field.height)
        cx, cy, half = result[0] + stray[1], result[1] + stray[2], RELOCATE_UGV_M
        region = (max(0.10, cx - half), max(0.10, cy - half), min(width - 0.10, cx + half), min(height - 0.10, cy + half))
        pose = None
        boxes, weights = self._fusion.structure(added=False)   # 드론 지도만으로 — 틀렸을지 모르는 자리에서 더한 상자에 기대지 않는다
        if region[0] < region[2] and region[1] < region[3]:
            step = 2*pi/SLOT_HEADINGS
            headings = [result[2] + k*step for k in range(-int(RELOCATE_YAW/step), int(RELOCATE_YAW/step) + 1)]
            pose = slot_search(field, scan, region, boxes=boxes, weights=weights, headings=headings)
        took = time.perf_counter() - started
        self._drone_next = timestamp + max(DRONE_RECHECK_NS, int(3 * took * 1e9))
        here = there = None
        if pose is not None:
            solid, box_half, wall_w, wall_h = structure(field, boxes=boxes)
            points = scan_points(scan) or []
            here = score_pose(points, result, solid, box_half, wall_w, wall_h)[0]
            there = score_pose(points, pose, solid, box_half, wall_w, wall_h)[0]
        better = (pose is not None and hypot(pose[0] - result[0], pose[1] - result[1]) >= DRONE_MIN_SHIFT_M
                  and there > 0 and here <= RELOCATE_CLEAR * there)
        again = (better and self._drone_pending is not None
                 and hypot(pose[0] - self._drone_pending[0], pose[1] - self._drone_pending[1]) <= RELOCATE_SAME_M
                 and abs(wrap(pose[2] - self._drone_pending[2])) <= 0.2)
        self._log_event('drone_recheck', here=None if here is None else int(here), there=None if there is None else int(there),
                        pose=[round(v, 3) for v in pose] if pose is not None else None, was=[round(v, 3) for v in result],
                        better=bool(better), moved=bool(again), ms=round(took * 1000, 1))
        if not better:
            self._drone_pending = None
            return None
        if not again:
            self._drone_pending = pose
            return None
        self._drone_pending = self._drone_stray = None
        self._pose_checks['drone_moved'] += 1
        undone = self._fusion.distrust(stray[4] - WALL_DISTRUST_NS)
        self.get_logger().warning('위치를 드론이 본 자리에서 다시 맞춰 옮김: (%.2f, %.2f, %.0f°) → (%.2f, %.2f, %.0f°), 맞은 점 %d → %d%s'
                                  % (result[0], result[1], result[2] * 180 / pi, pose[0], pose[1], pose[2] * 180 / pi, here, there,
                                     ' · 그 사이 고친 지도를 되돌림' if undone else ''))
        if undone:
            self._log_event('map_fusion', **self._fusion.summary())
            self._publish_lidar_boxes(changed_only=True)
        self._anchors.clear()
        self._accepted.clear()
        return pose

    def _without_ugv_points(self, points, pose, timestamp):
        """차체 좌표 점들에서, 드론이 본 다른 UGV(가운데에서 ugv_match_m 안)에 맞은 것을 뺀다. 지금 믿는 궤적만 (두 번 넘게 보였고 요즘 것 — UgvTracks.fresh)."""
        centres = [(u.x, u.y) for u in self._ugv_tracks.fresh(timestamp / 1e9)]
        if not centres:
            return points
        kept, dropped = without_points_near(points, pose, centres, self._ugv_tracks.p.ugv_match_m)
        self._ugv_masked += dropped
        return kept

    def _reset_slot(self):
        self._locked = None
        self._slot_pose = None
        self._slot_attempt = None
        self._relocate_next = self._relocate_pending = None

    def _receive_perception_map(self, message):
        if (self.latest_perception_map is not None
                and stamp_ns(message.header) <= stamp_ns(self.latest_perception_map.header)):
            return
        self.latest_perception_map = message
        width, height, size = (float(getattr(message, name, 0.0) or 0.0) for name in ('width', 'height', 'box_size'))
        self._fusion.set_map([(box.x, box.y) for box in getattr(message, 'boxes', [])], size, width, height,
                             fixed=fixed_pillars(width, height, size) if USE_PILLARS and size > 0 else ())
        if self.run_log.enabled:
            self._log_event('map', stamp=round(stamp_ns(message.header) / 1e9, 3), boxes=len(message.boxes),
                            vehicles=[[int(v.role), round(v.pose.position.x, 3), round(v.pose.position.y, 3)] for v in message.vehicles])
        selves = [planar_pose(vehicle.pose) for vehicle in message.vehicles
                  if vehicle.role == vehicle.SELF]
        if any(pose is not None for pose in selves):
            self._scan_anchor = None
            if len(selves) == 1:
                observed = stamp_ns(message.header)
                odom = self._odometry_at(observed)
                if odom is not None:
                    self._record_anchor(observed, 'field', selves[0], odom)
        if not self._publish_prediction() and stamp_ns(message.header) >= self._last_output_stamp:
            self.map_publisher.publish(self._fused(message))
            self._last_output_stamp = stamp_ns(message.header)

    def _receive_odometry(self, message):
        pose = planar_pose(message.pose.pose)
        timestamp = stamp_ns(message.header)
        frames = (message.header.frame_id, message.child_frame_id)
        if pose is None or timestamp <= 0 or not all(frames):
            return
        if frames != self._odom_frames:
            self._odometry.clear()
            self._path.clear()
            self._forget_checks()
            self._scan_anchor = None
            self._reset_calibration()
            self._reset_slot()
            self._odom_frames = frames
        if self._odometry and timestamp <= self._odometry[-1][0]:
            return
        if self._odometry and self._path:
            step = hypot(pose[0] - self._odometry[-1][1][0], pose[1] - self._odometry[-1][1][1])
            travelled = self._path[-1][1] + (step if step < 0.5 else 0.0)   # /odom 이 초기화돼 튄 한 번은 세지 않는다
        else:
            travelled = 0.0
        self._path.append((timestamp, travelled))
        self._odometry.append((timestamp, pose))
        if self._assignment is not None and self._assignment.phase.lower() in LOCKED_PHASES:
            self._locked = (pose[0], pose[1], self._imu[-1][1] if self._imu else None)
        self._publish_prediction()

    def _receive_wheels(self, message):
        data = message.data
        if len(data) < 2:
            return
        # odom_raw 에는 header 가 없다. 이 노드는 로봇에서 돌고 /odom 도 같은 시계로 찍히므로 받은 시각을 쓴다.
        self._on_wheels(self.get_clock().now().nanoseconds, float(data[0]), float(data[1]))

    def _on_wheels(self, timestamp, left, right):
        """좌·우 바퀴 누적 거리(m) → 이어 붙인 yaw. 방향 변화 = (오른쪽 − 왼쪽) / 바퀴 간격 (좌회전 +)."""
        if self.yaw_source != 'wheel' or not (isfinite(left) and isfinite(right)):
            return
        last, self._wheels = self._wheels, (left, right)
        if last is None:
            self._imu.append((timestamp, self._wheel_yaw))
            return
        if self._imu and timestamp <= self._imu[-1][0]:
            return
        dl, dr = left - last[0], right - last[1]
        if abs(dl) > 1.0 or abs(dr) > 1.0:   # 엔코더 초기화 등: 이 한 번은 버린다
            return
        self._wheel_yaw += (dr - dl) / self.wheel_base_m
        self._imu.append((timestamp, self._wheel_yaw))

    def _receive_imu(self, message):
        if self.yaw_source != 'imu':
            return
        yaw = planar_yaw(message.orientation)
        timestamp = stamp_ns(message.header)
        if yaw is None or timestamp <= 0:
            return
        if self._imu and timestamp <= self._imu[-1][0]:
            return
        if self._imu:
            last_yaw = self._imu[-1][1]
            yaw = last_yaw + wrap(yaw - last_yaw)   # ±π 를 넘어도 이어지게
        self._imu.append((timestamp, yaw))

    def _odometry_at(self, timestamp):
        if not self._odometry:
            return None
        matched_time, matched_pose = min(
            self._odometry, key=lambda sample: abs(sample[0] - timestamp)
        )
        return matched_pose if abs(matched_time - timestamp) <= SAMPLE_MATCH_NS else None

    def _imu_at(self, timestamp):
        if not self._imu:
            return None
        matched_time, yaw = min(self._imu, key=lambda sample: abs(sample[0] - timestamp))
        return yaw if abs(matched_time - timestamp) <= SAMPLE_MATCH_NS else None

    def _turn(self, start, end):
        """두 시각 사이 방향 변화. IMU 가 없으면 0 (방향을 관측 값 그대로 둔다)."""
        a, b = self._imu_at(start), self._imu_at(end)
        return 0.0 if a is None or b is None else b - a

    def _record_anchor(self, timestamp, source, pose, odom):
        """위치 관측과 그 시각의 /odom 을 쌓고, 충분히 퍼져 있으면 delta 를 다시 맞춘다."""
        self._anchors.append((timestamp, source, (pose[0], pose[1]), (odom[0], odom[1])))
        while self._anchors and timestamp - self._anchors[0][0] > CALIBRATION_WINDOW_NS:
            self._anchors.popleft()
        # 출처가 섞이면 서로 다른 오차가 섞이므로 가장 최근 관측과 같은 출처만 쓴다
        same = [anchor for anchor in self._anchors if anchor[1] == source]
        if len(same) < CALIBRATION_MIN_POINTS:
            return
        odom_points = [anchor[3] for anchor in same]
        if spread(odom_points) < CALIBRATION_MIN_SPREAD_M:
            return
        fit = rigid_fit(odom_points, [anchor[2] for anchor in same])
        if fit is not None and fit[3] <= CALIBRATION_MAX_RMS_M:
            self._delta = fit[2]

    def _still(self, odom):
        """잠긴 뒤로 거의 안 움직였나. 잠긴 것을 못 봤으면(예: 경기 중에 이 노드가 다시 켜짐) False."""
        if self._locked is None:
            return False
        x, y, yaw = self._locked
        if hypot(odom[0] - x, odom[1] - y) > SLOT_STILL_M:
            return False
        return yaw is None or not self._imu or abs(self._imu[-1][1] - yaw) <= SLOT_STILL_RAD

    def _slot_region(self, field, timestamp):
        """이번 경기 우리 출발 칸 (min_x, min_y, max_x, max_y). 지도 bases 의 name 이 slot(예 B-2)인 것 하나."""
        assignment = self._assignment
        if (assignment is None or not assignment.team_assigned
                or abs(timestamp - stamp_ns(assignment.header)) > 2_000_000_000):
            return None
        prefix = {'BLUE': 'B', 'RED': 'R'}.get(assignment.camp)
        if prefix is None or assignment.slot not in {f'{prefix}-{row}' for row in (1, 2, 3)}:
            return None
        regions = [base for base in field.bases if base.name == assignment.slot]
        if len(regions) != 1:
            return None
        region = regions[0]
        return (region.min_x, region.min_y, region.max_x, region.max_y)

    def _search_slot(self, field, scan, timestamp, odom):
        """처음 위치: 잠긴 뒤 안 움직였으면 출발 칸 전체를 라이다로 찾는다. 실패하면 SLOT_RETRY_NS 뒤에 다시."""
        if self._slot_pose is not None:
            return None
        if not self._still(odom):
            self._slot_note = '출발 칸을 쓰지 않음: ' + ('잠긴 단계(waiting·ready)를 보지 못함' if self._locked is None else '잠긴 뒤 움직임')
            return None
        if self._slot_attempt is not None and 0 <= timestamp - self._slot_attempt < SLOT_RETRY_NS:
            return None
        region = self._slot_region(field, timestamp)
        if region is None:
            self._slot_note = '출발 칸을 모름: 서버 단계 정보(진영·slot)가 없거나 2 s 넘게 묵었거나, 지도 bases 에 그 이름이 하나가 아님'
            return None
        self._slot_attempt = timestamp
        boxes, weights = self._fusion.structure()
        result = slot_search(field, scan, region, boxes=boxes, weights=weights)
        if result is not None:
            self._slot_pose = result
            self._slot_note = '출발 칸에서 찾음: (%.2f, %.2f, %.0f°)' % (result[0], result[1], result[2] * 180 / pi)
        else:
            self._slot_note = '출발 칸 탐색 실패 (라이다가 지도와 맞지 않거나 비슷한 자리가 둘 이상)'
        self.get_logger().info(self._slot_note)
        self._log_event('slot', note=self._slot_note, pose=[round(v, 3) for v in self._slot_pose] if self._slot_pose is not None else None)
        return result

    def _last_known(self, timestamp, odom):
        """마지막으로 받아들인 자세를 그 뒤 /odom 이 간 만큼 옮긴 것 (x, y, yaw 또는 None). 모르면 None.
        방향은 그때와 지금의 방향 표본(바퀴 · IMU)이 둘 다 남아 있을 때만 안다. odom 축(delta)을 모르면 거의 안 움직였을 때만 그 자리로 친다."""
        if self._scan_anchor is None:
            return None
        observed, anchor, start = self._scan_anchor
        dx, dy = odom[0] - start[0], odom[1] - start[1]
        if self._delta is not None:
            c, s = cos(self._delta), sin(self._delta)
            dx, dy = c*dx - s*dy, s*dx + c*dy
        elif hypot(dx, dy) > RELOCATE_NEAR_M / 2:
            return None
        else:
            dx = dy = 0.0
        a, b = self._imu_at(observed), self._imu_at(timestamp)
        return anchor[0] + dx, anchor[1] + dy, (wrap(anchor[2] + b - a) if a is not None and b is not None else None)

    def _relocate(self, field, scan, timestamp, odom):
        """잃은 위치 다시 찾기 (파일 위의 설명). 받을 자세 (x, y, yaw) 또는 None."""
        if self._relocate_next is not None and timestamp < self._relocate_next:
            return None
        started = time.perf_counter()
        self._pose_checks['lost_tries'] += 1
        width, height = float(field.width), float(field.height)
        last = self._last_known(timestamp, odom)
        seen = []
        if self._ugv_seen is not None and 0 <= timestamp - self._ugv_seen[0] <= UGV_MASK_AGE_NS:
            seen = list(self._ugv_seen[1])
        regions = []   # (어디서 온 자리, 가운데 x, y, 반 폭, 방향 또는 None)
        wide = RELOCATE_ALL_HEADINGS
        if last is not None:
            anchor = self._scan_anchor[1]
            if wide:
                # 마지막으로 정합한 자리가 먼저다 — /odom 으로 옮긴 자리는 밀리거나 헛돌면 틀린다 (파일 위의 설명). 둘이 떨어져 있으면 둘 다 본다
                regions.append(('anchor', anchor[0], anchor[1], RELOCATE_NEAR_M, None))
                if hypot(last[0] - anchor[0], last[1] - anchor[1]) > RELOCATE_SPLIT_M:
                    regions.append(('last', last[0], last[1], RELOCATE_NEAR_M, None))
            else:
                regions.append(('last', last[0], last[1], RELOCATE_NEAR_M, last[2]))
            near = sorted(seen, key=lambda q: hypot(q[0] - last[0], q[1] - last[1]))
            near = [q for q in near if all(hypot(q[0] - cx, q[1] - cy) > RELOCATE_NEAR_M for _, cx, cy, _, _ in regions)][:RELOCATE_MAX_UGVS]   # 그 안의 것은 위에서 훑는다
        else:
            near = seen   # 마지막 자리를 모른다 (경기 중에 켜졌다 등): 드론이 본 자리를 모두 본다
        regions += [('ugv', x, y, RELOCATE_UGV_M, None if wide or last is None else last[2]) for x, y in near]
        if wide:
            boxes, weights = None, None   # 드론 지도만으로 찾는다
        else:
            boxes, weights = self._fusion.structure()
        solid, box_half, wall_w, wall_h = structure(field, boxes=boxes)
        points = scan_points(scan) or []
        found = []   # (맞은 점 수 — 무게 없이, 어디서 온 자리, 자세)
        for source, cx, cy, half, yaw in regions:
            region = (max(0.10, cx - half), max(0.10, cy - half), min(width - 0.10, cx + half), min(height - 0.10, cy + half))
            if region[0] >= region[2] or region[1] >= region[3]:
                continue
            headings = None
            if yaw is not None:
                step = 2*pi/SLOT_HEADINGS
                headings = [yaw + k*step for k in range(-int(RELOCATE_YAW/step), int(RELOCATE_YAW/step) + 1)]
            pose = slot_search(field, scan, region, boxes=boxes, weights=weights, headings=headings)
            if pose is not None and all(hypot(pose[0] - other[0], pose[1] - other[1]) > 0.20 for _, _, other in found):
                found.append((score_pose(points, pose, solid, box_half, wall_w, wall_h)[0], source, pose))
        found.sort(key=lambda item: -item[0])
        took = time.perf_counter() - started
        self._relocate_next = timestamp + max(RELOCATE_RETRY_NS, int(3 * took * 1e9))
        result = None
        if not found or (len(found) > 1 and found[1][0] > RELOCATE_CLEAR * found[0][0]):
            # 맞보는 꼴의 경기장에서는 서로 다른 두 자리가 비슷하게 맞을 수 있다 — 뚜렷이 나은 한 곳이 없으면 버린다
            self._relocate_pending = None
            self._relocate_note = '잃은 위치 다시 찾기 실패: %s (훑은 곳 %d, 드론 UGV %d대)' % (
                '비슷하게 맞는 자리가 둘 이상' if found else '맞는 자리 없음', len(regions), len(seen))
        else:
            hits, source, pose = found[0]
            agree = any(hypot(q[0] - pose[0], q[1] - pose[1]) <= RELOCATE_AGREE_M for q in seen)
            again = (self._relocate_pending is not None
                     and hypot(pose[0] - self._relocate_pending[0], pose[1] - self._relocate_pending[1]) <= RELOCATE_SAME_M
                     and abs(wrap(pose[2] - self._relocate_pending[2])) <= 0.2)
            if wide:
                sure = source != 'ugv' and (agree or hits >= RELOCATE_SURE_SHARE * len(points))
            else:
                sure = source == 'last' and (last[2] is not None or agree)
            if sure or again:
                result = pose
                self._relocate_pending = None
                self._pose_checks['lost_found'] += 1
                self._relocate_note = '잃은 위치를 다시 찾음: (%.2f, %.2f, %.0f도) — %s%s' % (
                    pose[0], pose[1], pose[2] * 180 / pi, '마지막 자리 둘레' if source != 'ugv' else '드론이 본 UGV 자리 둘레',
                    ' · 드론도 그 자리' if agree else '')
                self.get_logger().warning(self._relocate_note)
            else:
                self._relocate_pending = pose
                self._relocate_note = '잃은 위치 후보 (%.2f, %.2f, %.0f도) — 다음에도 같으면 받는다' % (pose[0], pose[1], pose[2] * 180 / pi)
        self._log_event('relocate_lost', note=self._relocate_note, found=[[s, int(h), [round(v, 3) for v in q]] for h, s, q in found],
                        last=[None if v is None else round(v, 3) for v in last] if last is not None else None,
                        regions=len(regions), ugvs=len(seen), ms=round(took * 1000, 1))
        return result

    def _estimate(self, timestamp, odom_pose):
        field = self.latest_perception_map
        if self._scan_anchor is not None:
            observed, anchor, start = self._scan_anchor
            if 0 <= timestamp - observed <= PROPAGATE_NS:
                return propagate(anchor, start, odom_pose, self._delta, self._turn(observed, timestamp))
        observed = stamp_ns(field.header)
        if not 0 <= timestamp - observed <= PROPAGATE_NS:
            return None
        matched_pose = self._odometry_at(observed)
        if matched_pose is None:
            return None
        vehicles = [vehicle for vehicle in field.vehicles if vehicle.role == vehicle.SELF]
        if len(vehicles) != 1:
            return None
        anchor = planar_pose(vehicles[0].pose)
        if anchor is None:
            return None
        return propagate(anchor, matched_pose, odom_pose, self._delta, self._turn(observed, timestamp))

    def _forget_checks(self):
        self._accepted.clear()
        self._jump_rejects = 0
        self._wall_streak = []
        self._wall_doubt = False

    def _travelled_at(self, timestamp):
        if not self._path:
            return None
        matched_time, travelled = min(self._path, key=lambda sample: abs(sample[0] - timestamp))
        return travelled if abs(matched_time - timestamp) <= SAMPLE_MATCH_NS else None

    def _jumped(self, timestamp, result):
        """뜀 거르기: 이 정합 결과가 지난 JUMP_WINDOW_NS 안의 정합 자리에서 /odom 이 간 거리보다 멀리 가 있으면 (넘은 거리 m), 아니면 None."""
        travelled = self._travelled_at(timestamp)
        if travelled is None:
            return None
        worst = None
        for at, x, y, then in self._accepted:
            if not 0 < timestamp - at <= JUMP_WINDOW_NS:
                continue
            over = hypot(result[0] - x, result[1] - y) - (JUMP_GAIN * max(0.0, travelled - then) + JUMP_SLACK_M)
            if over > 0 and (worst is None or over > worst):
                worst = over
        return worst

    def _accept(self, timestamp, result):
        self._yaw_hint = None
        travelled = self._travelled_at(timestamp)
        if travelled is not None:
            self._accepted.append((timestamp, result[0], result[1], travelled))
        self._jump_rejects = 0

    def _walls_back(self, field, scan, seed, result, timestamp):
        """뛴 정합 result 를 외벽이 지키던 자리 seed 보다 뚜렷이 편드는가 (walls_prefer). 빠르게 도는 중의 스캔으로는 보지 않는다 (번진다)."""
        if seed is None or abs(self._turn(timestamp - 200_000_000, timestamp) / 0.2) > WALL_MAX_YAW_RATE:
            return False
        return walls_prefer(scan_points(scan, 2) or [], seed, result, float(field.width), float(field.height))

    def _wall_recheck(self, field, scan, result, timestamp):
        """벽으로 다시 보기: 옮겨야 하면 옮길 자세, 아니면 None. 의심이 이어지는 동안은 self._wall_streak 이 차 있다 (지도 고치기를 쉰다)."""
        if abs(self._turn(timestamp - 200_000_000, timestamp) / 0.2) > WALL_MAX_YAW_RATE:
            return None      # 도는 중의 스캔은 번진다. 이어진 판정은 그대로 둔다 (돌고 나서 이어 본다)
        points = scan_points(scan, 2)
        suspect = wall_suspect(points, result, float(field.width), float(field.height)) if points else None
        self._wall_doubt = suspect is not None
        second = None
        if suspect is not None:
            self._pose_checks['wall_suspect'] += 1
            boxes, weights = self._fusion.structure(added=False)
            second = confirm_wall_candidate(field, scan, result, suspect[2], boxes=boxes, weights=weights)
        if second is None:
            self._wall_streak = []
            return None
        shift = (second[0] - result[0], second[1] - result[1])
        if self._wall_streak and hypot(shift[0] - self._wall_streak[-1][1], shift[1] - self._wall_streak[-1][2]) > WALL_AGREE_M:
            self._wall_streak = []
        self._wall_streak.append((timestamp, shift[0], shift[1]))
        if len(self._wall_streak) < WALL_STREAK:
            return None
        began = self._wall_streak[0][0]
        self._wall_streak = []
        self._pose_checks['relocated'] += 1
        undone = self._fusion.distrust(began - WALL_DISTRUST_NS)
        self.get_logger().warning('위치를 외벽에 맞춰 옮김: (%.2f, %.2f, %.0f°) → (%.2f, %.2f, %.0f°), 벽 점수 %.0f → %.0f%s'
                                  % (result[0], result[1], result[2] * 180 / pi, second[0], second[1], second[2] * 180 / pi,
                                     suspect[0], suspect[1], ' · 그 사이 고친 지도를 되돌림' if undone else ''))
        self._log_event('relocate', before=[round(v, 3) for v in result], after=[round(v, 3) for v in second],
                        wall=[round(suspect[0], 1), round(suspect[1], 1)], map_undone=bool(undone))
        if undone:
            self._log_event('map_fusion', **self._fusion.summary())
            self._publish_lidar_boxes(changed_only=True)
        # 틀린 자리에서 모은 (정합 자리, /odom) 쌍으로 delta 를 다시 맞추지 않게 버린다. delta 값은 새 쌍이 모일 때까지 그대로 쓴다
        self._anchors.clear()
        self._accepted.clear()
        self._wall_doubt = False     # 옮긴 자리는 외벽에 맞는다
        self._fusion_hold = timestamp + FUSION_HOLD_NS
        return second

    def _receive_scan(self, scan):
        self._log_raw_scan(scan)
        field = self.latest_perception_map
        timestamp = stamp_ns(scan.header)
        if field is None or not self._odometry:
            self._scan_note = {'t': timestamp, 'result': '지도 없음' if field is None else '/odom 없음'}
            return
        if self._scan_anchor is not None and timestamp <= self._scan_anchor[0]:
            return
        matched_pose = self._odometry_at(timestamp)
        if matched_pose is None:
            self._scan_note = {'t': timestamp, 'result': '스캔 시각(±0.1 s)의 /odom 없음'}
            return
        started = time.perf_counter()
        seed = self._estimate(timestamp, matched_pose)
        if seed is None and self._scan_anchor is not None:
            # delta 를 아직 몰라 전파가 끊겼을 뿐이면 마지막 정합 위치에서 다시 찾는다 (초기화 아님)
            observed, anchor, _ = self._scan_anchor
            if 0 <= timestamp - observed <= PROPAGATE_NS:
                seed = (anchor[0], anchor[1], wrap(anchor[2] + self._turn(observed, timestamp)))
        if seed is None and self._slot_pose is not None and self._still(matched_pose):
            # 출발 칸에서 찾은 뒤 아직 그 자리다 (출발 칸 탐색이 오래 걸려 전파 시간이 지난 경우 등)
            seed = self._slot_pose
        if seed is not None:
            hint = self._yaw_hint
            if hint is not None and 0 < timestamp - hint[0] <= JUMP_WINDOW_NS:
                yaw = wrap(hint[1] + self._turn(hint[0], timestamp))   # 방금 버린 정합의 방향에서 이어 간다 (JUMP_YAW_HINT_MAX 의 설명)
                if abs(wrap(yaw - seed[2])) <= JUMP_YAW_HINT_MAX:
                    seed = (seed[0], seed[1], yaw)
                    self._pose_checks['yaw_hints'] += 1
            boxes, weights = self._fusion.structure()
            result = scan_match(field, scan, seed, False, boxes=boxes, weights=weights)
            how = '정합'
        else:
            result = self._search_slot(field, scan, timestamp, matched_pose)
            how = '처음 위치 찾기'
            if result is None and self.relocate_lost and (self._slot_pose is not None or not self._still(matched_pose)):
                # 출발 칸 탐색을 쓸 수 없다 (이미 썼다 · 출발 칸을 떠났다 · 경기 중에 켜졌다) → 마지막 자리 · 드론이 본 UGV 자리 둘레를 훑는다
                result = self._relocate(field, scan, timestamp, matched_pose)
                how = '잃은 위치 다시 찾기'
        self._scan_note = {'t': timestamp, 'result': how + (' 성공' if result is not None else ' 실패'),
                           'ms': round((time.perf_counter() - started) * 1000, 1)}
        if result is None and how == '정합' and self.pose_check:
            # 정합이 실패했다: 씨앗 자리가 틀렸을 수 있다 → 외벽이 가리키는 자리를 같은 방법으로 본다 (WALL_STREAK 번 이어 같으면 그 자리로)
            result = self._wall_recheck(field, scan, seed, timestamp)
            if result is not None:
                self._scan_note['result'] = '정합 실패 → 외벽에 맞춰 옮김'
                self._accept(timestamp, result)
        elif result is not None and self.pose_check:
            if how == '정합':
                over = self._jumped(timestamp, result)
                if over is not None and self._walls_back(field, scan, seed, result, timestamp):
                    # 뛰었지만 외벽이 새 자리를 뚜렷이 편든다: 지키던 자리가 틀린 쪽이다 → 받아들인다 (파일 위 「뜀 거르기」 의 22:39 판)
                    self._pose_checks['jump_wall_ok'] += 1
                    self._scan_note['result'] = '정합 성공 (뛰었으나 외벽이 새 자리를 편듦)'
                    self._accepted.clear()   # 앞의 자리들은 틀린 쪽이었다 — 다음 정합을 그것과 견주지 않는다
                    over = None
                if over is not None and self._jump_rejects < JUMP_MAX_REJECTS:
                    # 바퀴가 구른 것보다 멀리 뛴 정합: 버리고 마지막으로 받아들인 자리에서 /odom 으로 이어 간다
                    self._jump_rejects += 1
                    self._pose_checks['jump_rejected'] += 1
                    self._scan_note['result'] = '정합 버림 (/odom 이 간 거리보다 %.2f m 더 뜀)' % over
                    self._yaw_hint = (timestamp, result[2])
                    return
                if over is not None:
                    self._pose_checks['jump_forced'] += 1   # 이어서 같은 답: 위치가 정말 틀려 있었을 수 있다 → 받아들이고 벽으로 다시 본다
                    self._accepted.clear()
                moved = self._wall_recheck(field, scan, result, timestamp)
                if moved is not None:
                    result = moved
                    self._scan_note['result'] = '외벽에 맞춰 옮김'
                elif self.drone_pose_check:
                    moved = self._drone_recheck(field, scan, result, timestamp)
                    if moved is not None:
                        result = moved
                        self._scan_note['result'] = '드론이 본 자리에서 다시 맞춰 옮김'
            else:
                self._forget_checks()    # 처음 위치: 견줄 앞 자리가 없다
                if how == '잃은 위치 다시 찾기':
                    self._fusion_hold = timestamp + FUSION_HOLD_NS
            self._accept(timestamp, result)
        if result is not None:
            self._pose_trail.append((timestamp, result[0], result[1]))
            self._scan_anchor = (timestamp, result, matched_pose)
            self._record_anchor(timestamp, 'scan', result, matched_pose)
            self._publish_prediction()
            # 위치를 의심하는 동안에는 지도를 고치지 않는다. 2026-11-19 본선 경기장 기록: 위치가 0.45 m 틀린 30초 동안에도 map_fusion 의
            # 안전장치(설명되는 점 40 % · 믿는 점 10개)는 통과해, 진짜 드론 상자 2개를 빼고 없는 상자를 더했다. 그 구간의 스캔은 거의 모두
            # 외벽이 다른 자리를 가리켰다 (wall_suspect) — 맞는 구간에서는 스캔의 0 ~ 9 % 뿐이라 건너뛰어도 잃는 것이 적다
            if self._wall_streak or self._wall_doubt:
                self._pose_checks['fusion_skipped'] += 1
            elif timestamp < self._fusion_hold or not self._fits_drone_map(field, scan, result):
                self._pose_checks['fusion_held'] += 1
            else:
                self._observe_map(scan, result, timestamp)

    def _fits_drone_map(self, field, scan, pose):
        """이 스캔이 이 자세에서 드론 지도(라이다로 고치기 전의 것)에 FUSION_MIN_DRONE_SHARE 넘게 맞는가 — 아니면 위치가 틀렸을 수 있다 (지도를 고치지 않는다)."""
        points = scan_points(scan)
        if not points or len(points) < 12:
            return False
        boxes, half, width, height = structure(field)
        return score_pose(points, pose, boxes, half, width, height)[0] >= FUSION_MIN_DRONE_SHARE * len(points)

    def _observe_map(self, scan, pose, timestamp):
        """정합한 자세로 드론 지도 상자를 고친다 (map_fusion). 위치를 낸 뒤에 해서 위치 발행을 늦추지 않는다."""
        if not self._fusion.wants_scan():
            return
        points = scan_points(scan, self._fusion.params.scan_step)
        if not points:
            return
        points = self._without_ugv_points(points, pose, timestamp)   # 드론이 본 다른 UGV 에 맞은 점은 지도 고치기에 쓰지 않는다
        if not len(points):
            return
        started = time.perf_counter()
        yaw_rate = self._turn(timestamp - 200_000_000, timestamp) / 0.2
        if self._fusion.observe(np.array(points, dtype=float), pose, timestamp, sensor=(LIDAR_X, 0.0), yaw_rate=yaw_rate):
            summary = self._fusion.summary()
            self.get_logger().info('지도 고침: 드론 상자 %d 중 %d 뺌 (내보내는 지도), 정합에만 쓰는 상자 %d 더함'
                                   % (summary['drone'], summary['rejected'], summary['added']))
            self._log_event('map_fusion', **summary)
            self._publish_lidar_boxes(changed_only=True)   # 더한 상자가 바뀌었으면 1초를 기다리지 않고 바로 알린다
        self._scan_note['fusion_ms'] = round((time.perf_counter() - started) * 1000, 1)

    def _publish_lidar_boxes(self, changed_only=False):
        """라이다로 찾아 정합에만 쓰는 상자를 /team/navigation/lidar_boxes 로 낸다 (PoseArray, frame arena, 상자마다 가운데 x · y).

        1초 타이머가 부르고(목록이 그대로여도 낸다), 지도 고치기에서 상자가 바뀐 때에도 부른다(changed_only — 목록이 달라졌을 때만 낸다).
        지도(/team/navigation/map)의 boxes 는 건드리지 않는다. 지도 고치기를 껐으면 내지 않는다.
        낸 목록이 앞의 것과 다르면 기록 파일에도 한 줄 남긴다 (event lidar_boxes). 여기서 난 오류로 위치 계산이 멈추면 안 되므로 통째로 감싼다.
        """
        try:
            boxes = lidar_boxes(self._fusion)
            if boxes is None or (changed_only and boxes == self._lidar_boxes_sent):
                return
            message = PoseArray()
            message.header.frame_id = 'arena'
            message.header.stamp = self.get_clock().now().to_msg()
            poses = []
            for x, y in boxes:
                pose = Pose()
                pose.position.x, pose.position.y = x, y
                pose.orientation.w = 1.0
                poses.append(pose)
            message.poses = poses
            self.lidar_boxes_publisher.publish(message)
            if boxes != self._lidar_boxes_sent:
                self._lidar_boxes_sent = boxes
                self._log_event('lidar_boxes', boxes=[[x, y] for x, y in boxes])
        except Exception as error:  # 이 목록 때문에 위치 계산 · 스캔 처리가 멈추면 안 된다
            self.get_logger().warning('라이다로 찾은 상자 목록 발행 실패: %r' % (error,))

    def _log_raw_scan(self, scan):
        """기록 폴더가 있으면 라이다 스캔 원본을 log_raw_scan_s 초마다 한 줄 남긴다 (event scan_raw) — 경기 뒤에 스캔을 그대로 다시 돌려 보려고.

        stamp = 스캔 시각(초), angle_min · angle_increment (rad), ranges = 거리(m, mm 까지. inf · NaN 은 0),
        pose = 지금 내보내는 위치 [x, y, yaw] (낼 수 없는 때는 null). 한 줄 약 3.5 KB(505점) → 1초에 하나면 경기 한 판에 약 1 MB.
        기록에서 난 오류가 스캔 처리를 막지 않게 통째로 감싼다.
        """
        try:
            if self.log_raw_scan_s <= 0.0 or not self.run_log.enabled:
                return
            now = self.get_clock().now().nanoseconds
            # 스캔은 0.1초마다 온다 → 반 주기(0.05초)의 여유를 두어 「1초에 하나」 가 1.1초에 하나로 밀리지 않게 한다
            if self._raw_scan_at is not None and 0 <= now - self._raw_scan_at < (self.log_raw_scan_s - 0.05) * 1e9:
                return
            self._raw_scan_at = now
            odom = self._odometry[-1] if self._odometry else None
            estimate = self._estimate(odom[0], odom[1]) if odom is not None and self.latest_perception_map is not None else None
            self._log_event('scan_raw', stamp=round(stamp_ns(scan.header) / 1e9, 3),
                            angle_min=round(float(scan.angle_min), 6), angle_increment=round(float(scan.angle_increment), 7),
                            ranges=[round(float(r), 3) if isfinite(r) else 0 for r in scan.ranges],
                            pose=[round(v, 3) for v in estimate] if estimate is not None else None)
        except Exception as error:  # 기록 때문에 위치 계산이 멈추면 안 된다
            self.get_logger().warning('스캔 원본 기록 실패: %r' % (error,))

    def _fused(self, field):
        """내보낼 지도: 라이다로 뺀 상자가 없으면 field 그대로, 있으면 뺀 복사본 (바뀔 때마다 한 번만 만든다)."""
        if not self._fusion.changed():
            return field
        cache = self._fused_cache   # 같은 지도 객체 · 같은 고친 횟수면 다시 만들지 않는다 (ROS 메시지 == 는 칸을 다 비교해서 is 로 본다)
        if cache is None or cache[0] is not field or cache[1] != self._fusion.version:
            self._fused_cache = (field, self._fusion.version, fused_map(field, self._fusion.output()))
        return self._fused_cache[2]

    def _publish_prediction(self):
        field = self.latest_perception_map
        if field is None or not self._odometry:
            return False
        latest_time, latest_pose = self._odometry[-1]
        if latest_time <= stamp_ns(field.header) or latest_time < self._last_output_stamp:
            return False
        estimate = self._estimate(latest_time, latest_pose)
        if estimate is None:
            return False
        x, y, yaw = estimate
        output = deepcopy(self._fused(field))
        vehicles = [vehicle for vehicle in output.vehicles if vehicle.role == vehicle.SELF]
        if len(vehicles) == 1:
            vehicle = vehicles[0]
        else:
            output.vehicles = [item for item in output.vehicles if item.role != item.SELF]
            # 항공 지도가 우리 차를 UNKNOWN 으로만 잡았으면 그것을 SELF 로 바꾼다 (같은 차가 두 번 나오지 않게)
            near = [(hypot(item.pose.position.x - x, item.pose.position.y - y), index)
                    for index, item in enumerate(output.vehicles) if item.role != item.OTHER]
            near = [entry for entry in near if entry[0] <= SELF_MATCH_M]
            if near:
                vehicle = output.vehicles[min(near)[1]]
                vehicle.role = FieldVehicle.SELF
            else:
                vehicle = FieldVehicle(role=FieldVehicle.SELF)
                output.vehicles.append(vehicle)
        pose = vehicle.pose
        pose.position.x, pose.position.y = x, y
        pose.orientation.x = pose.orientation.y = 0.0
        pose.orientation.z, pose.orientation.w = sin(yaw / 2), cos(yaw / 2)
        output.header.stamp.sec, output.header.stamp.nanosec = divmod(
            latest_time, 1_000_000_000
        )
        self.map_publisher.publish(output)
        self.pose_publisher.publish(PoseStamped(header=output.header, pose=pose))
        self._last_output_stamp = latest_time
        if self._published == 0:
            self.get_logger().info('첫 위치: (%.2f, %.2f, %.0f°)' % (x, y, yaw * 180 / pi))
        self._published += 1
        return True

    def _publish_debug(self):
        """시험 화면용: 위치를 왜 내는지/못 내는지 볼 수 있는 내부 상태를 JSON 으로. 추정에는 쓰지 않는다."""
        self.debug_publisher.publish(String(data=json.dumps(self._debug_state(), ensure_ascii=False, separators=(',', ':'))))

    def _write_log(self):
        """log_hz 타이머: 같은 내부 상태를 기록 파일에 한 줄 (run_log.py)."""
        try:
            out = self._debug_state()
            out['event'] = 'state'
            self.run_log.write(out)
        except Exception as error:  # 기록 때문에 위치 계산이 멈추면 안 된다
            self.get_logger().warning('내부 상태 기록 실패: %r' % (error,))

    def _log_event(self, event, **fields):
        if self.run_log.enabled:
            self.run_log.write(dict({'event': event, 't': round(self.get_clock().now().nanoseconds / 1e9, 3)}, **fields))

    def _debug_state(self):
        """/team/navigation/debug 와 기록 파일에 쓰는 내부 상태."""
        now = self.get_clock().now().nanoseconds
        field = self.latest_perception_map
        assignment = self._assignment
        odom = self._odometry[-1] if self._odometry else None
        estimate = self._estimate(odom[0], odom[1]) if odom is not None and field is not None else None
        out = {
            'node': 'navigation_slam', 't': round(now / 1e9, 3),
            'map': field is not None, 'boxes': len(field.boxes) if field is not None else 0,
            'odom_age': round((now - odom[0]) / 1e9, 3) if odom is not None else None,
            'heading_age': round((now - self._imu[-1][0]) / 1e9, 3) if self._imu else None,
            'yaw_source': self.yaw_source,
            'phase': assignment.phase if assignment is not None else None,
            'slot': assignment.slot if assignment is not None else None,
            'locked': self._locked is not None, 'still': bool(odom is not None and self._still(odom[1])),
            'slot_pose': [round(v, 3) for v in self._slot_pose] if self._slot_pose is not None else None,
            'slot_note': self._slot_note,
            'relocate_note': self._relocate_note,
            'delta': round(self._delta, 4) if self._delta is not None else None,
            'pose': [round(v, 3) for v in estimate] if estimate is not None else None,
            'published': self._published,
            'map_fusion': self._fusion.summary(),
            'pose_check': dict(self._pose_checks, suspect_streak=len(self._wall_streak)) if self.pose_check else None,
            'ugvs': None if self._ugv_tracks.stamp is None else {
                'others': [[round(u.x, 2), round(u.y, 2)] for u in self._ugv_tracks.fresh(now / 1e9)], 'masked': self._ugv_masked,
                # 드론이 본 우리 자리 − SLAM 자리 (마지막으로 보였을 때) · 우리 차가 안 보이는 동안 우리를 따라오는 「모르는 것」 [차 x, y, 이어진 장 수]
                'self_off': [round(v, 3) for v in self._drone_off] if self._drone_off is not None else None,
                'stray': [round(self._drone_stray[1], 2), round(self._drone_stray[2], 2), self._drone_stray[3]] if self._drone_stray is not None else None},
            'scan': dict(self._scan_note, age=round((now - self._scan_note['t']) / 1e9, 3)) if self._scan_note else None,
        }
        if self._scan_anchor is not None:
            observed, anchor, _ = self._scan_anchor
            out['anchor'] = [round(v, 3) for v in anchor]
            out['anchor_age'] = round((now - observed) / 1e9, 3)
        return out


def main():
    rclpy.init()
    node = SlamNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass   # robot/team_node.py 는 SIGINT 로 끈다
    finally:
        node.run_log.close()
        node.destroy_node()
        rclpy.try_shutdown()

