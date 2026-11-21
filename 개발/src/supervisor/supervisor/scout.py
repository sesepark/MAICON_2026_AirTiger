"""정찰: 어디에 서서 어느 방향을 찍을지 정한다. 두 방식을 같은 모양으로 만들어 설정으로 바꿔 쓴다.

  PlaceScout  자리 기준 — 그때그때 「새 면이 가장 많이 보이는 자리」 로 가서 정면으로 한 장. 답 전체를 면들에 순서 · 위치로 짝짓는다
  FaceScout   면 기준 — 지도를 받으면 설 자리와 순서를 처음에 다 정한다. 자리마다 면을 하나씩 카메라(팬)로 겨눠 찍고,
              사진 가운데의 그림을 그 면의 것으로 적는다. 답을 기다리지 않고 다음으로 간다

둘 다 한 틱에 한 번 step() 을 부르면 지금 할 일 하나를 돌려준다:
  ("go", x, y, yaw)        이 자리로 가서 이 방향을 본다 (도착할 때까지 같은 값을 돌려준다)
  ("go", x, y, yaw, tol)   면 기준: 방향은 ±tol(rad) 안이면 된다 — 그 안이면 찍을 면이 모두 팬으로 닿는다
  ("shoot", pan_deg, tag)  지금 자리에서 팬을 이 각도로 하고 사진 한 장을 판독에 보낸다. 찍었으면 shot_taken(tag) 로 알려 준다
  ("wait",)                판독 답을 기다린다
  ("done",)                더 할 것이 없다
판독 답이 오면 on_answer(tag, pose, pan_deg, pictures) 로 넣는다. ROS 를 쓰지 않는다.
면 기준은 지나가며 찍은 사진의 답도 on_pass(tag, pose, pan_deg, pictures, at) 로 받는다 — 겨누지 않은 면의 「본 것」 (pick_votes · FaceBook.sight).
"""
from __future__ import annotations

import itertools
import math
from array import array
from dataclasses import dataclass, field

from . import arena as A
from .align import aligned
from .coverage import CoveragePlanner, ViewParams
from .faces import CameraParams, camera_pose, expected_faces, match, wrap

PAN_LIMIT_DEG = 50.0  # 팬 ±100 × 0.5°


@dataclass
class ScoutParams:
    mode: str = "face"  # "face" (면 기준) · "place" (자리 기준)
    arrive_m: float = 0.15  # 목표에서 이 안이면 도착
    # 자리에서 차체 방향이 목표 방향과 이만큼(°) 안이면 도착. 팬은 실제 방향(pose)으로 겨누므로 방향이 조금 틀려도 찍을 수 있다
    # (2026-11-18 팀 테스트 3차: 남은 방향 10 ~ 15° 에서 팬 −35° 로 찍은 사진이 판독됨). navigation 의 align_accept_rad(0.35 rad = 20°)
    # 가 방향을 못 맞추고 REACHED 로 끝내도 도착으로 받도록 그보다 넓게 둔다 (10° 이면 그 뒤 3 s 를 더 서 있다가 accept_heading)
    arrive_yaw_deg: float = 25.0
    answer_wait_s: float = 4.0  # 판독 답을 기다리는 가장 긴 시간
    retries: int = 2  # 한 자리 · 한 면에서 다시 찍는 횟수
    # 자리 기준
    place_view: ViewParams = field(default_factory=lambda: ViewParams(range_m=1.5, max_incidence_deg=50.0))
    # 면 기준
    close_range_m: float = 1.1  # 이 안에서 본 것을 「크게 찍혔다」 로 친다
    close_incidence_deg: float = 50.0
    pan_reach_deg: float = 45.0  # 차체를 돌리지 않고 팬으로 겨눌 수 있는 범위
    # 겨눈 면의 그림이 사진에서 그 면이 찍혀야 할 자리에서 폭의 이만큼 안에 있어야 한다.
    # 2026-11-19: 실차 사진 33장을 하나하나 보고 다시 맞춰 본 결과(정찰 방식 분석) — 면이 실제로 찍힌 자리와 계산한 자리의 차이는 가운데값 0.014,
    # 90% 가 0.04 안이었다. 0.15 일 때 틀리게 적힌 한 장(상자가 옮겨져 먼 상자의 그림을 적음)은 0.056 떨어져 있었고, 0.05 로 줄여도 맞게 적힌 15장은 그대로다
    center_tol: float = 0.05
    # 가까운 면은 허용을 넓힌다 (pick_centered 의 near): 0.03 + 0.03 / 거리(m), 가장 크게 0.12 → 1 m 0.06 · 0.5 m 0.09 · 0.4 m 0.105.
    # 2026-11-19 14:13 실차: 0.44 m 앞에서 찍은 탄약상자가 0.068 밀려 있어(크기는 1.05배로 맞음) 적지 못했다. 먼 상자의 그림이 그 자리에 찍힌 것은
    # 크기 검사(aim_height)가 거른다 (틀리게 적혔던 한 장은 0.34배)
    center_tol_near: tuple = (0.03, 0.03, 0.12)
    aim_incidence_deg: float = 62.0  # 가운데 그림으로 정한 겨눈 면을 적는 비스듬함의 한도 (면 기록의 보통 한도는 55°)
    center_sep: float = 0.08  # 두 번째로 가까운 그림은 이만큼 더 멀어야 한다 (헷갈리면 적지 않는다)
    # 다만 고른 그림의 세로 크기가 이 면이 찍힐 크기의 이 배수 안이고 가까이 있는 다른 그림들은 모두 밖이면 적는다 (pick_centered 의 rival). None 이면 끔
    center_rival: tuple | None = (0.85, 1.45)
    # 자리를 도는 순서 (order):
    #   "tour"  구역을 따지지 않고 가까운 자리부터 이어 가는 가장 짧은 길 (기본). 먼저 돌 구역(prefer — 지정표적이 있는 곳)의 자리만 맨 앞에 둔다
    #   "best"  정찰하는 동안 얻을 것으로 보이는 M3 점수가 가장 큰 순서를 찾는다 (_best_order — 구역을 일찍 끝내는 쪽으로 기운다)
    #   "near"  구역을 하나씩, 가까운 구역부터 (2026-11-19 낮까지의 방식)
    # 2026-11-19 의 경과: 낮에 "tour" → 15:28 "best" (시간 어림만으로 견줬을 때 150초까지 끝나는 구역이 1 → 2개) → 저녁에 다시 "tour".
    # 저녁에 가상 경기(오차를 넣은 12판)를 시한을 두고 끊어 「그때 맞은 구역 수」 로 견줬다 — 시한 90 · 110 · 130 · 150초에서
    #   tour 1.75 · 3.25 · 3.58 · 3.92    best 2.00 · 2.50 · 2.75 · 3.83    (다 도는 데 tour 139초 · best 160초)
    # 계획의 6할도 못 도는 판에서만 best 가 조금 낫고, 그보다 더 도는 판에서는 tour 가 크게 낫다. 16:01 실차는 계획의 자리를 거의 다 돌았다
    # (그 판에서 못 적은 두 면은 순서가 아니라 통로 첫 자리에서 팬이 안 닿은 탓이었다). 구역을 하나씩 끝내려고 오가는 길이 면을 모으는 속도를 깎는다.
    # order 를 빈 문자열로 두면 예전 설정 sector_first 를 따른다 (True = "near", False = "tour")
    #   "live"  (2026-11-19 밤 · 시험 중) 처음 계획은 "tour" 와 같고, 경기 도중에 다음 자리를 그때그때 다시 고른다 (FaceScout._live_pick):
    #           자리를 마쳤을 때 · 이동이 막혔을 때 · 가는 도중에 면이 적히거나(지나가며 찍은 사진) 라이다 상자가 바뀌었을 때.
    #           「거점 앞 대기를 시작할 때 맞아 있을 구역 점수」 의 기대값이 가장 큰 순서의 첫 자리로 간다 — 값은 아래 live_…
    order: str = "tour"
    # ---- 실시간 순서 (order = "live") 의 값. 시간 어림: 이동 live_speed (서서 도는 시간까지 친 속도) · 자리마다 live_stop_s · 면마다 live_face_s
    live_speed: float = 0.17
    live_stop_s: float = 2.0
    live_face_s: float = 1.5
    live_block_s: float = 9.0  # 막힌 이동 한 번에 잃는 시간 (2026-11-19 로봇 6대 경기: 5 ~ 13초)
    live_strike_s: float = 20.0  # 지정표적을 아직 못 찾았으면 남은 시간에서 사격에 쓸 이만큼을 뺀다
    live_hit: float = 0.9  # 자리에 서서 찍은 면이 적힐 가능성
    live_pass: float = 0.2  # 가지 않은 자리의 면이 지나가며 찍은 사진으로 적힐 가능성
    # 값을 보는 때 (남은 시간의 몇 배, 무게): 막힘 · 다시 찍기로 시간이 모자랄 것에 대비해 이른 때의 점수도 함께 본다 → 구역을 하나씩 끝내는 쪽으로 기운다.
    # 가상 경기(다른 로봇을 넣은 것 · 40 seed × 지도 3)로 견준 값: (0.6 · 0.8 · 1.0) 이나 (0.5 · 0.7 · 0.9) 처럼 더 짜게 보면 막힘이 적은 판에서 길만 길어졌다
    live_scenarios: tuple = ((0.75, 0.4), (1.0, 0.6))
    # 고르는 법: 지금 순서보다 기대 점수가 live_margin 넘게 큰 순서가 있으면 그리로 바꾼다 (가는 도중에는 그 두 배). 없으면 지금 순서와 점수가 같은
    # 순서들 가운데 다 도는 시간(어림)이 live_shorter_s 넘게 짧은 것이 있을 때만 바꾼다 (가는 도중에는 바꾸지 않는다).
    live_margin: float = 0.3
    live_shorter_s: float = 5.0
    # 몇 곳 안 남았고 시간이 넉넉할 때 (2026-11-20 01시): 고를 자리가 live_exact 곳 이하이고, 지금 순서로 다 도는 데 드는 시간(어림)이 남은 시간의
    # live_slack 배 안이면, 점수는 어느 순서나 같고 남은 것은 「얼마나 빨리 끝내느냐」 뿐이다 → 가장 짧은 순서를 빠짐없이 찾아(_shortest — 7곳이면
    # 이 PC 에서 2 ~ 3 ms) 견줄 순서에 넣고(live_all 곳 이하면 모든 순서 — 5곳에 120가지), 점수가 같은(live_tie 안) 순서들 가운데 live_easy_s 만
    # 짧아도 그리로 바꾼다. 가던 자리의 면이 가는 도중에 다 적혔을 때도 이렇게 다시 고른다.
    # 00:52 연습 판(R-1): 세 자리가 남고 시간은 108초(다 도는 데 49초)였는데, 견준 순서 넷 가운데 가장 짧은 것이 지금 순서보다 4.0초 짧아 그대로
    # 갔다 (정말 가장 짧은 순서 — 1.1 m 앞의 자리부터 — 는 견준 것에 없었다). 서 → 동 → 서로 8 m 를 오갔다.
    # 그 밖의 때(자리가 많이 남았다 · 시간이 빠듯하다)는 전과 똑같이 고른다. 「남은 자리」 는 지정표적을 찾느라 먼저 돌 구역의 자리만 추리기 전의
    # 수다 — 추린 뒤의 수로 보았더니 경기 첫머리부터(추린 자리가 서넛이다) 순서가 달라져 방해가 많은 가상 경기에서 1.2 ~ 1.7점을 잃었다
    live_slack: float = 0.8
    live_easy_s: float = 1.0
    live_tie: float = 0.05
    live_exact: int = 7
    live_all: int = 5
    # 뒤에 생긴 면(지도에 없던 라이다 상자의 면)의 자리도 순서를 고를 때 함께 견준다 — 그 상자의 면이 사진에 한 번이라도 찍힌 뒤부터
    # (「본 것」 이 있거나 적힌 면이 있는 상자). False 면 전처럼 계획한 자리를 다 돈 뒤에 간다.
    # 그 판에서 지도에 빠진 상자 둘(지정표적이 붙은 것 포함)의 자리 넷이 로봇에서 0.5 ~ 1 m 인데도 맨 뒤로 밀려, 동쪽 자리를 다녀온 뒤에야 갔다.
    # 사진에 찍힌 적이 없는 상자는 전처럼 맨 뒤에 둔다: 서 있는 로봇을 상자로 잘못 본 것일 수 있다 (그림이 없다). 그것까지 함께 견주면 가상 경기
    # (서 있는 로봇이 라이다 상자로 나오는 것)에서 3 ~ 5점을 잃었다
    live_late: bool = True
    live_robot_m: float = 0.55  # 라이다 상자(지도에 없는 것 — 서 있는 다른 로봇)가 자리에서 이 안에 있으면 그 자리는 막힐 가능성 live_robot_p 로 본다
    live_robot_p: float = 0.8
    # 위에서 본(드론 사진) 「서 있는 다른 로봇」 도 같은 값으로 본다 (FaceScout.robots — mission 이 넣어 준다. 2026-11-20: navigation 이 서 있는 로봇의
    # 라이다 상자를 목록에서 빼게 된 뒤로, 라이다 상자만으로는 로봇을 알 수 없다 — 00:00 연습 판: 가운데에 1분 넘게 서 있던 로봇 곁의 자리로 가다 19초를 잃었다).
    # 오래 서 있던 로봇은 곧 떠난다고 보지 않는다: 반으로 보는 시간을 「서 있은 시간」 까지 늘린다 (20초는 정찰 자리에 5 ~ 25초 서는 로봇의 값이다)
    # 서 있는 로봇 곁의 자리가 막힐 가능성을 「늦게 닿을수록 줄어든다」 고 보던 값 (반으로 주는 시간, 전 20초). 사실상 끔 (2026-11-20 새벽):
    # 03:28 연습 판에서 경기장 가운데 (2.71, 1.45) 에 다른 팀 로봇이 끝까지 서 있었는데, 그 곁의 세 자리가 「나중에 갈수록 점수가 높은」 자리가 되어
    # 가까운 자리를 두고 3 m 떨어진 자리부터 가며 경기장을 동서로 네 번 가로질렀다 (정찰 중 34.3 m). 기다린 값을 치르는 셈이 없어서다.
    # 줄지 않게 두면 그 판의 65.7초부터의 길이 24.6 m → 17.1 m (단순하게 이어 본 것), 가상 경기(다른 로봇이 있는 360판)에서 점수는 18.73 → 18.73,
    # 달린 길 −0.30 m, 막힌 이동 +0.16번. 서 있는 로봇 곁의 자리는 여전히 뒤로 가고, 다른 자리가 없으면 간다
    live_robot_half_s: float = 1.0e6
    live_line_m: float = 0.40  # 지금 선 곳에서 그 자리까지의 곧은 선에서 이 안에 라이다 상자가 있으면 가는 길이 막힐 가능성 live_line_p
    live_line_p: float = 0.5
    live_centre: tuple = (0.9, 60.0, 0.25)  # 경기장 한가운데에서 이 거리(m) 안의 자리는, 경기 이 시각(초)까지 막힐 가능성을 이만큼 더 본다 (처음 1분은 가운데 원에 로봇이 몰린다)
    sector_first: bool = False
    plan_budget_s: float = 170.0  # 순서를 정할 때 보는 「정찰에 쓸 수 있는 시간」 (경기 240초에서 지도 · 사격 · 거점 앞 대기를 뺀 어림)
    plan_horizons: tuple = ((0.6, 1.0), (0.8, 2.0), (1.0, 3.0), (1.15, 3.0))  # 값을 보는 때 (가장 짧은 길로 다 도는 시간의 몇 배, 무게)
    plan_speed: float = 0.30  # 시간 어림: 이동 속도 (m/s)
    plan_stop_s: float = 5.0  # 자리 하나에 서고 도는 데 드는 시간
    plan_face_s: float = 2.5  # 면 하나를 찍는 데 드는 시간
    # 2026-11-19: 자리에서 차체 방향을 정확히 맞추느라 쓴 시간이 이동보다 길었다 (3차 팀 테스트: 이동 50초, 자리에서 방향 맞추기 65초).
    # 카메라는 팬으로 겨누므로 차체는 「찍을 면이 모두 팬 범위에 드는 방향」 이면 된다 → 그 범위의 가운데를 목표로, 범위의 반을 허용으로 준다.
    # 팬 끝(±48°)에서 이만큼은 남긴다. 2026-11-19 저녁: 8 → −8 (면 하나를 찍는 자리의 차체 방향 허용이 ±40° → ±56°).
    # 16:01 실차의 시간을 쪼개 보니 자리에 와서 차체 방향을 맞추는 데 56초(정찰의 27%)를 썼다 — 도착할 때의 방향이 목표에서 가운데값 111° 벗어나
    # 있었고, 서서 도는 속도는 초당 24° 다. 팬 끝을 조금 넘는 면도 팬을 끝까지 돌려 찍으므로(aim_over_deg 16) 차체는 그만큼 덜 돌아도 된다
    aim_margin_deg: float = -8.0
    # 자리에 왔는데 면이 팬 끝을 조금 넘는 방향에 있으면, 버리지 않고 팬을 끝까지 돌려 찍는다 (_aim). 2026-11-19 14:52 실차: 벽 쪽 통로에서
    # 차체를 돌릴 자리가 없어 선 채로, 찍을 면이 팬 48.6° · 54.3° 방향에 있었는데(한도 48°) 사진을 찍지 않고 떠났다 — 둘 다 탄약상자였고
    # 그 면 하나씩이 모자라 R-S3 · CM 두 구역(16점)이 틀렸다. 12° 를 넘겨도 면 가운데는 화각(±64°) 안쪽 60° 에 온다
    aim_over_deg: float = 16.0  # 12 → 16 (2026-11-19 16:01 실차: 아래 통로에서 팬 63° 방향의 면을 3° 차이로 못 찍었다. 16° 를 넘겨도 면은 화각 안이다)
    # 벽 쪽 통로의 면 (상자의 면과 벽 사이가 corridor_gap_m 안): 통로 가운데에, 통로 방향 그대로 서서 팬만 옆으로 돌려 찍는다.
    # 2026-11-19 14:52 실차: 그런 면의 자리가 벽에서 0.18 m 인 곳(격자 0.2 m 의 줄)에 잡혔다. 로봇은 통로에 들어가기는 했는데 벽을 보고 서서
    # 차체를 돌리지 못했고(120° 남음), 다음 자리는 11.5초 뒤 시한으로 끝났다 — 위 통로의 두 면(미사일 · 박격포)을 못 적어 R-S1 이 0점이었다.
    # 통로 가운데(벽과 면에서 같은 거리)에서는 면 가운데보다 그 거리만큼 앞에 서면 면이 팬 45° 방향 · 비스듬함 45° 로 사진에 다 들어온다.
    # 들어온 쪽으로 향한 채 찍으므로 차체를 돌릴 일이 없다 (오는 쪽에 따라 면의 앞 · 뒤 어느 쪽에 설지를 그 자리로 갈 때 정한다)
    corridor_gap_m: tuple = (0.44, 0.86)  # 면과 벽 사이가 이 안이면 통로로 본다 (주최측: 상자는 벽에서 최소 40 cm)
    corridor_yaw_tol_deg: float = 20.0  # 정한 방향에서 이만큼 벗어나도 된다
    # 2026-11-19 16:01 실차(이 방식의 첫 판): 통로의 둘째 자리 둘(3N · 12S)은 찍혔고 첫 자리 둘(2N · 13S)은 못 찍었다. 첫 자리에는 통로 밖에서
    # 비스듬히 들어와 벽 쪽으로 18 ~ 21° 틀어진 채 섰고(허용 20° 안이라 그대로 찍으려 했다), 자리도 9 cm 지나쳐서 면이 팬 72° · 63° 방향이 됐다.
    # 면 가운데보다 「벽까지의 반」 만큼만 앞에 서면 방향이 맞아도 팬 45° 라 남는 것이 없었다. 그래서:
    #  · 더 뒤에 선다 (corridor_back × 벽까지의 반 → 0.6 m 통로에서 면 가운데보다 0.37 m 앞). 첫 자리가 통로 어귀 밖이 되어 차체를 돌릴 자리도 생긴다
    #  · 차체 방향을 통로 방향에서 상자 쪽으로 corridor_bias_deg 만큼 돌려 잡는다 (벽 쪽으로 틀어지는 것이 나쁘고 상자 쪽은 괜찮다)
    # 값은 「±12 cm 어긋나고 정한 방향에서 ±20° 틀어진」 모든 경우를 셈해 골랐다: 필요한 팬 최대 58° (한도 48 + aim_over_deg 16 = 64°),
    # 비스듬함 최대 59° (겨눈 면을 적는 한도 62°). 1.4 × · 8° 는 팬 57° · 비스듬함 61°, 1.15 × · 12° 는 팬 62° · 비스듬함 57°
    corridor_back: float = 1.25
    corridor_bias_deg: float = 12.0
    # 통로의 면 앞 통로 안으로, navigation 이 라이다로 찾은 상자(지도에는 없는 것 — /team/navigation/lidar_boxes)가 그 면의 선보다 벽 쪽으로
    # 이만큼 넘게 나와 있으면 그 통로 자리에는 가지 않는다 (맨 뒤로 미루고, 그때도 그대로면 지나친다 — FaceScout._corridor_narrowed). 0 이면 보지 않는다.
    # 2026-11-19 18:08 모의 경기 6차: 붙은 상자 둘을 하나로 그린 지도에서 아래 벽 통로의 폭을 0.79 m 로 믿고 그 어귀 (3.96, 0.39) 에 섰다. 실제 상자의
    # 면은 0.2 m 더 벽 쪽이었고, 로봇은 164초부터 경기 끝까지 그 통로에서 나오지 못해 거점 진입도 못 했다. navigation 은 44.9초부터 그 통로로
    # 0.21 m 나온 상자 (4.77, 0.77) 를 알리고 있었다 (navigation 담당의 제안 — docs/주행_막힘_우회_2026-11-19.md 3번).
    # 값: 지도의 나란한 두 상자도 잰 자리가 10 cm 까지 어긋난다 (같은 날 5차의 지도) — 지도의 상자끼리는 견주지 않고 라이다 상자만 본다
    corridor_intrude_m: float = 0.12
    # 구역을 떠나기 전에: 그 구역에 못 적은 면이 남았으면(팬이 안 닿아 미룬 면 · 답을 못 얻은 면) 그 면의 자리를 바로 다시 잡아 들른다 (구역마다 한 번).
    # M3 는 구역의 면을 다 맞혀야 점수다. 전에는 못 적은 면을 모든 자리를 돈 뒤에야 다시 봤는데, 거기까지 간 판이 없다
    # (2026-11-19 16:01 실차: 27초에 놓친 2N 을 210초에야 다시 가다가 거점 앞 대기로 끊겼다 — 그 면 하나로 R-S1 7점을 잃었다)
    retry_in_sector: bool = True
    # 여러 면을 찍는 자리에서도 방향 허용을 이보다 좁히지는 않는다. arrive_yaw_deg(25°)를 그대로 쓰면 navigation 이 21° 틀어진 채 도착으로 끝내
    # 가장자리 면이 팬 밖으로 나간다 (가상 경기에서 한 면을 놓쳤다) → 따로 둔다. navigation 이 세 번 돌아 보고 20° 안에서 끝낸 경우는
    # mission 이 기다리지 않고 팬이 닿는 면만 찍으므로(「도착」 을 알렸으면 바로 넘어간다) 서 있는 시간은 늘지 않는다
    # 10 → 14° (2026-11-19 사용자: 방향 허용을 조금 더 푼다). navigation 에는 4° 좁혀 넘기므로(mission) 여러 면 자리의 요청 허용이 6° → 10° 가 된다.
    # 값: 면들이 가운데에서 ±34° 안에 있으면 차체가 14° 틀어져도 팬 끝(±48°) 안이다. 그보다 넓게 벌어진 자리에서는 가장자리 면이
    # 팬 밖일 수 있고, 그 면은 겨누지 못하면 다시 돌 때로 미룬다 (_aim 이 None → 놓치지는 않는다)
    min_body_tol_deg: float = 14.0
    # 가려던 자리가 막혔을 때(다른 UGV 등): 지금 선 자리에서 보이는 남은 면을 찍는다. 그 조건
    here_range_m: float = 1.3
    here_min_m: float = 0.35
    here_incidence_deg: float = 55.0
    # 지정표적으로 보이는 면부터 (2026-11-19 저녁): 지정표적의 종류(comms · outpost · radar)가 찍힌 그림을, 적는 규칙보다 느슨하게 지도의 면에 대 본다
    # (확신 hint_conf 넘게 · 사진에서 hint_tol 안 · hint_range_m 안 · hint_incidence_deg 안, CM 이 아닌 아직 못 적은 면). 맞는 면이 있으면 적지는 않고
    # 그 면을 찍을 자리를 맨 앞으로 당긴다 — 가서 겨눠 찍은 사진이 적는 규칙을 통과해야 표적으로 친다. 한 판에 hint_max 번까지.
    # 오늘 실차 8판의 사진을 다시 넣어 보면: 그 종류가 한 번이라도 찍힌 7판 모두에서 처음 가리킨 면이 진짜 표적 면이었고,
    #   모의 5차 10.3초(실제로 찾은 때 100.3초) · 연습 14:52 판 13.9초(63.0초) · 모의 3차 25.6초(끝까지 못 찾음) · 4차 94.6초(끝까지 못 찾음).
    #   다른 면을 가리킨 것은 그 뒤에 4번 (같은 상자의 옆 면 2번 · 0.6 m 앞 72도 1번 · 3 m 밖 1번) — 순서만 바꾸므로 틀려도 자리 하나를 먼저 갈 뿐이다
    hint_on: bool = True
    hint_conf: float = 0.55
    hint_tol: float = 0.10
    hint_range_m: float = 4.5
    hint_incidence_deg: float = 75.0
    hint_max: int = 2
    avoid_m: float = 0.6  # 막혔던 자리에서 이 안의 다른 자리는 뒤로 미룬다 (같은 물체에 또 막힌다 — 3차: 0.4 m 떨어진 두 자리에서 21초씩)
    avoid_s: float = 40.0  # 막혔던 자리를 이만큼 동안 피한다
    # 한 번 「자리에 왔다」 고 본 뒤에는 위치 값이 이만큼 더 벗어나도 그 자리로 친다. 2026-11-19 실차(2경기장 B-1): 목표에서 0.12 ~ 0.17 m 인 곳에
    # 서서 위치 값이 도착 한도(0.15 m)를 넘나들 때마다 「가라 → 왔다(이동 거둠) → 가라」 가 되풀이됐다 — 같은 목표가 68초 동안 81번 나갔다
    arrive_slack_m: float = 0.12
    # ---- 선 자리에서 팬이 닿지 않는 면 (2026-11-19 14:52 · 16:01 실차 뒤): 다시 돌 때로 미루기 전에, 그 자리에서 차체를 그 면 쪽으로 한 번 돌려 본다.
    # 16:01 판(d74b578 — 통로 가운데에 통로 방향으로 서고 팬 끝을 12° 넘는 면까지 찍는 코드)에서 벽 쪽 통로의 두 자리가 이랬다: 차체 방향이
    # 허용(±20°)의 끝에 걸리고 선 자리가 8 cm 어긋나, 찍을 면이 팬 74° · 62° 방향이었다 (찍을 수 있는 한도 60°). 둘 다 「다시 돌 때」 로 미뤄졌고
    # 다시 돌러 가던 중 223초에 정찰이 끝났다 — 상자 2 의 N(미사일) · 상자 13 의 S(탄약상자) 하나씩이 모자라 R-S1 · R-S3 두 구역(14점)이 틀렸다
    # (30점 중 16점). 14:52 판에서도 채점 면 26개 가운데 4개가 같은 식으로 빠졌다. 다시 돌려면 경기장을 가로질러 와야 하지만 선 자리에서
    # 60° 쯤 더 도는 데는 2 ~ 3초다. 통로 가운데(벽 · 면에서 0.3 m)는 차체 반 대각선(0.17 m)보다 넓어 서서 돌 수 있다
    turn_to_face: bool = True  # False 면 전처럼 바로 다시 돌 때로 미룬다
    # ---- 자리에서 못 적은 면은 맨 뒤로 미루지 않고 바로 가까운 자리에서 다시 본다 (2026-11-19 밤, 팀장의 지적: 「인식에 실패하면 그것을 최후로
    # 미루니 왔다 갔다 하게 되고 구역마다 한두 면을 빠뜨려 0점」). 전에는 찍고 답을 기다리지 않고 떠났고, 못 적은 면은 모든 자리를 돈 뒤에야 다시 갔다.
    # 모의 5차: 상자 2 의 N 면을 63초에 못 찍고, 185초에 다시 가서 찍은 것은 안 적혔고, 세 번째로 가던 중(222초) 정찰이 끝났다 — R-S1 7점.
    # 16:01 연습 판: 통로의 두 면을 못 찍고 다시 가던 중 끝났다 — 14점. 14:52 판: 채점 면 26개 가운데 10개가 남았다.
    #   retry_nearby        자리를 떠나기 전에, 그 자리에서 못 적은 면(찍었는데 안 적힘 · 차체를 돌려도 팬이 안 닿음)을 더 가까이 · 더 정면에서 볼 자리를
    #                       바로 다음 차례에 끼워 넣는다 (면마다 한 번. 그래도 안 되면 전처럼 다 돈 뒤에 다시 본다). 벽 쪽 통로의 면은 통로의 반대편 자리로.
    #   stop_answer_wait_s  그 판단을 하려고, 자리의 마지막 사진의 판독 답을 이만큼까지 기다린다 (답은 0.06 ~ 0.25초면 온다 — 오늘 사진 1713장의 가장 긴 것 0.13초)
    retry_nearby: bool = True
    stop_answer_wait_s: float = 0.6
    roomy_first: bool = False  # (시험 중) 설 자리를 넓은 칸에서 먼저 고른다
    turn_tol_deg: float = 25.0  # 돌려 볼 때의 방향 허용 — 팬 끝(±48°)까지 넉넉히 남긴다 (navigation 은 허용의 가장자리에서 멈춘다)
    # ---- 덤: 겨눈 면 말고 같은 사진에 함께 찍힌 다른 면도 적는다 (pick_each).
    # 2026-11-19: 실차 사진 48장을 다시 맞춰 보았다. 지금 판독 모델은 사진 한 장에 그림을 5 ~ 8개 답한다 (벽의 배경 그림 · 먼 상자까지) →
    # 사진 전체 짝짓기(match)는 34장 가운데 1장만 받아들였고, 나머지는 겨눈 면 하나만 적고 다른 그림은 버렸다.
    # 그림 하나하나를 「자리」 와 「크기」 로 지도의 면에 붙인다. 맞게 적힌 그림의 크기는 계산한 크기의 0.93 ~ 1.25배였고,
    # 상자가 옮겨져 그 자리에 먼 상자의 그림이 찍힌 것은 0.22 ~ 0.60배였다
    each_tol: float = 0.05  # 그림의 자리가 면이 찍혀야 할 자리에서 사진 폭의 이만큼 안
    each_sep: float = 0.08  # 이 안에 크기가 맞는 다른 면이나 다른 그림이 또 있으면 헷갈리는 것으로 보고 적지 않는다
    each_height: tuple = (0.85, 1.45)  # 그림의 세로 크기 / 면이 찍힐 세로 크기 — 이 안이어야 한다
    each_width: tuple = (0.6, 1.6)  # 가로 크기도 같은 식으로 (아주 비스듬한 옆 면 · 가로로 긴 벽 그림을 거른다)
    each_fraction: float = 0.85  # 면이 이만큼은 보여야 한다 (가려진 면은 자리 · 크기 계산이 어긋난다)
    each_range_m: float = 1.6  # 이보다 먼 면은 덤으로 적지 않는다. 더 멀면 옆 상자의 그림과 크기로 가릴 수 없다 (2.7 m 에서 옆 상자 그림이 같은 크기로 찍힌 일이 있었다)
    each_incidence_deg: float = 55.0  # 이보다 비스듬한 면은 덤으로 적지 않는다 (겨눠 찍을 때의 한도와 같다)
    each_conf: float = 0.5  # 판독 모델의 확신이 이보다 낮은 그림은 덤으로 적지 않는다
    # 겨눈 면이 제자리에서 읽힌 사진에서만 덤을 적는다. 겨눈 면이 안 읽혔으면 사진이 통째로 밀렸을 수 있다 (팬이 덜 돈 사진 · 방향 오차 · 옮겨진 상자)
    # — 2026-11-18 의 팬이 도는 중에 찍힌 사진에서는 옆 상자의 그림이 이웃 면의 자리에 0.03 차이로 찍혔다
    each_need_aim: bool = True
    aim_height: tuple = (0.5, 2.0)  # 겨눈 면의 그림: 세로 크기가 이 밖이면 적지 않는다 (틀리게 적혔던 한 장은 0.34배 · 맞게 적힌 것은 0.53 ~ 1.40배)
    # ---- 본 것 (표): 겨누지 않았어도 사진에 또렷이 찍힌 면을, 다른 자리에서 찍은 여러 장이 같은 답일 때 적는다 (FaceBook.sight).
    # 2026-11-19 14:52 실차: 적은 면 16개는 모두 맞았는데 채점 면 26개 가운데 10개를 못 적어 네 구역 중 하나만 맞았다 (7 / 30점). 못 적은 면 가운데
    # 여럿은 이미 찍은 사진에 또렷이 있었다 — 13번 상자의 N 면(트럭)은 정찰 사진 13장 중 5장에, 13S · 12S · 2N · 9W 는 지나가며 찍은 사진에.
    # 사진 한 장의 그림 하나를 「본 것」 으로 치는 조건 (pick_votes): 그림 자리 가까이에 찍힐 면이 꼭 하나이고, 자리와 크기가 그 면에 맞고,
    # 잘리지 않았고, 확신 vote_conf 이상, 거리 vote_range_m 이하, 비스듬함 vote_incidence_deg 이하. 그 판의 기록(정찰 사진 13장 + 팬을 보낸 뒤의
    # 지나가며 찍은 사진 85장)에서 이 조건으로 고른 짝 가운데 틀린 것은 돌며 찍힌 한 장의 하나뿐이었고, 그것은 mission 이 돌던 사진으로 거른다.
    # 값은 그 한 판으로만 맞춘 것이다 — 다른 판의 기록이 생기면 다시 본다
    vote_on: bool = True  # False 면 「본 것」 으로는 적지 않는다 (겨눠 찍은 것 · 덤만)
    vote_pass: bool = True  # 지나가며 찍은 사진의 답도 「본 것」 으로 쓴다 (mission 이 팬을 한 번이라도 보낸 뒤 · 돌지 않을 때 찍은 것만 넘긴다)
    vote_need: int = 2  # 서로 떨어진 사진이 이만큼 같은 답이어야 적는다 (한 장으로는 적지 않는다)
    # 찍을 면이 모두 이미 「본 것」 (한 장뿐이라 아직 적히지는 않았지만 다른 답이 없는 것 — FaceBook.sighted) 인 자리는 맨 뒤로 미룬다.
    # 아무것도 모르는 면부터 보러 가고, 미룬 자리는 시간이 남으면 가서 확인한다 (못 가면 M3 개수에 그 「본 것」 이 들어간다 — FaceBook.count_sighted).
    # 정할 때는 그 자리를 처음 잡을 때뿐이다 — 이미 navigation 에 시킨 목표를 이것으로 바꾸지 않는다.
    # 기본은 끔 (2026-11-19 가상 경기, 지나가며 찍기를 흉내 낸 4가지 조건 × 48판): 맞은 구역이 192 → 191 · 165 → 167 · 162 → 159 · 132 → 136 으로
    # 묶음마다 오르내려 얻는 것이 분명하지 않았고, 한 장짜리로 세는 면(그만큼 틀릴 수 있는 면)만 늘었다. 미룬 자리를 끝에 다시 가느라 길이 길어진다
    defer_sighted: bool = False
    vote_apart: tuple = (0.10, 5.0, 0.3)  # 「서로 떨어진 사진」: 렌즈 자리가 이만큼(m) 다르거나, 카메라 방향이 이만큼(°) 다르거나, 찍은 때가 이만큼(초) 다르다
    vote_conf: float = 0.8
    vote_range_m: float = 2.5
    vote_incidence_deg: float = 60.0
    vote_tol: float = 0.05  # 그림의 자리가 면이 찍혀야 할 자리에서 사진 폭의 이만큼 안
    vote_sep: float = 0.08  # 이 안에 다른 면(크기가 맞든 안 맞든)이나 크기가 맞는 다른 그림이 또 있으면 헷갈리는 것으로 보고 치지 않는다
    vote_height: tuple = (0.85, 1.45)
    vote_width: tuple = (0.6, 1.6)
    vote_fraction: float = 0.85
    # 「한꺼번에 맞추기」 (align.aligned) 로 붙인 그림도 「본 것」 으로 친다. 그림을 하나씩 따로 붙이는 위의 규칙(pick_votes)은 나란히 붙은 상자의 면을
    # 2 m 밖에서는 가리지 못해 버린다 (두 면이 사진 폭의 0.09 안에 찍힌다). 사진의 그림 여럿이 면 여럿과 「같은 만큼 밀려」 맞는 자리를 찾으면 가릴 수 있다.
    # 2026-11-19 14:52 실차의 사진 101장을 정답표와 견줌 (확신 0.8 이상 · 비스듬함 55° 이하 · 3.2 m 안): 붙인 그림 116개가 모두 맞았다
    # (따로 붙이는 규칙은 65개). 60° · 3.5 m · 확신 0.5 로 풀면 120개 가운데 1개가 틀렸다 (2.4 m · 59° 에서 옆 상자의 미사일 그림)
    vote_align: bool = True
    vote_align_range_m: float = 3.2
    vote_align_incidence_deg: float = 55.0


def _size_of(pic):
    """그림의 (가로, 세로) 크기 — 사진 폭 · 높이에 대한 비율. 크기가 없는 답(옛 형식 · 가상 시험)이면 None."""
    if len(pic) >= 5 and pic[3] and pic[4]:
        return float(pic[3]), float(pic[4])
    return None


def _fits(pic, exp, heights, widths):
    """그림의 크기가 그 면이 찍힐 크기와 맞는가 (모르면 False)."""
    size = _size_of(pic)
    if size is None or exp.height <= 0.0 or exp.width <= 0.0:
        return False
    return heights[0] <= size[1] / exp.height <= heights[1] and widths[0] <= size[0] / exp.width <= widths[1]


SECTOR_POINTS = {"CM": 9.0}  # 그 밖의 채점 구역은 7점 (API 문서 7단계: 상대 진영 3구역 각 7 + CM 9)
# 못 본 면 하나가 「객체가 아니어서 보고에 영향이 없을」 가능성의 어림. 2026-11-19 연습 1경기장의 정답표(면 44개): 객체 26 · 지정표적 6 · 비객체 12 → 0.41
UNSEEN_OK = 0.4


def pick_centered(expected, pictures, target_key, tol=0.15, sep=0.08, heights=None, near=None, rival=None):
    """겨눠 찍은 사진에서 겨눈 면의 그림을 고른다: 그 면이 찍혀야 할 자리에 가장 가까운 그림 하나.

    pictures 의 그림은 (kind, x, cut) 또는 뒤에 (가로, 세로, 확신)이 더 붙은 것. heights = (가장 작게, 가장 크게) 를 주고 그림에 크기가 있으면
    세로 크기가 그 면이 찍힐 크기의 그 배수 안일 때만 적는다 (상자가 옮겨져 먼 상자의 그림이 그 자리에 찍힌 것을 거른다).
    near = (바탕, 1 m 에서 더하는 값, 가장 크게) 를 주면 가까운 면일수록 자리의 허용을 넓힌다: max(tol, min(가장 크게, 바탕 + 더하는 값 / 거리)).
      가까이서 찍으면 로봇 위치가 몇 cm 만 틀려도 그림이 사진에서 크게 밀린다 (0.44 m 앞에서 5 cm 는 사진 폭의 0.06).
    rival = (가장 작게, 가장 크게) 를 주면, 가까이 있는 다른 그림(sep 안)이 있어도 「고른 그림의 세로 크기는 이 면이 찍힐 크기의 그 배수 안인데
      그 다른 그림들은 모두 밖」 이면 헷갈리는 것으로 치지 않는다 (더 먼 상자의 그림 · 벽 그림이 옆에 찍힌 것). 크기를 모르는 그림은 전처럼 막는다.
      2026-11-19 모의 5차: 0.48 m 앞에서 찍은 면(전차 0.95, 자리 0.072 어긋남)을, 0.055 더 먼 자리의 뒤 상자 그림(세로 0.60배) 때문에 버려
      그 구역(7점)을 놓쳤다. 정답표가 있는 실차 8판의 서서 찍은 사진으로 견주면 맞게 적는 것 41 → 45장, 틀리게 적는 것 0 → 0장.
    돌려주는 값: (Expected, kind) 또는 None (그림이 없거나, 멀거나, 옆 그림과 헷갈리거나, 「모름」 일 때).
    """
    exp = next((e for e in expected if e.face.key == target_key), None)
    if exp is None or not pictures:
        return None
    if near is not None and exp.dist > 0.0:
        tol = max(tol, min(near[2], near[0] + near[1] / exp.dist))
    ranked = sorted(pictures, key=lambda q: abs(q[1] - exp.x))
    kind, x = ranked[0][0], ranked[0][1]
    d = abs(x - exp.x)
    if d > tol or kind == A.UNKNOWN:
        return None
    close = [q for q in ranked[1:] if abs(q[1] - exp.x) - d < sep]  # 고른 그림만큼 가까이 있는 다른 그림들
    if close:
        def tall(q):  # 세로 크기가 이 면이 찍힐 크기에 맞는가 (크기를 모르면 None)
            size = _size_of(q)
            return None if size is None or exp.height <= 0.0 else rival[0] <= size[1] / exp.height <= rival[1]

        if rival is None or tall(ranked[0]) is not True or any(tall(q) is not False for q in close):
            return None
    # 그 그림 자리에 더 가까운 다른 면이 있으면 (옆 면과 뒤바뀔 수 있다) 적지 않는다
    if any(o.face.key != target_key and abs(o.x - x) < d + sep * 0.5 for o in expected if o.dist <= exp.dist * 1.6):
        return None
    size = _size_of(ranked[0])
    if heights is not None and size is not None and exp.height > 0.0 and not heights[0] <= size[1] / exp.height <= heights[1]:
        return None
    return exp, kind


def pick_each(expected, pictures, p, skip=()):
    """사진의 그림 하나하나를 지도의 면에 붙인다 — 자리와 크기가 맞고 헷갈릴 것이 없는 그림만. 돌려주는 값: [(Expected, kind)].

    p = ScoutParams (each_… 값들). skip = 적지 않을 면의 key (겨눈 면은 pick_centered 가 따로 정한다).
    크기가 없는 답 (kind, x, cut) 은 붙이지 않는다 — 크기를 모르면 옮겨진 상자 · 벽 그림을 가릴 수 없다.
    그림 하나에 대해:
      1. 「모름」 · 가장자리에서 잘린 그림 · 확신이 낮은 그림은 뺀다
      2. 그림 자리에서 each_sep 안에 있고 크기가 맞는 면이 꼭 하나여야 한다 (둘이면 어느 면인지 모른다)
      3. 그 면이 each_tol 안이고, 충분히 보이고(each_fraction), 너무 멀거나 비스듬하지 않아야 한다
      4. 그 면 자리에서 each_sep 안에 크기가 맞는 다른 그림이 없어야 한다
    벽의 배경 그림 · 지도에 없는 상자의 그림은 붙을 면이 없어 그냥 넘어간다.
    """
    out = []
    sized = [q for q in pictures if _size_of(q) is not None]
    for q in sized:
        kind, x = q[0], q[1]
        if q[2] or kind == A.UNKNOWN or kind not in A.ALL_KINDS:
            continue
        if len(q) >= 6 and q[5] is not None and q[5] < p.each_conf:
            continue
        near = [e for e in expected if abs(x - e.x) <= p.each_sep and _fits(q, e, p.each_height, p.each_width)]
        if len(near) != 1:
            continue
        e = near[0]
        if e.face.key in skip or abs(x - e.x) > p.each_tol:
            continue
        if e.fraction < p.each_fraction or e.dist > p.each_range_m or e.incidence_deg > p.each_incidence_deg:
            continue
        if any(o is not q and abs(o[1] - e.x) <= p.each_sep and _fits(o, e, p.each_height, p.each_width) for o in sized):
            continue
        out.append((e, kind))
    return out


def pick_votes(expected, pictures, p):
    """사진 한 장의 그림들 가운데 「본 것」 으로 칠 것을 고른다 (겨누지 않은 면의 증거 — FaceBook.sight 에 넣는다). 돌려주는 값: [(Expected, kind)].

    p = ScoutParams (vote_… 값들). pick_each 보다 엄하다 — 이 답은 겨눈 면의 확인 없이, 움직이며 찍은 사진에서도 쓰기 때문이다:
      1. 「모름」 · 가장자리에서 잘린 그림 · 크기나 확신이 없는 그림 · 확신이 vote_conf 보다 낮은 그림은 뺀다
      2. 그림 자리에서 vote_sep 안에 찍힐 면이 꼭 하나여야 한다 — **크기가 맞든 안 맞든**. pick_each 는 크기가 맞는 면만 세는데, 그러면 나란히 놓인
         두 상자의 면이 겹쳐 찍힐 때 크기 검사가 엉뚱한 쪽을 골라 줄 수 있다 (2026-11-19 14:52 실차: 2.4 m 밖 59° 로 본 3S 자리에서 0.03 옆인
         2S 의 미사일 그림이 3S 의 크기 범위에 들어 3S 의 것으로 붙었다)
      3. 그 면이 vote_tol 안이고 크기가 맞고(vote_height · vote_width), 충분히 보이고(vote_fraction), 너무 멀거나 비스듬하지 않아야 한다
      4. 그 면 자리에서 vote_sep 안에 크기가 맞는 다른 그림이 없어야 한다
    """
    out = []
    sized = [q for q in pictures if _size_of(q) is not None]
    for q in sized:
        kind, x = q[0], q[1]
        if q[2] or kind == A.UNKNOWN or kind not in A.ALL_KINDS:
            continue
        if len(q) < 6 or q[5] is None or q[5] < p.vote_conf:
            continue
        near = [e for e in expected if abs(x - e.x) <= p.vote_sep]
        if len(near) != 1:
            continue
        e = near[0]
        if abs(x - e.x) > p.vote_tol or not _fits(q, e, p.vote_height, p.vote_width):
            continue
        if e.fraction < p.vote_fraction or e.dist > p.vote_range_m or e.incidence_deg > p.vote_incidence_deg:
            continue
        if any(o is not q and abs(o[1] - e.x) <= p.vote_sep and _fits(o, e, p.vote_height, p.vote_width) for o in sized):
            continue
        out.append((e, kind))
    return out


class _Base:
    def __init__(self, fmap, grid, book, cam=None, params=None):
        self.fmap, self.grid, self.book = fmap, grid, book
        self.cam = cam or CameraParams()
        self.any_heading = False  # 지금 자리에서 차체 방향을 따지지 않는다 (accept_heading)
        self.skipped = 0  # 가지 못해 건너뛴 자리 수
        self.p = params or ScoutParams()
        self.body_tol = math.radians(self.p.arrive_yaw_deg)  # 지금 자리에서 차체 방향의 허용 (면 기준은 자리마다 다시 정한다)
        self.log = []  # (글) — 판단 기록
        self.shots = 0
        self.rejected = 0
        self.bonus = 0  # 덤으로 적은 면의 수 (겨누지 않았는데 같은 사진에서 읽은 면)
        self.voted = 0  # 「본 것」 여럿으로 적은 면의 수 (_see)
        self.want = None  # 지정표적의 종류 (perception 이름: comms · outpost · radar). None = 모른다 · 이미 찾았다 (want_kind)
        self._lead = None  # 지정표적으로 보여 먼저 찍으러 갈 면의 key (_hint → FaceScout.step 이 순서를 바꾼다)
        self._hinted = set()  # 그렇게 가리킨 적이 있는 면
        self._hints_used = 0  # 그림으로 가리킨 횟수 (hint_max 까지. 이어서 가리킨 옆 면은 세지 않는다 — FaceScout._hint_followup)
        self._hint_closed = set()  # 가리킨 면 가운데 종류가 적혀 끝난 것
        self.trace = None  # 사진 한 장을 어떻게 읽었는지 받아 적을 함수 (현장에서 짝짓기가 맞는지 볼 때)

    def _note(self, tag, pose, pan_deg, expected, pictures, pairs, bonus=(), sights=None):
        """사진 한 장을 어떻게 읽었는지 적는다. pairs = 적은 짝, bonus = 그 가운데 덤(겨누지 않은 면)으로 적은 짝 — "bonus": true 가 붙는다.
        sights = 이 사진에서 「본 것」 으로 친 짝 [(Expected, kind, 이번에 그 면이 적혔는가)] — "sights" 로 따로 붙는다 (없으면 안 붙는다)."""
        if self.trace is None:
            return

        def pic(q):
            d = {"kind": q[0], "x": round(q[1], 3), "cut": bool(q[2])}
            if len(q) >= 5:  # 크기가 있는 답 (실차)
                d.update(w=round(q[3], 3), h=round(q[4], 3))
            if len(q) >= 6 and q[5] is not None:
                d["conf"] = round(q[5], 2)
            return d

        self.trace({
            "tag": [str(v) for v in tag], "pose": [round(pose[0], 3), round(pose[1], 3), round(math.degrees(pose[2]), 1)], "pan_deg": round(pan_deg, 1),
            "expected": [{"box": e.face.box, "side": e.face.side, "x": round(e.x, 3), "dist": round(e.dist, 2), "incidence": round(e.incidence_deg), "sure": e.sure,
                          "w": round(e.width, 3), "h": round(e.height, 3)} for e in expected],
            "pictures": None if pictures is None else [pic(q) for q in pictures],
            "pairs": None if pairs is None else [{"box": e.face.box, "side": e.face.side, "kind": k} for e, k in pairs]
            + [{"box": e.face.box, "side": e.face.side, "kind": k, "bonus": True} for e, k in bonus],
            **({"sights": [{"box": e.face.box, "side": e.face.side, "kind": k, "recorded": bool(new)} for e, k, new in sights]} if sights else {}),
        })

    def _at(self, pose, x, y, yaw=None):
        d = math.hypot(pose[0] - x, pose[1] - y)
        if self.any_heading:  # 이 자리에 이미 왔다고 봤다 (또는 차체가 더 돌지 못한다) → 방향은 따지지 않고, 위치도 조금 벗어난 것은 넘어간다
            return d <= self.p.arrive_m + self.p.arrive_slack_m
        if d > self.p.arrive_m:
            return False
        return yaw is None or abs(wrap(yaw - pose[2])) <= self.body_tol

    def near_stop(self, pose, x, y):
        """자리(위치)에는 와 있는가 — 방향은 보지 않는다."""
        return math.hypot(pose[0] - x, pose[1] - y) <= self.p.arrive_m + (self.p.arrive_slack_m if self.any_heading else 0.0)

    def accept_heading(self):
        """지금 자리에서는 차체 방향을 더 맞추지 않는다. 다음 자리로 넘어가면 풀린다."""
        if not self.any_heading:
            self.any_heading = True
            self.log.append("자리에 왔으나 방향이 덜 맞음 → 지금 방향에서 팬으로 닿는 면만 찍는다")

    def want_kind(self, kind):
        """지정표적의 종류(perception 이름)를 알린다. None 이면 끈다 (모른다 · 이미 찾았다)."""
        self.want = str(kind) if kind else None

    def _hint(self, expected, pictures):
        """지정표적의 종류가 찍힌 그림이 가리키는 면을 찾아 「먼저 찍으러 갈 면」 으로 표시한다 (적지는 않는다). 가리킨 면의 key 또는 None.

        값과 근거는 ScoutParams.hint_… 의 설명. 무슨 일이 있어도 예외를 밖으로 내지 않는다 (덤으로 하는 일이다).
        """
        p = self.p
        try:
            if not p.hint_on or self.want is None or not pictures or self._hints_used >= p.hint_max:
                return None
            left = set(self.book.remaining())
            for q in pictures:
                if q[0] != self.want or (len(q) >= 6 and q[5] is not None and q[5] < p.hint_conf):
                    continue
                near = [e for e in expected if e.face.key in left and getattr(e.face, "sector", None) != "CM"
                        and e.dist <= p.hint_range_m and e.incidence_deg <= p.hint_incidence_deg and abs(e.x - q[1]) <= p.hint_tol]
                if not near:
                    continue
                e = min(near, key=lambda c: abs(c.x - q[1]))
                if e.face.key in self._hinted:
                    continue  # 이미 가리킨 면의 그림이다 (같은 그림이 또 찍혔다고 그 옆 면을 가리키지 않는다)
                self._hinted.add(e.face.key)
                self._hints_used += 1
                self._lead = e.face.key
                self.log.append("지정표적(%s)으로 보이는 그림: 상자 %d 의 %s 면 (%.1f m · %.0f도) → 그 면부터 찍으러 간다" % (
                    self.want, e.face.box, e.face.side, e.dist, e.incidence_deg))
                return e.face.key
        except Exception as error:  # noqa: BLE001
            self.log.append("지정표적으로 보이는 면을 찾다 예외 (%s: %s)" % (type(error).__name__, error))
        return None

    def _see(self, pose, pan_deg, expected, pictures, at=None, each=True):
        """사진 한 장의 답에서 「본 것」 을 골라 면 기록에 넣는다 (FaceBook.sight). 돌려주는 값: [(Expected, kind, 이번에 그 면이 적혔는가)].

        그림을 고르는 것은 pick_votes. 채점 구역의 면만 넣는다. at = 사진을 찍은 때 (모르면 None).
        each=False 이면 그림을 하나씩 따로 붙이는 규칙(pick_votes)은 쓰지 않고 「한꺼번에 맞추기」 로 붙은 것만 넣는다 (방향 값이 덜 믿기는 사진).
        """
        p = self.p
        if not p.vote_on or not pictures:
            return []
        goal = {f.key for f in self.book.goal}
        cx, cy, heading = camera_pose(pose, pan_deg, self.cam)
        out = []
        picked = {e.face.key: (e, kind) for e, kind in pick_votes(expected, pictures, p)} if each else {}
        if p.vote_align:
            try:
                _shift, pairs = aligned(expected, pictures, min_conf=p.vote_conf)
            except Exception as error:  # noqa: BLE001 — 덤으로 하는 일이다. 여기서 난 예외로 사진의 다른 답까지 버리지 않는다
                self.log.append("한꺼번에 맞추기 예외 (%s: %s)" % (type(error).__name__, error))
                pairs = []
            for e, kind, _resid, _q in pairs:
                if e.fraction < p.vote_fraction or e.dist > p.vote_align_range_m or e.incidence_deg > p.vote_align_incidence_deg:
                    continue
                old = picked.get(e.face.key, False)
                if old is False:
                    picked[e.face.key] = (e, kind)
                elif old is not None and old[1] != kind:
                    picked[e.face.key] = None  # 두 방법의 답이 다르다 → 이 사진에서는 이 면을 치지 않는다
        for e, kind in [v for v in picked.values() if v is not None]:
            if e.face.key not in goal:
                continue
            new = self.book.sight(e.face.key, kind, (cx, cy, heading, at), need=p.vote_need, apart=p.vote_apart)
            if new is not None:
                self.voted += 1
                self.log.append("본 것으로 적음: 상자 %d 의 %s 면 = %s (사진 %d장)" % (e.face.box, e.face.side, new, len(self.book.sights[e.face.key])))
            out.append((e, kind, new is not None))
        return out

    def _aim(self, pose, face):
        """그 면을 찍을 팬 각도 (오른쪽 +). 보통은 면을 사진 가운데에 둔다. 팬 끝(±48°)을 aim_over_deg 까지 넘는 면은 팬을 끝까지 돌려 찍는다
        (면이 사진 가운데에서 그만큼 벗어날 뿐 화각 ±64° 안이다 — 면의 자리는 실제 팬으로 다시 계산하므로 적는 데는 문제가 없다). 그보다 멀면 None."""
        pan = -math.degrees(wrap(math.atan2(face.cy - pose[1], face.cx - pose[0]) - pose[2]))
        limit = PAN_LIMIT_DEG - 2.0
        if abs(pan) <= limit:
            return pan
        return math.copysign(limit, pan) if abs(pan) <= limit + self.p.aim_over_deg else None


class PlaceScout(_Base):
    """자리 기준: 남은 면 / (이동 + 돌기 + 머묾) 이 가장 큰 자리로 가서 정면 한 장 (sim-strategy 의 정찰과 같은 고르기)."""

    def __init__(self, fmap, grid, book, cam=None, params=None):
        super().__init__(fmap, grid, book, cam, params)
        self.cover = CoveragePlanner(fmap, grid, book.goal, self.p.place_view)
        self.visited = set()
        self.rounds = 0
        self.view = None  # (번호, x, y, 방향, 면들)
        self.pending = None  # 답을 기다리는 사진의 tag
        self.asked_at = None
        self.tries = 0

    def remaining(self):
        return self.book.remaining() - self.cover.unreachable

    def step(self, pose, now):
        if self.pending is not None:
            if now - self.asked_at < self.p.answer_wait_s:
                return ("wait",)
            self.pending = None  # 답이 안 온다 → 이 자리는 넘어간다
            self._leave()
        remaining = self.remaining()
        if not remaining:
            return ("done",)
        if self.view is None or not (self.view[4] & remaining):
            pick = self.cover.next_view(pose, remaining, banned=self.visited)
            if pick is None:
                if self.rounds >= 2:
                    self.log.append("더 볼 수 있는 자리가 없음 (남은 면 %d)" % len(remaining))
                    return ("done",)
                self.rounds += 1
                self.visited = set()
                return ("wait",)
            self.view, self.tries = pick, 0
        k, x, y, heading, _keys = self.view
        if not self._at(pose, x, y, heading):
            return ("go", x, y, heading)
        return ("shoot", 0.0, ("place", k, self.tries))

    def shot_taken(self, tag, now):
        self.shots += 1
        self.pending, self.asked_at = tag, now

    def on_answer(self, tag, pose, pan_deg, pictures):
        if tag != self.pending:
            return
        self.pending = None
        expected = expected_faces(self.fmap, pose, pan_deg, self.cam)
        pairs = None if pictures is None else match(expected, pictures)
        self._note(tag, pose, pan_deg, expected, pictures, pairs)
        if pairs is None:
            self.rejected += 1
            self.tries += 1
            if self.tries <= self.p.retries:
                return  # 같은 자리에서 다시 한 장
        else:
            self.book.add_all(pairs)
        self._leave()

    def _leave(self):
        if self.view is not None:
            self.visited.add(self.view[0])
        self.view = None
        self.any_heading = False

    def skip_stop(self, now=None, retry=True):
        """지금 가려던 자리를 포기하고 다음 자리를 고르게 한다 (가지 못했다)."""
        if self.view is not None:
            self.skipped += 1
            self.log.append("자리 %s 에 가지 못함 → 다른 자리로" % (self.view[0],))
            self._leave()


class FaceScout(_Base):
    """면 기준: 설 자리와 순서를 처음에 다 정하고, 자리마다 면을 하나씩 겨눠 찍는다. 답은 기다리지 않는다."""

    def __init__(self, fmap, grid, book, cam=None, params=None, start=None):
        super().__init__(fmap, grid, book, cam, params)
        self.by_key = book.by_key  # 면 기록과 같은 표 (key → 면). 뒤에 생긴 면(FaceBook.add_faces)도 여기서 바로 찾을 수 있다
        self.stops = []  # [(x, y, [면 key])] 순서대로
        self.i = 0
        self.queue = []  # 지금 자리에서 찍을 [면 key]
        self.body = None  # 지금 자리에서 차체가 볼 방향
        self.out = {}  # 답을 기다리는 사진: tag → 찍은 시각
        self.tried = {}  # 면 key → 찍은 횟수
        self.passes = 0
        self.unreachable = set()
        self._planned = False
        self.first = ()  # 먼저 돌 구역들 (임무 명령의 탐색 구역 가운데 상대 진영의 것 — prefer)
        self.corridor = {}  # 벽 쪽 통로의 면: key → ((통로를 +쪽으로 가며 설 자리 x, y, 방향), (−쪽으로 가며 설 자리 x, y, 방향))
        self._sectors_begun = set()  # 자리를 하나라도 시작한 구역
        self._sectors_retried = set()  # 떠나기 전에 다시 본 구역 (구역마다 한 번)
        self._reorder = False  # 먼저 돌 구역이 계획 뒤에 정해졌다 → 다음 자리로 넘어갈 때 남은 자리의 순서를 다시 정한다
        self._reorder_at = None  # 이 번호의 자리로 넘어갈 때 남은 자리의 순서를 다시 정한다 (지정표적으로 보이는 면의 자리를 앞으로 당긴 뒤)
        self._start = start
        self.blocks = []  # [(x, y, 시각)] 가지 못한 자리 — 그 둘레는 한동안 피한다
        self.deferred = set()  # 뒤로 미룬 적이 있는 자리 (한 번만 미룬다)
        self.here_shots = 0  # 막힌 자리 대신 선 자리에서 찍은 횟수
        self.turned = set()  # 지금 자리에서 「차체를 그 면 쪽으로 돌려 보라」 고 한 면 (면마다 한 번 — 자리를 떠나면 비운다)
        self.nearby = set()  # 못 적어서 「바로 가까운 자리에서 다시 보기」 를 한 번 쓴 면 (retry_nearby)
        self.nearby_stops = 0  # 그렇게 끼워 넣은 자리 수
        self.turns = 0  # 그렇게 돌려 본 횟수
        self.unaimed = 0  # 돌려 봐도(또는 돌려 보지 않아) 선 자리에서 끝내 겨누지 못해 다시 돌 때로 미룬 면의 수
        self.late = set()  # 뒤에 생긴 면의 key (지도에 없던 상자 — add_faces). 처음 계획에서는 빼 두었다가 맨 뒤에 붙인다
        self._gone_known = frozenset()  # 지금 자리를 잡을 때 이미 빠져 있던 상자의 번호 (step — 가는 동안 새로 빠진 상자와 가린다)
        self.lidar_boxes = None  # navigation 이 라이다로 찾았으나 지도에는 없는 상자의 가운데 [(x, y)] — mission 이 틱마다 넣어 준다 (_corridor_narrowed)
        # ---- 실시간 순서 (order = "live")
        self.robots = None  # 위에서 본, 서 있는 다른 로봇 [(x, y, 서 있은 시간 s)] — mission 이 틱마다 넣어 준다 (standing.Standing). 모르면 None
        self.clock = None  # (경기 시각, 거점 앞 대기를 시작할 경기 시각의 어림, 대기 자리 (x, y)) — mission 이 틱마다 넣어 준다. 없으면 plan_budget_s 로 어림한다
        self._fields = {}  # 자리의 칸 → 그 자리에서 모든 칸까지의 길 길이를 접어 둔 것 (_keep_field · _dist)
        self._pins = set()  # 순서를 바꾸지 않고 꼭 지금 갈 자리 (그 자리의 면들): 지정표적으로 보이는 면의 자리 · 못 적어 바로 다시 보는 자리
        self._live_at, self._live_seen, self._live_t0 = -1e9, None, None  # 마지막으로 고른 때 · 그때의 (못 적은 면 수, 라이다 상자) · 처음 고른 때
        self._live_errors, self._live_off, self._live_switched = 0, False, False
        self._live_pace, self._live_leg = 1.0, None  # 시간 어림에 곱하는 값 (실제로 걸린 시간 / 어림) · 지금 가는 길 (떠난 때, 어림한 시간, 그때 지정표적을 찾고 있었나)

    def remaining(self):
        return self.book.remaining() - self.unreachable

    # ---- 계획
    def plan(self, pose, only=None):
        """남은 면(또는 only)을 가까이서 볼 자리들을 고르고 순서를 정한다."""
        want = set(only) if only is not None else {f.key for f in self.book.goal}
        late = set()
        if only is None and self.late:  # 뒤에 생긴 면(지도에 없던 상자)은 처음 계획에 섞지 않고 맨 뒤에 붙인다
            late = want & self.late
            want = want - late
        chosen, unseen, left = self._choose(pose, want, again=only is not None)
        chosen, unseen = self._corridor_stops(pose, want, chosen, unseen)
        if only is None:
            self.unreachable = set(unseen)
        # 다시 도는 면은 가까운 것부터다 (먼저 돌 구역을 앞세우지 않는다 — 16:01 실차: 0.45 m 옆의 못 적은 면을 두고 4.3 m 떨어진
        # 「먼저 돌 구역」 의 면으로 가다가 시간이 끝났다)
        self.stops = self._order(pose, chosen, lead=only is None)
        self.i, self.queue, self.body = 0, [], None
        self.turned = set()
        self._planned = True
        self.log.append("정찰 계획: 설 자리 %d곳, 면 %d개 (설 자리가 없는 면 %d개)" % (len(self.stops), sum(len(s[3]) for s in self.stops), len(left) + len(unseen)))
        if late:
            self._append(pose, late)

    def _corridor_of(self, face):
        """면이 벽 쪽 통로를 보고 있으면 두 자리 ((x, y, 방향), (x, y, 방향)) — 통로를 + 쪽으로 가며 설 자리와 − 쪽으로 가며 설 자리. 아니면 None.

        자리: 통로 가운데(면과 벽의 한가운데)에서, 면 가운데보다 corridor_back × (벽까지의 반) 만큼 앞. 방향: 가는 쪽에서 상자 쪽으로 corridor_bias_deg.
        """
        p = self.p
        lo, hi = p.corridor_gap_m
        bias = math.radians(p.corridor_bias_deg)
        if abs(face.ny) > 0.5:
            gap = (self.fmap.height - face.cy) if face.ny > 0 else face.cy
            if not lo <= gap <= hi:
                return None
            y, back = face.cy + face.ny * gap / 2.0, p.corridor_back * gap / 2.0
            return (face.cx - back, y, wrap(-face.ny * bias)), (face.cx + back, y, wrap(math.pi + face.ny * bias))
        gap = (self.fmap.width - face.cx) if face.nx > 0 else face.cx
        if not lo <= gap <= hi:
            return None
        x, back = face.cx + face.nx * gap / 2.0, p.corridor_back * gap / 2.0
        return (x, face.cy - back, wrap(math.pi / 2.0 + face.nx * bias)), (x, face.cy + back, wrap(-math.pi / 2.0 - face.nx * bias))

    def _corridor_narrowed(self, key):
        """벽 쪽 통로의 면(key) 앞 통로 안으로, 지도에는 없고 navigation 이 라이다로 찾은 상자(self.lidar_boxes)가 나와 있는가.

        통로 자리는 「그 면에서 벽까지」 를 통로의 폭으로 믿고 그 가운데에 잡는다. 라이다 상자의 벽 쪽 변이 그 면의 선보다 벽 쪽으로
        corridor_intrude_m 넘게 나와 있으면 실제 통로는 그만큼 좁다 — 그 둘레의 지도가 틀렸다는 표시다 (붙은 상자 둘을 하나로 그린 지도 등).
        보는 범위: 통로를 따라 면 가운데에서 「설 자리까지의 거리 + 상자 한 변」 안. 벽 밖에 걸친 것(벽의 점으로 생긴 헛것)은 치지 않는다.
        지도의 상자끼리는 견주지 않는다 — 나란한 두 상자의 자리는 잰 값이 10 cm 까지 다르고, 20 cm 더 벽 쪽에 선 이웃 상자는 정상 배치다.
        """
        p, m = self.p, self.fmap
        face = self.by_key.get(key)
        if face is None or key not in self.corridor or p.corridor_intrude_m <= 0.0 or not self.lidar_boxes:
            return False
        half = m.box_size / 2.0
        across = abs(face.ny) > 0.5  # 위 · 아래 벽 쪽 통로 (통로는 x 방향으로 뻗는다)
        if across:
            gap = (m.height - face.cy) if face.ny > 0 else face.cy
        else:
            gap = (m.width - face.cx) if face.nx > 0 else face.cx
        reach = p.corridor_back * gap / 2.0 + m.box_size + half
        try:
            boxes = [(float(q[0]), float(q[1])) for q in self.lidar_boxes]
        except (TypeError, ValueError, IndexError):  # 목록의 모양이 다르다 → 보지 않는다 (덤으로 하는 확인이다)
            return False
        for bx, by in boxes:
            along = (bx - face.cx) if across else (by - face.cy)
            out = ((by - face.cy) * face.ny if across else (bx - face.cx) * face.nx) + half  # 그 상자의 벽 쪽 변이 면의 선보다 벽 쪽으로 나온 거리
            if abs(along) <= reach and p.corridor_intrude_m < out <= gap:
                return True
        return False

    def _corridor_stops(self, pose, want, chosen, unseen):
        """고른 자리들에서 벽 쪽 통로의 면을 빼고, 그 면마다 통로 가운데의 자리를 따로 넣는다. 돌려주는 값: (자리들, 설 자리가 없는 면)."""
        reach = self.grid.distances((pose[0], pose[1]))
        mine = {}
        for key in sorted(want):
            face = self.by_key.get(key)
            pair = self._corridor_of(face) if face is not None else None
            if pair is None:
                continue
            ok = [spot for spot in pair if self.grid.free(spot[0], spot[1]) and self.grid.cell_of(spot[0], spot[1]) in reach]
            if not ok:
                continue  # 통로 가운데에 설 수 없다 (기둥 · 다른 상자) → 전의 자리 그대로
            self.corridor[key] = pair if len(ok) == 2 else (ok[0], ok[0])
            mine[key] = min(ok, key=lambda spot: reach[self.grid.cell_of(spot[0], spot[1])])
        if not mine:
            return chosen, unseen
        out = []
        for x, y, heading, keys in chosen:
            rest = [k for k in keys if k not in mine]
            if rest:
                out.append((x, y, heading, rest))
        for key, (x, y, heading) in sorted(mine.items()):
            out.append((x, y, heading, [key]))
        self.log.append("벽 쪽 통로의 면 %d개 → 통로 가운데에서 통로 방향 그대로 찍는다: %s" % (len(mine), " ".join("%d%s" % k for k in sorted(mine))))
        return out, set(unseen) - set(mine)

    def next_face(self):
        """지금 가고 있는 자리에서 처음 찍을 면 (없으면 None) — mission 이 가까워지면 팬을 그쪽으로 미리 돌린다."""
        if self.i >= len(self.stops):
            return None
        remaining = self.book.remaining()
        for key in (self.queue or self.stops[self.i][3]):
            if key in remaining and key not in self.book.voted:
                return self.by_key.get(key)
        return None

    def _retry_before_leaving(self, pose):
        """다음 자리로 떠나기 전에: 돌던 구역에 못 적은 면이 남았고 앞으로 그 구역의 자리가 더 없으면, 그 면들의 자리를 지금 끼워 넣는다.

        넣었으면 True (구역마다 한 번). 다시 잡는 자리는 「다시 도는 면」 의 조건(더 가까이 · 더 정면)이고, 벽 쪽 통로의 면은 통로의 자리다
        (어느 쪽에서 들어갈지는 그 자리로 떠날 때 로봇이 있는 쪽으로 정한다).
        """
        if not self.p.retry_in_sector or not self._sectors_begun:
            return False
        # 이미 「본 것」 이 있는 면은 여기서 다시 끼워 넣지 않는다 (defer_sighted — 맨 뒤로 미룬 자리에서 확인한다). 미룬 그 자리가 앞에 남아 있다고
        # 그 구역을 「아직 갈 자리가 있다」 로 치지도 않는다 — 그러면 같은 구역의 정말 모르는 면을 떠나기 전에 다시 보는 일이 미뤄진다
        remaining = {k for k in self.book.remaining() if not self._all_sighted([k])}
        ahead = {self.by_key[k].sector for stop in self.stops[self.i:] for k in stop[3] if k in remaining}
        waiting = {tag[1] for tag in self.out if len(tag) > 1}  # 찍어 놓고 답을 기다리는 면 — 답이 오면 적힐 수 있다
        for name in sorted(self._sectors_begun - self._sectors_retried - ahead, key=str):
            mine = {f.key for f in self.book.goal if f.sector == name and f.key in remaining}
            if mine & waiting:
                continue  # 아직 모른다 → 다음 자리로 넘어갈 때 다시 본다 (답은 0.1초쯤이면 온다)
            self._sectors_retried.add(name)
            # 사진을 한 번도 못 찍은 면만 (팬이 안 닿아 미룬 면 · 가지 못한 자리의 면). 찍었는데 못 읽은 면은 전처럼 다 돈 뒤에 다시 본다 —
            # 가상 경기에서 그것까지 바로 다시 보면 다 도는 데 19초가 더 걸리고 맞은 구역은 늘지 않았다
            left = {k for k in mine if self.tried.get(k, 0) == 0 and k[0] not in self.fmap.gone and not self._corridor_narrowed(k)}
            if not left:
                continue
            chosen, unseen, _rest = self._choose(pose, left, again=True)
            chosen, _unseen = self._corridor_stops(pose, left, chosen, unseen)
            if not chosen:
                continue
            self.stops[self.i:self.i] = self._order(pose, chosen, lead=False)
            self._pins |= {tuple(stop[3]) for stop in chosen}
            self.log.append("구역 %s 에 못 적은 면 %d개 (%s) → 떠나기 전에 다시 본다" % (name, len(left), " ".join("%d%s" % k for k in sorted(left))))
            return True
        return False

    def _corridor_side(self, pose, stop):
        """통로의 면을 찍는 자리로 떠날 때: 로봇이 오는 쪽에 맞는 자리로 바꾼다 (면의 앞쪽에서 들어오면 앞에, 뒤쪽에서 오면 뒤에 선다)."""
        x, y, heading, keys = stop
        if len(keys) != 1 or keys[0] not in self.corridor:
            return stop
        face, (plus, minus) = self.by_key[keys[0]], self.corridor[keys[0]]
        along = (pose[0] - face.cx) if abs(face.ny) > 0.5 else (pose[1] - face.cy)
        spot = plus if along < 0.0 else minus
        if keys[0] in self.nearby:  # 한쪽 자리에서 못 적어 다시 보는 면 → 통로의 반대편 자리에서 본다 (retry_nearby)
            spot = minus if spot is plus else plus
        return (spot[0], spot[1], spot[2], keys)

    def _choose(self, pose, want, again=False):
        """want 의 면(key 집합)을 가까이서 볼 자리들을 고른다 — 면을 가장 많이 덮는 자리부터 (같으면 pose 에서 가까운 자리).

        again = 다시 도는 면: 더 가까이 · 더 정면에서 (옆 그림과 헷갈려 못 적은 것이 대부분이다).
        돌려주는 값: (고른 자리 [(x, y, 방향, [면 key])] — 순서는 아직 정하지 않은 것, 볼 자리가 아예 없는 면, 자리는 있으나 갈 수 없어 남은 면).
        """
        p = self.p
        faces = [self.by_key[k] for k in want]
        # 한 자리에서 차체를 돌리지 않고 팬으로 닿는 면들만 그 자리의 몫으로 친다 (half_fov = 팬이 닿는 범위)
        if not again:
            view = ViewParams(range_m=p.close_range_m, max_incidence_deg=p.close_incidence_deg, half_fov_deg=p.pan_reach_deg)
        else:
            view = ViewParams(range_m=0.85, max_incidence_deg=35.0, half_fov_deg=p.pan_reach_deg, min_range_m=0.4)
        loose = ViewParams(range_m=1.6, max_incidence_deg=60.0, half_fov_deg=p.pan_reach_deg)
        if getattr(p, "roomy_first", False):
            cover = CoveragePlanner(self.fmap, self.grid, faces, view, tight=False)
            steps = ((loose, False), (view, True), (loose, True))
        else:  # 전의 순서: 가까이 · 정면 조건(붙은 칸까지) → 조금 멀리 · 비스듬히(붙은 칸까지)
            cover = CoveragePlanner(self.fmap, self.grid, faces, view, fine=False)
            steps = ((loose, True),)
        for params, tight in steps:
            if not cover.unreachable:
                break
            more = CoveragePlanner(self.fmap, self.grid, [self.by_key[k] for k in cover.unreachable], params, tight=tight, fine=getattr(p, "roomy_first", False))
            cover.views += more.views
            cover.unreachable = more.unreachable
        reach = self.grid.distances((pose[0], pose[1]))
        spots = [(x, y, h, keys) for x, y, h, keys in cover.views if self.grid.cell_of(x, y) in reach]
        left = want - cover.unreachable
        chosen = []
        while left and spots:  # 면을 가장 많이 덮는 자리부터 (같으면 가까운 자리)
            best = max(spots, key=lambda s: (len(s[3] & left), -reach[self.grid.cell_of(s[0], s[1])]))
            gain = best[3] & left
            if not gain:
                break
            chosen.append((best[0], best[1], best[2], sorted(gain)))
            left -= gain
        return chosen, set(cover.unreachable), left

    # ---- 뒤에 생긴 면 (지도에 없던 상자 — mission 이 navigation 의 라이다 상자 목록에서 받아들인 것)
    def add_faces(self, pose, keys):
        """뒤에 생긴 면을 찍을 자리를 골라 계획의 맨 뒤에 붙인다. 이미 정한 자리와 그 순서는 건드리지 않는다. 붙인 자리 수를 돌려준다.

        덤이다: 계획한 면을 다 찍은 뒤에 간다 (서 있는 다른 로봇을 상자로 잘못 본 것이어도 맨 뒤에서 시간을 조금 쓸 뿐이다).
        면은 먼저 면 기록에 들어 있어야 한다 (FaceBook.add_faces — 목표에 든 면만 찍으러 간다).
        아직 처음 계획을 세우기 전이면 적어만 두고, 처음 계획(plan) 때 맨 뒤에 붙인다. 이미 적은 면 · 갈 자리가 이미 남아 있는 면은 다시 넣지 않는다.
        """
        keys = {k for k in keys if k in self.by_key}
        self.late |= keys
        if not self._planned:
            return 0
        waiting = {k for stop in self.stops[self.i:] for k in stop[3]}
        return self._append(pose, (keys & self.book.remaining()) - waiting)

    def _append(self, pose, want):
        """want 의 면을 볼 자리를 골라 self.stops 의 맨 뒤에 붙인다. 순서는 지금 계획의 마지막 자리(없으면 pose)에서 가까운 것부터 (곧은 거리).

        붙는 자리는 상자 하나 둘레의 몇 곳뿐이라 길 계산으로 순서를 정하지 않는다 (_order 는 자리마다 격자 전체의 거리를 다시 계산한다 —
        경기 도중의 한 틱에 하기에는 길다. 이 PC 에서 자리 셋에 약 90 ms).
        """
        if not want:
            return 0
        tail = self.stops[-1][:2] if self.i < len(self.stops) else (pose[0], pose[1])
        chosen, unseen, left = self._choose(tail, want)
        self.unreachable |= unseen
        more, cur = [], tail
        while chosen:
            nxt = min(chosen, key=lambda s: math.hypot(s[0] - cur[0], s[1] - cur[1]))
            chosen.remove(nxt)
            more.append(nxt)
            cur = nxt[:2]
        self.stops += more
        self.log.append("뒤에 생긴 면 %d개: 설 자리 %d곳을 계획의 맨 뒤에 붙임 (설 자리가 없는 면 %d개)" % (len(want), len(more), len(left) + len(unseen)))
        return len(more)

    def drop_faces(self, keys):
        """아직 가지 않은 자리들에서 이 면들을 뺀다 (라이다로만 찾았던 상자가 다시 안 보일 때). 자리 자체는 두고 — 찍을 면이 남지 않은 자리는 지나친다."""
        keys = set(keys)
        for n in range(self.i, len(self.stops)):
            x, y, heading, old = self.stops[n]
            if keys & set(old):
                self.stops[n] = (x, y, heading, [k for k in old if k not in keys])
        self.queue = [k for k in self.queue if k not in keys]

    def prefer(self, sectors):
        """먼저 돌 구역을 정한다 (임무 명령 M1 의 탐색 구역). 지정표적은 상대 진영의 구역에 있으므로(CM 에는 없다) CM 은 뺀다.

        2026-11-19 14:52 실차: 명령의 탐색 구역이 CM · R-S1 이었고 지정표적은 R-S1 에 있었다. 순서를 점수만으로 정하면 R-S1 이 맨 뒤로 밀려
        (출발 칸에서 가장 멀다) 표적을 찾지 못한 채 끝날 수 있다 — M2 는 15점으로 구역 둘보다 크다 → 그 구역을 먼저 돈다.
        계획이 이미 서 있으면 지금 자리는 그대로 마치고, 다음 자리로 넘어갈 때 남은 자리의 순서를 다시 정한다.
        """
        first = tuple(sorted({str(name) for name in (sectors or ()) if name and str(name) != "CM"}))
        if first != self.first:
            self.first = first
            self._reorder = self._planned
            self.log.append("먼저 돌 구역: %s" % (", ".join(first) or "없음"))

    def _order(self, pose, stops, lead=True):
        """자리들을 잇는 순서: 구역 하나씩(가까운 구역부터), 구역 안에서는 가까운 것부터 잡고 엇갈린 길을 푼다."""
        if not stops:
            return []
        pts = [(pose[0], pose[1])] + [(s[0], s[1]) for s in stops]
        n = len(pts)
        dist = []
        live = self.p.order == "live"
        for i in range(n):
            field_ = self.grid.distances(pts[i])
            dist.append([field_.get(self.grid.cell_of(*pts[j]), 1e6) for j in range(n)])
            if live and i:
                self._keep_field(pts[i], field_)  # 경기 도중에 다음 자리를 다시 고를 때 쓴다 (_live_pick)

        def sector_of(stop):
            names = [self.by_key[k].sector for k in stop[3]]
            return max(sorted(set(names), key=str), key=names.count)  # 같은 수면 이름 순 (집합의 순서는 실행마다 달라진다)

        mode = self.p.order or ("near" if self.p.sector_first else "tour")
        if live:
            mode = "tour"  # 처음 순서는 가장 짧은 길이다

        def tour(start, members):
            """그 무리의 자리들을 start 에서부터 도는 순서 (가까운 것부터 잡고 엇갈린 길을 푼다)."""
            todo, part = list(members), []
            while todo:
                nxt = min(todo, key=lambda j: (dist[part[-1] if part else start][j], j))
                part.append(nxt)
                todo.remove(nxt)
            return self._untangle(start, part, dist)

        if mode == "best":
            best = self._best_order(stops, dist, tour, lead)
            if best is not None:
                return [stops[j - 1] for j in best]
            mode = "tour"
        if mode == "tour":
            # 구역을 따지지 않는 가장 짧은 길. 먼저 돌 구역(prefer — 지정표적이 있는 곳)의 면이 있는 자리들만 맨 앞에 둔다
            ahead = [idx for idx, s in enumerate(stops, start=1) if lead and {self.by_key[k].sector for k in s[3]} & set(self.first)]
            head = tour(0, ahead) if ahead else []
            rest = [idx for idx in range(1, n) if idx not in head]
            return [stops[j - 1] for j in head + (tour(head[-1] if head else 0, rest) if rest else [])]
        groups = {}
        for idx, s in enumerate(stops, start=1):
            groups.setdefault(sector_of(s), []).append(idx)
        order, cur = [], 0
        while groups:
            name = min(sorted(groups, key=str), key=lambda g: min(dist[cur][j] for j in groups[g]))
            part = tour(cur, groups.pop(name))
            order += part
            cur = part[-1]
        return [stops[j - 1] for j in order]

    def _best_order(self, stops, dist, tour, lead=True):
        """자리들을 도는 순서 가운데 「정찰하는 동안 얻을 것으로 보이는 M3 점수」 가 가장 큰 것을 찾는다. 자리가 둘이 안 되면 None.

        M3 는 구역 단위로, 그 구역의 종류 · 개수가 모두 맞아야 점수다 (상대 구역 7 · CM 9). 못 본 면이 있으면 그 면이 객체가 아닐 때만 맞는다
        → 어느 때의 값 = Σ 구역 점수 × UNSEEN_OK ^ (그 구역의 못 본 면 수). 경기가 언제 끊길지 모르므로(막힘 · 사격 · 거점 앞 대기)
        plan_budget_s 의 0.5 ~ 1.5배의 여러 때의 값을 고루 본다. 시간은 어림한다 (자리 plan_stop_s + 거리 / plan_speed + 면마다 plan_face_s).

        찾는 법: ① 구역을 따지지 않는 가장 짧은 길과 ② 구역을 하나씩 끝내는 차례들(구역이 넷이면 24가지)에서 출발해, 자리 하나를 다른 곳으로
        옮기거나 한 토막을 뒤집어 값이 커지면 받아들이기를 되풀이한다. 가장 짧은 길은 면을 빨리 모으지만 구역이 끝에 가서야 끝나고,
        구역을 하나씩 끝내면 길이 길어진다 — 2026-11-19 가상 경기(오차를 넣은 12판)를 시한을 두고 끊어 보니 시한 110초에서 맞은 구역이
        가장 짧은 길 3.25 · 구역 하나씩 2.67 이었다. 둘 사이의 가장 좋은 순서를 값으로 고른다.
        먼저 돌 구역(prefer)의 면이 있는 자리들은 맨 앞에 둔다 (지정표적이 거기 있다 — M2 15점).
        """
        p = self.p
        n = len(stops)
        if n < 2:
            return None
        left_faces = self.book.remaining()
        need = {}
        for face in self.book.goal:
            if face.key in left_faces:
                need.setdefault(face.sector, set()).add(face.key)
        keys = [set(stop[3]) for stop in stops]
        in_stop = [{self.by_key[k].sector for k in stop[3]} for stop in stops]
        names = sorted({name for names_ in in_stop for name in names_}, key=str)
        cost = [p.plan_stop_s + p.plan_face_s * len(stop[3]) for stop in stops]
        horizons = []  # 값을 보는 때들 — 아래에서 가장 짧은 길로 다 도는 시간에 견줘 정한다
        points = {name: SECTOR_POINTS.get(name, 7.0) for name in need}
        ahead = [j for j in range(1, n + 1) if lead and in_stop[j - 1] & set(self.first)]  # 먼저 돌 구역의 자리들

        def value(seq):
            cur, t, seen, total, h = 0, 0.0, set(), 0.0, 0
            unseen = {name: len(ks) for name, ks in need.items()}
            now = sum(points[name] * UNSEEN_OK ** unseen[name] for name in need)
            for j in seq:
                t += cost[j - 1] + dist[cur][j] / p.plan_speed
                while h < len(horizons) and horizons[h][0] < t:
                    total += horizons[h][1] * now
                    h += 1
                cur = j
                new = keys[j - 1] - seen
                if new:
                    seen |= new
                    for k in new:
                        name = self.by_key[k].sector
                        if name in unseen and k in need[name]:
                            unseen[name] -= 1
                    now = sum(points[name] * UNSEEN_OK ** unseen[name] for name in need)
            while h < len(horizons):
                total += horizons[h][1] * now
                h += 1
            return total, -t

        def improve(seq, fixed):
            """fixed 개의 앞 자리는 그대로 두고, 자리 하나 옮기기 · 토막 뒤집기로 값을 키운다."""
            best = value(seq)
            for _ in range(6):
                moved = False
                for i in range(fixed, len(seq)):
                    for j in range(fixed, len(seq)):
                        if i == j:
                            continue
                        cand = seq[:i] + seq[i + 1:]
                        cand.insert(j, seq[i])
                        got = value(cand)
                        if got > best:
                            seq, best, moved = cand, got, True
                for i in range(fixed, len(seq) - 1):
                    for j in range(i + 1, len(seq)):
                        cand = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                        got = value(cand)
                        if got > best:
                            seq, best, moved = cand, got, True
                if not moved:
                    break
            return seq, best

        head = tour(0, ahead) if ahead else []
        rest = [j for j in range(1, n + 1) if j not in head]
        start = head[-1] if head else 0
        seeds = [head + tour(start, rest)]  # ① 가장 짧은 길
        whole, cur = 0.0, 0
        for j in seeds[0]:
            whole += cost[j - 1] + dist[cur][j] / p.plan_speed
            cur = j
        # 값을 보는 때: 가장 짧은 길로 다 도는 시간(whole)의 plan_horizons 배. 다 돌 시간이 넉넉하면 짧은 길이 이기고(늦은 때의 값이 같아 걸린 시간으로
        # 가른다), 빠듯하면 구역을 일찍 끝내는 순서가 이긴다. 정찰에 쓸 수 있는 시간(plan_budget_s)이 whole 보다 짧으면 그만큼 앞당겨 본다
        scale = min(1.0, p.plan_budget_s / whole) if whole > 0.0 else 1.0
        horizons[:] = [(f * whole * scale, w) for f, w in p.plan_horizons]
        if 1 < len(names) <= 5:
            import itertools

            for perm in itertools.permutations(names):  # ② 구역을 하나씩
                seq, cur, left = list(head), start, set(rest)
                for name in perm:
                    part = tour(cur, [j for j in sorted(left) if name in in_stop[j - 1]])
                    if part:
                        seq += part
                        left -= set(part)
                        cur = part[-1]
                seq += tour(cur, sorted(left)) if left else []
                seeds.append(seq)
        ranked = sorted({tuple(seq) for seq in seeds}, key=lambda seq: value(list(seq)), reverse=True)
        best = None
        for seq in ranked[:3]:
            got = improve(list(seq), len(head))
            if best is None or got[1] > best[1]:
                best = got
        return best[0]

    @staticmethod
    def _untangle(first, part, dist):
        """2-opt: 길이 서로 엇갈린 곳을 뒤집어 푼다 (first 에서 출발하는 열린 길)."""
        def length(seq):
            return sum(dist[a][b] for a, b in zip([first] + seq[:-1], seq))

        improved = True
        while improved:
            improved = False
            for i in range(len(part) - 1):
                for j in range(i + 1, len(part)):
                    new = part[:i] + part[i:j + 1][::-1] + part[j + 1:]
                    if length(new) < length(part) - 1e-9:
                        part, improved = new, True
        return part

    # ---- 한 틱
    def step(self, pose, now):
        if not self._planned:
            self.plan(self._start or pose)
        for tag, at in list(self.out.items()):  # 너무 늦는 답은 버린다
            if now - at > self.p.answer_wait_s * 2.0:
                del self.out[tag]
        live_left = 12  # 한 틱에 실시간 순서를 다시 고르는 횟수의 상한 (그 뒤로는 그 틱에서 목록의 차례대로 간다 — 아래 되풀이가 길어지는 일이 없게)
        while True:
            if self.i >= len(self.stops):
                if self.out:
                    return ("wait",)
                left = {k for k in self.remaining() if self.tried.get(k, 0) <= self.p.retries and k[0] not in self.fmap.gone and not self._corridor_narrowed(k)}
                if not left or self.passes >= 2:
                    return ("done",)
                self.passes += 1  # 답을 못 얻은 면만 다시 돈다
                self.plan(pose, only=left)
                self.log.append("못 적은 면 %d개 → 다시 돈다" % len(left))
                continue
            self._hint_followup(pose)
            if self._lead is not None and not self.any_heading:
                self._take_lead()
            if self._reorder_at is not None and self.i >= self._reorder_at and self.body is None:
                self._reorder_at, self._reorder = None, True  # 당겨 온 자리를 마쳤다 → 남은 자리는 지금 선 곳에서부터 다시 잇는다
            if self._reorder and self.body is None:
                # 먼저 돌 구역이 계획 뒤에 정해졌다 → 남은 자리의 순서를 다시 정한다 (뒤로 미뤄 둔 자리는 그대로 맨 뒤에 둔다)
                self._reorder = False
                rest = self.stops[self.i:]
                late = [stop for stop in rest if (round(stop[0], 2), round(stop[1], 2)) in self.deferred]
                self.stops = self.stops[:self.i] + self._order(pose, [stop for stop in rest if stop not in late]) + late
            if self.body is None and self._retry_before_leaving(pose):
                continue
            if self.p.order == "live" and live_left > 0:
                live_left -= 1
                if self.body is None:
                    self._live(pose, now)  # 다음 자리를 지금 사정으로 다시 고른다 (남은 자리의 순서만 바뀐다)
                elif (not self.any_heading and not self._live_switched and self._live_due(now)
                      and math.hypot(pose[0] - self.stops[self.i][0], pose[1] - self.stops[self.i][1]) > 0.5 and self._live(pose, now, moving=True)):
                    # 가는 도중에 더 나은 자리가 생겼다 → 그리로 간다. 한 번 떠나서는 한 번만 바꾼다 (라이다 상자가 나왔다 들어갔다 할 때 두 자리 사이를 오가지 않게)
                    self.queue, self.body, self._live_switched = [], None, True
                    self.turned = set()
            x, y, heading, keys = self.stops[self.i]
            todo = [k for k in keys if k in self.book.remaining()]
            if not todo:
                self._leave_stop()
                continue
            if self.body is None:  # 이 자리에 처음 왔다
                absent = [k for k in todo if k[0] in self.fmap.gone]
                if absent and self._defer((x, y, heading, absent)):
                    # navigation 이 라이다로 「없다」 고 본 상자의 면 → 버리지 않고 맨 뒤로 미룬다 (잘못 뺀 것이면 다른 면을 다 찍은 뒤에 가 본다).
                    # 2026-11-18 · 19 실차 사진 33장 중 5장이 이미 없어진 상자를 겨눈 것이었고, 틀리게 적힌 한 장도 거기서 나왔다
                    self.log.append("자리 (%.2f, %.2f) 의 면 %d개는 라이다로 없다고 본 상자의 것 → 맨 뒤로 미룬다" % (x, y, len(absent)))
                    todo = [k for k in todo if k not in absent]
                    if not todo:
                        self.i += 1
                        continue
                if self._all_sighted(todo) and self._defer(self.stops[self.i]):
                    # 이 자리에서 찍을 면을 모두 이미 다른 사진으로 한 번씩 봤다 (다른 답 없음) → 모르는 면부터 보러 가고, 여기는 맨 뒤에 확인하러 온다.
                    # 이미 미뤘던 자리면(_defer 가 False) 이번에는 간다
                    self.log.append("자리 (%.2f, %.2f) 의 면 %d개는 이미 본 것 (%s) → 맨 뒤로 미룬다" % (
                        x, y, len(todo), " ".join("%d%s=%s" % (k[0], k[1], self.book.sighted(k)) for k in todo)))
                    self.i += 1
                    continue
                if self._near_block(x, y, now):
                    if self._defer(self.stops[self.i]):
                        self.log.append("자리 (%.2f, %.2f) 는 방금 막힌 곳 가까이 → 뒤로 미룬다" % (x, y))
                        self.i += 1
                        continue
                    if self._clear_stop_ahead(now):
                        # 이미 한 번 미뤘던 자리인데 막힌 지 avoid_s 가 지나지 않았다. 막히지 않은 자리가 남아 있으면 그것부터 간다 (그 뒤에 다시 본다).
                        # navigation: 「BLOCKED [OBSTACLE] 은 돌아갈 길까지 다 찾아본 뒤의 답이라, 같은 목표를 바로 다시 보내면 같은 답이 온다」
                        # (docs/주행_막힘_우회_2026-11-19.md). 모의 6차: 같은 자리에 120 · 125 · 139초에 세 번 갔고 세 번 다 막혔다
                        self.stops.append(self.stops[self.i])
                        self.log.append("자리 (%.2f, %.2f) 는 아직 막힌 곳 가까이 → 막히지 않은 자리부터 간다" % (x, y))
                        self.i += 1
                        continue
                if len(keys) == 1 and self._corridor_narrowed(keys[0]):
                    # 벽 쪽 통로가 지도보다 좁다 (라이다 상자가 통로 안으로 나와 있다) → 들어가지 않는다. 한 번은 맨 뒤로 미루고(그 사이 목록에서
                    # 사라질 수 있다 — 지나가던 로봇), 미룬 뒤에도 그대로면 이 자리는 지나친다 (그 면은 못 적은 면으로 남는다)
                    again = self._defer(self.stops[self.i])
                    self.log.append("자리 (%.2f, %.2f): 통로 안으로 라이다 상자가 나와 있다 (지도보다 좁다) → %s" % (x, y, "뒤로 미룬다" if again else "가지 않는다"))
                    self.i += 1
                    continue
                self.queue = list(todo)
                self._gone_known = frozenset(k[0] for k in todo if k[0] in self.fmap.gone)  # 빠진 줄 알면서 가 보는 상자 (미뤘다가 한 번 가 보는 자리)
                self._sectors_begun |= {self.by_key[k].sector for k in todo}
                if len(keys) == 1 and keys[0] in self.corridor:
                    # 벽 쪽 통로의 면: 오는 쪽에 맞는 자리로 바꾸고, 차체는 통로 방향 그대로 둔다 (팬만 옆으로 돌려 찍는다)
                    self.stops[self.i] = self._corridor_side(pose, self.stops[self.i])
                    x, y, heading, keys = self.stops[self.i]
                    self.body, self.body_tol = heading, math.radians(self.p.corridor_yaw_tol_deg)
                else:
                    self.body, self.body_tol = self._body_for(x, y, heading, todo)
            else:
                # 이 자리로 가는 동안(또는 서서 찍는 동안) navigation 이 그 상자를 지도에서 뺐다 → 가서 찍어도 적을 수 없다 (사진 짝짓기는 뺀 상자의
                # 면을 치지 않는다 — faces.expected_faces). 자리를 처음 잡을 때와 같게 그 면만 맨 뒤로 미루고, 찍을 면이 남지 않으면 바로 다음 자리로 간다.
                # 2026-11-19 18:08 모의 경기 6차: 151.7초에 떠난 아래 벽 통로의 자리 (3.96, 0.39) 는 상자 9 의 S 면 하나를 보는 자리였다. 156.2초에
                # navigation 이 상자 9 를 뺐는데도 끝까지 가서 164초에 찍었고(적히지 않았다), 그 통로에서 경기 끝(246초)까지 나오지 못했다
                lost = [k for k in self.queue if k[0] in self.fmap.gone and k[0] not in self._gone_known]
                if lost:
                    self.queue = [k for k in self.queue if k not in lost]
                    self._defer((x, y, heading, lost))  # 이미 미뤘던 자리면 다시 붙이지 않는다 (못 적은 면으로 남아, 상자가 되살아나면 다시 돌 때 본다)
                    self.log.append("자리 (%.2f, %.2f) 로 가는 동안 navigation 이 상자를 뺀 면 %d개 → 맨 뒤로 미룬다%s" % (
                        x, y, len(lost), "" if self.queue else " (이 자리는 지나친다)"))
                    if not self.queue:
                        self._leave_stop()
                        continue
            if not self._at(pose, x, y, self.body):
                return ("go", x, y, self.body, self.body_tol)
            self.any_heading = True  # 한 번 맞았으면 이 자리에서는 다시 따지지 않는다 (위치 · 방향 값이 흔들려 왔다 갔다 하지 않게)
            # 이 자리로 오는 동안 「본 것」 으로 적힌 면은 찍지 않는다 (다른 자리의 사진 두 장이 같은 답이었다). 덤으로 적힌 면은 전처럼 겨눠 찍어 확인한다 —
            # 덤은 사진 한 장의 답이고, 그 사진에 함께 찍히는 옆 면이 덤으로 적히는 몫도 크다 (가상 경기: 그것까지 건너뛰면 지정표적을 70초 늦게 찾았다)
            self.queue = [k for k in self.queue if k not in self.book.voted]
            for key in self.queue:  # 지금 방향에서 팬이 닿는 면부터 찍는다
                pan = self._aim(pose, self.by_key[key])
                if pan is not None:
                    if key != self.queue[0]:  # 다음 틱에도 같은 면을 겨누게 맨 앞에 둔다 (팬이 두 면 사이를 오가지 않게)
                        self.queue.remove(key)
                        self.queue.insert(0, key)
                    return ("shoot", pan, ("face", key, self.tried.get(key, 0)))
            # 남은 면은 모두 팬이 닿지 않는다 (선 자리 · 방향이 어긋났다). 아직 못 적은 면이면 그 면 쪽으로 차체를 한 번 돌려 보고(_turn_to),
            # 그래도 안 되면 전처럼 다시 돌 때로 미룬다. 그 사이 덤으로 이미 적힌 면은 돌려 보지 않고 넘어간다
            self.queue = [k for k in self.queue if k in self.book.remaining()]
            if self.queue and self._turn_to(pose):
                continue
            if self._answers_due(keys, now):
                return ("wait",)  # 이 자리에서 찍은 사진의 답을 보고 떠난다 (못 적었으면 바로 가까운 자리에서 다시 본다 — retry_nearby)
            again = self._retry_nearby(pose, keys)
            for key in self.queue:
                self.unaimed += 1
                self.log.append("면 %s%s: 선 방향에서 팬이 닿지 않음 → %s" % (key[0], key[1], "바로 가까운 자리에서 다시 본다" if key in again else "다시 돌 때로 미룬다"))
            self._leave_stop()

    # ---- 실시간 순서 (order = "live"): 남은 자리의 순서를 경기 도중에 다시 정한다. 자리를 고르고 찍는 나머지 일은 그대로다
    def _keep_field(self, xy, field_=None):
        """자리 xy 에서 모든 칸까지의 길 길이(Grid.distances)를 작게 접어 둔다 — 칸 번호 → m, 갈 수 없는 칸은 1e9 (자리 하나에 약 47 kB)."""
        if field_ is None:
            field_ = self.grid.distances(xy)
        if len(self._fields) > 80:
            self._fields.clear()
        ny = self.grid.ny
        flat = array("f", [1e9]) * (self.grid.nx * ny)
        for (i, j), d in field_.items():
            flat[i * ny + j] = d
        self._fields[self.grid.cell_of(xy[0], xy[1])] = flat

    def _dist(self, a, b):
        """a → b 의 길 길이 (m). 둘 중 하나가 거리장을 접어 둔 자리면 거기서 읽는다 (길은 어느 쪽으로 가나 거의 같다). 아니면 곧은 거리의 1.3배."""
        ny = self.grid.ny
        ca, cb = self.grid.cell_of(a[0], a[1]), self.grid.cell_of(b[0], b[1])
        for src, dst in ((cb, ca), (ca, cb)):
            flat = self._fields.get(src)
            if flat is not None and flat[dst[0] * ny + dst[1]] < 1e8:
                return flat[dst[0] * ny + dst[1]]
        return 1.3 * math.hypot(a[0] - b[0], a[1] - b[1])

    def _lidar_mark(self):
        """라이다 상자 · 위에서 본 서 있는 로봇의 자리를 0.2 m 눈금으로 줄인 것 (목록이 바뀌었는지 볼 때)."""
        try:
            return frozenset((round(float(q[0]) * 5.0), round(float(q[1]) * 5.0)) for q in list(self.lidar_boxes or ()) + list(self.robots or ()))
        except (TypeError, ValueError, IndexError):
            return frozenset()

    def _robots_now(self, pose):
        """막고 있을 만한 것들 [(x, y, 서 있은 시간 s 또는 None)]: 라이다 상자(서 있은 시간은 모른다)와 위에서 본 서 있는 로봇.
        같은 것(0.30 m 안)은 위에서 본 쪽으로 하나만 둔다. 우리 자리에서 0.15 m 안의 것만 뺀다 (우리 차 자신이다 — 두 차의 가운데가 그만큼 가까울 수는 없다).
        더 넓게 빼면 안 된다: 2026-11-20 00:00 연습 판에서 막혀 서 있던 우리 차와 막은 로봇의 가운데가 0.24 ~ 0.27 m 였다 (우리 차를 골라내는 일은 mission 이 한다)."""
        out = []
        try:
            seen = [(float(q[0]), float(q[1]), float(q[2])) for q in self.robots or ()]
        except (TypeError, ValueError, IndexError):
            seen = []
        try:
            boxes = [(float(q[0]), float(q[1]), None) for q in self.lidar_boxes or ()]
        except (TypeError, ValueError, IndexError):
            boxes = []
        # 지도에 이미 있는 상자(mission 이 받아들인 라이다 상자도 지도에 들어 있다)는 「서 있는 로봇」 으로 치지 않는다 — 라이다 목록에는 그 상자가 계속
        # 나오고, 그 상자의 면을 찍을 자리는 바로 그 곁이다 (00:52 연습 판: 상자 12 의 W 면을 찍을 자리가 상자 13 의 가운데에서 0.43 m)
        solid = [b for number, b in enumerate(self.fmap.boxes) if number not in self.fmap.gone]
        boxes = [q for q in boxes if all(math.hypot(q[0] - bx, q[1] - by) > 0.30 for bx, by in solid)]
        for q in seen + [b for b in boxes if all(math.hypot(b[0] - r[0], b[1] - r[1]) > 0.30 for r in seen)]:
            if math.hypot(q[0] - pose[0], q[1] - pose[1]) > 0.15:
                out.append(q)
        return out

    def _live_due(self, now):
        """가는 도중에 다시 고를 일이 생겼는가: 그 사이 면이 적혔거나(지나가며 찍은 사진) 라이다 상자가 바뀌었다. 1초에 한 번만 본다."""
        return now - self._live_at >= 1.0 and (len(self.book.remaining()), self._lidar_mark()) != self._live_seen

    def _live(self, pose, now, moving=False):
        """실시간 순서로 다음 자리를 고른다 (_live_pick). 맨 앞 자리가 바뀌었으면 True.
        무슨 일이 있어도 예외를 밖으로 내지 않는다 — 예외면 계획한 순서 그대로 가고, 세 번이면 이 경기에서는 더 쓰지 않는다."""
        if self._live_off:
            return False
        try:
            return self._live_pick(pose, now, moving)
        except Exception as error:  # noqa: BLE001
            self._live_errors += 1
            self._live_off = self._live_errors >= 3
            self.log.append("실시간 순서 예외 (%s: %s) → 계획한 순서대로 간다%s" % (type(error).__name__, error, " (이 경기에서는 더 쓰지 않는다)" if self._live_off else ""))
            return False

    def _live_pick(self, pose, now, moving=False):
        """남은 자리 가운데 「거점 앞 대기를 시작할 때의 기대 점수」 가 가장 큰 순서를 찾아 self.stops 의 남은 부분을 그 순서로 바꾼다. 맨 앞 자리가 바뀌었으면 True.

        기대 점수 = Σ 구역 점수 × Π (그 구역의 못 적은 면마다: 적힐 가능성 r + (1 − r) × UNSEEN_OK). r 은 그 면의 자리를 시한 안에 마치면
        live_hit × (1 − 막힐 가능성), 아니면 live_pass. 시한은 남은 시간의 live_scenarios 배 — 자리를 마친 뒤 대기 자리까지 가는 시간도 넣어 견준다.
        시간 어림: 길 / live_speed + live_stop_s + 면마다 live_face_s + 막힐 가능성 × live_block_s — 지난 길이 어림보다 더 걸렸으면 그 비율(_live_pace)을 곱한다.
        막힐 가능성: 라이다 상자나 위에서 본 서 있는 로봇(_robots_now)이 자리 곁에 있다 · 지금 선 곳에서 가는 곧은 선 위에 있다 · 경기 첫머리의 한가운데 자리.
        견주는 순서들: 지금 순서 · 가까운 자리부터 · 자리 하나를 맨 앞으로 당긴 것 · 구역을 하나씩 끝내는 차례들(구역 넷이면 24가지).
        고르는 법: 지금 순서보다 기대 점수가 live_margin 넘게 큰 순서가 있으면 그 가운데 가장 빨리 끝나는 것. 없으면 점수가 같은 순서들 가운데
        live_shorter_s 넘게 빨리 끝나는 것이 있을 때만 그것으로. 가는 도중에는 (가던 자리가 아직 고를 자리이면) 점수가 margin 의 두 배 넘게 커질 때만 바꾼다.
        고를 자리가 live_exact 곳 이하이고 시간이 넉넉하면(지금 순서로 다 도는 시간이 남은 시간의 live_slack 배 안) 가장 짧은 순서(_shortest)를
        — live_all 곳 이하면 모든 순서를 — 견줄 것에 더하고, 점수가 같은(live_tie 안) 순서로는 live_easy_s 만 빨라도 바꾼다.
        고르지 않는 자리 (전과 같이 뒤에 남는다): 찍을 면이 없는 자리 · 사진에 찍힌 적 없는 「뒤에 생긴 상자」 의 면만 남은 자리 · 라이다로 없다고 본 상자의 면만 남은 자리 ·
        통로가 좁아진 자리 · 방금 막힌 곳 가까이의 자리(_near_block). 고를 자리가 하나도 없거나, 꼭 지금 갈 자리(_pins)가 맨 앞이면 순서를 건드리지 않는다.
        지정표적을 아직 못 찾았으면 먼저 돌 구역(prefer)의 자리들 가운데서만 고른다 ("tour" 와 같다).
        """
        p = self.p
        rest = self.stops[self.i:]
        left = self.book.remaining()
        self._live_at, self._live_seen = now, (len(left), self._lidar_mark())
        if not moving:
            # 방금 마친 길이 어림보다 얼마나 더 걸렸는지로 앞으로의 시간 어림을 고친다 (막힘 · 돌아가기가 잦은 판에서는 남은 시간을 짜게 본다).
            # 그 사이 지정표적을 찾았으면(사격으로 서 있었다) 그 길은 치지 않는다
            leg, self._live_leg = self._live_leg, None
            if leg is not None and now > leg[0] and leg[2] == (self.want is not None):
                self._live_pace = 0.6 * self._live_pace + 0.4 * min(3.0, max(0.5, (now - leg[0]) / max(leg[1], 3.0)))
        if not rest or tuple(rest[0][3]) in self._pins:
            return False
        gone = self.fmap.gone
        todo = [[k for k in stop[3] if k in left and k[0] not in gone] for stop in rest]
        # 뒤에 생긴 면 가운데 아직 견주지 않을 것: 사진에 한 번도 찍힌 적이 없는 상자의 면 (live_late 의 설명)
        proven = {k[0] for k in self.late if k in self.book.votes or self.book.sights.get(k)} if p.live_late else set()
        unsure = {k for k in self.late if k[0] not in proven}
        cand = [n for n, stop in enumerate(rest) if todo[n] and not all(k in unsure for k in todo[n])
                and not (len(stop[3]) == 1 and self._corridor_narrowed(stop[3][0]))]
        cand = [n for n in cand if not self._near_block(rest[n][0], rest[n][1], now)]
        if not cand:
            # 고를 자리가 모두 방금 막힌 곳 가까이다 → 순서를 건드리지 않는다. 그런 자리는 step 이 차례대로 뒤로 보내거나(막히지 않은 자리가 남아 있으면)
            # 그대로 간다. 2026-11-20 가상 경기에서 찾은 것: 여기서 그 가운데 하나를 맨 앞으로 당기면, step 이 「뒤에 막히지 않은 자리가 있다」
            # (뒤에 생긴 면의 자리 — 여기서는 고르지 않는 자리다) 며 그것을 다시 맨 뒤에 붙이고, 다음 되풀이에서 또 당겨 와 한 틱에 수십 번을 돌았다.
            # 붙인 것도 고를 자리라 자리 목록이 결정 때마다 곱절로 불었다 (한 판에 순서 바꿈 8269번 · 틱 하나에 0.1초 넘게)
            return False
        sector_of = {n: {self.by_key[k].sector for k in todo[n]} for n in cand}
        few = len(cand) <= p.live_exact  # 남은 일 전체가 몇 곳뿐인가 (아래에서 먼저 돌 구역의 자리만 추리기 전의 수로 본다)
        if self.want is not None and self.first:
            cand = [n for n in cand if sector_of[n] & set(self.first)] or cand
        if not cand:
            return False
        # 남은 시간 · 대기 자리
        if self.clock is not None and self.clock[0] is not None:
            time_left, stage, centre_s = self.clock[1] - self.clock[0], self.clock[2], p.live_centre[1] - self.clock[0]  # centre_s = 한가운데를 피할 남은 시간
        else:
            self._live_t0 = now if self._live_t0 is None else self._live_t0
            time_left, stage, centre_s = p.plan_budget_s - (now - self._live_t0), None, 0.0
        time_left = max(1.0, time_left - (p.live_strike_s if self.want is not None else 0.0))
        lack = [xy for xy in ([stage] if stage is not None else []) + [rest[n][:2] for n in cand] if self.grid.cell_of(xy[0], xy[1]) not in self._fields]
        if lack:
            self._keep_field(lack[0])  # 한 번 고를 때 거리장은 하나만 새로 만든다 (이 PC 에서 약 20 ms)
        robots = self._robots_now(pose)
        cx, cy = self.fmap.width / 2.0, self.fmap.height / 2.0
        here, eta, robot, half, centre, line = {}, {}, {}, {}, {}, {}
        for n in cand:
            x, y = rest[n][0], rest[n][1]
            here[n] = self._dist((pose[0], pose[1]), (x, y))
            beside = [stood for bx, by, stood in robots if math.hypot(x - bx, y - by) <= p.live_robot_m]
            robot[n] = p.live_robot_p if beside else 0.0
            half[n] = max([p.live_robot_half_s] + [stood for stood in beside if stood is not None])  # 오래 서 있던 로봇일수록 늦게 떠난다
            centre[n] = math.hypot(x - cx, y - cy) <= p.live_centre[0]
            eta[n] = self._dist((x, y), stage) / p.plan_speed + 3.0 if stage is not None else 0.0
            # 지금 선 곳에서 그 자리까지의 곧은 선 가까이에 서 있는 로봇이 있는가 (그 자리로 바로 갈 때만 친다)
            ux, uy = x - pose[0], y - pose[1]
            span = ux * ux + uy * uy
            near = False
            for bx, by, _stood in robots:
                k = max(0.0, min(1.0, ((bx - pose[0]) * ux + (by - pose[1]) * uy) / span)) if span > 1e-6 else 0.0
                near = near or math.hypot(pose[0] + k * ux - bx, pose[1] + k * uy - by) <= p.live_line_m
            line[n] = near
        need = {}
        for face in self.book.goal:
            if face.key in left and face.key not in unsure and face.box not in gone:
                need.setdefault(face.sector, []).append(face.key)
        points = {name: SECTOR_POINTS.get(name, 7.0) for name in need}
        between = {}

        def d(a, b):  # 자리 a → b 의 길 (a 가 None 이면 지금 선 곳에서)
            if a is None:
                return here[b]
            key = (a, b) if a < b else (b, a)
            if key not in between:
                between[key] = self._dist(rest[a][:2], rest[b][:2])
            return between[key]

        def chain(cur, members):  # 가까운 자리부터 잇는다
            members, out = list(members), []
            while members:
                nxt = min(members, key=lambda n: (d(cur, n), n))
                members.remove(nxt)
                out.append(nxt)
                cur = nxt
            return out

        def value(seq):
            t, cur, when = 0.0, None, {}
            for n in seq:
                new = [k for k in todo[n] if k not in when]
                t += d(cur, n) / p.live_speed * self._live_pace  # 그 자리에 닿는 때
                r = robot[n] * 0.5 ** (t / half[n])
                if centre[n] and t < centre_s:
                    r = max(r, p.live_centre[2])
                if cur is None and line[n]:
                    r = max(r, p.live_line_p)
                t += (p.live_stop_s + p.live_face_s * len(new)) * self._live_pace + r * p.live_block_s
                for k in new:
                    when[k] = (t + eta[n], p.live_hit * (1.0 - r))
                cur = n
            total = 0.0
            for scale, weight in p.live_scenarios:
                limit = time_left * scale
                for name, keys in need.items():
                    ok = points[name]
                    for k in keys:
                        got = when.get(k)
                        r = got[1] if got is not None and got[0] <= limit else p.live_pass
                        ok *= r + (1.0 - r) * UNSEEN_OK
                    total += weight * ok
            return total, t

        seqs = [list(cand), chain(None, cand)] + [[n] + [m for m in cand if m != n] for n in cand[1:]]
        names = sorted({name for n in cand for name in sector_of[n]}, key=str)
        if 1 < len(names) <= 4:
            for perm in itertools.permutations(names):
                seq, pool = [], list(cand)
                for name in perm:
                    part = chain(seq[-1] if seq else None, [n for n in pool if name in sector_of[n]])
                    seq += part
                    pool = [n for n in pool if n not in part]
                seqs.append(seq + chain(seq[-1] if seq else None, pool))
        now_value, now_time = value(seqs[0])
        if few and len(cand) > 2 and now_time <= p.live_slack * time_left:
            # 몇 곳 안 남았고 지금 순서로 다 돌고도 시간이 넉넉하다 → 가장 빨리 끝나는 순서를 꼼꼼히 찾는다 (live_slack 의 설명)
            seqs += [list(seq) for seq in itertools.permutations(cand)] if len(cand) <= p.live_all else [self._shortest(cand, d)]
            scored = [value(seq) + (seq,) for seq in seqs]
            top = max(got for got, _t, _seq in scored)
            going = moving and cand[0] == 0  # 가는 도중이고, 가던 자리가 아직 고를 자리다 (그 자리의 면이 그 사이 다 적혔으면 자리를 마친 것과 같게 고른다)
            if top > now_value + p.live_margin * (2.0 if going else 1.0):
                best_value, best_time, best = min((q for q in scored if q[0] >= top - p.live_tie), key=lambda q: q[1])  # 점수가 뚜렷이 큰 순서들 가운데 가장 빠른 것
            else:
                best_value, best_time, best = min((q for q in scored if q[0] >= now_value - p.live_tie), key=lambda q: q[1])  # 점수가 같은 순서들 가운데 가장 빠른 것
                if going or best_time >= now_time - p.live_easy_s:
                    best, best_value = seqs[0], now_value
        else:  # 그 밖에는 전과 똑같이 고른다
            best, best_value, best_time = seqs[0], now_value, now_time
            for seq in seqs[1:]:
                got, t = value(seq)
                if (got, -t) > (best_value, -best_time):
                    best, best_value, best_time = seq, got, t
            better = best_value > now_value + p.live_margin * (2.0 if moving else 1.0)
            if not better and not (not moving and best_value >= now_value - 1e-6 and best_time < now_time - p.live_shorter_s):
                best, best_value = seqs[0], now_value
        order = best + [n for n in range(len(rest)) if n not in best]
        if not moving:
            self._live_leg = (now, here[best[0]] / p.live_speed + p.live_stop_s + p.live_face_s * len(todo[best[0]]), self.want is not None)
        if order == list(range(len(rest))):
            return False
        self.stops[self.i:] = [rest[n] for n in order]
        if order[0] == 0:
            return False
        x, y = rest[order[0]][0], rest[order[0]][1]
        self.log.append("실시간 순서: 다음 자리를 (%.2f, %.2f) [%s] 로 (기대 점수 %.1f → %.1f · 남은 시간 %.0f초 · 고를 자리 %d곳%s)" % (
            x, y, " ".join(sorted(sector_of[order[0]], key=str)), now_value, best_value, time_left, len(cand), " · 가는 도중" if moving else ""))
        return True

    @staticmethod
    def _shortest(cand, d):
        """cand 의 자리를 지금 선 곳에서부터 모두 도는 가장 짧은 순서 (끝은 어디든 좋다). d(a, b) = 길 길이 (a 가 None 이면 지금 선 곳에서).

        부분집합마다 「그 자리들을 돌고 어느 자리에서 끝날 때의 가장 짧은 길」 을 차례로 채운다 (자리 n 곳이면 n² × 2ⁿ 번 — 7곳에 6천 번쯤).
        가까운 자리부터 잇는 것(chain)은 첫걸음을 잘못 디디면 되돌아온다: 00:52 연습 판에서 0.92 m 앞의 자리부터 가는 순서가 5.88 m,
        1.14 m 앞의 자리부터 가는 순서가 5.20 m 였다.
        """
        n = len(cand)
        leg = [[d(a, b) if a != b else 0.0 for b in cand] for a in cand]
        best = {(1 << k, k): (d(None, cand[k]), None) for k in range(n)}  # (돈 자리들, 끝 자리) → (길이, 그 앞 자리)
        for mask in range(1, 1 << n):
            for last in range(n):
                here = best.get((mask, last))
                if here is None:
                    continue
                for nxt in range(n):
                    if mask & (1 << nxt):
                        continue
                    key, total = (mask | (1 << nxt), nxt), here[0] + leg[last][nxt]
                    if key not in best or total < best[key][0]:
                        best[key] = (total, last)
        full = (1 << n) - 1
        last = min(range(n), key=lambda k: best[(full, k)][0])
        seq, mask = [], full
        while last is not None:
            seq.append(cand[last])
            mask, last = mask & ~(1 << last), best[(mask, last)][1]
        return seq[::-1]

    def _answers_due(self, keys, now):
        """이 자리(keys 의 면)에서 찍은 사진 가운데 판독 답을 아직 기다릴 것이 있는가 — 찍은 지 stop_answer_wait_s 안인 것만."""
        wait = self.p.stop_answer_wait_s
        if not self.p.retry_nearby or wait <= 0.0:
            return False
        return any(len(tag) > 1 and tag[0] == "face" and tag[1] in keys and now - at < wait for tag, at in self.out.items())

    def _retry_nearby(self, pose, keys):
        """이 자리에서 못 적은 면을 더 가까이 · 더 정면에서 볼 자리를 바로 다음 차례에 끼워 넣는다 (면마다 한 번). 그렇게 한 면들을 돌려준다.

        못 적은 면 = 이 자리의 면 가운데 아직 종류가 없고, 여기서 찍었거나(답이 왔는데 안 적힘 · 답이 늦음) 팬이 닿지 않아 못 찍은 것.
        벽 쪽 통로의 면은 통로의 반대편 자리로 간다 (_corridor_side 가 nearby 에 든 면은 반대편을 고른다).
        끼워 넣을 자리가 지금 선 자리와 같으면(같은 곳 · 같은 방향) 넣지 않는다. 무슨 일이 있어도 예외를 밖으로 내지 않는다.
        """
        if not self.p.retry_nearby:
            return set()
        try:
            left = self.book.remaining()
            failed = {k for k in keys if k in left and k not in self.nearby and k[0] not in self.fmap.gone
                      and (self.tried.get(k, 0) >= 1 or k in self.queue) and not self._corridor_narrowed(k)}
            if not failed:
                return set()
            self.nearby |= failed
            chosen, unseen, _rest = self._choose(pose, failed, again=True)
            chosen, _unseen = self._corridor_stops(pose, failed, chosen, unseen)
            here = self.stops[self.i]
            chosen = [stop for stop in chosen if stop[3] and (len(stop[3]) == 1 and stop[3][0] in self.corridor
                      or math.hypot(stop[0] - here[0], stop[1] - here[1]) > 0.12 or abs(wrap(stop[2] - here[2])) > math.radians(20.0))]
            if not chosen:
                return set()
            self.stops[self.i + 1:self.i + 1] = self._order(pose, chosen, lead=False)
            self._pins |= {tuple(stop[3]) for stop in chosen}
            self.nearby_stops += len(chosen)
            got = {k for stop in chosen for k in stop[3]}
            self.log.append("자리 (%.2f, %.2f) 에서 못 적은 면 %d개 (%s) → 미루지 않고 바로 가까운 자리 %d곳에서 다시 본다" % (
                here[0], here[1], len(got), " ".join("%d%s" % k for k in sorted(got)), len(chosen)))
            return got
        except Exception as error:  # noqa: BLE001 — 덤으로 하는 일이다 (실패하면 전처럼 다 돈 뒤에 다시 본다)
            self.log.append("가까운 자리에서 다시 보기 예외 (%s: %s)" % (type(error).__name__, error))
            return set()

    def _leave_stop(self):
        """지금 자리를 끝내고 다음 자리로 넘어간다."""
        if self.i < len(self.stops):
            self._pins.discard(tuple(self.stops[self.i][3]))
        self._live_switched = False
        self.i += 1
        self.queue, self.body, self.any_heading = [], None, False
        self.turned = set()

    def _turn_to(self, pose):
        """선 자리에서 팬이 닿지 않는 남은 면 쪽으로 차체를 돌리게 한다 (이 자리에서 면마다 한 번). 그렇게 했으면 True.

        지금 선 곳을 그대로 설 자리로 삼는다 (돌기만 하면 된다 — 원래 자리로 몇 cm 다시 가게 하지 않는다). 다음 step 이
        ("go", 지금 자리, 그 면을 보는 방향, turn_tol_deg) 를 돌려준다. navigation 이 「돌 자리 없음」 을 알렸던 자리면 mission 이 보내지 않고
        바로 accept_heading 을 부르므로(mission._scout 의 no_turn) 기다리는 시간 없이 다시 돌 때로 미룬다.
        """
        key = next((k for k in self.queue if k not in self.turned), None) if self.p.turn_to_face else None
        if key is None:
            return False
        self.turned.add(key)
        self.turns += 1
        face = self.by_key[key]
        heading = math.atan2(face.cy - pose[1], face.cx - pose[0])
        self.queue.remove(key)
        self.queue.insert(0, key)
        self.stops[self.i] = (pose[0], pose[1], heading, list(self.stops[self.i][3]))
        self.body, self.body_tol, self.any_heading = heading, math.radians(self.p.turn_tol_deg), False
        self.log.append("면 %s%s: 선 방향에서 팬이 닿지 않음 (선 자리 (%.2f, %.2f) 의 차체 방향에서 %.0f°) → 차체를 그 면 쪽으로 돌려 본다" % (
            face.box, face.side, pose[0], pose[1], abs(math.degrees(wrap(heading - pose[2])))))
        return True

    def _hint_followup(self, pose):
        """가리킨 면(_hint)이 지정표적이 아닌 것으로 적혔으면, 같은 상자의 아직 못 적은 다른 면(가까운 것부터)을 이어서 가리킨다.

        그림을 면에 느슨하게 대다 보니 같은 상자의 옆 면을 가리키는 일이 있다. 2026-11-19 21:54 연습 판(2경기장 R-3): 19.9초의 사진에 찍힌
        레이더(상자 1 의 E 면)를 S 면으로 가리켜 그 자리로 갔고, 30초에 S 면이 전차로 적힌 뒤에는 실마리가 없어 표적을 141초에야 찾았다
        (E 면을 찍을 자리는 0.3 m 옆이었다). 이어서 가리키는 것은 hint_max 에 세지 않는다.
        """
        if self.want is None or not self._hinted:
            return
        try:
            left = self.book.remaining()
            for key in sorted(self._hinted - self._hint_closed):
                kind = self.book.kind(key)
                if kind is None:
                    continue
                self._hint_closed.add(key)
                sides = [k for k in left if k[0] == key[0] and k not in self._hinted]
                if kind == self.want or not sides:
                    continue
                nxt = min(sides, key=lambda k: (math.hypot(self.by_key[k].cx - pose[0], self.by_key[k].cy - pose[1]), k[1]))
                self._hinted.add(nxt)
                self._lead = nxt
                self.log.append("가리킨 상자 %d 의 %s 면은 %s 였다 → 같은 상자의 %s 면을 이어서 찍으러 간다" % (key[0], key[1], kind, nxt[1]))
                return
        except Exception as error:  # noqa: BLE001 — 덤으로 하는 일이다
            self.log.append("가리킨 면의 옆 면을 찾다 예외 (%s: %s)" % (type(error).__name__, error))

    def _take_lead(self):
        """지정표적으로 보이는 면(_hint)을 찍을 자리를 맨 앞으로 당긴다. 지금 자리에서 찍는 중(any_heading)이면 부르지 않는다 — 그 자리를 마친 뒤에 한다.
        가던 자리는 버리지 않고 바로 뒤에 남는다. 그 면을 찍을 자리가 계획에 없으면(이미 찍었다 · 갈 수 없다) 아무것도 하지 않는다."""
        key, self._lead = self._lead, None
        if key not in self.book.remaining():
            return
        j = next((j for j in range(self.i, len(self.stops)) if key in self.stops[j][3]), None)
        if j is None:
            return
        stop = self.stops[j]
        if j != self.i:
            del self.stops[j]
            self.stops.insert(self.i, stop)
            self.queue, self.body, self.any_heading = [], None, False
            self.turned = set()
        self._reorder_at = self.i + 1
        self._pins.add(tuple(stop[3]))
        self.log.append("지정표적으로 보이는 상자 %d 의 %s 면을 찍을 자리 (%.2f, %.2f) 를 맨 앞으로" % (key[0], key[1], stop[0], stop[1]))

    def _body_for(self, x, y, heading, keys):
        """자리 (x, y) 에서 keys 의 면을 모두 팬으로 겨눌 수 있는 차체 방향의 가운데와, 거기서 벗어나도 되는 각도 (rad).

        면 하나면 그 면을 똑바로 보는 방향 ± 40°, 여러 면이면 그만큼 좁아진다 (min_body_tol_deg 보다 좁아지지는 않는다).
        """
        reach = math.radians(PAN_LIMIT_DEG - 2.0 - self.p.aim_margin_deg)
        offs = [wrap(math.atan2(self.by_key[k].cy - y, self.by_key[k].cx - x) - heading) for k in keys]
        lo, hi = min(offs), max(offs)
        return wrap(heading + (lo + hi) / 2.0), max(math.radians(self.p.min_body_tol_deg), reach - (hi - lo) / 2.0)

    def _near_block(self, x, y, now):
        return any(now - t <= self.p.avoid_s and math.hypot(bx - x, by - y) <= self.p.avoid_m for bx, by, t in self.blocks)

    def _clear_stop_ahead(self, now):
        """지금 자리 뒤에, 막힌 곳 가까이가 아니면서 찍을 면이 남은 자리가 있는가."""
        left = self.book.remaining()
        return any(not self._near_block(stop[0], stop[1], now) and any(k in left and k[0] not in self.fmap.gone for k in stop[3])
                   for stop in self.stops[self.i + 1:])

    def _all_sighted(self, keys):
        """keys 가 모두 「본 것」 이 있는(표는 없고 본 답이 모두 같은) 면인가 (defer_sighted 가 꺼져 있으면 늘 False)."""
        sighted = getattr(self.book, "sighted", None)
        return bool(keys) and self.p.defer_sighted and sighted is not None and all(sighted(k) is not None for k in keys)

    def _defer(self, stop):
        """자리를 맨 뒤로 미룬다 (나중에 한 번 더 가 본다). 이미 미뤘던 자리면 False."""
        mark = (round(stop[0], 2), round(stop[1], 2))
        if mark in self.deferred:
            return False
        self.deferred.add(mark)
        self.stops.append(stop)
        return True

    def visible_here(self, pose):
        """지금 선 자리에서 팬으로 겨눠 찍을 수 있는 남은 면들의 key (가까운 것부터)."""
        p = self.p
        cos_inc = math.cos(math.radians(p.here_incidence_deg))
        out = []
        for key in self.remaining():
            if self.tried.get(key, 0) > p.retries or key[0] in self.fmap.gone:
                continue
            face = self.by_key[key]
            dx, dy = pose[0] - face.cx, pose[1] - face.cy
            d = math.hypot(dx, dy)
            if not (p.here_min_m <= d <= p.here_range_m) or (dx * face.nx + dy * face.ny) / d < cos_inc:
                continue
            if self._aim(pose, face) is None or not self.fmap._visible(pose[0], pose[1], face):
                continue
            out.append((d, key))
        return [key for _d, key in sorted(out)]

    def shoot_here(self, pose, now):
        """가려던 자리에 가지 못했다 (막힘 · 시한). 지금 선 자리에서 보이는 남은 면이 있으면 여기서 찍는다 (없으면 False).

        2026-11-18 3차 팀 테스트: 정찰 자리에 다른 팀 로봇이 서 있어 20 cm 앞에서 21초를 서 있다가 그냥 떠났다 — 볼 면은 거기서도 보였다.
        """
        if self.i >= len(self.stops):
            return False
        x, y, heading, old = self.stops[self.i]
        self.blocks.append((x, y, now))
        keys = self.visible_here(pose)
        if not keys:
            return False
        rest = [k for k in old if k not in keys and k in self.book.remaining()]
        self.stops[self.i] = (pose[0], pose[1], pose[2], keys)
        if rest:  # 여기서 안 보이는 면은 원래 자리에서 나중에 한 번 더
            self._defer((x, y, heading, rest))
        self.queue, self.body, self.body_tol, self.any_heading = list(keys), pose[2], math.pi, True
        self.here_shots += 1
        self.log.append("자리 (%.2f, %.2f) 에 가지 못함 → 선 자리 (%.2f, %.2f) 에서 보이는 면 %d개를 찍는다" % (x, y, pose[0], pose[1], len(keys)))
        return True

    def skip_stop(self, now=None, retry=True):
        """지금 가려던 자리를 포기하고 다음 자리로 간다 (가지 못했다). retry 면 맨 뒤에 넣어 나중에 한 번 더 가 본다
        (다른 UGV 는 움직인다). 그래도 못 적은 면은 다시 돌 때 다른 자리에서 본다."""
        if self.i < len(self.stops):
            stop = self.stops[self.i]
            self.skipped += 1
            if now is not None and not any(abs(bx - stop[0]) < 1e-6 and abs(by - stop[1]) < 1e-6 and t == now for bx, by, t in self.blocks):
                self.blocks.append((stop[0], stop[1], now))
            again = retry and self._defer(stop)
            self.log.append("자리 (%.2f, %.2f) 에 가지 못함 → %s" % (stop[0], stop[1], "나중에 한 번 더 가 본다" if again else "다음 자리로"))
            self._leave_stop()

    def shot_taken(self, tag, now):
        self.shots += 1
        self.out[tag] = now
        key = tag[1]
        self.tried[key] = self.tried.get(key, 0) + 1
        if key in self.queue:
            self.queue.remove(key)

    def on_answer(self, tag, pose, pan_deg, pictures):
        at = self.out.pop(tag, None)  # 찍은 때 (shot_taken 이 적어 둔 것)
        if pictures is None:
            self.rejected += 1
            return
        expected = expected_faces(self.fmap, pose, pan_deg, self.cam)
        # 먼저 사진 전체를 순서 · 위치로 짝짓는다 (로봇 방향이 조금 틀려 사진이 통째로 밀린 것을 여러 그림으로 바로잡는다).
        # 사진이 면 목록과 안 맞아 버려졌을 때 (실차에서는 거의 늘 그렇다 — 벽 그림 · 먼 상자까지 답이 와서):
        #   겨눈 면은 사진 가운데의 그림 하나로 분명하면 그것을 쓰고 (pick_centered),
        #   다른 그림은 하나하나 자리와 크기로 면에 붙여 헷갈리지 않는 것만 덤으로 적는다 (pick_each)
        pairs = match(expected, pictures)
        if pairs is not None:
            got = next(((e, k) for e, k in pairs if e.face.key == tag[1] and k != A.UNKNOWN), None)
            extra = [(e, k) for e, k in pairs if e.face.key != tag[1] and k != A.UNKNOWN and e.dist <= self.p.close_range_m * 1.2]
            limit = None
            centred = False
            if got is None:
                centred = True
                # 전체 짝짓기는 「됐다」 는데 겨눈 면이 그 짝에 없다 → 겨눈 면은 사진 가운데의 그림으로 따로 정한다.
                # 2026-11-19 실차: 지정표적을 한가운데에 찍고 모델도 맞게 읽었는데(자리 차이 0.004 · 크기 1.02배) 전체 짝짓기가 다른 면들에 맞춰
                # 사진을 밀어 놓고는 겨눈 면을 「차이가 크다」 고 뺐다. 그 판은 지정표적을 못 찾은 것이 되어 한 발도 쏘지 못했다
                got = pick_centered(expected, pictures, tag[1], self.p.center_tol, self.p.center_sep, self.p.aim_height, self.p.center_tol_near, self.p.center_rival)
        else:
            centred = True
            got = pick_centered(expected, pictures, tag[1], self.p.center_tol, self.p.center_sep, self.p.aim_height, self.p.center_tol_near, self.p.center_rival)
            extra = pick_each(expected, pictures, self.p, skip=(tag[1],)) if got is not None or not self.p.each_need_aim else []
            limit = (self.p.each_range_m, self.p.each_incidence_deg)
        if got is None:
            self.rejected += 1
        elif centred:
            # 가운데 그림으로 정한 겨눈 면은 자리와 크기를 함께 확인한 것이라 조금 더 비스듬해도(aim_incidence_deg) 적는다.
            # 정찰은 설 자리가 마땅치 않으면 60° 까지 비스듬한 자리에서도 찍는데(plan 의 느슨한 조건), 면 기록의 한도가 55° 라 그 답을 버리고 있었다
            # (2026-11-19 14:13 실차: 55° · 56° 에서 찍은 두 장). 사진 전체 짝짓기로 얻은 답은 전처럼 55° 까지만 (가상 시험: 그 너머는 옆 면과 바뀐다)
            self.book.add(*got, limit=(self.book.record_range_m, self.p.aim_incidence_deg))
        elif not self.book.add(*got):
            # 사진 전체 짝짓기로 얻은 겨눈 면의 답이 한도(55°) 밖이라 적히지 않았다. 사진 가운데의 그림(자리 · 크기 확인)도 같은 답이면 그것으로 적는다.
            # 벽 쪽 통로의 자리는 일부러 45 ~ 59° 비스듬히 찍는다 — 2026-11-19 저녁 가상 경기: 통로의 면이 제자리에 맞게 읽혔는데도(자리 차이 0.01)
            # 57° 라 세 번을 찍고도 적지 못했다. 가운데 그림의 답이 다르면 적지 않는다 (55° 너머의 전체 짝짓기는 옆 면과 바뀔 수 있다)
            again = pick_centered(expected, pictures, tag[1], self.p.center_tol, self.p.center_sep, self.p.aim_height, self.p.center_tol_near, self.p.center_rival)
            if again is not None and again[1] == got[1]:
                self.book.add(*again, limit=(self.book.record_range_m, self.p.aim_incidence_deg))
        # 덤은 표를 작게 주고, 그 면에 이미 다른 종류가 적혀 있으면 적지 않는다 (FaceBook.add 의 bonus)
        bonus = [(exp, kind) for exp, kind in extra if self.book.add(exp, kind, bonus=True, limit=limit)]
        self.bonus += len(bonus)
        # 사진에 함께 찍힌 모든 면을 「본 것」 으로 모은다 (겨눈 면 · 덤으로 이미 적힌 면은 FaceBook.sight 가 건드리지 않는다)
        self._hint(expected, pictures)
        try:
            sights = self._see(pose, pan_deg, expected, pictures, at)
        except Exception as error:  # noqa: BLE001 — 덤으로 하는 일이다. 여기서 난 예외로 겨눈 면의 기록 · 경기 흐름이 멈추면 안 된다
            sights = []
            self.log.append("본 것을 모으다 예외 (%s: %s) → 이 사진에서는 넘어간다" % (type(error).__name__, error))
        self._note(tag, pose, pan_deg, expected, pictures, [got] if got else [], bonus, sights)

    def on_pass(self, tag, pose, pan_deg, pictures, at=None, steady=True):
        """지나가며 찍은 사진의 답: 겨눈 면이 없으므로 「본 것」 으로만 쓴다. 이번에 적힌 면 [(면 key, 종류)] 을 돌려준다.

        mission 이 걸러서 넘긴다 — 팬을 한 번이라도 보낸 뒤에 찍었고(그 전에는 팬이 어디를 보는지 모른다), 찍을 때 차체가 돌고 있지 않았던 사진만.
        steady=False = 찍을 때 방향이 조금 흔들린 사진 (방향 값이 몇 도 늦을 수 있다 → 사진 전체가 같은 만큼 밀린다). 이런 사진은 밀림을 스스로 찾는
        「한꺼번에 맞추기」 (vote_align) 로 붙은 그림만 쓴다. vote_align 이 꺼져 있으면 아무것도 쓰지 않는다.
        """
        if not pictures or pose is None:
            return []
        expected = expected_faces(self.fmap, pose, pan_deg, self.cam)
        self._hint(expected, pictures)
        if not self.p.vote_on or not self.p.vote_pass:
            return []
        sights = self._see(pose, pan_deg, expected, pictures, at, each=bool(steady))
        # 사진 기록(photos.jsonl)에는 넣지 않는다 — 그 파일을 읽는 도구들이 정찰 사진의 tag 모양을 전제로 한다. 적힌 면은 log 와 mission 의 글로 남는다
        return [(e.face.key, kind) for e, kind, new in sights if new]


def make_scout(fmap, grid, book, cam=None, params=None, start=None):
    params = params or ScoutParams()
    if params.mode == "place":
        return PlaceScout(fmap, grid, book, cam, params)
    return FaceScout(fmap, grid, book, cam, params, start=start)
