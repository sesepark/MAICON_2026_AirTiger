"""경기 흐름: 지금 무엇을 할지 정하고, 주행 · 사진 판독 · 팬틸트 · 발사 · 보고를 시킨다. ROS 를 쓰지 않는다.

한 틱(0.1초)에 tick(facts, ports) 를 한 번 부른다.
  facts   지금 알고 있는 것 (Facts) — node.py 가 메시지에서 모아 준다
  ports   시키는 창구 (아래 Ports 의 함수들) — node.py 가 메시지로 내보낸다. 시험에서는 가짜가 받는다

움직이는 일은 한 번에 하나다 (위에서부터 먼저):
  assault  목표거점이 게시됐다 → 그 거점 안으로
  stage    게시 시각이 가깝다 → 우리 줄의 상대 거점 앞에서 기다린다 (게시 6초 전부터 빔 바로 앞)
  strike   지정표적 면을 찾았다 → 쏠 자리로 가서 조준하고 쏜다 (계산한 각도로 겨눈 뒤 사진으로 고친다 — aim.py)
  scout    정찰 (scout.py — 자리 기준 / 면 기준)
  rest     할 일이 없다 → 그 자리에 선다
보고는 움직임과 따로 돈다: 준비완료, M1(해석을 받으면), M3(개수가 바뀌면 · 주기적으로), M6(보고 단계가 되면).

판단의 순서와 값은 sim-strategy 브랜치 team/navigation/navigator.py (기본 전략)에서 가져왔다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import arena as A
from .aim import AimLoop, AimParams, find_receiver, laser_pixel, surface_z
from .ugv_track import TrackParams, UgvTracker, choose_ugv, foe_spots
from .faces import CameraParams, FaceBook, wrap
from .grid import Grid, path_length
from .scout import PAN_LIMIT_DEG, ScoutParams, make_scout
from .standing import Standing

TILT_LIMIT_DEG = 30.0  # 틸트 ±100 × 0.3° (공식 team_node.py 뼈대 2026-11-18 — 현장에서 잰다)


@dataclass
class MissionParams:
    scout: ScoutParams = field(default_factory=ScoutParams)
    robot_radius: float = 0.15
    base_keepout_m: float = 0.45  # 거점 앞은 게시 전까지 들어가지 않는다 (벽에서 이만큼)
    # M2
    target_shots: int = 3  # 한 표적에 쏘는 발 수의 상한
    # 발사 뒤 사진에서 레이저 점을 찾아 수광부에 닿았는지 본다 (laser_spot.py · 창구 watch_laser). 닿은 것으로 보이면 더 쏘지 않는다 —
    # 표적은 처음 부서질 때 한 번만 인정된다 (경기서버연동API 6단계). 닿았는지 볼 수 없었으면(레이저 점이 안 보임 · 창구 없음):
    # 조준이 허용 안에 든 채 쏜 발이 unconfirmed_shots 번이면 그만 쏜다 (전에는 늘 target_shots 번)
    unconfirmed_shots: int = 2
    hit_px: float = 6.0  # 레이저 점과 수광부의 차이가 이 안이면 맞은 것으로 본다 (탐지한 수광부가 크게 찍혔으면 그 크기의 hit_size_frac 까지)
    hit_size_frac: float = 0.5
    spot_wait_s: float = 3.5  # 발사 뒤 레이저 점의 답을 기다리는 시간
    spot_fix_max_px: float = 30.0  # 잰 레이저 점이 식의 자리와 이보다 멀면 잘못 찾은 것으로 보고 조준을 고치는 데 쓰지 않는다
    # 사진으로 표적 찾기 (보완 — 제어 콘솔의 M2 시험과 같은 방식): 계산한 각도로 겨눈 사진에서 표적을 못 찾으면 팬을 search_pans 로
    # 돌려 가며 사진에서 표적 그림을 찾고, 찾으면 그쪽으로 겨눈 뒤 사진 조준을 잇는다. 거기서도 못 찾으면 쏘지 않고 정찰로 돌아간다
    search_pans: tuple = (0.0, -35.0, 35.0)
    strike_tries: int = 2  # 표적을 못 찾고 물러난 뒤 정찰을 마치고 다시 해 보는 것까지 합친 횟수
    standoff_m: tuple = (0.9, 0.7, 1.2, 1.5)
    emitter_height: float = 0.29  # 레이저 발광부 높이 [추정] — 전의 틸트 식(camera_height = 0 일 때)에만 쓴다
    # 틸트 0 일 때 카메라가 수평으로 보는 높이. 2026-11-18 측정(8호기): 렌즈 앞 1.07 m 의 수광부가 사진 가운데보다 30픽셀(6.0°) 위 →
    # 수광부보다 0.112 m 아래 → 0.32 − 0.112 = 0.21 m. 계산한 틸트 각도에 쓴다 (aim_angles). 0 으로 두면 전의 식으로 계산한다
    camera_height: float = 0.21
    # 표적 수광부 높이: 2026-11-19 실차 사진 4장에서 수광 돔의 가운데가 면 위 끝에서 높이의 18 ~ 21% 아래였다 → 40 cm 면에서 약 0.32 m (그 전 추정 0.35)
    target_receiver_height: float = 0.32
    aim_settle_s: float = 0.8  # 팬틸트를 보낸 뒤 쏘기까지 기다리는 시간 (ok 는 접수일 뿐)
    laser_on_s: float = 5.0
    fire_interval_s: float = 10.0
    strike_give_up_s: float = 40.0  # 쏠 자리에 이만큼 지나도 못 가면 포기
    strike_total_s: float = 75.0  # 지정표적에 쓰는 시간의 상한 (정찰이 멈춰 있는 시간)
    strike_yaw_tol_deg: float = 25.0  # 쏠 자리에서 차체 방향의 허용 — 조준은 팬틸트가 한다 (팬 ±50°)
    strike_keep_m: float = 0.35  # 쏠 자리에 한 번 선 뒤에는 위치 값이 이 안이면 그 자리로 친다 (처음 설 때는 0.2 m)
    reaim_deg: float = 3.0  # 조준한 뒤 계산한 각도가 이만큼 넘게 달라졌을 때만 다시 조준한다 (위치 값의 잔떨림에 따라가지 않는다)
    fire_reply_wait_s: float = 4.0  # 발사 요청의 답을 이만큼 기다려도 없으면 다시 쏠 수 있게 한다 (답이 한 번 빠지면 그 뒤로 못 쏘던 것)
    aim: AimParams = field(default_factory=AimParams)  # 사진으로 조준 고치기 (aim.py — 레이저 점 식 · 허용 픽셀 · 횟수)
    # 상대 UGV 쏘기 (ugv_track.py). 정찰(M3)이 다 끝나 서 있는 동안(rest)에만 한다 — 그동안 M6 용으로 상대 UGV 를 살핀다 (watch_pans_deg)
    # 2026-11-20 사용자 결정(사격 담당): 기본으로 켠다. 멈춰 있는 UGV 만 쏜다 — 콘솔 시험에서 멈춘 UGV 는 잘 맞았고, 움직이는 UGV 는 잘 겨눠도 인정되지 않았다
    engage_ugvs: bool = True  # False 면 서 있는 동안 살피기만 하고 쏘지 않는다
    engage_still_only: bool = True  # True 면 멈춰 있는 것으로 보이는 UGV 만 쏜다 (track.still_*). False 면 움직이는 것도 앞을 겨눠 쏜다 (전의 동작)
    engage_unknown_camp: bool = False  # 진영(안테나 색)을 못 읽은 UGV 도 쏘는가 — 아군일 수 있으므로 끔
    ugv_shots: int = 3  # 상대 UGV 에 쏠 발 수의 상한. 지정표적을 다 쏜 뒤에만 쏜다 (engage_ugv 의 owed)
    ugv_hit_watch_s: float = 2.5  # 쏜 뒤 이만큼 지켜본다: 그동안의 사진에서 레이저 점이 수광부를 벗어나지 않았으면 맞힌 것으로 본다 (유효 피격 2초)
    ugv_hits_lower_lives: bool = True  # 우리가 맞힌 팀은 LED 판독이 안 바뀌었어도 목숨을 그만큼 낮춰 센다 → M6 순위에서 내려간다
    # M6 순찰에서 표적을 붙들고 있는 시간: 멈췄거나 조금씩 움직이는 표적은 그 자리에서 계속 겨누고(최대 m6_hold_max_s),
    # 움직이는 표적은 이만큼만 따라가 보고(멈추기를 기다린다), 많이 움직이는 표적은 기다리지 않는다
    m6_moving_wait_s: float = 6.0
    m6_turn_to_target: bool = True  # 멈춘 표적이 팬 범위 끝(±m6_turn_at_pan_deg 밖)에 있으면 차체를 그쪽으로 돌린다 (설 때마다 한 번)
    m6_turn_at_pan_deg: float = 42.0
    # 서 있는 동안 상대 UGV 를 찾으려고 팬을 차례로 돌려 보는 각도 (도, + 오른쪽)와 한 각도에 머무는 시간. 빈 목록이면 돌리지 않는다.
    # 드론 영상은 쓰지 않는다 (M6 때는 없다고 본다) — 카메라로 본 것만.
    watch_pans_deg: tuple = (0.0, -40.0, 40.0)
    watch_dwell_s: float = 3.0
    # M6 순찰 (2026-11-20 사용자 결정): 정찰(M3)이 다 끝나면 우리 진영으로 돌아가, 자리를 옮겨 가며 상대 UGV 를 찾아 살피고 멈춘 것을 쏜다.
    # False 면 정찰이 끝난 자리에 그대로 서서 살핀다.
    #
    # m6_hunt = 이 「M6 사냥」 전체의 스위치 (2026-11-20 사용자 결정: 옵션으로 둔다). False 면 정찰(M3)이 일찍 끝나도 main 의 기존 흐름 그대로다:
    #   정찰이 끝난 자리에 서서 3초마다 read_ugvs 만 부르고(순찰 · 팬 돌리기 · 상대 UGV 사격 없음), 4:00 전 거점 앞 대기 → 거점 진입(M5)을 하고,
    #   M6 순위는 LED 판독의 가장 최근 값만 쓴다 (세 번의 다수 · 맞힌 상대 낮추기 없음). 아래의 m6_* · engage_* · watch_* 값은 쓰이지 않는다.
    m6_hunt: bool = True
    m6_patrol: bool = True
    # M6(살피기 · 순찰 · 쏘기)는 정찰(M3)이 경기 시각 이 값(초) 안에 다 끝났을 때만 한다 — 4:00 목표거점 게시까지 1분 넘게 남았을 때.
    # 더 늦게 끝났으면 M6 를 하지 않고 서 있는다. 시간을 따지지 않으려면 큰 값(예 1000)을 준다
    m6_scout_done_by_s: float = 180.0
    m6_instead_of_m5: bool = True  # M6 를 하는 판에서는 M5(거점 앞 대기 · 거점 진입)를 하지 않고 경기 끝까지 M6 를 한다. 끄면 M5 가 먼저다
    # M6 순환 경로 (2026-11-20 사용자 결정: 자리에 서서 보지 않고, 멈추지 않고 돌며 찾는다). 정찰(M3)의 격자로 만든다 (mission._m6_post_list):
    # 우리 벽에서 폭의 m6_route_x_fracs 만큼 떨어진 두 세로줄 × 높이의 m6_route_y_fracs 지점 (우리 거점에는 들어가지 않는다 — 빈 칸으로 옮긴다)
    m6_route_x_fracs: tuple = (0.26, 0.44)
    m6_route_y_fracs: tuple = (0.2, 0.5, 0.8)
    m6_reach_m: float = 0.35  # 지점에 이만큼 다가가면 서지 않고 다음 지점으로 간다
    m6_go_give_up_s: float = 25.0  # 지점에 이만큼 지나도 못 가면 다음 지점으로
    m6_scan_gap_s: float = 0.7  # 도는 중에 locate_sensors 를 부르는 간격
    m6_hold_s: float = 6.0  # 상대 수광부를 보고 선 뒤, 겨눌 것이 이만큼 없으면 다시 돈다
    m6_hold_max_s: float = 28.0  # 한 번 서서 겨누는 시간의 상한
    m6_resight_s: float = 8.0  # 다시 돌기 시작한 뒤 이만큼은 수광부가 찍혀도 서지 않는다 (쏠 수 없던 같은 상대 앞에 곧바로 다시 서지 않게)
    # 다 쏜 뒤(또는 겨눌 것이 없어진 뒤) 그 멈춘 상대의 호출부호 · 목숨을 아직 모르면 곁 → 뒤 → 다른 곁으로 돌아가 가까이에서 읽는다 (mission._m6_inspect).
    # 2026-11-20 07시 연습 판: 쏜 거리에서는 진영만 읽히고 호출부호 · 목숨이 한 번도 안 읽혀, 맞히고도 M6 순위에 넣지 못했다
    m6_inspect: bool = True
    m6_inspect_range_m: float = 0.80  # 상대에서 이만큼 떨어진 자리에 선다
    m6_inspect_go_s: float = 14.0  # 한 자리에 이만큼 지나도 못 가면 다음 자리로
    m6_inspect_read_s: float = 3.0  # 한 자리에 서서 읽는 시간
    m6_inspect_read_gap_s: float = 1.0  # 그동안 read_ugvs 를 부르는 간격
    m6_inspect_max_s: float = 40.0  # 한 상대에 쓰는 시간의 상한
    m6_inspect_until_s: float = 265.0  # 경기 시각이 이보다 늦으면 시작하지 않는다
    track: TrackParams = field(default_factory=TrackParams)
    locate_timeout_s: float = 3.0  # 수광부 자리를 물은 뒤 이만큼 답이 없으면 버린다
    # M5
    target_at_s: float = A.TARGET_AT_S
    shoot_settle_s: float = 0.6  # 멈춘 뒤(도착했거나 이동을 거둔 뒤) 이만큼 지나야 사진을 찍는다 — 차체가 아직 흔들릴 때 찍으면 흐리다
    stall_s: float = 3.0  # 정찰 자리 가까이에서 이만큼 서 있는데도 「도착」 이 안 오면, 방향은 지금 것으로 받아들이고 찍는다
    # navigation 이 「도착」 으로 끝낸 곳이 정찰 자리에서 arrive_m 을 조금 넘었을 때 (REACHED [POS_NOT_MET]): ScoutParams.arrive_slack_m 안이면
    # 온 것으로 치고 그 자리에서 찍는다. False 면 전처럼 「가지 못함」 으로 다룬다 (선 자리에서 보이는 면만 찍거나 그 자리를 뒤로 미룬다)
    short_arrival_ok: bool = True
    # 빠져나오기: 이동 목표가 escape_fails 번 잇달아 「못 갔다」 로 끝났는데 그 사이 escape_stuck_m 도 못 움직였으면, 지나온 길로
    # escape_back_m 되돌아간 자리로 한 번 간다 (2026-11-19 실차: 다른 로봇 · 어긋난 상자 사이에 들어가 100초, 85초를 그 자리에 있었다)
    escape_fails: int = 2
    escape_stuck_m: float = 0.25
    escape_back_m: float = 0.8
    escape_limit_s: float = 14.0
    # 돌 자리가 없어 출발하지 못했다고 navigation 이 알려 주면(BLOCKED + 「돌 자리 없음(출발)」), 돌지 않고 차체 방향 그대로 앞이나 뒤로 이만큼 곧게 간다
    # (navigation 의 MODE_STRAIGHT). 지나온 길로 되돌아가라고 하면 대개 뒤쪽 목표라 다시 서서 돌아야 해서 같은 자리에서 또 막힌다
    straight_out_m: float = 0.40
    # 지나온 길로 되돌아가는 빠져나오기도 실패했고 그 사이 자리도 그대로면, 차체 방향 그대로 뒤로 back_off_m 곧게 물러난다 (0 이면 하지 않는다).
    # 2026-11-19 17:26 모의 경기 3차(본선 경기장 · 로봇 6대): 위쪽 통로에서 다른 로봇과 0.25 m 를 두고 마주 선 채 130초를 서 있었다 —
    # navigation 은 「장애물이 비키기를 기다림」 만 되풀이했고(상대도 같았을 것이다), 지나온 길로 가는 목표도 「그 길도 못 감」 이었다.
    # 막은 것은 가던 쪽(앞)에 있으므로 뒤로 물러나면 사이가 벌어져 둘 다 돌아갈 길이 생긴다. 충돌에는 벌점이 없다 (본선 기본정보)
    back_off_m: float = 0.40
    # 빠져나오기 목표는 우리 진영 거점(사각형 + 이만큼 = 차체 반 대각선 0.17 + 여유) 안에 두지 않는다 — 목표거점 게시(4:00) 뒤에는
    # 우리 UGV 가 우리 거점에 들어가도 상대 연합의 점령이 된다 (경기서버연동API.md 「우리 거점에 우리 UGV가 다시 들어가도 상대 연합의 점령」, 2026-11-19).
    # 게시 escape_own_base_lead_s 전부터 (빠져나오기 한 번이 escape_limit_s 걸리므로). 그 전에는 거점에 들어가도 아무 일 없어 출발 칸으로도 물러날 수 있다
    escape_own_base_m: float = 0.20
    escape_own_base_lead_s: float = 20.0
    straight_out_half_w: float = 0.135  # 곧게 갈 띠의 반폭: 차체 반폭 0.115 + 여유 0.02
    no_turn_keep_s: float = 60.0  # 「돌 자리 없는 자리」 를 기억하는 시간
    # 지나가며 찍기: 정찰 자리로 가는 동안 곧게 가고 있으면 이 간격으로 사진을 한 장씩 판독에 보낸다. 0 = 끔 (기본).
    # 2026-11-19: 처음에는 답을 쓰지 않고 기록만 했다 (팀원 제안: 계속 인식). 14:52 실차 기록을 보고 저녁부터 면 기록의 「본 것」 으로 쓴다 (아래 pass_vote_…)
    pass_gap_s: float = 0.5  # 지나가며 찍는 간격 (초). 0 이면 찍지 않는다. 2026-11-19 저녁부터 켠다 — 답을 면 기록의 「본 것」 으로 쓴다 (판독은 장당 약 0.06초)
    # 정찰 자리로 가는 동안에는 카메라를 정면(팬 0 · 틸트 0)으로 돌려 둔다. 그대로 두면 앞 자리에서 겨눴던 쪽(옆으로 39° 등)을 본 채 달린다 (14:52 실차).
    # 정면이면 앞의 면과 양옆의 면이 멀리서부터 찍혀 「본 것」 이 쌓인다 (멀리서 본 나란한 면은 scout 의 vote_align 이 가린다).
    # 가까운 자리(pass_forward_min_m 안)로 갈 때는 돌리지 않는다 — 곧 그 자리의 면을 겨눠야 한다
    pass_forward: bool = True
    pass_forward_min_m: float = 0.7
    # 자리에서 사진까지의 기다림을 겹친다 (2026-11-19 16:01 실차의 시간을 쪼갠 결과 — 정찰 207초 가운데 자리에 서서 쓴 것이 111초였다:
    # 방향 맞추기 56초 · 팬이 돌기를 기다린 것 30초 · 「차체가 멈춘 뒤에야 팬을 보내는」 기다림 12초. 면이 팬에 닿게 된 때부터 사진까지 가운데값 4.3초).
    #  · shoot_early: 자리 · 방향이 됐으면 navigation 의 「도착」 이나 「1초 서 있음」 을 기다리지 않고 바로 이동을 거둔다
    #    (벽 쪽 통로의 자리는 빼고 — 거기는 어긋남의 여유가 작아 navigation 이 끝까지 가게 둔다)
    #  · 팬은 차체가 멈추기를 기다리지 않고 바로 보낸다. 자리에 가까워지면(pan_track_m 안) 찍을 면 쪽으로 미리 돌려 둔다
    #  · pan_keep_deg: 미리 돌려 둔 팬이 찍을 각도에서 이 안이면 다시 돌리지 않고 그대로 찍는다 (면이 사진 가운데에서 그만큼 벗어날 뿐이다)
    shoot_early: bool = True
    pan_track_m: float = 0.6
    pan_track_step_deg: float = 8.0
    pan_keep_deg: float = 6.0
    pass_turn_deg: float = 4.0  # 지난 0.5초 동안 방향이 이보다 많이 바뀌었으면 찍지 않는다 (도는 중의 사진은 흐리고, 방향 값도 늦다)
    # 지나가며 찍은 사진의 답을 면 기록의 「본 것」 으로 쓴다 (scout.on_pass — 끄는 것은 ScoutParams 의 vote_on · vote_pass). 쓰기 전에 두 가지를 본다:
    #  · 팬을 한 번이라도 보낸 뒤에 찍은 사진인가. 팬 각도는 읽을 수 없어 보내기 전에는 어디를 보는지 모른다 (2026-11-19 14:52 실차: 첫 팬 명령 전의
    #    사진은 기록의 팬 0° 와 실제가 약 50° 달랐다). 그래서 경기 첫머리에 팬을 가운데로 한 번 보내고(tick), pan_home_s 뒤부터 믿는다
    #  · 찍을 때 차체가 돌고 있지 않았는가: 찍기 0.6초 전부터 pass_vote_wait_s 뒤까지의 방향이 사진의 방향에서 pass_vote_turn_deg 넘게 벗어나면 버린다.
    #    찍기 전 0.5초만 보는 pass_turn_deg 로는 막 돌기 시작한 때의 사진이 걸러지지 않는다 (같은 판: 돌며 찍혀 흐린 사진에서 콘크리트 벽을 탄약상자 0.86 으로 읽음)
    pan_home_s: float = 2.5
    pass_vote_turn_deg: float = 6.0
    pass_vote_wait_s: float = 0.5
    # 방향이 pass_vote_turn_deg 보다는 많이, 이 값까지 벗어난 사진은 버리지 않고 「한꺼번에 맞추기」 로만 쓴다 (scout.on_pass 의 steady=False).
    # 차체는 곧게 갈 때도 방향을 계속 조금씩 고친다 — 2026-11-19 16:01 실차: 지나가며 찍은 133장 가운데 59장이 6° 한도에 걸려 버려졌다 (적힌 면 5개).
    # 방향 값이 늦어 생기는 것은 사진 전체가 같은 만큼 밀리는 것이고, 한꺼번에 맞추기(align.aligned)는 그 밀림을 스스로 찾아 뺀다. 그림 하나를 면 하나에
    # 따로 붙이는 규칙(pick_votes)은 밀림을 모르므로 이런 사진에는 쓰지 않는다. 16:01 · 14:52 두 판을 정답표와 견줌: 6 ~ 15° 의 사진 50 + 17장에서 얻은
    # 「본 것」 80개가 모두 맞았고, 틀린 것은 20° 를 넘는 사진에서만 나왔다 (3개 — 면이 한 칸씩 밀려 맞음). 기록의 방향은 0.5초마다뿐이라 이 「흔들림」 은
    # 실차(매 틱)보다 작게 잡힌 값이다 → 실차의 15° 는 이보다 엄하다. 0 이하 또는 pass_vote_turn_deg 이하이면 전처럼 6° 를 넘는 사진을 버린다
    pass_vote_turn_align_deg: float = 15.0
    trail_step_m: float = 0.20  # 지나온 길을 이 간격으로 적어 둔다
    # 지도에 없는 상자 (2026-11-19): navigation 이 라이다로 찾아 /team/navigation/lidar_boxes 로 알려 주는 상자. 지도(드론 사진)는 위 · 아래 벽 쪽이 잘려
    # 벽에 붙은 상자가 빠진다 (실차 한 판에서 셋). 오래 보인 것을 지도의 맨 뒤 번호로 넣고, 그 면을 찍을 자리를 정찰 계획의 맨 뒤에 붙인다 —
    # 덤이다: 계획한 면을 다 찍은 뒤에 간다. 서 있는 다른 로봇을 상자로 잘못 본 것이어도 맨 뒤에서 시간을 조금 쓸 뿐이다 (면 기준 정찰에서만 쓴다)
    extra_boxes: bool = True  # False 면 통째로 끈다
    extra_min_age_s: float = 4.0  # 목록에 이만큼 이어서 있어야 받아들인다
    extra_lost_s: float = 6.0  # 받아들인 상자가 목록에서 이만큼 빠져 있으면 없는 것으로 친다 (다시 나오면 되살린다)
    # 위에서 본 UGV 자리 (2026-11-20): 한자리에 robot_stand_s 넘게 서 있는 다른 로봇을 정찰의 실시간 순서에 알린다 (그 곁의 자리를 뒤로 돌린다).
    # 우리 차는 그 관측의 때에서 robot_self_window_s 앞까지 우리가 있던 자리로 골라낸다 (사진은 받은 때보다 0.3 ~ 0.7초 앞서 찍혔다 — navigation 과 같은 값)
    overhead_robots: bool = True  # False 면 쓰지 않는다
    robot_stand_s: float = 4.0
    robot_self_window_s: float = 1.5
    extra_max: int = 6  # 받아들인 상자(지금 있는 것으로 치는 것) 수의 상한. 없어진 것까지 합쳐서는 이 값의 3배까지만 받는다
    extra_match_m: float = 0.15  # 목록의 점이 앞의 후보에서 이 안이면 같은 후보다
    extra_near_m: float = 0.30  # 지도의 상자 · 이미 받아들인 상자와 가운데가 이 안이면 그 상자를 본 것이다 (새 상자가 아니다)
    extra_wall_gap_m: float = 0.30  # 면 앞에서 벽까지가 이보다 좁은 면은 만들지 않는다 (벽에 붙은 쪽 — 찍을 수 없다)
    # 기다리는 자리 · 발표 직전에 다가서는 자리 모두 벽에서 0.80 m (2026-11-19 밤, 저녁에는 0.72, 그 전 0.60). 모의 5차: 0.60 m 자리에 비스듬히(28°) 서서
    # 차체 앞 모서리가 거점 네모의 선을 2 cm 넘어 있었다 (지도는 가로 6.12 로 그리는데 실제는 6.08 이라 4 cm 더 넘었을 수 있다).
    # 그 거점이 목표로 게시되면 「게시 전부터 안에 있던 것」 이 되어 점령으로 인정되지 않을 수 있다 (Q&A Q4).
    # 0.72 로 시킨 22:40 연습 판에서는 0.62 m 에 섰다 (navigation 의 위치 허용이 0.10 m — 모의 5차도 0.60 에 0.55). 그러면 앞 모서리(가운데에서
    # 0.17 m)가 거점 네모의 선(0.40 m)에서 5 cm 밖뿐이다. 0.80 으로 시키면 0.10 m 더 들어가 서도 선에서 13 cm 밖이다. 돌입 거리는 8 cm (0.13초) 늘 뿐이다.
    # 거점 센서는 노란 기둥에 붙어 있고(팀장이 현장에서 봄) 기둥은 거점 네모보다 얕다 (드론 사진으로 벽에서 0.2 ~ 0.25 m [추정]).
    stage_from_wall_m: float = 0.80
    # 발표를 기다릴 거점. 빈 글자(기본)면 늘 상대 진영의 가운데(2번 줄) 거점 앞에서 기다린다.
    # 종류(예 "COMMAND_POST")를 주면, 지도가 상대 진영의 거점 종류를 읽어 알려 줬을 때(FieldMap.kinds_read) 그 종류의 거점 앞에서 기다린다 (못 읽었으면 가운데).
    # 잰 것 (2026-11-19): 기다리던 거점이 게시되면 들어가는 데 1.5초 (21:54 연습 판), 가운데에서 기다리다 끝 거점이 게시되면 9.2 ~ 9.8초 (모의 5차 ·
    # 22:40 연습 판). 끝 거점 앞에서 기다리다 반대쪽 끝이 게시되면 가상 경기로 15.9초 (가운데에서였다면 6.5초).
    # 주최측이 만든 판(본선 경기장 모의 경기 여섯 판 · 포털의 팀 테스트 기본 임무 여섯 자리)에서 게시된 목표거점은 모두 작전지휘소였다. 그래서 23:05 에
    # 기본값을 COMMAND_POST 로 두었으나, 팀장 판단(같은 날 밤): 모의 경기는 대충 정해 둔 값일 수 있고 본선에서 작전지휘소가 게시될 가능성은 1/3 로 본다
    # → 세 거점이 같은 가능성이면 가운데가 낫다 (가장 늦는 경우가 9.8초, 끝에서 기다리면 15.9초). 기본값을 빈 글자로 되돌림
    stage_kind: str = ""
    stage_margin_s: float = 8.0
    stage_speed: float = 0.30  # 대기 지점까지 걸리는 시간을 어림할 때 쓰는 평균 속도
    # 발표 직전에 다가서는 자리. 0.42 였을 때는 차체 앞(가운데 − 0.125)이 벽에서 0.295 m — 거점 네모(깊이 0.40 m) 안이었다.
    # 센서 빔의 깊이를 모르는데([추정] 0.20 m) 발표 전에 이미 빔을 가리고 있으면 「새로 들어간 진입」 이 아니게 된다 → 차체 앞이 거점 밖(0.475 m)에 있게 한다
    close_from_wall_m: float = 0.80
    close_lead_s: float = 6.0
    stage_yaw_tol_deg: float = 30.0  # 거점 앞에서 기다릴 때 차체 방향의 허용 (정확히 맞추려다 제자리에서 시간을 쓰지 않게)
    # 기다리는 거점 안 · 바로 앞에 다른 로봇이 서 있으면 벽을 보되 그 로봇에서 **먼 쪽으로** 이만큼 기울여 선다 (2026-11-20 05:30, 04:48 연습 판 뒤의 가상 주행).
    # 대기 자리에는 보통 그 로봇의 반대쪽에서 와서 방향 허용(30°)에 들자마자 멈추므로 차체가 그 로봇 **쪽으로** 20°쯤 기울어 선다 (04:48 판 −158° ·
    # 02:39 판 −154° — 벽 쪽은 180°). navigation 의 가상 주행(고정 기둥이 있는 지도, 그 로봇의 자리 30가지 × 들어갈 자리 둘)에서 첫 돌진이 들어간 판:
    # 벽을 똑바로 보거나 먼 쪽으로 기울면(0 ~ 40°) 60/60, 그 로봇 쪽으로 3° 54/60 · 5° 48/60 · 10° 42/60 · 22° 38/60
    # (나머지는 「지도에 없는 물체가 길을 막음」 으로 1초 만에 BLOCKED — 차체 앞 띠에 그 로봇이 걸린다. 실차 02:39 판 22번이 그랬다).
    # 0 이하 = 기울이지 않음 (전과 같음: 벽 쪽, 허용 stage_yaw_tol_deg)
    stage_lean_deg: float = 20.0
    stage_lean_tol_deg: float = 10.0  # 그때의 방향 허용 — 끝난 방향이 먼 쪽으로 10 ~ 30° 가 된다
    # 거점 안에 들어갈 깊이: 차체 가운데가 벽에서 robot_half_length + wall_margin_m (= 0.20 m). 앞이 벽에서 7.5 cm.
    # 0.05 였을 때는 비스듬히 들어가면 모서리가 벽에 닿았고(가상), 서서 돌 자리도 없었다 (차체 대각선의 반이 0.17 m)
    wall_margin_m: float = 0.075
    robot_half_length: float = 0.125
    # 들어갈 자리는 거점의 양 끝에서 이만큼 안쪽 — 끝에는 고정 기둥이 있다 (0.25 였을 때는 기둥 면에서 0.29 m)
    beam_inset_m: float = 0.40
    # 들어갈 거점 안 · 바로 앞에 다른 로봇이 서 있을 때 (위에서 본 것 — scout.robots, 벽에서 assault_robot_near_m 안 · 거점의 세로 범위 안):
    # 그 로봇에서 세로로 assault_robot_gap_m 넘게 떨어진 자리부터 들어가 본다 (지금 자리에서 가까운 것부터). 그런 자리를 만들려고 거점 끝에서
    # beam_inset_tight_m 인 자리도 후보에 넣는다. 2026-11-20 01:50 연습 판: 가운데 거점 안 (5.86, 2.45) 에 다른 팀 로봇이 서 있었는데 후보 자리
    # (y 2.2 ~ 2.6)가 모두 그 로봇에서 0.25 m 안이라 네 번 막힌 뒤 33초 만에 y 2.79 로 들어갔다 (그 로봇에서 0.34 m — 그 자리가 처음부터 답이었다).
    # 주행 담당의 제안 3 (docs/서_있는_UGV_곁의_목표_제안_2026-11-20.md). 드론 사진이 끊겼거나 거점이 화면 밖이면 전과 같다
    assault_robot_near_m: float = 0.90
    assault_robot_gap_m: float = 0.30
    beam_inset_tight_m: float = 0.22
    # 거점 안의 로봇은 드론 화면의 끝(벽 곁)에 있어 「서 있는 로봇」 목록에 들어왔다 빠졌다 한다 (02:24 연습 판: 가운데 거점 안 (5.81, 2.16) 의
    # 로봇이 4 ~ 41초씩 잡혔다 놓였고, 거점으로 떠나는 순간에는 목록에 없어 그 로봇 바로 곁 자리로 들어가다 막혔다).
    # → 요즘 이만큼 안에 서 있는 것으로 본 적이 있는 자리도 함께 본다
    assault_robot_memory_s: float = 60.0
    assault_in_m: float = 0.10  # 들어갈 깊이에서 이 안까지 왔으면 「들어갔다」 (navigation 의 도착 허용과 같다)
    # 거점 안에 들어온 뒤 이만큼 지나도 경기가 끝나지 않으면 그 거점은 목표가 아니다 → 다음 거점으로 (2026-11-19 공지: 거점 종류의 배치가 바뀔 수 있다)
    capture_wait_s: float = 2.0
    # 돌진(MODE_ASSAULT — navigation 이 0.60 m/s 를 지키고 장애물 앞에서 줄이지 않는다)을 쓰는 때:
    #   · 첫 거점(들어가 볼 순서의 맨 앞)으로 갈 때 — 거점 쪽 벽에서 assault_from_m 안(거점 앞 대기 자리가 여기다)에 있으면 어느 거점이든 바로 돌진한다.
    #     M5 는 몇 초 싸움이고 충돌 감점은 없다 (팀장 2026-11-19: 거점으로 갈 때는 돌진이 낫다).
    #   · 그 거점의 바로 앞에 있을 때 — 차체가 거점의 세로 범위(± assault_side_m) 안이고 벽에서 assault_from_m 안.
    # 그 밖(첫 거점이 목표가 아니어서 다른 거점으로 옮겨 갈 때 · 경기장 안쪽에서 늦게 올 때)에는 거점 앞(벽에서 close_from_wall_m)까지 보통 주행으로
    # 간 뒤 돌진한다. 2026-11-19 21:54 연습 판: B-3 거점 안에서 B-1 거점으로 돌진으로 옮겨 가다(3 m, 기둥 둘을 돌아) 4:28 에 B-S2 의 바깥 상자
    # 모서리를 밀었다 (상자가 18도 돌고 10 cm 밀림) — 거점 깊숙한 곳에서 기둥을 돌아 나오며 0.6 m/s 로 크게 휘었다. 보통 주행(0.45 m/s)으로는 1초쯤 더 든다.
    # 모든 구간을 돌진으로 하려면 assault_from_m 을 경기장 가로보다 크게, assault_side_m 을 세로보다 크게 둔다.
    assault_from_m: float = 1.10
    assault_side_m: float = 0.05
    # 들어갈 거점 안 · 바로 앞에 다른 로봇이 서 있을 때는 돌진을 바로 내지 않는다 (2026-11-20 아침 — docs/돌진_기울기_확인_요청 의 「남은 것」 1번을 여기서 피한다).
    # 차체가 그 로봇 쪽으로 기운 채 돌진을 받으면 navigation 이 1초 만에 BLOCKED 로 끝낸다 (가상 주행 60판: 벽을 똑바로 보거나 먼 쪽으로 기울면 60,
    # 그 로봇 쪽으로 5° 48 · 22° 38 · 60 ~ 90° 32. 실차 02:39 판 22번: 60° 기운 채 받은 돌진이 두 번 막혀 그 거점에는 끝내 못 들어갔다).
    # 거점 사이를 옮겨 갈 때 · 다른 자리로 다시 들어갈 때가 그렇다 — 통로를 따라오면 차체가 벽과 나란히, 그 로봇 쪽을 보고 있다.
    # → 그런 거점에서는 먼저 들어갈 자리의 바로 앞(벽에서 close_from_wall_m)에 보통 주행으로 가서 「벽 쪽에서 그 로봇의 먼 쪽으로 stage_lean_deg」 로 서고
    #   (허용 stage_lean_tol_deg — 대기 자리와 같은 규칙), 아래의 「설 준비가 됐다」 가 되면 돌진한다. 이미 그렇게 서 있으면(대기 자리에서 떠날 때) 같은 틱에 바로 돌진한다.
    #   · 준비가 됐다 = 거점 쪽 벽에서 assault_from_m 안 · 세로로 자리에서 assault_guard_side_m 안 · 차체가 벽 쪽에서 그 로봇 쪽으로 assault_guard_toward_deg 까지,
    #     먼 쪽으로 assault_guard_away_deg 까지 기울어 있다.
    #   · 끝없이 기다리지 않는다: navigation 이 그 앞 자리에 도착했다고 알리면(방향을 못 맞췄어도), 또는 「거리 ÷ stage_speed + assault_guard_wait_s」 가 지났고
    #     전 같으면 돌진을 냈을 자리(거점의 바로 앞 · 첫 거점이면 벽 가까이)면 그대로 돌진한다.
    #   · 서 있는 로봇이 있는지 · 어느 쪽으로 기울지는 그 거점의 자리를 고를 때 한 번만 정한다 (목록이 흔들려도 목표가 오락가락하지 않게).
    #     한 자리에 돌진을 냈으면 navigation 이 결과를 알릴 때까지 그대로 둔다.
    #   · 거점 **안에** 서 있는 로봇일 때만 한다 — 그 로봇의 가운데가 벽에서 assault_guard_depth_m 안 (거점 네모의 깊이는 0.40 m).
    #     거점 앞 통로 쪽(벽에서 0.60 ~ 0.90 m)에 서 있는 로봇이면 전의 돌진이 거의 다 한 번에 들어가고(가상 주행 60판 가운데 59판), 그 로봇 곁의
    #     앞 자리까지 보통 주행으로 가는 쪽이 1 ~ 4초 더 걸렸다 → 그때는 전과 같이 한다 (자리의 차례만 바꾼다 — assault_robot_gap_m).
    # 가상 주행 (Mission 을 그대로 돌리고 주행은 navigation 실행기 · 그 로봇의 자리 30가지 × 두 방향 = 60판, 거점의 바로 앞에 들어선 때부터 들어갈 때까지):
    #   통로를 따라 옮겨 올 때 — 전: 54판 들어감(6판은 자리 다섯 곳이 다 막혀 그 거점을 그만둠) · 첫 돌진에 32판 · 가운데값 3.0초 · 가장 늦은 판 10.4초
    #                          새: 60판 · 첫 돌진에 60판 · 가운데값 2.7초 · 가장 늦은 판 4.5초
    #   가운데 대기 자리에서 게시된 끝 거점으로 (게시된 때부터) — 전: 60판 · 첫 돌진에 31판 · 가운데값 6.0초 · 가장 늦은 판 21.3초 / 새: 60판 · 60판 · 5.0초 · 6.9초
    #   전의 첫 돌진이 들어가던 판(60판 가운데 27판)에서는 새 쪽이 0.2 ~ 1.4초 늦다 (서서 도는 시간 — 실차는 더 걸릴 수 있다).
    # 서 있는 로봇이 없는 거점은 전과 똑같다. False = 이 장치를 끔 (전과 같음)
    assault_guard: bool = True
    assault_guard_depth_m: float = 0.50
    assault_guard_side_m: float = 0.12
    assault_guard_toward_deg: float = 2.0
    assault_guard_away_deg: float = 45.0
    # 실차는 서서 도는 데 가상보다 오래 걸린다 (가운데에서 기다리다 끝 거점으로 돌진해 들어가는 데 가상 3.3 ~ 4.8초 · 실차 9.2 ~ 9.8초) → 넉넉히.
    # 못 가는 자리면 navigation 이 5 ~ 8초 안에 스스로 BLOCKED 로 끝낸다 — 이 값은 그래도 끝나지 않을 때의 마지막 문턱이다
    assault_guard_wait_s: float = 6.0
    assault_guard_grace_s: float = 0.25  # 그 앞 자리를 시켰는데 「가는 중」 을 한 번도 못 본 채 이만큼 서 있으면 도착한 것으로 친다 (같은 목표는 창구가 다시 보내지 않는다)
    # 보고
    m3_min_gap_s: float = 2.0  # M3 를 이보다 자주 보내지 않는다
    m3_refresh_s: float = 20.0  # 바뀐 것이 없어도 이만큼마다 다시 보낸다 (서버에 남은 것을 확실히)
    retry_gap_s: float = 2.0
    m1_reask_s: float = 9.0  # 명령문 해석을 다시 묻는 간격 (답이 없거나, LLM 이 아직 확인하지 못했을 때)
    m1_max_asks: int = 12  # 한 경기에서 묻는 횟수의 상한
    ugv_look_gap_s: float = 3.0  # UGV 상태 판독을 이 간격으로
    # M6 모르는 값은 이것으로 친다
    prior_lives: float = 3.5
    prior_ammo: float = 6.0


@dataclass
class Facts:
    now: float  # 흐르는 시계 (초)
    t: float | None = None  # 경기 시작부터 초 (팀이 잰다). 시작 전이면 None
    phase: str = "idle"  # idle · standby · running · paused · report · stop
    control: bool = False  # 지금 로봇이 명령을 받는가 (연습 모드이거나 경기 진행 중)
    slot: str | None = None  # 우리 호출부호 B-1 … (서버가 알려 준다)
    pose: tuple | None = None  # (x, y, yaw) — navigation
    arrived: bool = False  # 지금 목표에 도착해 멈췄는가 — navigation
    boxes: list | None = None  # [(x, y)] 상자 가운데 목록 (fmap 이 없을 때)
    fmap: object | None = None  # 지도(FieldMap) — navigation 의 /team/navigation/map 을 받은 것. 있으면 이것을 쓴다
    order_text: str | None = None  # M1 명령문 원문 — server_bridge
    parsed: dict | None = None  # M1 해석 {"target", "search_sectors", "base"} — llm_worker
    parsed_pending: bool = False  # 위 해석이 낱말 규칙만으로 낸 것이다 (LLM 이 아직 확인하지 못함) → 다시 물어본다
    objective: str | None = None  # 게시된 목표거점 보고값 — server_bridge
    hp: int | None = None
    nodes_ready: bool = True  # 다른 프로그램이 준비됐는가
    # 이번 틱에 온 그림 판독 답 [(tag, pose, pan_deg, pictures 또는 None)]. 정찰 사진의 pictures 는 [(kind, x, cut, 가로 크기, 세로 크기, 확신)]
    # (크기 없이 앞의 셋만 있어도 된다 — 가상 시험). tag 가 ("aim", …) 인 것은 조준 확인 사진의 답이고,
    # pictures 자리에 {"pictures": [{kind, x, y, w, h, conf, cut}], "sensors": [{x, y, w, h, conf}] 또는 None} 이 온다
    answers: list = field(default_factory=list)
    ugv_answers: list = field(default_factory=list)  # 이번 틱에 온 UGV 상태 [{"camp", "callsign", "lives", "ammo"}]
    fire_reply: tuple | None = None  # (message, left_ms) — 이번 틱에 온 발사 답
    sensor_answers: list = field(default_factory=list)  # 이번 틱에 온 수광부 자리 [(tag, 찍을 때의 팬, 틸트, 수광부 목록 또는 None)] — 목록은 SensorSpot 의 값을 담은 dict
    laser_spots: list = field(default_factory=list)  # 이번 틱에 온 레이저 점 확인 [(tag, {"u", "v", …} 또는 None)] — 발사 전후 사진을 견준 것 (창구 watch_laser)
    results: list = field(default_factory=list)  # 이번 틱에 온 보고 결과 [(종류 "ready"·1·3·6, 받아들여졌는가)]
    # 이번 틱에 「못 갔다」 로 끝난 이동 목표 (x, y, 까닭 번호[, 돌 자리가 없어 출발하지 못했는가]) — navigation 의 시한 넘김 · 막힘
    go_failed: tuple | None = None
    no_turn: tuple | None = None  # 이번 틱에 「자리에는 왔는데 도착 방향으로 돌 자리가 없다」 로 끝난 목표 (x, y) — navigation 의 REACHED + detail
    gone: frozenset | None = None  # navigation 이 라이다로 없다고 보아 지도에서 뺀 상자의 번호 (처음 지도의 번호). 모르면 None
    # navigation 이 라이다로 찾았으나 지도에는 넣지 않은 상자의 가운데 [(x, y)] — /team/navigation/lidar_boxes 의 가장 새 목록. 받은 적이 없으면 None
    extra_boxes: list | None = None
    # 위에서 본 UGV 자리(/team/perception/ugvs — perception 이 드론 사진에서 찾은 것, 우리 차 포함)의 요즘 관측들
    # [(그 관측의 때 — now 와 같은 시계, [(x, y, 점수)])] 오래된 것부터. 받은 적이 없으면 None. 같은 관측이 여러 틱에 다시 와도 된다 (한 번만 쓴다)
    ugv_sights: list | None = None


class Ports:
    """시키는 창구. 모두 바로 돌아온다 (기다리지 않는다). 못 보냈으면 False."""

    def go(self, x, y, yaw, yaw_free=False, mode="normal", yaw_tol=None, pos_tol=None): ...  # mode: normal · assault(거점 진입). yaw_tol · pos_tol: 허용(rad · m), 없으면 설정값
    def stop(self): ...
    def aim(self, pan_deg, tilt_deg): ...
    def fire(self): ...
    def shoot(self, tag, pan_deg): ...
    def shoot_aim(self, tag): ...  # 조준 확인: 팬틸트를 건드리지 않고 지금 사진 한 장을 판독에 보낸다 (없는 창구면 사진 조준을 하지 않는다)
    def locate(self, tag): ...  # 지금 사진에서 수광부 자리와 임자(지정표적 · UGV)를 찾아 달라 — 답은 Facts.sensor_answers (없는 창구면 상대 UGV 를 따라가지 않는다)
    def watch_laser(self, tag, near=None): ...  # 다음 발사의 레이저 점이 사진 어디에 찍히는지 봐 달라 (near = 식으로 계산한 레이저 점, 그 둘레만 본다) — 발사 직전에 부른다. 답은 Facts.laser_spots (없는 창구면 맞았는지 보지 않는다)
    def shoot_pass(self, tag): ...  # 지나가며 찍기: 팬틸트를 건드리지 않고, 판독 줄이 비어 있을 때만 사진 한 장을 보낸다 (없는 창구면 하지 않는다)
    def look_ugvs(self): ...
    def analyze_order(self, text): ...
    def set_ready(self): ...
    def report(self, kind, payload): ...


class Mission:
    def __init__(self, params=None, arena=None, base_kinds=None, cam=None):
        self.p = params or MissionParams()
        self.arena = arena or A.ArenaPlan()
        self.base_kinds = base_kinds or A.DEFAULT_BASE_KINDS
        self.cam = cam or CameraParams()
        self.fmap = self.grid = self.book = self.scout = None
        self.camp = self.foe = None
        self.row = None
        self.task = "wait"
        self.log = []  # (경기 시각, 글)
        # 보고
        self.ready_ok = False
        self._ready_at = -1e9
        self.designated = None
        self._order_at = -1e9  # 다음에 명령문 해석을 물어도 되는 시각
        self._order_asks = 0
        self._sent = {}  # 종류 → 서버가 받아들인 마지막 내용
        self._flying = {}  # 종류 → (보낸 내용, 보낸 시각)
        self._try_at = {}  # 종류 → 다음에 보내도 되는 시각
        self._m3_at = -1e9
        self.m1 = None
        # 정찰
        self.scout_done_at = None
        self._shot_wait = None  # 찍으려고 멈춰 기다리는 중인 사진 tag
        self._scout_goal = None  # ((x, y, 방향), 보낸 시각) — navigation 에 시켜 둔 정찰 자리
        self._trail = []  # 지나온 길 [(x, y)] — trail_step_m 간격
        self._fails = []  # 잇달아 「못 갔다」 로 끝난 때의 (시각, x, y)
        self._escape = None  # 빠져나가는 중: {"goal": (x, y), "since": 시각[, "straight": 차체 방향 — 돌지 않고 곧게]}
        self._no_turn = None  # (x, y, 시각) navigation 이 「도착 방향으로 돌 자리가 없다」 고 알린 자리
        self._pass_at, self._pass_n = -1e9, 0  # 지나가며 찍기: 마지막으로 찍은 때 · 장 수
        self._pan_homed = False  # 경기 첫머리에 팬을 가운데로 보냈는가
        self._pan_fwd_for = None  # 카메라를 정면으로 돌려 둔 정찰 목표 (목표마다 한 번)
        self._pan_sent = None  # 정찰하며 마지막으로 보낸 팬 (각도, 보낸 때) — 미리 돌리기 · 따라 돌리기를 너무 자주 보내지 않게
        self._pan_at = None  # 이때부터 팬 각도를 안다 (한 번이라도 보낸 뒤). None = 아직 모른다
        self._pass_info = {}  # 지나가며 찍은 사진 번호 → (찍은 때, 그때 팬 각도를 알았는가)
        self._pass_wait = []  # 답이 온 지나가며 찍은 사진 [(tag, pose, pan, pictures)] — 찍은 뒤의 방향까지 보고 쓴다 (_pass_votes)
        self._lit_due = None  # 발사 뒤 레이저가 켜진 동안의 사진을 찍을 때
        self.escapes = 0
        # M2
        self.target_face = None
        self.target_shots = 0
        self.target_done = False
        self._strike = None  # {"spot", "since", "aimed_at", "pan", "tilt", "geo": 계산한 각도, "aim": 사진 조준 상태(_aim_state)}
        self._still = []  # 멈춰 있는 동안의 최근 위치들 (조준 각도는 이것의 평균으로 계산한다)
        self._arrived_since = None  # 「서 있음」(도착했거나 목표가 없음)이 이어진 시작 시각
        self._stage_y = None  # [거점, 대기 자리의 세로 자리] — 대기하러 떠날 때 한 번 정한다 (그 거점에 서 있는 로봇을 피해 — _stage)
        self._stage_sent = None  # 마지막으로 보낸 대기 목표 (x, y, 방향) — 거점 진입의 「자리 앞 목표」 가 이것과 같으면 navigation 이 이미 끝낸 목표다 (_guard_spot)
        self._dash_base = None  # 돌진 목표를 한 번 낸 거점 — 그 거점에서는 「거점 앞까지 보통 주행」 으로 되돌아가지 않는다 (_assault 의 설명)
        self._assault_goal = None  # [거점, 들어갈 자리 후보들, 지금 후보 번호, 못 간 횟수] — 거점마다 한 번 정하면 바꾸지 않는다
        self._assault_plan = None  # [게시된 목표, 들어가 볼 거점 순서, 지금 몇 번째]
        self._assault_asked = None  # [(x, y), 시킨 시각, 그 뒤 「가는 중」 을 봤는가] — navigation 에 시켜 둔 거점 안 자리
        # [거점, 그 거점에 서 있던 로봇들 [(x, y)], 지금 자리의 상태 또는 None] — 서 있는 로봇이 있는 거점에서만 (없으면 None). 자리를 고를 때 한 번 정한다 (assault_guard 의 설명)
        self._assault_guard = None
        self._entered_at = None  # 지금 거점 안에 들어온 시각
        self._recent = []  # 최근 몇 초의 (시각, 위치) — navigation 의 「도착」 과 상관없이 실제로 서 있는지 본다
        self._staging = False
        self._last_fire = -1e9
        self._laser_until = -1e9
        self._fire_asked = False
        self._fire_asked_at = -1e9
        self._fire_for = "target"  # 지금 기다리는 발사 답이 어디에 쏜 것인가: "target" 지정표적 · "ugv" 상대 UGV
        self.target_hits = 0  # 레이저 점이 수광부에 닿은 것으로 보인 발 수
        self._good_shots = 0  # 조준이 허용 안에 든 채 쐈는데 닿았는지는 볼 수 없었던 발 수
        self._spot = None  # 기다리는 레이저 점 확인: {"tag", "until", "rec": 수광부, "laser": 식의 레이저 점, "good": 조준이 허용 안이었나}
        self._laser_fix = (0.0, 0.0)  # 잰 레이저 점 − 식의 레이저 점 (픽셀). 빗나간 발에서 배워 다음 조준에 더한다
        self._strike_tries = 0  # 표적을 못 찾고 물러난 횟수
        self._strike_wait = False  # 물러났다 → 정찰을 마친 뒤에 다시 한다
        self._strike_bad = []  # 물러난 쏠 자리들 (다시 할 때는 다른 자리에서)
        if isinstance(self.p.track, dict):  # 설정 파일에서 온 값
            self.p.track = TrackParams(**self.p.track)
        self._tracker = UgvTracker(self.p.track, self.p.aim, self.p.aim.deg_px(self.cam.hfov_deg))
        self._engage = None  # 상대 UGV 를 따라가는 중의 상태
        self._fire_sign = ""  # 지금 기다리는 발사 답이 쏜 상대 팀의 호출부호 (모르면 "")
        self._ugv_shot = None  # 상대 UGV 에 쏜 한 발을 지켜보는 중: {"at", "sign", "n": 쏜 뒤 본 사진 수, "off": 그 가운데 수광부를 벗어난 수}
        self._foe_seen = (-1e9, [])  # (시각, read_ugvs 가 마지막으로 읽은 상대 호출부호들)
        self.ugv_hits = {}  # 호출부호("" = 모름) → 우리가 맞힌 것으로 본 횟수
        self._watch_at, self._watch_i = None, 0  # 서서 살필 때: 팬을 마지막으로 돌린 때와 그 차례 (watch_pans_deg)
        self._m6, self._m6_posts = None, None  # M6 순찰: 지금 자리 · 방향 · 서기 시작한 때 / 순찰 자리 목록 (지도가 생긴 뒤 한 번 정한다)
        self._locate_no = 0
        self.ugv_shots = 0
        # M5
        self.stage_base = None
        self._eta = 30.0
        self._eta_at = -1e9
        # M6
        self.ugvs = {}  # 호출부호 → {"lives", "ammo", "t"}
        self._ugv_at = -1e9
        self._restored_book = None  # 다시 켜졌을 때: 지도가 오면 면 기록에 되살릴 것
        # 지도에 없는 상자 (navigation 이 라이다로 찾은 것 — _extra_boxes)
        self._extra_seen = []  # 후보 [{"xy": (x, y), "since": 목록에 처음 나온 시각, "why": 넣지 않는 까닭을 적었으면 그 글}]
        self._extras = {}  # 받아들인 상자: 번호 → {"xy": (x, y), "seen": 마지막으로 목록에 있던 시각, "lost": 지금 없는 것으로 치는가}
        self._extra_gone = set()  # 받아들인 상자 가운데 지금 없는 것으로 치는 것의 번호
        self._extra_off = False  # 처리하다 예외가 나서 이 경기에서는 더 받지 않는다
        self._parked = Standing()  # 위에서 본 UGV 자리 → 서 있는 다른 로봇 (정찰의 실시간 순서가 쓴다 — _robots)
        self._parked_mark = frozenset()  # 마지막으로 기록에 남긴 서 있는 로봇들의 자리 (0.2 m 눈금)
        self._stood = {}  # 서 있는 것으로 본 적이 있는 자리: (0.2 m 눈금 x, y) → (x, y, 마지막으로 그렇게 본 때) — 거점 진입이 쓴다 (assault_robot_memory_s)
        self.photo_trace = None  # 사진 한 장을 어떻게 읽었는지 받아 적을 함수 (node.py 가 파일에 쓴다)

    # ================================================================== 한 틱
    def tick(self, f, ports):
        self._build(f)
        self._absorb(f)
        if f.pose is not None and f.arrived:
            self._still = (self._still + [f.pose])[-10:]
        else:
            self._still = []
        if f.pose is not None:
            self._recent = [(when, q) for when, q in self._recent if f.now - when <= 8.0] + [(f.now, f.pose)]
        else:
            self._recent = []
        if self._pass_wait:
            self._pass_votes(f)
        if not f.arrived:
            self._arrived_since = None
        elif self._arrived_since is None:
            self._arrived_since = f.now
        if f.pose is not None and (not self._trail or math.hypot(f.pose[0] - self._trail[-1][0], f.pose[1] - self._trail[-1][1]) >= self.p.trail_step_m):
            self._trail = (self._trail + [(f.pose[0], f.pose[1])])[-80:]
        self._reports(f, ports)
        if self._lit_due is not None and f.now >= self._lit_due:
            # 팀 테스트 포털은 M2 명중을 알려 주지 않는다 → 레이저가 켜진 사진을 남겨 두고 본다. 판독 줄이 비었을 때만, 2.5초 안에 못 찍으면 그만둔다
            shoot = getattr(ports, "shoot_pass", None)
            if shoot is None or f.now - self._lit_due > 2.5 or shoot(("lit", self.target_shots)):
                self._lit_due = None
        if f.phase == "report" or not f.control or f.phase not in ("running", "idle"):
            self.task = "report" if f.phase == "report" else "wait"
            return self.task
        if self.fmap is None or f.pose is None or f.t is None:
            self.task = "wait-map" if self.fmap is None else "wait-pose"
            return self.task
        if not self._pan_homed and getattr(ports, "aim", lambda pan, tilt: False)(0.0, 0.0):
            # 경기 첫머리(다시 켜졌을 때도)에 팬틸트를 가운데로 한 번 보낸다. 팬 각도는 읽을 수 없어, 한 번도 보내지 않았으면 「0°」 라는 값은 믿을 수 없다
            self._pan_homed, self._pan_at = True, f.now + self.p.pan_home_s
        if self._stuck(f, ports):
            return self.task
        # M6 를 하는 판에서는 M5(4:00 전 거점 앞 대기 · 게시된 거점 진입)를 하지 않고 경기 끝까지 M6 를 한다 (m6_instead_of_m5, 2026-11-20 사용자 결정)
        m6_only = self.m6_replaces_m5(f)
        if f.objective and not m6_only:
            return self._assault(f, ports)
        if not m6_only and (self._staging or self._time_to_stage(f)):
            self._staging = True  # 한 번 대기하러 가기 시작하면 되돌아가지 않는다
            return self._stage(f, ports)
        if self._want_strike(f):
            task = self._do_strike(f, ports)
            if task:
                # 지정표적을 쏘는 동안에는 정찰 자리로 가던 목표가 쏠 자리의 목표로 바뀐다 → 「그 정찰 자리를 navigation 에 시켜 두었다」 는 표시를 지운다.
                # 그대로 두면 사격을 마치고 정찰로 돌아온 첫 틱에 「시켰고 · 서 있고(도착) · 자리에서 멀다」 로 읽어, 가 보지도 않은 그 자리를
                # 「가지 못함」 으로 맨 뒤에 미룬다 (2026-11-19 18:08 모의 경기 6차: 13.2초에 시킨 (5.23, 3.23) 을 28.0초에 사격이 끝나자마자 미뤘다 —
                # 지나가며 찍은 사진으로 표적을 찾으면 늘 「가는 중」 에 사격이 끼어든다)
                self._scout_goal = None
                return task
        if self.scout_done_at is None:
            return self._scout(f, ports)
        # 정찰(M3)이 다 끝났다 → M6: 우리 진영으로 돌아가 상대 UGV 를 찾아 살피고(진영 · 호출부호 · 목숨) 멈춘 것을 쏜다.
        # 단, 정찰이 3분 안에 끝났을 때만 한다 (m6_scout_done_by_s, 2026-11-20 사용자 결정) — 더 늦게 끝났으면 그 자리에 서 있는다 (살피지도 쏘지도 않는다)
        if not self.p.m6_hunt:
            # M6 사냥을 껐다 → main 의 기존 흐름 그대로: 그 자리에 서서 UGV 상태만 읽는다 (M5 는 tick 의 위쪽에서 전처럼 먼저 본다)
            if f.now - self._ugv_at >= self.p.ugv_look_gap_s and ports.look_ugvs():
                self._ugv_at = f.now
            ports.stop()
            self.task = "rest"
            return self.task
        if not self.m6_allowed(f):
            ports.stop()
            self.task = "rest"
            return self.task
        if self.p.m6_patrol and self._m6_post_list():
            return self._m6_patrol(f, ports)
        self.task = "engage" if self._look_and_engage(f, ports) else "rest"
        return self.task

    def m6_allowed(self, f):
        """M6(상대 UGV 살피기 · 순찰 · 쏘기)를 해도 되는가: 정찰(M3)이 경기 시각 m6_scout_done_by_s(3분) 안에 다 끝났다.

        4:00 의 목표거점 게시까지 1분 넘게 남기고 정찰을 마쳤을 때만 M6 를 한다는 뜻이다 (2026-11-20 사용자 결정). 정찰이 그보다 늦게 끝났으면
        M6 를 하지 않고 서 있다가 거점 앞 대기(M5)로 넘어간다. 한 번 허락되면 경기 끝까지 계속한다 (m6_replaces_m5 — M5 는 하지 않는다).
        정찰이 다시 시작됐다가 끝나면(지도에 없던 상자) 그 끝난 시각으로 다시 따진다.
        """
        return self.scout_done_at is not None and self.scout_done_at <= self.p.m6_scout_done_by_s

    def m6_replaces_m5(self, f):
        """이 판은 M5(거점 앞 대기 · 거점 진입) 대신 M6 를 하는가: M6 가 허락됐고(m6_allowed), 실제로 할 M6 가 켜져 있다 (2026-11-20 사용자 결정).

        이미 대기하러 떠났으면(_staging) 되돌리지 않는다. m6_instead_of_m5 를 끄면 전처럼 M5 가 먼저다 (M6 는 대기하러 떠날 때까지만).
        """
        p = self.p
        return p.m6_hunt and p.m6_instead_of_m5 and not self._staging and (p.m6_patrol or p.engage_ugvs) and self.m6_allowed(f)

    def _look_and_engage(self, f, ports):
        """서서 상대 UGV 를 살핀다: read_ugvs 로 읽고, 보이는 상대의 수광부를 겨누고(멈춰 있으면 쏜다), 겨눌 것이 없으면 팬을 돌려 본다. 겨누는 중이면 True.

        그림 모델(M3)은 부르지 않는다 — read_ugvs · locate_sensors 둘 다 perception 의 UGV 부품 모델만 쓴다. 팬틸트가 돌고 있을 때의 사진은 흔들려서 읽지 않는다.
        """
        if f.now - self._ugv_at >= self.p.ugv_look_gap_s and getattr(ports, "aim_ready", lambda: True)() and ports.look_ugvs():
            self._ugv_at = f.now
        ports.stop()
        if self.p.engage_ugvs and f.arrived and self.engage_ugv(f, ports):
            return True
        if f.arrived:
            self._watch(f, ports)
        return False

    def _m6_post_list(self):
        """M6 순환 경로의 지점들 [(x, y)] — 우리 진영 안을 한 바퀴 도는 차례. 지도가 없거나 지점이 둘 미만이면 빈 목록.

        정찰(M3)의 격자를 그대로 쓴다: 우리 벽에서 폭의 m6_route_x_fracs 만큼 떨어진 두 세로줄 위, 높이의 m6_route_y_fracs 지점을
        가장 가까운 빈 칸으로 옮기고(grid.nearest_free — 상자 · 벽 · 우리 거점을 피한다), 우리 진영 밖으로 밀려난 것 · 서로 붙은 것 ·
        앞 지점에서 길이 없는 것(grid.plan)은 버린다. 차례: 벽 쪽 줄을 따라 올라갔다가 가운데 쪽 줄을 따라 내려온다.
        """
        if self._m6_posts is None:
            self._m6_posts = []
            if self.fmap is not None and self.grid is not None and self.camp in ("BLUE", "RED"):
                p, w, h = self.p, self.fmap.width, self.fmap.height
                xs = [w * float(fx) if self.camp == "BLUE" else w * (1.0 - float(fx)) for fx in p.m6_route_x_fracs]
                ys = [h * float(fy) for fy in p.m6_route_y_fracs]
                ring = []
                for k, x in enumerate(xs):
                    ring += [(x, y) for y in (ys if k % 2 == 0 else list(reversed(ys)))]
                for x, y in ring:
                    spot = self.grid.nearest_free(x, y)
                    if spot is None or A.side_of(self.fmap, spot[0]) != self.camp:
                        continue
                    if any(math.hypot(spot[0] - q[0], spot[1] - q[1]) <= 0.5 for q in self._m6_posts):
                        continue
                    if self._m6_posts and not self.grid.plan(self._m6_posts[-1], (spot[0], spot[1])):
                        continue
                    self._m6_posts.append((spot[0], spot[1]))
                if len(self._m6_posts) < 2:
                    self._m6_posts = []
        return self._m6_posts

    def _m6_patrol(self, f, ports):
        """우리 진영을 멈추지 않고 돌며 상대 UGV 의 수광부를 찾고, 보이면 그 자리에 서서 겨눠 쏜 뒤 다시 돈다 (2026-11-20 사용자 결정).

        - 도는 중("m6 go"): 경로의 다음 지점으로 간다 (방향은 요구하지 않는다 — 서지 않고 지나간다). 팬은 가운데에 두고, m6_scan_gap_s 마다
          locate_sensors 를, ugv_look_gap_s 마다 read_ugvs 를 부른다. 드론 영상은 쓰지 않는다.
        - 상대 수광부가 찍혔다: 선다("m6 watch"). 달리며 찍은 사진의 각도는 쓰지 않고 선 뒤에 처음부터 다시 본다 (_look_and_engage).
          멈췄거나 조금씩 움직이는 상대는 계속 겨눠 쏘고(최대 m6_hold_max_s), 움직이는 상대는 m6_moving_wait_s 만 기다리고, 많이 움직이는 상대는
          기다리지 않는다. 멈춘 상대가 팬 범위 끝에 있으면 차체를 그쪽으로 돌린다 (m6_turn_to_target, 한 번).
        - m6_hold_s 동안 겨눌 것이 없으면 다시 돈다. 같은 상대에 곧바로 다시 서지 않게 m6_resight_s 동안은 찍혀도 서지 않는다.
        - 쏠 수 없을 때(발 수를 다 썼다 · 지정표적을 아직 쏴야 한다)는 서지 않고 계속 돌며 read_ugvs 로 상태만 읽는다.
        """
        p, route = self.p, self._m6_post_list()
        m = self._m6
        if m is None:
            near = min(range(len(route)), key=lambda i: math.hypot(route[i][0] - f.pose[0], route[i][1] - f.pose[1]))
            m = self._m6 = {"i": near, "go_at": f.now, "hold": None, "scan": None, "scan_at": -1e9, "cool": -1e9, "homed": False, "inspect": None, "seen": []}
            self._say(f, "M6: 우리 진영을 돌며 상대 UGV 를 찾는다 (경로 지점 %d곳)" % len(route))
        lit = f.now < self._last_fire + p.laser_on_s
        aim = getattr(ports, "aim", lambda pan, tilt: False)
        if m.get("inspect") is not None:
            return self._m6_inspect(f, ports, m)
        hold = m["hold"]

        if hold is None:
            # ---------------------------------------------------------------- 도는 중
            self.task = "m6 go"
            if not m["homed"]:
                m["homed"] = bool(aim(0.0, 0.0))  # 팬을 가운데로 (앞을 본다)
            x, y = route[m["i"]]
            failed = f.go_failed is not None and math.hypot(f.go_failed[0] - x, f.go_failed[1] - y) <= 0.3
            if math.hypot(f.pose[0] - x, f.pose[1] - y) <= p.m6_reach_m:
                m["i"], m["go_at"] = (m["i"] + 1) % len(route), f.now
                x, y = route[m["i"]]
            elif failed or f.now - m["go_at"] >= p.m6_go_give_up_s:
                self._say(f, "M6: 지점 (%.2f, %.2f) 에 가지 못함 → 다음 지점" % (x, y))
                m["i"], m["go_at"] = (m["i"] + 1) % len(route), f.now
                x, y = route[m["i"]]
            if m.get("goal") != (x, y):
                # 방향 값은 지점마다 한 번만 정한다 — 틱마다 바뀌면 같은 지점을 navigation 에 거듭 새 목표로 보내게 된다 (방향은 요구하지 않는다)
                m["goal"], m["yaw"] = (x, y), math.atan2(y - f.pose[1], x - f.pose[0])
            ports.go(x, y, m["yaw"], yaw_free=True, pos_tol=max(0.10, p.m6_reach_m - 0.10))
            ready = getattr(ports, "aim_ready", lambda: True)()
            if f.now - self._ugv_at >= p.ugv_look_gap_s and ready and ports.look_ugvs():
                self._ugv_at = f.now
            locate = getattr(ports, "locate", None)
            if locate is None or self.foe is None or not p.engage_ugvs:
                return self.task
            sighted = False
            if m["scan"] is not None:
                got = next((a for a in f.sensor_answers if a[0] == m["scan"]), None)
                if got is not None:
                    dead = tuple(s for s in self.ugvs if self.lives_of(s) == 0)
                    sighted = bool(foe_spots(got[3], self.foe, p.track, p.engage_unknown_camp, dead))
                    m["scan"] = None
                elif f.now - m["scan_at"] > p.locate_timeout_s:
                    m["scan"] = None
            owed = self.target_face is not None and not self.target_done and self.target_shots < p.target_shots
            if sighted and self.ugv_shots < p.ugv_shots and not owed and f.now >= m["cool"]:
                ports.stop()
                m["hold"] = {"since": f.now, "at": f.now, "xy": (f.pose[0], f.pose[1]), "turn": None, "turned": False, "asked": False,
                             "target": None, "hits0": self.ugv_hits.get("", 0)}
                self._engage, self._watch_at = None, None
                self._say(f, "M6: 상대 UGV 의 수광부가 보인다 → (%.2f, %.2f) 에 서서 겨눈다" % (f.pose[0], f.pose[1]))
                self.task = "m6 watch"
                return self.task
            if m["scan"] is None and ready and f.now - m["scan_at"] >= p.m6_scan_gap_s:
                tag = ("hunt", self._locate_no)
                if locate(tag):
                    self._locate_no += 1
                    m["scan"], m["scan_at"] = tag, f.now
            return self.task

        # -------------------------------------------------------------------- 서서 겨누는 중
        if hold["turn"] is not None:
            # 팬 범위 끝에 멈춰 있는 표적 쪽으로 차체를 돌리는 중
            facing = abs(wrap(f.pose[2] - hold["turn"])) <= math.radians(25.0)
            if f.arrived and (facing or hold["asked"]) or f.now - hold["go_at"] >= p.m6_go_give_up_s:
                hold.update(turn=None, since=f.now)
                self._watch_at, self._watch_i = f.now, 0  # 표적 쪽으로 돌았다 → 팬은 가운데에 둔 채로 다시 찾는다
            else:
                ports.go(hold["xy"][0], hold["xy"][1], hold["turn"], yaw_tol=math.radians(20.0), pos_tol=0.30)
                hold["asked"] = True
                self.task = "m6 go"
                return self.task
        engaged = self._look_and_engage(f, ports)
        self.task = "engage" if engaged else "m6 watch"
        if engaged:
            e = self._engage or {}
            kind, seen_for = e.get("kind"), f.now - (e.get("since") if e.get("since") is not None else f.now)
            if not p.engage_still_only:
                worth = True
            else:
                worth = kind in ("still", "creep") or lit or (kind in ("moving", "new") and seen_for < p.m6_moving_wait_s)
            if worth and f.now - hold["at"] < p.m6_hold_max_s:
                hold["since"] = f.now  # 겨누는 동안에는 머문다
            seen = self._tracker.aim_point(f.now, 180.0, 90.0)
            if seen is not None and seen[2] in ("still", "creep"):
                # 멈춘 표적이 어디 있는지 적어 둔다 (본 때 · 우리 자세 · 팬 · 거리 어림 · 호출부호) — 다 쏜 뒤 가까이 돌아가 읽을 때 쓴다 (_m6_inspect)
                hold["target"] = (f.now, tuple(f.pose), seen[3], self._tracker.range_m, e.get("sign") or "")
            point =self._tracker.aim_point(f.now, PAN_LIMIT_DEG, TILT_LIMIT_DEG) if p.m6_turn_to_target and not hold["turned"] and not lit else None
            if point is not None and point[2] in ("still", "creep") and abs(point[3]) > p.m6_turn_at_pan_deg and self.ugv_shots < p.ugv_shots:
                # 멈춘 표적이 팬 범위 끝(또는 밖)에 있다 → 차체를 그쪽으로 돌려 가운데에서 겨눈다 (팬 + 는 오른쪽 = 방향각이 줄어드는 쪽). 한 번만
                hold.update(turn=wrap(f.pose[2] - math.radians(point[3])), turned=True, go_at=f.now, asked=False)
                self._engage = None
                aim(0.0, 0.0)
                self._say(f, "M6: 멈춘 상대가 팬 %.0f° 에 있다 → 차체를 그쪽으로 돌린다" % point[3])
                self.task = "m6 go"
                return self.task
        out_of_shots = self.ugv_shots >= p.ugv_shots
        if not lit and self._ugv_shot is None and (f.now - hold["since"] >= p.m6_hold_s or out_of_shots):
            m.update(hold=None, go_at=f.now, cool=f.now + p.m6_resight_s, scan=None, homed=False, goal=None)
            self._engage = None
            if self._m6_inspect_start(f, ports, m, hold):
                self.task = "m6 around"
        return self.task

    # ---------------------------------------------------------------------- 멈춘 상대에게 가까이 돌아가 읽기
    def _m6_inspect_start(self, f, ports, m, hold):
        """방금 겨눈 멈춘 상대의 호출부호 · 목숨을 아직 모르면, 그 상대의 곁과 뒤로 돌아가 가까이에서 읽기로 한다. 시작했으면 True.

        2026-11-20 07시 연습 판: 멈춘 상대를 세 발 다 맞혔지만 쏜 거리(차체가 사진에 120 픽셀쯤)에서는 read_ugvs 가 진영만 읽고 호출부호 · 목숨을
        한 번도 못 읽어, 누구를 맞혔는지 몰라 M6 순위에 넣지 못했다. 덮개의 띠는 가까워야(차체 200 픽셀 넘게) 읽히고, LED 바는 한쪽(뒤)에만 보인다.

        상대의 자리: 우리 자세 · 팬 각도 · 거리 어림(차체가 찍힌 폭)으로 계산하고, 위에서 본 「서 있는 로봇」 이 그 곁(1 m 안)에 있으면 그 자리를 쓴다.
        갈 자리: 상대에서 m6_inspect_range_m 떨어진 원 위의 옆 → 뒤(우리 반대편) → 다른 옆. 빈 칸이고 길이 있는 것만.
        """
        p, target = self.p, hold.get("target")
        if not p.m6_inspect or target is None or f.now - target[0] > 30.0 or self.grid is None:
            return False
        if f.t is not None and f.t > p.m6_inspect_until_s:
            return False
        _at, pose, pan, dist, sign = target
        sign = sign or self._one_foe_sign(f)
        if sign and self.ugvs.get(sign, {}).get("lives_seen"):
            return False  # 누구인지도, 목숨도 이미 읽었다
        dist = max(0.4, min(4.0, dist or 1.5))
        way = pose[2] - math.radians(pan)  # 팬 + 는 오른쪽 = 방향각이 줄어드는 쪽
        fx, fy = pose[0] + dist * math.cos(way), pose[1] + dist * math.sin(way)
        robots = [q for q in (getattr(self.scout, "robots", None) or []) if math.hypot(q[0] - fx, q[1] - fy) <= 1.0]
        if robots:
            fx, fy = min(robots, key=lambda q: math.hypot(q[0] - fx, q[1] - fy))[:2]
        if any(math.hypot(q[0] - fx, q[1] - fy) <= 1.0 for q in m["seen"]):
            return False  # 이 상대는 이미 한 번 돌아가 봤다
        back = math.atan2(f.pose[1] - fy, f.pose[0] - fx)  # 상대에서 우리 쪽
        cands = {}
        for name, turn in (("left", 90.0), ("rear", 180.0), ("right", -90.0)):
            a = back + math.radians(turn)
            want = (fx + p.m6_inspect_range_m * math.cos(a), fy + p.m6_inspect_range_m * math.sin(a))
            spot = self.grid.nearest_free(want[0], want[1])
            if spot is None or math.hypot(spot[0] - want[0], spot[1] - want[1]) > 0.30 or math.hypot(spot[0] - fx, spot[1] - fy) < 0.55:
                continue
            path = self.grid.plan(f.pose[:2], (spot[0], spot[1]))
            if not path:
                continue
            cands[name] = (spot[0], spot[1], math.atan2(fy - spot[1], fx - spot[0]), path_length(f.pose[:2], path))
        if not cands:
            return False
        sides = sorted((k for k in ("left", "right") if k in cands), key=lambda k: cands[k][3])  # 가까운 옆부터 → 뒤 → 다른 옆
        order = sides[:1] + (["rear"] if "rear" in cands else []) + sides[1:]
        m["seen"].append((fx, fy))
        m["inspect"] = {"foe": (fx, fy), "cands": [cands[k][:3] for k in order], "names": order, "k": 0, "since": None, "go_at": f.now, "start": f.now,
                        "asked": False, "sign": sign, "pending": self.ugv_hits.get("", 0) - hold.get("hits0", 0), "lives": [], "loose": []}
        getattr(ports, "aim", lambda pan, tilt: False)(0.0, 0.0)
        self._say(f, "M6: 멈춘 상대 (%.2f, %.2f) 의 %s 를 아직 모른다 → 가까이 돌아가 읽는다 (%s)" % (
            fx, fy, "호출부호 · 목숨" if not sign else "목숨", " → ".join({"left": "왼쪽 옆", "rear": "뒤", "right": "오른쪽 옆"}[k] for k in order)))
        return True

    def _m6_inspect(self, f, ports, m):
        """멈춘 상대의 옆 · 뒤로 돌아가 가까이에서 read_ugvs 로 호출부호 · 목숨을 읽는다 ("m6 around" 가는 중 · "m6 read" 서서 읽는 중).

        호출부호가 읽히면 누구인지 몰라 미뤄 둔 명중(pending)을 그 팀에 넣고, 호출부호 없이 읽힌 목숨도 그 팀의 것으로 적는다 (이 상대만 보고 있으므로).
        호출부호와 목숨 두 번을 읽었거나, 자리를 다 돌았거나, m6_inspect_max_s 가 지나면 다시 순환 경로로 돌아간다.
        """
        p, ins = self.p, m["inspect"]
        for u in f.ugv_answers:
            if (u.get("camp") or "") != self.foe:
                continue
            if u.get("callsign") and not ins["sign"]:
                ins["sign"] = u["callsign"]
                self._say(f, "M6: 가까이에서 읽은 호출부호 %s" % ins["sign"])
            if u.get("lives", -1) >= 0:
                ins["lives"].append(int(u["lives"]))
                if not u.get("callsign"):
                    ins["loose"].append(int(u["lives"]))  # 호출부호 없이 읽힌 목숨 — 누구인지 알게 되면 그 팀의 것으로 적는다
        if ins["sign"]:
            if ins["pending"] > 0:
                self._claim_hits(f, ins["sign"], ins["pending"])
                ins["pending"] = 0
            for lives in ins["loose"]:
                self._note_ugv(f, ins["sign"], lives, -1)
            ins["loose"] = []
        done = bool(ins["sign"]) and len(ins["lives"]) >= 2
        if done or ins["k"] >= len(ins["cands"]) or f.now - ins["start"] >= p.m6_inspect_max_s:
            self._say(f, "M6: 가까이 읽기 끝 — 호출부호 %s · 목숨 판독 %d번%s" % (ins["sign"] or "못 읽음", len(ins["lives"]), "" if done else " (다 읽지 못함)"))
            m.update(inspect=None, go_at=f.now, cool=f.now + p.m6_resight_s, scan=None, homed=False, goal=None)
            self.task = "m6 go"
            return self.task
        ready = getattr(ports, "aim_ready", lambda: True)()
        if f.now - self._ugv_at >= p.m6_inspect_read_gap_s and ready and ports.look_ugvs():
            self._ugv_at = f.now
        x, y, heading = ins["cands"][ins["k"]]

        def next_spot():
            ins.update(k=ins["k"] + 1, since=None, go_at=f.now, asked=False)

        if ins["since"] is None:
            self.task = "m6 around"
            at = math.hypot(f.pose[0] - x, f.pose[1] - y) <= 0.30
            facing = abs(wrap(f.pose[2] - heading)) <= math.radians(25.0)
            failed = f.go_failed is not None and math.hypot(f.go_failed[0] - x, f.go_failed[1] - y) <= 0.3
            if at and f.arrived and (facing or ins["asked"]):
                ins["since"] = f.now
            elif failed or f.now - ins["go_at"] >= p.m6_inspect_go_s:
                self._say(f, "M6: 상대의 %s 자리 (%.2f, %.2f) 에 가지 못함" % ({"left": "왼쪽 옆", "rear": "뒤", "right": "오른쪽 옆"}[ins["names"][ins["k"]]], x, y))
                next_spot()
            else:
                ports.go(x, y, heading, yaw_tol=math.radians(20.0), pos_tol=0.20)
                ins["asked"] = True
            return self.task
        self.task = "m6 read"
        ports.stop()
        if f.now - ins["since"] >= p.m6_inspect_read_s:
            next_spot()
        return self.task

    def _claim_hits(self, f, sign, count):
        """누구인지 몰라 미뤄 둔 명중 count 번을 그 팀의 것으로 옮긴다 (ugv_hits 의 "" → sign) — 목숨의 상한도 그만큼 낮춘다."""
        count = min(int(count), self.ugv_hits.get("", 0))
        if count <= 0 or not sign:
            return
        self.ugv_hits[""] -= count
        if self.ugv_hits[""] <= 0:
            del self.ugv_hits[""]
        self.ugv_hits[sign] = self.ugv_hits.get(sign, 0) + count
        old = self.ugvs.get(sign, {})
        cap = max(0, 4 - self.ugv_hits[sign])
        cap = cap if old.get("cap") is None else min(cap, old["cap"])
        self.ugvs[sign] = dict(old, lives=old.get("lives"), lives_seen=list(old.get("lives_seen") or []), ammo=old.get("ammo"), t=f.t, cap=cap)
        self._say(f, "앞서 맞힌 %d발은 %s 였다 → 목숨을 %s 로 낮춰 센다" % (count, sign, self.lives_of(sign)))

    def _watch(self, f, ports):
        """서 있는 동안 팬을 차례로 돌려 상대 UGV 를 찾는다 (watch_pans_deg). 겨누는 표적이 있을 때 · 레이저가 켜져 있을 때는 돌리지 않는다."""
        p = self.p
        if not p.watch_pans_deg or f.now < self._last_fire + p.laser_on_s:
            return
        if self._watch_at is not None and f.now - self._watch_at < p.watch_dwell_s:
            return
        nxt = 0 if self._watch_at is None else (self._watch_i + 1) % len(p.watch_pans_deg)
        if getattr(ports, "aim", lambda pan, tilt: False)(float(p.watch_pans_deg[nxt]), 0.0):
            self._watch_i, self._watch_at = nxt, f.now
            self._ugv_at = -1e9  # 다 돌면 바로 한 장 읽는다
            if self._engage is not None:
                self._engage["next_ask"] = f.now

    def _say(self, f, text):
        self.log.append((round(f.t, 1) if f.t is not None else None, text))

    def reset_map(self):
        """벽만 있는 임시 지도로 시작했는데 상자가 든 지도가 왔다 → 지도 · 면 기록 · 정찰 계획을 다음 틱(_build)에 새로 만든다."""
        self.fmap = self.grid = self.book = self.scout = None
        self.scout_done_at = None
        self._m6, self._m6_posts = None, None
        self._extra_seen, self._extras, self._extra_gone = [], {}, set()  # 앞 지도의 번호로 받아들인 상자는 버린다 (새 지도에도 없으면 다시 받아들인다)
        self.target_face, self._strike, self.target_done = None, None, False
        self._assault_plan = self._assault_goal = self._assault_asked = self._assault_guard = None
        self._scout_goal = None
        self._staging = False
        self.log.append((None, "상자가 든 지도가 와서 지도 · 정찰 계획을 새로 만든다"))

    # ================================================================== 준비
    def _build(self, f):
        """자리(진영 · 줄)와 상자 목록을 처음 알게 되면 지도 · 면 기록 · 정찰 계획을 만든다."""
        if self.fmap is not None or not f.slot or (f.fmap is None and not f.boxes):
            return
        self.camp, self.row = A.camp_of(f.slot), A.row_of(f.slot)
        self.foe = A.opponent(self.camp)
        self.fmap = f.fmap if f.fmap is not None else A.field_map(f.boxes, self.arena, self.base_kinds)
        self.grid = Grid(self.fmap, radius=self.p.robot_radius)
        k = self.p.base_keepout_m
        for value, (_x0, y0, _x1, y1) in self.fmap.bases.items():
            rect = (0.0, y0 - 0.15, k, y1 + 0.15) if self.fmap.base_camp(value) == "BLUE" else (self.fmap.width - k, y0 - 0.15, self.fmap.width, y1 + 0.15)
            self.grid.add_zone(value, rect, closed=True)
        self.book = FaceBook(self.fmap, self.camp)
        self.scout = make_scout(self.fmap, self.grid, self.book, self.cam, self.p.scout, start=f.pose)
        self.scout.trace = self.photo_trace
        self._prefer_sectors()
        self._want_target()
        if self._restored_book:
            self.book.restore(self._restored_book)
            self._say(f, "저장해 둔 면 기록을 되살림: 채점 구역 면의 %d%%" % round(100 * self.book.coverage()))
            self._restored_book = None
        # 목표거점 발표를 기다릴 자리: 상대 진영의 가운데(2번 줄) 거점 앞. 어느 자리가 목표가 될지 모르고(배치도 바뀔 수 있다 — 2026-11-19 공지),
        # 세 거점을 차례로 들어가 봐야 할 수도 있으니 어느 쪽으로도 가까운 가운데에서 기다린다 (전에는 우리 줄의 상대 거점 앞)
        self.stage_base = self.fmap.base_in_row(self.foe, 2) or self.fmap.base_in_row(self.foe, self.row)
        why = "가운데 거점 (거점 종류를 읽지 못함)" if self.p.stage_kind else "가운데 거점 (어느 거점이 게시될지 모르므로 양쪽에 가까운 곳)"
        likely = "%s_%s" % (self.foe, self.p.stage_kind)
        if self.p.stage_kind and self.foe in getattr(self.fmap, "kinds_read", ()) and likely in self.fmap.bases:
            # 게시될 가능성이 가장 큰 거점 앞에서 기다린다 (stage_kind 의 설명). 거점 종류를 읽지 못한 지도에서는 이름이 기본 배치라 믿지 않는다
            self.stage_base, why = likely, "게시될 가능성이 큰 거점 (%d번 줄)" % self.fmap.base_rows.get(likely, 0)
        self._say(f, "지도: 상자 %d개, 면 %d개 (채점 구역의 면 %d개). 우리는 %s%s" % (
            len(self.fmap.boxes), len(self.fmap.faces), len(self.book.goal), f.slot, "" if self.fmap.boxes else " — 상자 없는 임시 지도 (정찰은 못 하고 거점 진입만 한다)"))
        self._say(f, "목표거점 발표를 기다릴 자리: %s 앞 — %s" % (self.stage_base, why))

    def _absorb(self, f):
        # navigation 이 지도에서 뺀 상자(f.gone — 처음 지도의 번호)에, 라이다 목록에서 사라진 「지도에 없는 상자」(_extra_gone)를 합쳐서 넣는다
        if f.gone is not None and self.fmap is not None and self.fmap.set_gone(frozenset(f.gone) | self._extra_gone):
            self._say(f, "navigation 이 라이다로 없다고 본 상자: %s → 그 면은 맨 뒤로 미루고 사진 짝짓기에서 뺀다" % (
                ", ".join("%d번" % i for i in sorted(self.fmap.gone) if i not in self._extra_gone) or "없음 (뺐던 상자가 다시 보임)"))
        if self.p.overhead_robots and f.ugv_sights is not None and hasattr(self.scout, "robots"):
            self._robots(f)
        if self.p.extra_boxes and f.extra_boxes is not None and hasattr(self.scout, "lidar_boxes"):
            # 라이다 상자 목록을 정찰에도 그대로 넘긴다 (받아들이지 않은 것까지) — 벽 쪽 통로가 지도보다 좁은지 볼 때 쓴다 (FaceScout._corridor_narrowed)
            self.scout.lidar_boxes = f.extra_boxes
        if self.p.extra_boxes and f.extra_boxes is not None and self.fmap is not None and not self._extra_off:
            try:
                self._extra_boxes(f)
            except Exception as error:  # noqa: BLE001 — 덤으로 하는 일이다. 여기서 난 예외로 경기 흐름이 멈추면 안 된다
                self._extra_off = True
                self._say(f, "지도에 없는 상자를 다루다 예외 (%s: %s) → 이 경기에서는 더 받지 않고, 받아들였던 상자도 없는 것으로 친다" % (type(error).__name__, error))
                self._extra_abandon()
        if f.no_turn is not None:
            self._no_turn = (f.no_turn[0], f.no_turn[1], f.now)
        if f.parsed and dict(f.parsed) != self.m1:
            first, old = self.m1 is None, self.designated
            self.m1 = dict(f.parsed)
            self.designated = self.m1.get("target") or None
            self._say(f, ("M1 해석: %s" if first else "M1 해석이 바뀜: %s") % self.m1)
            self._prefer_sectors()
            if not first and self.designated != old:  # 지정표적이 달라졌다 → 앞의 표적에 하던 일을 버리고 새 표적을 찾는다
                self.target_face, self._strike, self.target_done, self.target_shots = None, None, False, 0
                self.target_hits, self._good_shots, self._spot = 0, 0, None
                self._strike_tries, self._strike_wait, self._strike_bad = 0, False, []
            self._want_target()
        for tag, pose, pan, pictures in f.answers:
            if tag and tag[0] == "pass":
                self._pass_wait.append((tag, pose, pan, pictures))  # 지나가며 찍은 사진: 찍은 뒤의 방향까지 본 다음 「본 것」 으로 쓴다 (_pass_votes)
                continue
            if tag and tag[0] == "lit":
                continue  # 레이저가 켜진 동안의 사진: 여기서는 쓰지 않는다 (기록에는 node 가 남긴다)
            if tag and tag[0] == "aim":  # 조준 확인 사진의 답은 정찰 기록에 넣지 않는다
                self._aim_answer(f, tag, pictures)
            elif self.scout is not None:
                self.scout.on_answer(tag, pose, pan, pictures)
        if f.ugv_answers:
            self._foe_seen = (f.now, sorted({u.get("callsign") for u in f.ugv_answers if u.get("camp") == self.foe and u.get("callsign")}))
        shot = self._ugv_shot
        if shot is not None and f.now - shot["at"] >= self.p.ugv_hit_watch_s:
            # 쏜 한 발을 다 지켜봤다. 유효 피격은 수광부에 2초 넘게 이어서 닿아야 한다 → 그동안 찍은 사진에서 한 번도 벗어나지 않았으면 맞힌 것으로 본다.
            # 그동안 사진을 못 봤으면(표적은 멈춰 있었다) 맞힌 것으로 친다. 서버는 명중을 알려 주지 않으므로 어림이다
            self._ugv_shot = None
            sign = shot["sign"]
            if shot["off"] == 0:
                self.ugv_hits[sign] = self.ugv_hits.get(sign, 0) + 1
                if sign:
                    old = self.ugvs.get(sign, {})
                    before = self.lives_of(sign)
                    cap = max(0, (4 if before is None else int(before)) - 1)
                    self.ugvs[sign] = dict(old, lives=old.get("lives"), lives_seen=list(old.get("lives_seen") or []), ammo=old.get("ammo"), t=f.t, cap=cap)
                self._say(f, "상대 UGV %s 를 맞힌 것으로 본다 (쏜 뒤 사진 %d장 모두 수광부 위) → 목숨을 %s 로 낮춰 센다" % (
                    sign or "(호출부호 모름)", shot["n"], self.lives_of(sign) if sign else "?"))
            else:
                self._say(f, "상대 UGV %s 에 쏜 발: 쏜 뒤 사진 %d장 가운데 %d장에서 수광부를 벗어남 → 맞힌 것으로 치지 않는다" % (
                    sign or "(호출부호 모름)", shot["n"], shot["off"]))
        for u in f.ugv_answers:
            if u.get("callsign"):
                self._note_ugv(f, u["callsign"], u.get("lives", -1), u.get("ammo", -1))
        if f.fire_reply is not None and self._fire_asked:
            self._fire_asked = False
            message, left_ms = f.fire_reply
            if message == "fired" and self._fire_for == "ugv":
                self.ugv_shots += 1
                self._ugv_shot = {"at": f.now, "sign": self._fire_sign, "n": 0, "off": 0}
                self._last_fire = f.now  # _laser_until 은 건드리지 않는다 (지정표적용 — 그동안 서 있게 한다). 레이저가 켜진 동안에도 계속 따라간다
                self._say(f, "상대 UGV 에 발사 %d" % self.ugv_shots)
            elif message == "fired":
                self.target_shots += 1
                self._last_fire = f.now
                self._laser_until = f.now + self.p.laser_on_s
                self._say(f, "발사 %d" % self.target_shots)
                self._lit_due = f.now + 1.0  # 레이저가 켜져 있는 동안 사진 한 장을 남긴다 (레이저 점이 수광부에 닿았는지 기록에서 눈으로 본다)
                if self._strike is not None:
                    self._aim_round(self._strike)  # 다음 발 앞에서 사진으로 다시 확인한다
                if self._spot is not None:
                    self._spot["until"] = f.now + self.p.spot_wait_s  # 레이저 점의 답을 이때까지 기다린다
                    if self._spot.get("watching") is False:  # 레이저 점을 볼 창구가 없다 → 바로 「볼 수 없었음」 으로 센다
                        self._judge_shot(f, None)
            elif message == "empty":
                self.target_done = True
                self._say(f, "탄 없음")
            else:  # locked · reloading → left_ms 뒤에 다시
                self._last_fire = f.now + left_ms / 1000.0 - self.p.fire_interval_s
            if message != "fired" and self._fire_for == "target":
                self._spot = None  # 쏘지 못했다 → 볼 레이저 점이 없다
        if self._spot is not None and self._spot.get("until") is not None:
            got = next((sp for tag, sp in f.laser_spots if tuple(tag) == tuple(self._spot["tag"])), False)
            if got is not False:
                self._judge_shot(f, got)
            elif f.now > self._spot["until"]:
                self._judge_shot(f, None)
        if self.book is not None and self.target_face is None and self.designated:
            face = self.book.find_target(self.designated)
            if face is not None:
                self.target_face = face
                self._say(f, "지정표적 %s 발견: 상자 %d 의 %s 면" % (self.designated, face.box, face.side))
                self._want_target()  # 찾았다 → 정찰은 더 찾지 않는다

    def _note_ugv(self, f, sign, lives, ammo):
        """read_ugvs 가 읽은 한 대의 상태를 그 호출부호에 적는다 (lives · ammo 는 못 읽었으면 -1)."""
        old = self.ugvs.get(sign, {})
        # 목숨: 사진 한 장의 판독은 틀릴 수 있다 (흔들린 사진 · 다른 불빛) → 최근 세 번 가운데 가장 많이 나온 값, 같으면 가장 최근 값
        seen = list(old.get("lives_seen") or [])
        if lives is not None and lives >= 0:
            seen = (seen + [int(lives)])[-3:]
        value = max(reversed(seen), key=seen.count) if seen else old.get("lives")
        if seen and not self.p.m6_hunt:
            value = seen[-1]  # M6 사냥을 껐다 → main 처럼 가장 최근 판독
        self.ugvs[sign] = {"lives": value, "lives_seen": seen, "ammo": ammo if ammo is not None and ammo >= 0 else old.get("ammo"), "t": f.t}
        if old.get("cap") is not None:
            self.ugvs[sign]["cap"] = old["cap"]  # 우리가 맞혀서 낮춘 상한은 판독이 와도 남는다 (lives_of)

    def _robots(self, f):
        """위에서 본 UGV 자리(f.ugv_sights)에서 「서 있는 다른 로봇」 을 골라 정찰에 넘긴다 (scout.robots — 실시간 순서가 그 곁의 자리를 뒤로 돌린다).

        우리 차는 그 관측 무렵 우리가 있던 자리(_recent — 지난 8초의 위치)로 골라낸다. 관측이 끊기면 빈 목록이 된다 (standing.Standing).
        덤으로 하는 일이다 — 예외가 나면 그 틱에는 모르는 것으로 둔다.
        """
        try:
            tracker, window = self._parked, self.p.robot_self_window_s
            for at, points in f.ugv_sights:
                if tracker.stamp is None or at > tracker.stamp:
                    tracker.add(at, points, [(q[0], q[1]) for when, q in self._recent if at - window <= when <= at + 0.1])
            self.scout.robots = robots = tracker.robots(f.now, self.p.robot_stand_s)
            mark = frozenset((round(x * 5.0), round(y * 5.0)) for x, y, _s in robots)
            for x, y, _s in robots:
                self._stood[(round(x * 5.0), round(y * 5.0))] = (x, y, f.now)
            if len(self._stood) > 40:  # 오래된 것은 버린다
                self._stood = {k: v for k, v in self._stood.items() if f.now - v[2] <= 120.0}
            if mark != self._parked_mark:  # 서 있는 로봇의 자리가 바뀐 때에만 기록에 남긴다
                self._parked_mark = mark
                self._say(f, "위에서 본 서 있는 다른 로봇 %d대%s" % (len(robots), "".join(" · (%.2f, %.2f) %.0f초째" % q for q in robots)))
        except Exception as error:  # noqa: BLE001
            self.scout.robots = None
            self._say(f, "위에서 본 UGV 자리를 쓰다 예외 (%s: %s) → 이번에는 모르는 것으로 둔다" % (type(error).__name__, error))

    # ================================================================== 보고 (움직임과 따로)
    def _reports(self, f, ports):
        p = self.p
        for kind, ok in f.results:
            sent = self._flying.pop(kind, None)
            if kind == "ready":
                self.ready_ok = self.ready_ok or ok
            elif ok and sent is not None:
                self._sent[kind] = sent[0]
            self._try_at[kind] = f.now + (0.0 if ok else p.retry_gap_s)
        for kind, (_payload, at) in list(self._flying.items()):
            if f.now - at > 6.0:  # 답이 없다 → 버리고 다시 보낼 수 있게
                del self._flying[kind]
        if f.phase == "standby" and not self.ready_ok and f.nodes_ready and "ready" not in self._flying and f.now >= self._try_at.get("ready", 0.0):
            if ports.set_ready():
                self._flying["ready"] = (None, f.now)
        if f.order_text and (self.m1 is None or f.parsed_pending) and f.now >= self._order_at and self._order_asks < p.m1_max_asks:
            if ports.analyze_order(f.order_text):
                self._order_at = f.now + p.m1_reask_s
                self._order_asks += 1
        if f.phase == "running":
            if self.m1 is not None:
                self._send(f, ports, 1, self.m1)
            if self.book is not None:
                counts = self.book.counts()
                changed = counts != self._sent.get(3)
                if (changed and f.now - self._m3_at >= p.m3_min_gap_s) or f.now - self._m3_at >= p.m3_refresh_s:
                    if self._send(f, ports, 3, counts, force=True):
                        self._m3_at = f.now
        if f.phase == "report":
            self._send(f, ports, 6, self.ranking())

    def _send(self, f, ports, kind, payload, force=False):
        if kind in self._flying or f.now < self._try_at.get(kind, 0.0):
            return False
        if not force and self._sent.get(kind) == payload:
            return False
        if ports.report(kind, payload):
            self._flying[kind] = (payload, f.now)
            return True
        return False

    def lives_of(self, sign):
        """그 팀의 남은 목숨으로 치는 값: LED 판독(최근 세 번의 다수)과 「우리가 맞혀서 낮춘 상한」 가운데 작은 쪽. 둘 다 없으면 None.

        2026-11-20 사용자 결정: 상대를 쏘는 것은 우리뿐일 것이므로, 우리가 맞힌 팀은 판독이 아직 안 바뀌었어도 목숨이 줄어든 것으로 센다.
        ugv_hits_lower_lives 를 끄면 판독만 쓴다.
        """
        u = self.ugvs.get(sign, {})
        lives, cap = u.get("lives"), u.get("cap") if self.p.ugv_hits_lower_lives and self.p.m6_hunt else None
        if cap is None:
            return lives
        return cap if lives is None else min(lives, cap)

    def _one_foe_sign(self, f):
        """방금(6초 안) read_ugvs 가 읽은 상대 호출부호가 하나뿐이면 그것 — 수광부 답에 호출부호가 없을 때 누구를 쐈는지 어림한다. 아니면 ""."""
        at, signs = self._foe_seen
        return signs[0] if len(signs) == 1 and f.now - at <= 6.0 else ""

    def ranking(self):
        """M6: 상대 세 팀을 위험한 순서로. 위험도 = 남은 목숨/4 + 남은 탄/8 (+ M3 점수/30 은 볼 수 없어 셋이 같다고 둔다).

        남은 목숨은 lives_of — 우리가 맞힌 팀은 그만큼 낮춰 센다 (덜 위험한 쪽으로 내려간다).
        셋을 가를 것이 없으면 호출부호 3 · 2 · 1 순이다 (2026-11-20 팀장 결정: 「어차피 운인 것 같으니 3, 2, 1 로」 — 전에는 1 · 2 · 3).
        2026-11-20 05:40 까지의 모든 판(모의 경기 6판 · 연습 11판)에서 상대의 목숨 · 탄을 읽은 적이 없어, 실제로 나간 것은 늘 이 「가를 것이 없을 때」 의 순서였다.
        """
        signs = A.foe_callsigns(self.camp or "BLUE")

        def risk(sign):
            u = self.ugvs.get(sign, {})
            lives = self.lives_of(sign) if self.lives_of(sign) is not None else self.p.prior_lives
            ammo = u.get("ammo") if u.get("ammo") is not None else self.p.prior_ammo
            return lives / 4.0 + ammo / 8.0

        return sorted(sorted(signs, reverse=True), key=lambda s: -risk(s))  # 위험도가 같으면 호출부호가 큰 것부터 (정렬이 차례를 지킨다)

    # ================================================================== 지도에 없는 상자 (navigation 이 라이다로 찾은 것)
    def _extra_boxes(self, f):
        """navigation 의 라이다 상자 목록(f.extra_boxes)에서 지도에 없는 상자를 받아들이고, 목록에서 사라진 것은 없는 것으로 친다.

        받아들이는 조건: 목록에 extra_min_age_s 동안 이어서 있었고(같은 후보인지는 extra_match_m 로 본다), 경기장 안이고, 지도의 살아 있는 상자 ·
        이미 받아들인 상자와 가운데가 extra_near_m 넘게 떨어져 있고, 고정 기둥과 겹치지 않고, 채점 구역에 찍을 면이 하나는 있고, 받아들인 수가 상한 아래
        (지금 있는 것으로 치는 상자 extra_max 개 — 없어진 것이 자리를 차지하지 않게. 없어진 것까지 합쳐서는 그 3배).
        받아들이면 (_extra_accept): 지도의 맨 뒤 번호로 넣고(앞의 번호는 그대로) → 면을 만들고 → 격자에서 그 둘레를 막고 → 채점 구역의 면을 면 기록의
        목표에 넣고 → 그 면을 찍을 자리를 정찰 계획의 맨 뒤에 붙인다. 정찰을 끝낸 뒤였으면 다시 정찰한다 (거점 앞 대기를 시작하기 전일 때만).
        받아들인 상자가 목록에서 extra_lost_s 넘게 빠져 있으면 없는 것으로 치고(fmap.gone) 격자를 풀고 갈 자리에서 그 면을 뺀다. 다시 나오면 되살린다.
        자리 기준 정찰(PlaceScout)에서는 하지 않는다. 위치를 모르는 틱에는 아무것도 하지 않는다 (설 자리를 고를 수 없다).
        거점 앞 대기 · 거점 진입을 시작한 뒤에도 하지 않는다 — 정찰로 돌아가지 않으니 쓸 데가 없고, 그때는 틱이 길어지는 일(자리 고르기)을 하지 않는다.
        """
        p, m, now = self.p, self.fmap, f.now
        if getattr(self.scout, "add_faces", None) is None or f.pose is None or self._staging or f.objective:
            return
        seen = [(float(x), float(y)) for x, y in f.extra_boxes]

        def near(a, b, limit):
            return math.hypot(a[0] - b[0], a[1] - b[1]) <= limit

        for number, e in self._extras.items():  # 받아들인 상자가 지금도 목록에 있는가
            if any(near(q, e["xy"], p.extra_near_m) for q in seen):
                e["seen"] = now
                if e["lost"]:
                    self._extra_back(f, number, e)
            elif not e["lost"] and now - e["seen"] > p.extra_lost_s:
                self._extra_lose(f, number, e)
        kept = []  # 후보: 목록에 이어서 있는 것만 남긴다 (한 번 빠지면 처음부터 다시 센다)
        for q in seen:
            if any(near(q, e["xy"], p.extra_near_m) for e in self._extras.values()):
                continue
            old = next((c for c in self._extra_seen if near(q, c["xy"], p.extra_match_m)), None)
            if old is None:
                kept.append({"xy": q, "since": now, "why": None})
            elif not any(old is c for c in kept):
                kept.append(old)
        self._extra_seen = kept
        for c in kept:
            if now - c["since"] < p.extra_min_age_s:
                continue
            x, y = c["xy"]
            why = self._extra_refusal(x, y)
            faces = []
            if why is None and (sum(1 for e in self._extras.values() if not e["lost"]) >= p.extra_max or len(self._extras) >= 3 * p.extra_max):
                why = "받아들이는 상자 수의 상한(%d개)에 닿음" % p.extra_max
            if why is None:
                faces = m.box_faces(x, y, len(m.boxes), p.extra_wall_gap_m)
                if not any(face.sector in self.book.scored for face in faces):
                    why = "채점 구역에 찍을 면이 없음"
            if why is None:
                self._extra_accept(f, x, y, faces)
            elif c["why"] != why:  # 까닭은 후보마다 한 번만 적는다 (바뀌면 다시)
                c["why"] = why
                self._say(f, "라이다로 찾은 상자 (%.2f, %.2f) 는 넣지 않는다: %s" % (x, y, why))

    def _extra_refusal(self, x, y):
        """그 자리의 라이다 상자를 받아들이지 않는 까닭 (자리로 보아 받아들여도 되면 None)."""
        m, p = self.fmap, self.p
        edge = m.box_size / 2.0 - 0.10  # 상자의 몸이 벽 밖으로 10 cm 넘게 나가 있으면 상자가 아니다 (그 안은 위치 오차로 본다)
        if not (edge <= x <= m.width - edge and edge <= y <= m.height - edge):
            return "경기장 벽 밖으로 걸쳐 있음"
        if any(i not in m.gone and math.hypot(bx - x, by - y) <= p.extra_near_m for i, (bx, by) in enumerate(m.boxes)):
            return "지도에 있는 상자의 자리"
        if any(abs(px - x) < m.box_size - 0.05 and abs(py - y) < m.box_size - 0.05 for px, py in m.pillars):
            return "고정 기둥의 자리"
        return None

    def _extra_accept(self, f, x, y, faces):
        m = self.fmap
        number = m.add_box(x, y, faces)
        half = m.box_size / 2.0 + getattr(self.grid, "grow", self.p.robot_radius + 0.02)  # 지도의 상자와 같은 폭으로 막는다
        self.grid.add_zone("extra%d" % number, (x - half, y - half, x + half, y + half), closed=True, soft=True)
        goal = self.book.add_faces(faces)
        self._extras[number] = {"xy": (x, y), "seen": f.now, "lost": False}
        planned = getattr(self.scout, "_planned", True)
        stops = self.scout.add_faces(f.pose, [face.key for face in goal])
        self._say(f, "지도에 없는 상자를 라이다로 찾음 (%.2f, %.2f) → 상자 %d번으로 넣음. 채점 구역의 면 %s 을 찍을 자리%s" % (
            x, y, number, " · ".join(face.side for face in goal),
            " %d곳을 정찰 계획의 맨 뒤에 붙임" % stops if planned else "는 정찰 계획을 세울 때 맨 뒤에 붙인다"))
        self._extra_resume(f, stops)

    def _extra_lose(self, f, number, e):
        e["lost"] = True
        self._extra_gone.add(number)
        self.fmap.set_gone(self.fmap.gone | {number})
        self.grid.set_zone("extra%d" % number, False)
        self.scout.drop_faces([face.key for face in self.fmap.faces if face.box == number])
        self._say(f, "라이다로 찾았던 상자 %d번 (%.2f, %.2f) 이 %.0f초 넘게 목록에 없음 → 없는 것으로 치고 그 면은 찍으러 가지 않는다" % (
            number, e["xy"][0], e["xy"][1], self.p.extra_lost_s))

    def _extra_back(self, f, number, e):
        e["lost"] = False
        self._extra_gone.discard(number)
        self.fmap.set_gone(self.fmap.gone - {number})
        self.grid.set_zone("extra%d" % number, True)
        stops = self.scout.add_faces(f.pose, [face.key for face in self.book.goal if face.box == number])
        self._say(f, "상자 %d번 (%.2f, %.2f) 이 라이다 목록에 다시 나옴 → 되살림 (아직 못 적은 면을 찍을 자리 %d곳을 맨 뒤에 붙임)" % (number, e["xy"][0], e["xy"][1], stops))
        self._extra_resume(f, stops)

    def _extra_abandon(self):
        """처리하다 예외가 났다: 받아들였던 상자를 모두 없는 것으로 치고 갈 자리에서 그 면을 뺀다 — 남은 정찰이 그 면을 다시 다루지 않게 (지도의 상자는 그대로다)."""
        for number, e in self._extras.items():
            e["lost"] = True
            self._extra_gone.add(number)
        try:
            self.fmap.set_gone(self.fmap.gone | self._extra_gone)
            self.scout.drop_faces([face.key for face in self.fmap.faces if face.box in self._extra_gone])
        except Exception:  # noqa: BLE001
            pass

    def _extra_resume(self, f, stops):
        """새 자리를 붙였는데 정찰을 이미 끝낸 뒤였으면 다시 정찰한다 (거점 앞 대기 · 거점 진입을 시작한 뒤에는 여기까지 오지 않는다 — _extra_boxes)."""
        if stops and self.scout_done_at is not None:
            self.scout_done_at = None
            self._say(f, "정찰을 끝낸 뒤였다 → 새로 붙인 자리를 찍으러 다시 정찰한다")

    # ================================================================== 빠져나오기
    def _stuck(self, f, ports):
        """이동이 잇달아 실패했는데 그 자리에서 못 벗어났으면, 지나온 길로 조금 되돌아간다. 되돌아가는 중이면 True (이번 틱의 일은 이것뿐).

        navigation 은 목표 쪽으로 가장 짧은 길만 다시 짠다 — 그 길이 다른 로봇이나 지도와 어긋난 상자에 막혀 있으면 어떤 목표를 줘도 같은 곳에서 선다.
        지나온 길은 방금 지나온 곳이라 비어 있었다 → 그쪽의 목표를 주면 길이 그리로 난다.
        """
        p = self.p
        avoid = self._own_base_risk(f)
        if f.go_failed is not None and len(f.go_failed) > 3 and f.go_failed[3] and f.pose is not None and self._escape is None:
            # navigation: 「목표 쪽으로 가려면 서서 돌아야 하는데 돌 자리가 없다」 (비켜 보기 3번도 실패). 못 움직인 것이 확실하니 두 번을 기다리지 않는다
            goal = self._straight_out(f.pose, avoid)
            if goal is not None:
                self._escape = {"goal": goal, "since": f.now, "straight": f.pose[2]}
                self.escapes += 1
                self._fails = []
                self._tell_scout_it_failed(f)
                self._say(f, "돌 자리가 없어 출발하지 못함 → 돌지 않고 (%.2f, %.2f) 로 곧게 %s 빠져나간다" % (
                    goal[0], goal[1], "나아가" if (goal[0] - f.pose[0]) * math.cos(f.pose[2]) + (goal[1] - f.pose[1]) * math.sin(f.pose[2]) > 0 else "물러나"))
        if f.go_failed is not None and f.pose is not None and self._escape is None:
            self._fails = [q for q in self._fails if f.now - q[0] <= 40.0 and math.hypot(q[1] - f.pose[0], q[2] - f.pose[1]) <= p.escape_stuck_m]
            self._fails.append((f.now, f.pose[0], f.pose[1]))
            if len(self._fails) >= p.escape_fails:
                goal = next(((x, y) for x, y in reversed(self._trail)
                             if math.hypot(x - f.pose[0], y - f.pose[1]) >= p.escape_back_m and not avoid(x, y)), None)
                if goal is not None:
                    self._escape = {"goal": goal, "since": f.now, "from": (f.pose[0], f.pose[1])}
                    self.escapes += 1
                    self._fails = []
                    self._tell_scout_it_failed(f)
                    self._say(f, "이동이 %d번 잇달아 실패했고 그 자리 그대로다 → 지나온 길 (%.2f, %.2f) 로 되돌아가 빠져나간다" % (p.escape_fails, goal[0], goal[1]))
        if self._escape is None:
            return False
        gx, gy = self._escape["goal"]
        failed = f.go_failed is not None and abs(f.go_failed[0] - round(gx, 2)) < 0.02 and abs(f.go_failed[1] - round(gy, 2)) < 0.02
        straight = self._escape.get("straight")
        if math.hypot(gx - f.pose[0], gy - f.pose[1]) <= (0.10 if straight is not None else 0.25) or failed or f.now - self._escape["since"] > p.escape_limit_s:
            ok = not failed and f.now - self._escape["since"] <= p.escape_limit_s
            self._say(f, "빠져나오기 끝 (%s)" % ("그 자리에 옴" if ok else "그 길도 못 감"))
            began = self._escape.get("from")
            self._escape = None
            self._scout_goal = None
            if not ok and began is not None and math.hypot(began[0] - f.pose[0], began[1] - f.pose[1]) <= p.escape_stuck_m:
                # 지나온 길로도 못 갔고 자리도 그대로다 → 뒤로 곧게 물러난다 (back_off_m 의 설명)
                goal = self._back_off(f.pose, avoid)
                if goal is not None:
                    self._escape = {"goal": goal, "since": f.now, "straight": f.pose[2]}
                    self._say(f, "앞이 막힌 채 굳었다 → 차체 방향 그대로 뒤로 (%.2f, %.2f) 까지 곧게 물러난다" % goal)
                    self.task = "escape"
                    ports.go(goal[0], goal[1], f.pose[2], yaw_free=True, mode="straight", pos_tol=0.08)
                    return True
            return False
        self.task = "escape"
        if straight is not None:
            ports.go(gx, gy, straight, yaw_free=True, mode="straight", pos_tol=0.08)
        else:
            ports.go(gx, gy, 0.0, yaw_free=True)
        return True

    def _tell_scout_it_failed(self, f):
        """빠져나오기가 이번 틱을 가져가면 정찰은 「그 자리에 못 갔다」 를 듣지 못한다 → 여기서 알려 그 자리를 뒤로 미룬다.

        2026-11-19 모의 6차: 자리 (3.23, 0.82) 에 120 · 125 · 139초에 세 번 갔고 세 번 다 막혔다. 두 번째 실패에서 빠져나오기가 시작돼
        정찰이 실패를 모른 채, 빠져나온 뒤 같은 자리를 또 시켰다. 자리에 와 있는데 방향만 못 맞춘 경우(near)는 건드리지 않는다.
        """
        goal, scout = self._scout_goal, self.scout
        if goal is None or f.go_failed is None or scout is None or f.pose is None:
            return
        gx, gy = goal[0][0], goal[0][1]
        if abs(f.go_failed[0] - gx) >= 0.02 or abs(f.go_failed[1] - gy) >= 0.02:
            return
        near, skip = getattr(scout, "near_stop", None), getattr(scout, "skip_stop", None)
        if skip is None or (near is not None and near(f.pose, gx, gy)):
            return
        skip(f.now)
        self._scout_goal = None
        self._say(f, "정찰 자리 (%.2f, %.2f) 에 가지 못함 → 뒤로 미룸 (빠져나온 뒤에는 다른 자리부터)" % (gx, gy))

    def _own_base_risk(self, f):
        """(x, y) 에 서면 우리 진영 거점의 빔을 가릴 수 있는가 — 목표거점 게시 escape_own_base_lead_s 전부터만 (경기 시각을 모르면 늘).
        그 전이거나 지도 · 진영을 모르면 늘 False 를 돌려주는 함수."""
        p = self.p
        if self.fmap is None or self.camp is None:
            return lambda x, y: False
        if f.t is not None and f.t < p.target_at_s - p.escape_own_base_lead_s:
            return lambda x, y: False
        rects = list(self.fmap.bases_of(self.camp).values())
        m = p.escape_own_base_m
        return lambda x, y: any(x0 - m <= x <= x1 + m and y0 - m <= y <= y1 + m for x0, y0, x1, y1 in rects)

    def _back_off(self, pose, avoid=lambda x, y: False):
        """차체 방향 그대로 뒤로 곧게 물러날 자리 (x, y) — 지도(벽 · 상자 · 고정 기둥)로 보아 빈 만큼, 가장 멀리 back_off_m. 물러날 데가 없으면 None."""
        p, m = self.p, self.fmap
        if p.back_off_m <= 0.0 or m is None:
            return None
        half = m.box_size / 2.0
        solids = [b for i, b in enumerate(m.boxes) if i not in m.gone] + list(m.pillars)
        c, s = math.cos(pose[2]), math.sin(pose[2])
        w = p.straight_out_half_w

        def clear(px, py):
            if not (w <= px <= m.width - w and w <= py <= m.height - w):
                return False
            return not any(abs(px - bx) < half + w and abs(py - by) < half + w for bx, by in solids)

        run = 0.0
        while run < p.back_off_m:
            step = run + 0.05
            if not clear(pose[0] - (step + 0.125) * c, pose[1] - (step + 0.125) * s):
                break
            run = step
        run = min(run, p.back_off_m)
        goal = (pose[0] - run * c, pose[1] - run * s)
        return goal if run >= 0.15 and not avoid(*goal) else None

    def _straight_out(self, pose, avoid=lambda x, y: False):
        """차체 방향 그대로 앞이나 뒤로 곧게 갈 자리 (x, y). 지도(벽 · 상자 · 고정 기둥)로 보아 차체가 지나갈 띠가 빈 쪽을 고르고, 둘 다 비면 더 멀리 빈 쪽.
        avoid(x, y) 가 참인 자리(우리 거점 — _own_base_risk)로는 가지 않고 다른 쪽을 본다.
        둘 다 막혔으면 None (그때는 전처럼 두 번 실패한 뒤 지나온 길로 되돌아간다)."""
        p, m = self.p, self.fmap
        half = m.box_size / 2.0
        solids = [b for i, b in enumerate(m.boxes) if i not in m.gone] + list(m.pillars)
        c, s = math.cos(pose[2]), math.sin(pose[2])

        def clear(px, py):
            if not (p.straight_out_half_w <= px <= m.width - p.straight_out_half_w and p.straight_out_half_w <= py <= m.height - p.straight_out_half_w):
                return False
            return not any(abs(px - bx) < half + p.straight_out_half_w and abs(py - by) < half + p.straight_out_half_w for bx, by in solids)

        def free_run(sign):  # 차체 가운데를 그쪽으로 곧게 옮길 수 있는 거리 (차체 앞 · 뒤 끝 0.125 m 까지 본다, 최대 0.8 m)
            run = 0.0
            while run < 0.8:
                step = run + 0.05
                if not clear(pose[0] + sign * (step + 0.125) * c, pose[1] + sign * (step + 0.125) * s):
                    break
                run = step
            return run

        runs = {sign: free_run(sign) for sign in (1.0, -1.0)}
        for sign in sorted(runs, key=lambda k: (runs[k], k), reverse=True):  # 더 멀리 빈 쪽부터, 같으면 앞으로
            dist = min(p.straight_out_m, runs[sign])
            goal = (pose[0] + sign * dist * c, pose[1] + sign * dist * s)
            if dist >= 0.15 and not avoid(*goal):
                return goal
        return None

    # ================================================================== 정찰
    def _want_target(self):
        """정찰에 지정표적의 종류를 알린다 — 그 종류가 찍힌 그림이 가리키는 면부터 찍으러 간다 (scout._hint). 표적 면을 찾았으면 끈다."""
        want = getattr(self.scout, "want_kind", None)
        if want is None:
            return
        kind = A.target_kind(self.designated) if self.designated and self.target_face is None else None
        want(kind[1] if kind else None)

    def _prefer_sectors(self):
        """임무 명령의 탐색 구역을 정찰에 알린다 — 그 가운데 상대 진영의 구역을 먼저 돈다 (지정표적이 거기 있다 → FaceScout.prefer)."""
        prefer = getattr(self.scout, "prefer", None)
        if prefer is None or not self.m1:
            return
        try:
            names = [str(name) for name in (self.m1.get("search_sectors") or ()) if str(name) in A.scored_sectors(self.camp)]
        except Exception:  # 해석 값의 모양이 다르다 → 순서는 점수만으로 정한다
            names = []
        prefer(names)

    def _scout(self, f, ports):
        told = len(self.scout.log)
        if hasattr(self.scout, "clock"):
            # 실시간 순서(ScoutParams.order = "live")가 남은 시간을 셈하게: 경기 시각 · 거점 앞 대기를 시작할 때의 어림 · 대기 자리
            self.scout.clock = (f.t, self.p.target_at_s - self.p.stage_margin_s, self._stage_point(self.stage_base, self.p.stage_from_wall_m)[0] if self.stage_base else None)
        act = self.scout.step(f.pose, f.now)
        for line in self.scout.log[told:]:
            # 정찰이 이번 틱에 정한 것(계획 · 미룬 자리 · 지나친 자리와 그 까닭)을 경기 기록에도 남긴다. 전에는 scout.log 에만 있어 기록에서는
            # 어느 자리를 왜 건너뛰었는지 볼 수 없었다 (2026-11-19 18:08 모의 경기 6차의 기록을 읽을 때 계획을 다시 계산해야 했다)
            self._say(f, "정찰: %s" % line)
        self.task = "scout %d" % len(self.scout.remaining())
        if act[0] == "done":
            self.scout_done_at = f.t
            self._say(f, "정찰 끝 (채점 구역 면의 %d%% 를 적음)" % round(100 * self.book.coverage()))
            ports.stop()
        elif act[0] == "go":
            self._shot_wait = None
            near = self.scout.near_stop(f.pose, act[1], act[2])
            failed = f.go_failed is not None and abs(f.go_failed[0] - act[1]) < 0.02 and abs(f.go_failed[1] - act[2]) < 0.02
            key = (round(act[1], 2), round(act[2], 2), round(act[3], 2))
            asked = self._scout_goal is not None and self._scout_goal[0] == key  # 이 자리 · 이 방향을 navigation 에 이미 시켰는가
            no_turn = (near and self._no_turn is not None and f.now - self._no_turn[2] <= self.p.no_turn_keep_s
                       and math.hypot(act[1] - self._no_turn[0], act[2] - self._no_turn[1]) <= 0.15)
            # navigation 이 「도착」 으로 끝냈는데(못 갔다는 알림이 아니다) 자리에서 arrive_m 을 조금 넘은 곳에 서 있다 — 「허용을 조금 넘었지만 고치려면
            # 크게 돌아야 해서 이 자리에서 끝냄」 (REACHED [POS_NOT_MET]). 막힌 것이 아니므로, 한 번 왔다고 본 뒤의 한도(arrive_slack_m) 안이면
            # 온 것으로 치고 선 자리에서 팬으로 겨눠 찍는다 (아래 accept_heading 의 길). 그보다 멀면 전처럼 「가지 못함」 이다.
            # 2026-11-20 03:28 연습 판(2경기장 R-3): 83초에 자리 (0.83, 2.02) 에서 0.19 m 인 곳에 방향까지 맞춰 서 있었는데 「가지 못함」 으로 쳤다.
            # 볼 면은 0.67 m 앞에 56° 로 보였지만 「선 자리에서 찍기」 의 한도(55°)에 걸려 그 자리를 뒤로 미루고 막힌 곳으로 적었고,
            # 153초에 다시 와서 찍었다 (그 자리를 들렀다 다음 자리로 가는 데 5.2 m · 19초 — 들르지 않는 곧은 거리는 2.1 m)
            short = (self.p.short_arrival_ok and not near and not failed and asked and f.arrived
                     and math.hypot(f.pose[0] - act[1], f.pose[1] - act[2]) <= self.p.scout.arrive_m + self.p.scout.arrive_slack_m)
            if (failed or (asked and f.arrived)) and not near and not short:
                # navigation 이 이 자리에 가지 못했다 (시한 넘김 · 막힘), 또는 끝났다는데 자리에서 멀다. 같은 목표를 또 보내면 같은 일이 되풀이된다.
                # 선 자리에서 보이는 남은 면이 있으면 여기서 찍고(면 기준), 없으면 이 자리를 뒤로 미루고 다음 자리로 간다
                here = getattr(self.scout, "shoot_here", None)
                if here is not None and here(f.pose, f.now):
                    self._say(f, "정찰 자리 (%.2f, %.2f) 에 가지 못함 → 선 자리에서 보이는 면을 찍는다" % (act[1], act[2]))
                    ports.stop()
                else:
                    self._say(f, "정찰 자리 (%.2f, %.2f) 에 가지 못함 → 뒤로 미룸" % (act[1], act[2]))
                    self.scout.skip_stop(f.now)
                self._scout_goal = None
            elif short or (near and (no_turn or failed or (asked and (f.arrived or (f.now - self._scout_goal[1] >= self.p.stall_s and self._standing(f, self.p.stall_s)))))):
                # no_turn: navigation 이 방금 이 자리에서 「돌 자리 없음(도착 방향)」 을 알렸다 → 방향만 다른 다음 목표를 시키지 않는다 (또 같은 답이 온다)
                # 자리에는 왔는데 끝의 방향 맞추기가 끝나지 않는다 (2026-11-18 팀 테스트: 제자리 회전이 조금 모자란 채 멈춰 3분을 그 자리에 있었다).
                # navigation 이 「도착」 을 알렸는데도 방향이 허용 밖인 경우(세 번 돌아 보고 20° 안에서 끝낸 때)에는 기다리지 않고 바로 넘어간다.
                # 팬으로 겨누므로 차체 방향은 정확하지 않아도 된다 → 이동을 거두고 지금 방향에서 닿는 면을 찍는다.
                # 「이 자리 · 이 방향을 이미 시켰을 때」 만이다 — 같은 자리에서 방향만 다른 다음 자리는 먼저 돌려 봐야 한다
                # (2026-11-19 고침: 앞 자리에서 3초 넘게 서 있었으면 다음 자리의 방향을 시키지도 않고 넘어가, 그 자리의 면을 놓쳤다)
                if short:
                    self._say(f, "정찰 자리 (%.2f, %.2f) 에서 %.2f m 인 곳에서 navigation 이 도착으로 끝냄 → 온 것으로 치고 여기서 찍는다" % (
                        act[1], act[2], math.hypot(f.pose[0] - act[1], f.pose[1] - act[2])))
                self.scout.accept_heading()
                ports.stop()
                self._scout_goal = None
            else:
                # 면 기준 정찰은 방향 허용을 같이 준다 (찍을 면이 팬으로 닿는 범위). navigation 에는 조금 좁혀 넘긴다 — 「도착」 이면 확실히 그 안이게
                tol = act[4] if len(act) > 4 else None
                if not asked:
                    self._scout_goal = (key, f.now)
                # 위치 허용은 「자리에 왔다」 한도(arrive_m)보다 조금 좁게 — navigation 이 도착이라 하면 여기서도 도착이게
                ports.go(act[1], act[2], act[3], yaw_tol=None if tol is None else max(0.10, tol - math.radians(4.0)),
                         pos_tol=max(0.10, self.p.scout.arrive_m - 0.03))
                self._pan_forward(f, ports, key, act[1], act[2])
                self._pass_shot(f, ports)
        elif act[0] == "shoot":
            pan = act[1]
            now_pan = getattr(ports, "pan_now", lambda: None)()
            if now_pan is not None and abs(now_pan - pan) <= self.p.pan_keep_deg:
                pan = now_pan  # 미리 돌려 둔 팬이 거의 맞다 → 다시 돌리지 않는다 (다시 돌리면 멈추기를 또 기다린다)
            if not f.arrived:
                # 자리 · 방향은 됐는데 navigation 이 아직 「도착」 을 알리지 않았다 → 이동을 거두고 찍는다.
                # 면 기준 정찰에서만이다 (자리에 한 번 왔다고 본 뒤에는 값이 흔들려도 그 자리로 친다 — FaceScout.any_heading. 자리 기준 정찰에는
                # 그것이 없어, 일찍 거두면 「가라 · 왔다」 가 되풀이된다). 벽 쪽 통로의 자리는 어긋남의 여유가 작아 navigation 이 끝까지 가게 둔다
                by_face = hasattr(self.scout, "corridor")
                corridor = by_face and len(act[2]) > 1 and act[2][1] in self.scout.corridor
                if self.p.shoot_early and by_face and not corridor:
                    ports.stop()  # 기다리지 않고 바로 (navigation 은 좁은 허용까지 더 돌려고 한다 — 그 사이를 기다리지 않는다)
                    # 거둔 뒤에 자리 · 방향 값이 흔들려 정찰이 다시 「가라」 고 하면 목표를 새로 보낸다
                    # (그대로 두면 「도착이라는데 자리에서 멀다」 로 읽어 그 자리를 건너뛴다 — 가상 경기에서 면 넷을 놓쳤다)
                    self._scout_goal = None
                else:
                    # 막 보낸 목표는 거두지 않는다 (보낸 직후에는 아직 안 움직여 「서 있다」 로 보인다)
                    fresh = self._scout_goal is not None and f.now - self._scout_goal[1] < 1.5
                    if not fresh and self._standing(f, 1.0):
                        ports.stop()
                self._pan_to(f, ports, pan)  # 팬은 차체가 멈추기를 기다리지 않고 보낸다
            elif f.now - (self._arrived_since if self._arrived_since is not None else f.now) < self.p.shoot_settle_s:
                self._pan_to(f, ports, pan)  # 막 멈췄다 → 차체가 가라앉을 때까지 잠깐 기다린다 (2026-11-18: 흐린 사진이 있었다). 팬은 그동안 돌린다
            elif ports.shoot(act[2], pan):
                self.scout.shot_taken(act[2], f.now)
                if self._pan_at is None or f.now < self._pan_at:
                    self._pan_at = f.now  # 팬을 보내고 다 돈 뒤에 찍혔다 → 이제 팬 각도를 안다
        return self.task

    def _pan_to(self, f, ports, pan, step=2.0, gap=0.3):
        """정찰하며 팬을 미리 돌린다 (틸트 0). 마지막으로 보낸 각도에서 step 넘게 다르고 gap 초가 지났을 때만 보낸다."""
        aim = getattr(ports, "aim", None)
        if aim is None:
            return
        last = self._pan_sent
        if last is not None and (abs(last[0] - pan) <= step or f.now - last[1] < gap):
            return
        if aim(pan, 0.0):
            self._pan_sent = (pan, f.now)

    def _pan_forward(self, f, ports, key, x, y):
        """정찰 자리로 가는 동안: 멀면 카메라를 정면으로 돌려 두고(목표마다 한 번 — pass_forward), 가까워지면 찍을 면 쪽으로 미리 돌린다 (pan_track_m)."""
        p = self.p
        dist = math.hypot(x - f.pose[0], y - f.pose[1])
        if p.shoot_early and dist <= p.pan_track_m:
            face = getattr(self.scout, "next_face", lambda: None)()
            if face is not None:
                want = -math.degrees(wrap(math.atan2(face.cy - f.pose[1], face.cx - f.pose[0]) - f.pose[2]))
                limit = PAN_LIMIT_DEG - 2.0
                if abs(want) <= limit + 20.0:  # 아직 옆 · 뒤에 있는 면은 따라가지 않는다
                    self._pan_to(f, ports, max(-limit, min(limit, want)), step=p.pan_track_step_deg, gap=0.4)
            self._pan_fwd_for = key
            return
        if not p.pass_forward or p.pass_gap_s <= 0.0 or self._pan_fwd_for == key:
            return
        if dist < p.pass_forward_min_m:
            self._pan_fwd_for = key
            return
        aim = getattr(ports, "aim", None)
        if aim is not None and aim(0.0, 0.0):
            self._pan_fwd_for = key
            self._pan_sent = (0.0, f.now)

    def _pass_shot(self, f, ports):
        """지나가며 찍기 (pass_gap_s > 0 일 때): 정찰 자리로 곧게 가는 중이면 사진 한 장을 판독에 보낸다. 답은 면 기록의 「본 것」 으로 쓴다 (_pass_votes)."""
        p = self.p
        shoot = getattr(ports, "shoot_pass", None)
        if p.pass_gap_s <= 0.0 or shoot is None or f.arrived or f.now - self._pass_at < p.pass_gap_s:
            return
        pts = [q for when, q in self._recent if f.now - when <= 0.5]
        if len(pts) < 3 or abs(wrap(pts[-1][2] - pts[0][2])) > math.radians(p.pass_turn_deg):
            return
        if shoot(("pass", self._pass_n)):
            self._pass_info[self._pass_n] = (f.now, self._pan_at is not None and f.now >= self._pan_at)
            self._pass_at, self._pass_n = f.now, self._pass_n + 1

    def _pass_votes(self, f):
        """답이 온 지나가며 찍은 사진을 면 기록의 「본 것」 으로 넘긴다 (scout.on_pass). 한 장으로는 적히지 않는다 — 다른 자리의 사진과 같은 답일 때만.

        찍은 뒤 pass_vote_wait_s 가 지나야 쓴다 (그 사이의 방향을 보려고). 버리는 사진: 팬을 보내기 전에 찍은 것 · 찍을 때 앞뒤로 돌고 있던 것 ·
        판독이 안 된 것 · 우리가 찍은 기록이 없는 것.
        """
        p, keep = self.p, []
        for item in self._pass_wait:
            tag, pose, pan, pictures = item
            info = self._pass_info.get(tag[1]) if len(tag) > 1 else None
            on_pass = getattr(self.scout, "on_pass", None)
            if info is None or pictures is None or pose is None or on_pass is None:
                if len(tag) > 1:
                    self._pass_info.pop(tag[1], None)
                continue
            at, pan_known = info
            if f.now - at < p.pass_vote_wait_s:
                keep.append(item)
                continue
            del self._pass_info[tag[1]]
            if not pan_known:
                continue
            yaws = [q[2] for when, q in self._recent if at - 0.6 <= when <= at + p.pass_vote_wait_s]
            if len(yaws) < 3:
                continue
            turned = max(abs(wrap(v - pose[2])) for v in yaws)
            steady = turned <= math.radians(p.pass_vote_turn_deg)
            if not steady and turned > math.radians(p.pass_vote_turn_align_deg):
                continue
            try:
                # 방향이 조금 흔들린 사진: 사진 전체의 밀림을 스스로 찾는 「한꺼번에 맞추기」 로만 쓴다 (pass_vote_turn_align_deg 의 설명)
                new = on_pass(tag, pose, pan, pictures, at=at) if steady else on_pass(tag, pose, pan, pictures, at=at, steady=False)
            except Exception as error:  # noqa: BLE001 — 덤으로 하는 일이다. 여기서 난 예외로 경기 흐름이 멈추면 안 된다
                self._say(f, "지나가며 찍은 사진을 쓰다 예외 (%s: %s) → 이 사진은 버린다" % (type(error).__name__, error))
                continue
            for key, kind in new:
                self._say(f, "지나가며 찍은 사진들로 면을 적음: 상자 %d 의 %s 면 = %s" % (key[0], key[1], kind))
        self._pass_wait = keep

    def _standing(self, f, seconds, move_m=0.03, turn_deg=3.0):
        """지난 seconds 초 동안 실제로 서 있었는가. 기록이 그만큼 없으면 False.

        그 동안의 앞쪽 1/3 과 뒤쪽 1/3 의 평균 자리 · 방향을 견준다 (3 cm · 3도 안이면 서 있다). 위치 값은 서 있어도 몇 cm 씩 튄다 —
        표본 하나하나를 지금 값과 견주면 한 번 튄 값 때문에 「움직인다」 로 보고 그 자리에서 기다리기만 한다 (2026-11-19 가상 시험: 잡음 3 cm 에서 사진 2장).
        """
        if f.pose is None or not self._recent or f.now - self._recent[0][0] < seconds:
            return False
        pts = [q for when, q in self._recent if f.now - when <= seconds]
        if len(pts) < 6:
            return False
        k = max(2, len(pts) // 3)

        def mean(part):
            return (sum(q[0] for q in part) / len(part), sum(q[1] for q in part) / len(part),
                    math.atan2(sum(math.sin(q[2]) for q in part), sum(math.cos(q[2]) for q in part)))

        old, new = mean(pts[:k]), mean(pts[-k:])
        return math.hypot(new[0] - old[0], new[1] - old[1]) <= move_m and abs(wrap(new[2] - old[2])) <= math.radians(turn_deg)

    # ================================================================== M2 지정표적
    def _want_strike(self, f):
        owed = self.target_face is not None and not self.target_done and self.target_shots < self.p.target_shots
        if owed and self._strike_wait and self.scout_done_at is None:
            owed = False  # 사진에서 표적을 못 찾아 물러났다 → 정찰을 마친 뒤에 다시 한다
        return owed or f.now < self._laser_until

    def _judge_shot(self, f, spot):
        """쏜 발이 수광부에 닿았는지: 레이저가 켜진 사진에서 찾은 레이저 점(spot, 못 찾았으면 None)과 조준 사진의 수광부를 견준다.

        - 닿은 것으로 보인다 → 이 표적은 끝 (더 쏘지 않는다. 표적은 처음 부서질 때 한 번만 인정된다)
        - 빗나갔다 → 잰 레이저 점과 식의 차이를 다음 조준에 더하고 다시 쏜다
        - 볼 수 없었다 → 조준이 허용 안에 든 채 쏜 발이 unconfirmed_shots 번이면 그만 쏜다
        """
        p, sp = self.p, self._spot
        self._spot = None
        rec, laser = sp.get("rec"), sp.get("laser")
        if spot is not None and rec is not None:
            du, dv = spot["u"] - rec.u, spot["v"] - rec.v
            miss = math.hypot(du, dv)
            if miss <= max(p.hit_px, p.hit_size_frac * rec.size):
                self.target_hits += 1
                self.target_done = True
                self._say(f, "명중으로 보임: 레이저 점이 수광부에서 %.1f 픽셀 (발사 %d) → 지정표적 끝" % (miss, self.target_shots))
                return
            fix = (spot["u"] - laser[0], spot["v"] - laser[1]) if laser is not None else (0.0, 0.0)
            if math.hypot(*fix) <= p.spot_fix_max_px:
                self._laser_fix = (self._laser_fix[0] + fix[0], self._laser_fix[1] + fix[1])
            self._say(f, "빗나간 것으로 보임: 레이저 점이 수광부에서 오른쪽 %+.0f · 아래 %+.0f 픽셀 → %s"
                      % (du, dv, "레이저 점 자리를 고쳐 다시 겨눈다" if math.hypot(*fix) <= p.spot_fix_max_px else "다시 겨눈다"))
            return
        if sp.get("good"):
            self._good_shots += 1
        if self._good_shots >= p.unconfirmed_shots:
            self.target_done = True
            self._say(f, "맞았는지는 볼 수 없었음 (레이저 점이 사진에 안 보임) — 조준이 맞은 채 %d발 쐈으므로 지정표적 끝" % self._good_shots)

    def _strike_defer(self, f, why):
        """사진에서 표적을 찾지 못했다 → 쏘지 않고 물러난다. 정찰(M3)로 돌아가고, 정찰을 마친 뒤 다른 자리에서 한 번 더 해 본다."""
        s = self._strike
        if s is not None:
            self._strike_bad.append(s["spot"])
        self._strike = None
        self._strike_tries += 1
        if self._strike_tries >= self.p.strike_tries:
            self.target_done = True
            self._say(f, "지정표적: %s — %d번 해 보고 그만둔다 (발사 %d)" % (why, self._strike_tries, self.target_shots))
        else:
            self._strike_wait = True
            self._say(f, "지정표적: %s → 쏘지 않고 정찰로 돌아간다 (정찰을 마친 뒤 다른 자리에서 다시)" % why)

    def _do_strike(self, f, ports):
        p = self.p
        face = self.target_face
        self.task = "strike"
        if f.now < self._laser_until:
            ports.stop()  # 레이저가 켜진 5초 동안 움직이지 않는다
            return self.task
        if self._strike is None:
            spot = self._standoff(face, avoid=self._strike_bad)
            if spot is None:
                self.target_done = True
                self._say(f, "지정표적을 쏠 자리가 없음")
                return None
            self._strike = {"spot": spot, "since": f.now, "aimed_at": None, "bad": list(self._strike_bad)}
        s = self._strike
        if f.go_failed is not None and abs(f.go_failed[0] - round(s["spot"][0], 2)) < 0.02 and abs(f.go_failed[1] - round(s["spot"][1], 2)) < 0.02 \
                and math.hypot(s["spot"][0] - f.pose[0], s["spot"][1] - f.pose[1]) > 0.2:
            # 쏠 자리에 가지 못했다 (다른 UGV 가 서 있는 등) → 그 자리를 빼고 다른 자리를 고른다. 전에는 40초 동안 같은 자리만 보내다 포기했다
            if not self._strike_elsewhere(f, face, "에 가지 못함"):
                return None
        if f.now - s["since"] > p.strike_total_s:
            self.target_done = True
            self._strike = None
            self._say(f, "지정표적에 쓸 시간을 다 씀 (발사 %d)" % self.target_shots)
            return None
        sx, sy = s["spot"]
        heading = math.atan2(face.cy - sy, face.cx - sx)
        x, y, yaw = f.pose
        # 쏠 자리에 한 번 선 뒤에는 위치 값이 조금 벗어나도 그 자리로 친다 (strike_keep_m). 전에는 쏘는 도중 위치 값이 한 번 0.2 m 넘게 튀면
        # 그 자리로 다시 가려 했고, 그때가 시작 40초 뒤면 「쏠 자리에 못 감」 으로 남은 발을 버렸다. 조준은 사진으로 고치므로 선 자리가 조금 달라도 된다
        near = math.hypot(sx - x, sy - y) <= (p.strike_keep_m if s.get("reached") else 0.2)
        if near and not f.arrived and self._standing(f, 1.5) and self.aim_angles(f.pose, face) is not None:
            # 쏠 자리에는 왔는데 끝의 방향 맞추기가 끝나지 않는다 → 이동을 거두고 팬틸트로 겨눈다 (정찰과 같은 까닭).
            # 지금 방향에서 표적이 팬틸트 범위 안일 때만이다 — 실차는 서서 돌라는 명령 뒤 1.5 ~ 2초 가만히 있다 돌기 시작한다
            # (2026-11-19 기록: 방향 150° 남은 채 1.6초). 그때 거두면 표적이 팬 밖이라 멀쩡한 자리를 버린다. navigation 은 2.5초 안 돌면 스스로 끝낸다
            ports.stop()
            return self.task
        if s.get("asked") == s["spot"] and not f.arrived:
            s["moving"] = True  # 이 자리로 「가는 중」 을 봤다
        if f.arrived and not near and not s.get("reached") and s.get("asked") == s["spot"] and s.get("moving"):
            # navigation 은 끝났다는데(REACHED) 쏠 자리에서 멀다 — 목표 칸이 막혀 navigation 이 0.30 m 안의 빈 칸으로 옮겨 선 경우 등.
            # 같은 목표는 창구가 다시 보내지 않아(node.go) 전에는 여기서 strike_give_up_s(40초)를 서 있다가 지정표적을 버렸다 (2026-11-19 주행 통합 뒤 정리)
            return self.task if self._strike_elsewhere(f, face, "에서 먼 곳에 섬 (navigation 은 도착이라 함)") else None
        if not f.arrived or not near:
            if not s.get("reached") and f.now - s["since"] > p.strike_give_up_s:
                self.target_done = True
                self._strike = None
                self._say(f, "쏠 자리에 못 감")
                return None
            if not f.arrived:
                self._aim_round(s, moved=True)  # 차체가 움직인다 → 서면 계산한 각도부터 다시 겨눈다 (위치 값이 한 번 튄 것으로는 지우지 않는다)
            if s.get("asked") != s["spot"]:
                s.update(asked=s["spot"], moving=False)
            ports.go(sx, sy, heading, yaw_tol=math.radians(p.strike_yaw_tol_deg))
            return self.task
        s["reached"] = True
        if len(self._still) < 5:
            return self.task  # 멈춘 뒤 위치 값이 몇 번 모일 때까지
        n = len(self._still)
        mean = (sum(q[0] for q in self._still) / n, sum(q[1] for q in self._still) / n,
                math.atan2(sum(math.sin(q[2]) for q in self._still), sum(math.cos(q[2]) for q in self._still)))
        angles = self.aim_angles(mean, face)
        if angles is None:
            # 선 자리 · 방향에서는 팬틸트가 닿지 않는다. navigation 은 방향을 못 맞춰도 도착(REACHED [YAW_NOT_MET] · [NO_ROOM_AT_GOAL])으로 끝낸다 —
            # 가상에서 요청 방향에 57° 못 미친 채 도착한 예가 있다 (docs/주행_결과_코드_확인_2026-11-19.md 1번). 전에는 여기서 지정표적을 통째로 버렸다.
            # 그 자리는 돌 자리가 없는 곳일 수 있으니 다른 자리를 고른다 (없으면 그때 포기. 쓰는 시간은 strike_total_s 가 막는다)
            if s.get("asked") != s["spot"]:
                # 다만 이 자리로 가라고 시킨 적이 없으면(앞의 일을 하던 곳이 마침 쏠 자리였다) 먼저 그 자리에서 표적 쪽으로 돌려 달라고 한다.
                # 모의 5차: 가장 좋은 자리(정면 0.97 m)에서 0.14 m 인 곳에 서 있었는데 팬이 57도 필요해(한도 50도) 그 자리를 버리고
                # 10.5초를 들여 비스듬한 자리(0.66 m · 46도)로 옮겨 쐈다. 돌려 달라고 했는데도 방향이 안 맞으면 전처럼 다른 자리를 고른다
                s.update(asked=s["spot"], moving=False, reached=False)
                self._aim_round(s, moved=True)
                ports.go(sx, sy, heading, yaw_tol=math.radians(p.strike_yaw_tol_deg))
                return self.task
            return self.task if self._strike_elsewhere(f, face, "에서는 표적이 팬틸트 범위 밖 (방향 %.0f°)" % math.degrees(mean[2])) else None
        pan, tilt = angles
        # 계산한 각도로 겨눈다: 처음에, 그리고 위치 값이 크게 달라졌을 때. 사진에서 수광부를 한 번 잡은 뒤에는 계산한 각도로 되돌아가지 않는다
        # (사진이 맞다 — 위치 값이 흔들려도 팬틸트를 되돌리지 않는다. 차체가 움직이면 위에서 풀린다).
        # 사진 조준을 쓸 때는 잡기 전에도 3배 넘게 달라져야 다시 겨눈다 — 그보다 작은 차이는 사진이 고치고, 도중에 돌리면 찍어 둔 사진을 버려야 한다
        geo = s.get("geo")
        held = s.get("aim") is not None and s["aim"]["held"]
        by_sight = p.aim.enabled and getattr(ports, "shoot_aim", None) is not None
        reaim = p.reaim_deg * (3.0 if by_sight else 1.0)
        if s["aimed_at"] is None or geo is None or (not held and (abs(pan - geo[0]) > reaim or abs(tilt - geo[1]) > reaim)):
            if ports.aim(pan, tilt):
                s.update(aimed_at=f.now, pan=pan, tilt=tilt, geo=(pan, tilt))
                if s.get("aim") is not None:  # 기다리던 조준 사진은 돌리기 전의 것이다 → 버리고 다시 찍는다 (못 찾은 횟수 · 쓴 시간은 이어 센다)
                    s["aim"]["loop"].interrupted()
                    s["aim"].update(state="look", tag=None, to=None)
            return self.task
        if not getattr(ports, "aim_ready", lambda: True)():
            return self.task  # 팬틸트가 아직 돌고 있다 (멀리 돌릴수록 오래 걸린다) — 도는 중에 쏘면 빗나간다
        if f.now - s["aimed_at"] < p.aim_settle_s:
            return self.task
        if self._fire_asked and f.now - self._fire_asked_at > p.fire_reply_wait_s:
            self._fire_asked = False  # 발사 요청의 답이 오지 않았다 → 다시 쏠 수 있게
        if self._fire_asked:
            return self.task
        if self._spot is not None:
            return self.task  # 앞 발이 닿았는지 보는 중 — 그 답을 보고 더 쏠지 정한다
        if not self._aim_by_image(f, ports, mean, face):
            return self.task if self._strike is not None else None  # 사진으로 조준을 고치는 중. 물러났으면(_strike_defer) 정찰로 넘긴다
        if f.now < self._last_fire + p.fire_interval_s:
            return self.task
        aim = s.get("aim") or {}
        watch = getattr(ports, "watch_laser", None)
        watching = bool(watch(("spot", self.target_shots + 1), aim.get("laser"))) if watch is not None else False
        if ports.fire():
            self._fire_asked, self._fire_asked_at = True, f.now
            self._fire_for = "target"
            # 이 발이 닿았는지 볼 재료: 조준 사진에서 본 수광부와 그때의 레이저 점. 조준이 허용 안에 들었는지도 적어 둔다
            # (사진 조준을 쓰지 않을 때는 계산한 각도를 믿고 쏜 것이므로 「조준이 맞은 채」 로 센다 — 전의 방식에서도 발 수가 준다)
            good = aim["loop"].converged if aim.get("loop") is not None and aim.get("held") else not by_sight
            self._spot = {"tag": ("spot", self.target_shots + 1), "until": None, "rec": aim.get("rec"), "laser": aim.get("laser"),
                          "good": bool(good), "watching": watching}
        return self.task

    def _strike_elsewhere(self, f, face, why):
        """지금 쏠 자리를 버리고 다른 자리를 고른다 (버린 자리의 둘레 0.35 m 는 다시 고르지 않는다). 다른 자리가 없으면 지정표적을 그만두고 False."""
        s = self._strike
        s["bad"].append(s["spot"])
        spot = self._standoff(face, avoid=s["bad"])
        if spot is None:
            self.target_done = True
            self._strike = None
            self._say(f, "지정표적: 쏠 자리 (%.2f, %.2f) %s — 다른 자리도 없어 그만둔다" % (s["spot"][0], s["spot"][1], why))
            return False
        self._say(f, "쏠 자리 (%.2f, %.2f) %s → (%.2f, %.2f) 로" % (s["spot"][0], s["spot"][1], why, spot[0], spot[1]))
        s.update(spot=spot, aimed_at=None, reached=False)
        return True

    # ---- 사진으로 조준 고치기 (aim.py). 못 쓰면 위에서 계산한 각도 그대로 쏜다
    def _aim_state(self, s):
        """이 지정표적의 사진 조준 상태 (없으면 만든다).

        state  look 사진을 찍을 차례 · wait 판독 답을 기다림 · turn 팬틸트를 고칠 차례 · ready 쏴도 된다
        held   사진에서 수광부를 잡았다 (계산한 각도로 되돌아가지 않는다)
        """
        a = s.get("aim")
        if a is None:
            loop = AimLoop(self.p.aim, self.p.aim.deg_px(self.cam.hfov_deg), PAN_LIMIT_DEG, TILT_LIMIT_DEG)
            a = s["aim"] = {"loop": loop, "state": "look", "since": None, "tag": None, "asked_at": 0.0, "n": 0, "to": None, "laser": None,
                            "checked_at": -1e9, "held": False, "told": 0}
        return a

    def _aim_round(self, s, moved=False):
        """사진 조준을 처음부터 다시 한다 (새 발 앞 · 차체가 움직인 뒤 · 확인한 뒤 오래 못 쐈을 때). 배운 「1픽셀에 몇 도」 는 남긴다."""
        a = s.get("aim")
        if a is None:
            return
        a["loop"].reset()
        a.update(state="look", since=None, tag=None, to=None)
        if moved:
            a.update(held=False, search=None, z=None)  # 차체가 움직였다 → 사진으로 잡은 것 · 찾던 것 · 어림한 거리를 버린다

    def _aim_by_image(self, f, ports, pose, face):
        """사진으로 조준을 고친다. 쏴도 되면 True, 아직 고치는 중이면 False.

        한 차례: 지금 팬틸트로 사진 한 장 → 수광부가 레이저 점(aim.laser_pixel)에서 벗어난 픽셀 → 팬틸트를 그만큼 돌림 → 다시 사진 …
        허용(tol_px) 안에 들어오거나 max_corrections 번 고쳤으면 쏜다. 발사 뒤에는 다음 발 앞에서 한 번 더 확인한다.
        사진을 쓸 수 없으면(설정으로 끔 · 사진 창구가 없음 · max_misses 장 잇달아 못 찾음 · budget_s 를 넘김) 지금 각도로 쏜다 —
        한 번도 못 찾았으면 그것은 계산한 각도다 (전의 방식).
        """
        p, s = self.p, self._strike
        shoot = getattr(ports, "shoot_aim", None)
        if not p.aim.enabled or shoot is None:
            return True
        a = self._aim_state(s)
        loop = a["loop"]
        if a["state"] == "ready":
            if f.now - a["checked_at"] <= p.aim.fresh_s:
                return True
            self._aim_round(s)  # 확인한 뒤 오래 못 쐈다 (잠김 · 재장전) → 다시 본다
        if a["since"] is None:
            a["since"] = f.now
        if a["state"] == "turn":
            if ports.aim(a["to"][0], a["to"][1]):
                s.update(aimed_at=f.now, pan=a["to"][0], tilt=a["to"][1])
                a["state"] = "look"
            return False
        if a["state"] == "wait":
            if f.now - a["asked_at"] <= p.aim.answer_wait_s:
                return False
            loop.miss()  # 판독 답이 오지 않았다
            if a.get("search") is not None:
                a["search"]["looking"] = False  # 표적을 찾던 사진이었다 → 다음 팬으로
            a["state"] = "look"
        search = a.get("search")
        if search is None and (loop.misses >= p.aim.max_misses or f.now - a["since"] > p.aim.budget_s):
            if a["held"] or a.get("answers", 0) == 0:
                # 사진으로 잡았다가 놓쳤다 → 고치던 각도로 쏜다. 판독 답이 한 번도 없었다(perception 이 죽었다) → 계산한 각도로 쏜다 (전의 방식)
                a.update(state="ready", checked_at=f.now)
                self._say(f, "조준 사진에서 수광부를 찾지 못함 (잇달아 %d장, %.0f초) → %s 각도로 쏜다"
                          % (loop.misses, f.now - a["since"], "사진으로 고치던" if a["held"] else "계산한"))
                return True
            # 판독은 되는데 계산한 각도의 사진에 표적이 없다 (위치 값이 틀렸다) → 팬을 돌려 가며 사진에서 표적 그림을 찾는다
            search = a["search"] = {"pans": [q for q in p.search_pans if abs(q - s["pan"]) > 8.0], "i": 0}
            self._say(f, "계산한 각도의 사진에 표적이 없음 (잇달아 %d장) → 팬을 돌려 가며 사진에서 표적 그림을 찾는다" % loop.misses)
        if search is not None and not search.get("looking"):
            if search["i"] >= len(search["pans"]):
                self._strike_defer(f, "사진에서 표적 그림을 찾지 못함")
                return False
            if ports.aim(search["pans"][search["i"]], s["tilt"]):
                s.update(aimed_at=f.now, pan=search["pans"][search["i"]])
                search.update(i=search["i"] + 1, looking=True)
                loop.interrupted()
            return False  # 팬틸트가 다 돈 뒤(다음 틱들)에 찍는다
        dist = math.hypot(face.cx - pose[0], face.cy - pose[1])
        tag = ("aim", self.target_shots + 1, a["n"])
        if shoot(tag):
            z = a.get("z") or surface_z(dist, self.cam.forward_m, p.aim)  # 사진에서 표적을 찾았으면 그림 상자의 높이로 어림한 거리
            u, v = laser_pixel(z, p.aim)
            a.update(state="wait", tag=tag, asked_at=f.now, n=a["n"] + 1, laser=(u + self._laser_fix[0], v + self._laser_fix[1]))
        return False

    def _search_answer(self, f, a, found, want):
        """표적을 찾는 사진의 답 (팬을 돌려 가며). 표적 그림이 보이면 그쪽으로 겨누고 사진 조준을 잇는다. 안 보이면 다음 팬으로."""
        p, s, loop, search = self.p, self._strike, a["loop"], a["search"]
        search["looking"] = False
        a["state"] = "look"
        pics = [q for q in (found or {}).get("pictures") or () if q.get("kind") == want and q.get("w", 0.0) > 0.0 and q.get("h", 0.0) > 0.0]
        if not pics:
            return
        pic = min(pics, key=lambda q: abs(q["x"] - 0.5))  # 여럿이면 사진 가운데에 가까운 것
        # 지도의 거리는 위치 값이 틀리면 같이 틀린다 → 그림 상자의 높이(상자 한 변)로 렌즈에서의 거리를 어림한다 (제어 콘솔의 방식)
        tall = math.radians(max(1.0, pic["h"] * p.aim.image_h * p.aim.deg_px(self.cam.hfov_deg)))
        lens = self.fmap.box_size / (2.0 * math.tan(min(1.4, tall / 2.0)))
        z = min(max(lens - p.aim.z_offset_m, p.aim.z_min_m), p.aim.z_max_m)
        u, v = laser_pixel(z, p.aim)
        laser = (u + self._laser_fix[0], v + self._laser_fix[1])
        rec = find_receiver(found.get("pictures"), found.get("sensors"), want, (pic["x"] * p.aim.image_w, pic["y"] * p.aim.image_h), p.aim)
        if rec is None:
            return  # 그림은 보이는데 수광부 자리를 정하지 못했다 (잘린 상자) → 다음 팬에서 더 잘 보일 수 있다
        right, up = rec.u - laser[0], laser[1] - rec.v
        pan, tilt = s["pan"] + loop.k[0] * right, s["tilt"] + loop.k[1] * up
        if abs(pan) > PAN_LIMIT_DEG + 2.0 or abs(tilt) > TILT_LIMIT_DEG + 2.0:
            self._say(f, "사진에서 표적 그림을 찾았으나 팬틸트 범위 밖 (팬 %.0f°)" % pan)
            search["i"] = len(search["pans"])  # 이 자리에서는 못 쏜다 → 물러난다
            return
        pan, tilt = max(-PAN_LIMIT_DEG, min(PAN_LIMIT_DEG, pan)), max(-TILT_LIMIT_DEG, min(TILT_LIMIT_DEG, tilt))
        self._say(f, "사진에서 표적 그림을 찾음 (팬 %.0f° 의 사진, 오른쪽 %+.0f · 위 %+.0f 픽셀, 거리 약 %.2f m) → 팬 %.1f° 틸트 %.1f° 로 겨눈다"
                  % (s["pan"], right, up, lens, pan, tilt))
        loop.reset()
        a.update(search=None, held=True, z=z, state="turn", to=(pan, tilt), since=f.now, rec=rec)

    def _aim_answer(self, f, tag, found):
        """조준 확인 사진의 판독 답. found = {"pictures": [...], "sensors": [...] 또는 None} — 판독이 실패했거나 답이 없었으면 None."""
        s = self._strike
        a = s.get("aim") if s is not None else None
        if a is None or a["state"] != "wait" or a["tag"] is None or tuple(tag) != tuple(a["tag"]):
            return  # 기다리던 사진이 아니다 (늦게 온 답 · 이미 끝난 지정표적)
        loop = a["loop"]
        want = A.target_kind(self.designated)
        rec = None
        if isinstance(found, dict):
            a["answers"] = a.get("answers", 0) + 1  # 판독이 되기는 한다 (표적이 안 보이는 것과 perception 이 죽은 것을 가른다)
            if a.get("search") is not None:
                self._search_answer(f, a, found, want[1] if want else None)
                return
            rec = find_receiver(found.get("pictures"), found.get("sensors"), want[1] if want else None, a["laser"], self.p.aim)
        elif a.get("search") is not None:
            a["search"]["looking"] = False  # 판독 실패 → 다음 팬으로
            a["state"] = "look"
            return
        if rec is not None:
            a["rec"] = rec  # 쏜 뒤 레이저 점과 견줄 수광부 자리
        act = loop.see(rec, a["laser"], s["pan"], s["tilt"])
        if act[0] == "look":
            a["state"] = "look"
            return
        a["held"] = True
        how = {"sensor": "탐지", "near": "탐지 · 그림 없이", "box": "그림 상자에서 어림"}.get(rec.how, rec.how)
        if act[0] == "turn":
            a.update(state="turn", to=(act[1], act[2]))
            self._say(f, "조준 사진 %d: 수광부(%s)가 레이저 점에서 오른쪽 %+.0f · 위 %+.0f 픽셀 → 팬 %+.1f° · 틸트 %+.1f° 더"
                      % (loop.corrections, how, loop.offset[0], loop.offset[1], act[1] - s["pan"], act[2] - s["tilt"]))
        else:
            a.update(state="ready", checked_at=f.now)
            self._say(f, "조준 %s: 수광부(%s)와 레이저 점의 차이 %.1f 픽셀 (고침 %d번)"
                      % ("맞춤" if loop.converged else "덜 맞은 채 쏜다", how, loop.err, loop.corrections))
        for name, k in loop.relearned[a["told"]:]:
            self._say(f, "팬틸트 %s: 1픽셀 옮기는 데 %+.3f° 로 다시 정함 (처음 값 %.3f°) — 단위나 방향이 설정과 다르다" % (name, k, loop.k0))
        a["told"] = len(loop.relearned)

    # ================================================================== 상대 UGV 따라가며 쏘기 (선택 사항, ugv_track.py)
    def engage_ugv(self, f, ports):
        """보이는 상대 UGV 의 수광부를 따라가며 쏜다. 한 틱에 한 번 부른다 (로봇은 서 있어야 한다). 따라가는 중이면 True.

        기본 흐름(tick)에서는 engage_ugvs 가 True 일 때 할 일이 없어 서 있는 동안(rest)만 부른다. 언제 쓸지는 팀이 정한다 — 다른 때에 쓰려면
        그 자리에서 이 함수를 부르면 된다. 아군과 진영을 못 읽은 UGV 는 쏘지 않는다 (engage_unknown_camp). 지정표적을 아직 쏴야 하면 쏘지 않고
        (탄과 재장전 시간을 지정표적에 먼저 쓴다), 발 수는 ugv_shots 까지다.
        """
        p = self.p
        locate = getattr(ports, "locate", None)
        if locate is None or self.foe is None:
            return False
        e = self._engage
        if e is None:
            e = self._engage = {"asked": None, "asked_at": 0.0, "pan": None, "tilt": None, "next_ask": -1e9, "sign": "", "since": None, "kind": None}
            self._tracker.reset()
        if e["asked"] is not None:
            got = next((a for a in f.sensor_answers if a[0] == e["asked"]), None)
            if got is not None:
                _tag, pan, tilt, spots = got
                # 목숨이 0 으로 읽힌 팀은 쏘지 않는다. 여러 대가 보이면 따라가던 표적을 계속 따라간다 (UgvTracker.choose)
                dead = tuple(s for s in self.ugvs if self.lives_of(s) == 0)
                spot, switched = self._tracker.choose(foe_spots(spots, self.foe, p.track, p.engage_unknown_camp, dead), e["asked_at"], pan, tilt)
                if spot is not None:
                    if switched:
                        # 다른 표적으로 바뀌었다 (또는 한 번에 gate_deg 넘게 뛸 만큼 빨리 움직인다) → 처음부터 다시 본다.
                        # 붙든 시간(since)은 그대로 둔다 — 빠르게 오가는 표적 하나가 매번 「새 표적」 이 되어 끝없이 붙들지 않게
                        self._tracker.reset()
                        e["sign"] = ""
                    self._tracker.observe(e["asked_at"], spot, pan, tilt)
                    e["pan"], e["tilt"] = pan, tilt
                    e["sign"] = spot.get("callsign") or e["sign"]
                    if e["since"] is None:
                        e["since"] = e["asked_at"]
                    shot = self._ugv_shot
                    if shot is not None and e["asked_at"] >= shot["at"]:
                        # 쏜 뒤의 사진: 레이저 점이 수광부 위에 있고, 표적이 움직이기 시작하지 않았는가 (움직이면 2초를 이어서 닿지 못한다)
                        shot["n"] += 1
                        left = switched or not self._tracker.on_sensor() or self._tracker.motion(e["asked_at"]) in ("moving", "fast")
                        shot["off"] += 1 if left else 0
                e["asked"] = None
            elif f.now - e["asked_at"] > p.locate_timeout_s:
                e["asked"] = None
        if self._fire_asked and f.now - self._fire_asked_at > p.fire_reply_wait_s:
            self._fire_asked = False  # 발사 요청의 답이 오지 않았다 → 다시 쏠 수 있게
        ready = getattr(ports, "aim_ready", lambda: True)()
        if p.engage_still_only:
            # 멈춘 표적 · 조금씩 움직이는 표적만 쏜다 (UgvTracker.motion):
            #   still  멈춤 — 본 것들의 평균 자리를 겨눠 쏜다
            #   creep  조금씩 — 곧은 선으로 내다본 자리를 겨눠 쏘고, 레이저가 켜진 동안 계속 고쳐 겨눈다
            #   그 밖  움직임 · 많이 움직임 — 마지막으로 본 자리를 따라가기만 한다 (멈추면 바로 쏠 수 있게). 쏘지 않는다
            point = self._tracker.aim_point(f.now, PAN_LIMIT_DEG, TILT_LIMIT_DEG)
            want = None if point is None else point[:2]
            e["kind"] = kind = None if point is None else point[2]
            on_target = (kind in ("still", "creep") and e["pan"] is not None
                         and math.hypot(point[3] - e["pan"], point[4] - e["tilt"]) <= min(p.track.tol_deg, self._tracker.tol()))
            # 멈춘 표적은 판독 잡음(1 ~ 2 픽셀)마다 다시 돌리지 않는다 — 레이저가 켜진 2초 동안 수광부에 머물러야 한다
            slack = 0.5 if kind == "still" else 0.3
        else:
            want = self._tracker.command(f.now, PAN_LIMIT_DEG, TILT_LIMIT_DEG)
            on_target = e["pan"] is not None and self._tracker.aligned(f.now, e["pan"], e["tilt"])
            slack = 0.3
        tracking = want is not None
        if tracking and ready and e["pan"] is not None:
            owed = self.target_face is not None and not self.target_done and self.target_shots < p.target_shots  # 지정표적을 아직 쏴야 한다
            can_fire = self.ugv_shots < p.ugv_shots and not owed and not self._fire_asked and f.now >= self._last_fire + p.fire_interval_s
            if can_fire and on_target:
                if ports.fire():
                    self._fire_asked, self._fire_asked_at = True, f.now
                    self._fire_for = "ugv"
                    self._fire_sign = e.get("sign") or self._one_foe_sign(f)  # 어느 팀을 쐈는가 (모르면 "")
            elif math.hypot(want[0] - e["pan"], want[1] - e["tilt"]) > slack and ports.aim(want[0], want[1]):
                e["pan"], e["tilt"] = want
                e["next_ask"] = f.now  # 돌린 뒤(aim_ready)에 바로 다시 찍는다
        if e["asked"] is None and ready and f.now >= e["next_ask"]:
            tag = ("ugv", self._locate_no)
            if locate(tag):
                self._locate_no += 1
                e.update(asked=tag, asked_at=f.now, next_ask=f.now + (0.0 if tracking else 1.0))  # 안 보일 때는 1초에 한 번만 찾아본다
        return tracking

    def aim_angles(self, pose, face):
        """그 면의 수광부를 겨누는 (팬, 틸트) 도 — 위치와 높이로 계산한 값. 범위 밖이면 None.

        틸트는 잰 값 둘로 계산한다: 수광부가 사진에서 레이저 점의 높이에 오게 하는 각도.
          · 틸트 0 에서 카메라는 camera_height 높이에서 수평을 본다 (2026-11-18 측정)
          · 레이저 점은 거리 Z 에서 사진의 v = 248.6 − 13.4 / Z 에 찍힌다 (2026-11-19 측정, aim.laser_pixel)
        전의 식(발광부 높이 emitter_height 에서 틸트 방향으로 곧게 나간다고 봄)은 이 두 측정으로 보면 0.9 m 앞에서 2 ~ 4° 낮게 겨눈다
        (수광부 높이를 0.35 m 로 두면 약 3 cm, 사진에서 잰 0.32 m 로 두면 약 6 cm 아래). camera_height 를 0 으로 두면 전의 식이다.
        """
        p = self.p
        x, y, yaw = pose
        dist = math.hypot(face.cx - x, face.cy - y)
        pan = -math.degrees(wrap(math.atan2(face.cy - y, face.cx - x) - yaw))
        if p.camera_height > 0.0:
            _u, v = laser_pixel(surface_z(dist, self.cam.forward_m, p.aim), p.aim)
            lens = max(dist - self.cam.forward_m, 0.05)
            up = math.degrees(math.atan2(p.target_receiver_height - p.camera_height, lens)) - (p.aim.image_h / 2.0 - v) * p.aim.deg_px(self.cam.hfov_deg)
        else:
            up = math.degrees(math.atan2(p.target_receiver_height - p.emitter_height, max(dist, 0.05)))
        tilt = up * p.aim.tilt_sign  # tilt_sign: 틸트 + 가 위면 1
        if abs(pan) > PAN_LIMIT_DEG or abs(tilt) > TILT_LIMIT_DEG:
            return None
        return pan, tilt

    def _standoff(self, face, avoid=()):
        """표적 면을 쏠 자리: 정면 가까이부터, 없으면 비스듬히. avoid = 가지 못했던 자리들 (그 둘레 0.35 m 는 고르지 않는다)."""
        import numpy as np

        from .coverage import _blocked_any

        segs = np.array(self.fmap.segments(), dtype=np.float64)
        base = math.atan2(face.ny, face.nx)
        qx, qy = face.cx + face.nx * 0.01, face.cy + face.ny * 0.01
        tries = [(d, off) for d in self.p.standoff_m for off in (0.0, 20.0, -20.0, 35.0, -35.0)]
        tries += [(d, off) for d in (0.5, 0.4, 0.6) for off in (0.0, 25.0, -25.0, 45.0, -45.0)]
        for dist, off in tries:
            ang = base + math.radians(off)
            sx, sy = face.cx + math.cos(ang) * dist, face.cy + math.sin(ang) * dist
            if not self.grid.free(sx, sy) or any(math.hypot(sx - ax, sy - ay) < 0.35 for ax, ay in avoid):
                continue
            if _blocked_any(np.array([sx]), np.array([sy]), np.array([qx]), np.array([qy]), segs)[0]:
                continue
            return sx, sy
        return None

    # ================================================================== M5 거점
    def _stage_point(self, value, from_wall):
        _x0, y0, _x1, y1 = self.fmap.bases[value]
        inward = self.fmap.inward(value)
        wall = 0.0 if inward > 0 else self.fmap.width
        heading = math.pi if inward > 0 else 0.0  # 거점(벽) 쪽을 본다
        y = self._stage_y[1] if self._stage_y is not None and self._stage_y[0] == value else (y0 + y1) / 2.0
        return (wall + inward * from_wall, y), heading

    def _time_to_stage(self, f):
        p = self.p
        if self.stage_base is None or f.t >= p.target_at_s + 30.0:
            return False
        if f.now - self._eta_at >= 1.0:  # 길 계산은 1초에 한 번
            goal, _h = self._stage_point(self.stage_base, p.stage_from_wall_m)
            path = self.grid.plan(f.pose[:2], goal)
            self._eta = path_length(f.pose[:2], path) / p.stage_speed + 3.0 if path else 30.0
            self._eta_at = f.now
        return f.t >= p.target_at_s - self._eta - p.stage_margin_s

    def _stage(self, f, ports):
        p = self.p
        if self._stage_y is None or self._stage_y[0] != self.stage_base:
            # 대기 자리의 세로 자리를 한 번 정한다: 그 거점 안 · 바로 앞에 다른 로봇이 서 있으면 그 로봇에서 떨어진 쪽 (들어갈 자리를 고르는 것과 같은 규칙).
            # 2026-11-20 03:28 연습 판: 가운데 거점 안 (0.27, 2.39) 에 다른 팀 로봇이 서 있었고 대기 자리 (0.80, 2.40) 은 그 로봇에서 0.53 m 였다.
            # 0.4 m 앞까지 갔다가 navigation 이 길을 4.2 m 짜리 우회로로 다시 잡아 상자 쌍을 돌아 나갔고, 4:00 에 대기 자리에서 1.6 m 떨어져 있어
            # 들어가는 데 12초가 걸렸다. 떨어진 쪽에서 기다리면 그 로봇에서 0.65 m 이고, 게시되면 그 자리에서 곧장 들어간다
            _x0, y0, _x1, y1 = self.fmap.bases[self.stage_base]
            mid = round((y0 + y1) / 2.0, 2)
            y = self._spots_clear_of_robots(f, self.stage_base, [mid], why="대기")[0]
            # 그 로봇에서 먼 쪽(+1 = y 가 커지는 쪽, −1 = 작아지는 쪽, 0 = 서 있는 로봇이 없다)도 함께 정해 둔다 — 그쪽으로 기울여 선다 (stage_lean_deg)
            near = min(self._robots_at_base(f, self.stage_base), key=lambda r: abs(r[1] - y), default=None)
            lean = 0 if near is None else (1 if y >= near[1] else -1)
            self._stage_y = [self.stage_base, y, lean]
        close = f.t >= p.target_at_s - p.close_lead_s
        goal, heading = self._stage_point(self.stage_base, p.close_from_wall_m if close else p.stage_from_wall_m)
        self.task = "stage close" if close else "stage"
        tol = p.stage_yaw_tol_deg
        lean = self._stage_y[2] if len(self._stage_y) > 2 else 0
        if lean and p.stage_lean_deg > 0.0:
            a = math.radians(p.stage_lean_deg)
            heading = math.atan2(lean * math.sin(a), math.cos(heading) * math.cos(a))
            tol = p.stage_lean_tol_deg
        self._stage_sent = (round(goal[0], 2), round(goal[1], 2), round(heading, 2))  # 창구가 「같은 목표」 를 가리는 값과 같게 (_guard_spot 이 본다)
        ports.go(goal[0], goal[1], heading, yaw_tol=math.radians(tol))
        return self.task

    def _assault_order(self, f):
        """들어가 볼 거점의 순서 [보고값, ...]. 2026-11-19 주최측 공지: 「M5 목표거점 — 위치는 고정, 작전지휘소 · 정비소 · 군수보급소의 배치 위치는 변경 가능」.

        게시된 목표(예 RED_MAINTENANCE)가 세 자리 중 어디인지 지도만으로는 알 수 없게 됐다 (지도의 이름은 연습 포털의 배치 — 위부터
        작전지휘소 · 정비소 · 군수보급소 — 를 따른 것이다). 목표가 아닌 상대 거점에 들어가도 아무 일도 없고(API 문서: 「목표가 아닌 거점에 들어가면
        점령이 아닙니다 — 아무 UGV도 멈추지 않고 경기는 계속되며, 점수도 없습니다」), 목표 거점에 들어가면 그 순간 경기가 끝난다.
        그래서 지도의 이름대로인 거점부터 들어가 보고, 경기가 안 끝나면 같은 진영의 나머지 거점을 가까운 것부터 차례로 들어간다.
        우리 진영의 거점은 절대 넣지 않는다 (우리 거점에 들어가면 상대의 점령이 될 수 있다).
        """
        camp = self.fmap.base_camp(f.objective) if f.objective else self.foe
        if camp != self.foe:  # 목표는 늘 상대 진영의 거점이다 — 값이 이상하면 상대 진영으로 본다
            camp = self.foe
        mine = self.fmap.bases_of(camp)
        first = f.objective if f.objective in mine else None

        def centre(value):
            _x0, y0, _x1, y1 = mine[value]
            return (y0 + y1) / 2.0

        ref = centre(first) if first is not None else f.pose[1]
        rest = sorted((v for v in mine if v != first), key=lambda v: (abs(centre(v) - ref), abs(centre(v) - f.pose[1])))
        return ([first] if first is not None else []) + rest

    def _assault(self, f, ports):
        """게시된 목표거점 안으로. 가는 길과 속도는 navigation 이 정한다 (목표가 거점 안이면 진입).

        - 어느 거점부터 들어갈지는 _assault_order (배치가 바뀔 수 있어, 경기가 안 끝나면 다음 거점으로 간다).
        - 들어갈 자리는 거점마다 한 번만 고른다 (3차 팀 테스트: 매 틱 「지금 내 y」 로 다시 계산해 50초에 60번 새 목표가 나갔다).
          거점의 양 끝에는 고정 기둥이 있으므로 끝에서 beam_inset_m 안쪽만 쓴다. navigation 이 못 갔다고 알리면(막힘 · 시한) 다음 자리를 보고,
          그 거점의 자리를 다 못 가면 다음 거점으로 간다.
        - 방향은 요구하지 않는다 (yaw_free). 벽에서 0.2 m 인 자리에서는 차체가 돌 자리가 없어, 방향을 맞추려다 모서리가 벽에 걸린 채 시한까지 간다 (가상).
          점령은 거점 센서가 가려지는 순간으로 정해지고 차체 방향과는 상관없다.
        - 「들어갔다」 = 차체 가운데가 들어갈 깊이(벽에서 0.20 m)에서 assault_in_m 안. 거점 네모(깊이 0.40 m)에 발만 들인 것으로는 치지 않는다 —
          센서 빔의 깊이를 모르므로([추정] 0.20 m) 차체가 그 깊이를 훑고 지나가게 끝까지 들어간다.
        """
        p = self.p
        self.task = "assault"
        if self._assault_plan is None or self._assault_plan[0] != f.objective:
            order = self._assault_order(f)
            self._assault_plan = [f.objective, order, 0]
            self._assault_goal = self._assault_asked = None
            self._entered_at = None
            self._say(f, "목표거점 %s — 들어가 볼 순서: %s" % (f.objective, " → ".join(order)))
        _objective, order, k = self._assault_plan
        if not order:
            ports.stop()
            return self.task
        value = order[k % len(order)]
        _x0, y0, _x1, y1 = self.fmap.bases[value]
        inward = self.fmap.inward(value)
        wall = 0.0 if inward > 0 else self.fmap.width
        depth = p.robot_half_length + p.wall_margin_m
        gx = wall + inward * depth
        heading = math.pi if inward > 0 else 0.0
        if self._assault_goal is None or self._assault_goal[0] != value:
            inset = min(p.beam_inset_m, (y1 - y0) / 2.0)
            near = min(max(f.pose[1], y0 + inset), y1 - inset)  # 지금 자리에서 가장 가까운 쪽
            spots = []
            for y in (near, (y0 + y1) / 2.0, y1 - inset if near < (y0 + y1) / 2.0 else y0 + inset, y0 + inset, y1 - inset):
                if not any(abs(y - other) < 0.05 for other in spots):
                    spots.append(round(y, 2))
            spots = self._spots_clear_of_robots(f, value, spots)
            self._assault_goal = [value, spots, 0, 0]  # [거점, 자리 후보들, 지금 후보 번호, 이 거점에서 못 간 횟수]
            # 이 거점에 서 있는 로봇이 있는지도 여기서 한 번만 정한다 — 있으면 자리 앞에 기울여 선 뒤에 돌진한다 (assault_guard 의 설명)
            stood = [r for r in self._robots_at_base(f, value) if abs(r[0] - wall) <= p.assault_guard_depth_m] if p.assault_guard else []
            self._assault_guard = [value, stood, None] if stood else None
        _value, spots, i, failed = self._assault_goal
        if y0 <= f.pose[1] <= y1 and abs(f.pose[0] - wall) <= depth + p.assault_in_m:
            self._assault_asked = None
            if not f.arrived:
                ports.stop()  # 끝까지 들어왔다 → 더 밀지 않고 선다
            if self._entered_at is None:
                self._entered_at = f.now
                self._say(f, "거점 %s 안에 들어옴 (%.2f, %.2f)" % (value, f.pose[0], f.pose[1]))
            elif f.now - self._entered_at >= p.capture_wait_s and len(order) > 1:
                # 목표 거점이었다면 들어온 순간 경기가 끝났을 것이다. 아직 진행 중 → 이 자리는 목표가 아니다 (배치가 바뀌었다) → 다음 거점으로
                self._assault_plan[2] = k + 1
                self._assault_goal = None
                self._entered_at = None
                self._say(f, "%.1f초가 지나도 경기가 끝나지 않음 → %s 은 목표가 아닌 것으로 보고 다음 거점 %s 로" % (p.capture_wait_s, value, order[(k + 1) % len(order)]))
            self.task = "assault in"
            return self.task
        self._entered_at = None
        lane = abs(f.pose[0] - wall) <= p.assault_from_m  # 거점 쪽 벽 가까이 (거점 앞 대기 자리 · 거점 안)
        in_front = lane and y0 - p.assault_side_m <= f.pose[1] <= y1 + p.assault_side_m
        # 한 번 돌진을 낸 거점에서는 되돌리지 않는다. 2026-11-20 02:23 연습 판(주행 담당의 확인 요청 — docs/거점_앞_목표_번갈아_보냄): 차체가
        # 「거점의 바로 앞」 을 가르는 금(y = 거점 끝 − assault_side_m) 위에서 서서 도는 동안 위치가 몇 mm 흔들릴 때마다 보통 목표와 돌진 목표를
        # 1.3초에 여섯 번 바꿔 보냈고, 그동안 제자리였다
        plain = self._dash_base == value or in_front or (k == 0 and lane)  # 전부터의 규칙: 여기서는 돌진을 낸다
        # 이 거점에 서 있는 로봇이 있으면(guard) 돌진은 「자리 앞에 기울여 선 뒤」 에 낸다 — 지금 자리에 이미 돌진을 냈으면 그대로 둔다 (assault_guard 의 설명)
        guard = self._guard_spot(f, value, spots, i, failed, wall, inward)
        if not (plain if guard is None else guard["dash"]):
            # 첫 거점도 아니고 이 거점의 바로 앞도 아니다 (다른 거점에서 옮겨 감 · 경기장 안쪽에서 옴) → 거점 앞까지는 보통 주행으로 (assault_from_m 의 설명)
            ax = wall + inward * p.close_from_wall_m
            if f.go_failed is not None and abs(f.go_failed[0] - round(ax, 2)) < 0.02 and abs(f.go_failed[1] - spots[i]) < 0.02:
                failed += 1
                if failed >= len(spots) and len(order) > 1:
                    self._assault_plan[2] = k + 1
                    self._assault_goal = None
                    self._say(f, "거점 %s 앞까지 %d번 못 감 → 다음 거점 %s 로" % (value, failed, order[(k + 1) % len(order)]))
                    return self.task
                self._assault_goal[2] = i = (i + 1) % len(spots)
                self._assault_goal[3] = failed
                self._say(f, "거점 %s 앞 (y = %.2f) 으로 다시 가 본다" % (value, spots[i]))
                guard = self._guard_spot(f, value, spots, i, failed, wall, inward)  # 새 자리의 기울일 쪽
            self._assault_asked = None
            if guard is None:
                ports.go(ax, spots[i], heading, yaw_free=True)
                return self.task
            if not self._guard_ready(f, guard, value, lane, plain):
                ports.go(ax, spots[i], guard["yaw"], yaw_tol=math.radians(p.stage_lean_tol_deg))
                return self.task
        ask = self._assault_asked
        if ask is not None and ask[0] != (round(gx, 2), spots[i]):
            ask = self._assault_asked = None
        if ask is not None and not f.arrived:
            ask[2] = True
        # navigation 은 끝났다는데(REACHED) 위의 「들어갔다」 깊이에 못 미쳤다: 돌진으로 벽 · 기둥 곁에 서서 목표에서 0.25 m 안이면 도착으로 끝내고
        # (REACHED [POS_NOT_MET]), 위치 허용 0.10 m 끝에 걸려 선 경우도 있다. 같은 목표는 창구가 다시 보내지 않아(node.go) 전에는 여기서 경기 끝까지
        # 서 있었다 (2026-11-19 주행 통합 뒤 정리). 「못 갔다」 와 같이 다음 자리로 다시 들어간다. 「가는 중」 을 못 봤으면 5초 뒤에 (목표가 안 나간 경우)
        short = ask is not None and f.arrived and (ask[2] or f.now - ask[1] >= 5.0)
        if short:
            ports.stop()
            self._say(f, "navigation 은 도착이라는데 거점 %s 안 깊이에 못 미침 (%.2f, %.2f)" % (value, f.pose[0], f.pose[1]))
        if short or (f.go_failed is not None and abs(f.go_failed[0] - round(gx, 2)) < 0.02 and abs(f.go_failed[1] - spots[i]) < 0.02):
            self._assault_asked = None
            failed += 1
            if failed >= len(spots) and len(order) > 1:  # 이 거점의 자리를 다 못 갔다 (안에 다른 로봇이 있는 등) → 다음 거점부터 가 본다
                self._assault_plan[2] = k + 1
                self._assault_goal = None
                self._say(f, "거점 %s 의 자리 %d곳에 다 못 감 → 다음 거점 %s 로" % (value, len(spots), order[(k + 1) % len(order)]))
                return self.task
            self._assault_goal[2] = i = (i + 1) % len(spots)
            self._assault_goal[3] = failed
            self._say(f, "거점 안의 다른 자리로 다시 들어가 본다 (y = %.2f)" % spots[i])
            if guard is not None:
                # 서 있는 로봇이 있는 거점: 새 자리에도 그 앞에 기울여 선 뒤에 돌진한다 (전에는 서 있던 자리에서 바로 다음 돌진을 냈다)
                guard = self._guard_spot(f, value, spots, i, failed, wall, inward)
                if guard is not None and not self._guard_ready(f, guard, value, lane, plain):
                    ports.go(wall + inward * p.close_from_wall_m, spots[i], guard["yaw"], yaw_tol=math.radians(p.stage_lean_tol_deg))
                    return self.task
        if self._assault_asked is None:
            self._assault_asked = [(round(gx, 2), spots[i]), f.now, False]
        self._dash_base = value
        if guard is not None:
            guard["dash"] = True  # 이 자리에는 돌진을 냈다 → navigation 이 결과를 알릴 때까지 보통 목표로 되돌리지 않는다
        ports.go(gx, spots[i], heading, yaw_free=True, mode="assault")
        return self.task

    def _guard_spot(self, f, value, spots, i, failed, wall, inward):
        """서 있는 로봇이 있는 거점에서 지금 들어갈 자리의 상태 (dict) — 그런 거점이 아니면 None (전과 같이 한다). assault_guard 의 설명.

        자리가 바뀌거나(못 가서 다음 자리 — failed 가 는다) 빠져나오기가 끼어들었으면(escapes) 새로 만든다: 그 자리 앞에 다시 서는 것부터 한다.
        기울일 쪽(lean: +1 = y 가 커지는 쪽)은 그 자리에서 세로로 가장 가까운 로봇의 먼 쪽 — 대기 자리(_stage)와 같은 규칙. 예외를 밖으로 내지 않는다.
        """
        g = self._assault_guard
        if g is None or g[0] != value:
            return None
        try:
            p, st, y = self.p, g[2], spots[i]
            if st is None or (st["i"], st["y"], st["failed"], st["escapes"]) != (i, y, failed, self.escapes):
                near = min(g[1], key=lambda r: abs(r[1] - y))
                lean = 1 if y >= near[1] else -1
                a = math.radians(max(p.stage_lean_deg, 0.0))
                ax = wall + inward * p.close_from_wall_m
                face = -1.0 if inward > 0 else 1.0
                yaw = math.atan2(lean * math.sin(a), face * math.cos(a))
                away = math.hypot(ax - f.pose[0], y - f.pose[1])
                # 이 자리의 앞 목표가 방금까지의 대기 목표(_stage)와 같고 navigation 이 그 목표를 이미 끝냈다(도착 · 그 둘레에 서 있다) → 다시 보내도 창구가
                # 보내지 않으니(같은 목표) 더 맞춰질 것이 없다. 준비가 안 됐어도(0.12 m 넘게 어긋나 섬 · 방향을 못 맞춤) 전처럼 그 틱에 돌진한다
                sent, self._stage_sent = self._stage_sent, None
                same = bool(f.arrived and away <= 0.25 and sent == (round(ax, 2), round(y, 2), round(yaw, 2)))
                st = g[2] = {"i": i, "y": y, "failed": failed, "escapes": self.escapes, "lean": lean, "face": face, "yaw": yaw,
                             "since": f.now, "limit": away / max(p.stage_speed, 0.05) + p.assault_guard_wait_s,
                             "asked": same, "moving": same, "dash": False}
            return st
        except Exception as error:  # noqa: BLE001 — 덤으로 하는 일이다. 여기서 난 예외로 거점 진입이 멈추면 안 된다
            self._assault_guard = None
            self._say(f, "거점 %s 의 자리 앞에 기울여 서는 것을 정하다 예외 (%s: %s) → 이 거점은 전처럼 바로 돌진한다" % (value, type(error).__name__, error))
            return None

    def _guard_ready(self, f, st, value, lane, plain):
        """서 있는 로봇이 있는 거점의 자리 st 에 지금 돌진을 내도 되는가. False 면 그 자리의 앞(통로)으로 보통 주행을 더 한다 (부른 쪽이 목표를 보낸다).

        True 인 때: 설 준비가 됐다 · navigation 이 앞 자리에 도착했다고 알렸다(방향을 못 맞췄어도 더 기다리지 않는다) · 너무 오래 걸렸고 전 같으면 돌진을 냈을 자리다(plain).
        무슨 일이 있어도 예외를 밖으로 내지 않는다 — 예외면 전처럼 한다 (plain).
        """
        p = self.p
        try:
            # 차체가 벽 쪽에서 기운 각 (+ = 그 로봇의 먼 쪽, − = 그 로봇 쪽). 벽을 등지면 ±90° 를 넘는다
            tilt = math.degrees(math.atan2(st["lean"] * math.sin(f.pose[2]), st["face"] * math.cos(f.pose[2])))
            side = abs(f.pose[1] - st["y"])
            waited = f.now - st["since"]
            if st["asked"] and not f.arrived:
                st["moving"] = True
            why = None
            if lane and side <= p.assault_guard_side_m and -p.assault_guard_toward_deg <= tilt <= p.assault_guard_away_deg:
                why = "자리 앞에 섰다"
            elif st["asked"] and f.arrived and (st["moving"] or waited >= p.assault_guard_grace_s):
                why = "navigation 이 자리 앞에 도착했다고 알렸는데 설 준비는 안 됐다 — 더 기다리지 않는다"
            elif plain and waited >= st["limit"]:
                why = "자리 앞에 서는 데 %.1f초 넘게 걸렸다 — 더 기다리지 않는다" % st["limit"]
            if why is None:
                if not st["asked"]:
                    st["asked"] = True
                    self._say(f, "거점 %s 에 서 있는 로봇이 있어 자리 y = %.2f 의 앞에 먼저 선다: 방향 %.0f° (그 로봇의 먼 쪽으로 %.0f°)" % (
                        value, st["y"], math.degrees(st["yaw"]), max(p.stage_lean_deg, 0.0)))
                return False
            self._say(f, "거점 %s 자리 y = %.2f 로 돌진: %s (세로로 %.2f m · 기울기 %+.0f° — + 가 그 로봇의 먼 쪽 · %.1f초 걸림)" % (value, st["y"], why, side, tilt, waited))
            return True
        except Exception as error:  # noqa: BLE001
            self._assault_guard = None  # 이 거점에서는 더 하지 않는다 (틱마다 같은 예외가 나지 않게)
            self._say(f, "거점 %s 의 자리 앞에 섰는지 보다 예외 (%s: %s) → 이 거점은 전처럼 한다" % (value, type(error).__name__, error))
            return bool(plain)

    def _robots_at_base(self, f, value):
        """거점 value 안 · 바로 앞에 서 있는 다른 로봇들 [(x, y), ...] (위에서 본 것 · 요즘 assault_robot_memory_s 안에 본 것). 모르면 빈 목록.
        예외를 밖으로 내지 않는다."""
        p = self.p
        try:
            _x0, y0, _x1, y1 = self.fmap.bases[value]
            wall = 0.0 if self.fmap.inward(value) > 0 else self.fmap.width
            known = [(r[0], r[1]) for r in (getattr(self.scout, "robots", None) or ())]
            known += [(x, y) for x, y, at in self._stood.values() if f.now - at <= p.assault_robot_memory_s]
            inside = []
            for rx, ry in known:
                rx, ry = float(rx), float(ry)
                if (y0 - 0.20 <= ry <= y1 + 0.20 and abs(rx - wall) <= p.assault_robot_near_m
                        and not any(math.hypot(rx - ox, ry - oy) < 0.25 for ox, oy in inside)):
                    inside.append((rx, ry))
            return inside
        except Exception:  # noqa: BLE001
            return []

    def _spots_clear_of_robots(self, f, value, spots, why="진입"):
        """들어갈 거점 안 · 바로 앞에 다른 로봇이 서 있으면 그 로봇에서 떨어진 자리부터 들어가 보게 후보의 차례를 바꾼다 (assault_robot_gap_m 의 설명).
        서 있는 로봇을 모르면(드론 사진이 끊김 · 화면 밖) 받은 차례 그대로. 무슨 일이 있어도 예외를 밖으로 내지 않는다."""
        p = self.p
        try:
            _x0, y0, _x1, y1 = self.fmap.bases[value]
            inside = self._robots_at_base(f, value)
            if not inside:
                return spots
            tight = min(p.beam_inset_tight_m, (y1 - y0) / 2.0)
            cand = list(spots)
            for y in (y0 + tight, y1 - tight):
                if not any(abs(y - other) < 0.05 for other in cand):
                    cand.append(round(y, 2))

            def gap(y):
                return min(abs(y - ry) for _rx, ry in inside)

            good = sorted((y for y in cand if gap(y) >= p.assault_robot_gap_m), key=lambda y: abs(y - f.pose[1]))
            rest = sorted((y for y in cand if gap(y) < p.assault_robot_gap_m), key=lambda y: -gap(y))
            if why == "대기":
                self._say(f, "거점 %s 에 다른 로봇이 서 있음%s → 그 로봇에서 떨어진 쪽에서 기다린다: y = %.2f" % (
                    value, "".join(" (%.2f, %.2f)" % q for q in inside), (good + rest)[0]))
            else:
                self._say(f, "거점 %s 에 다른 로봇이 서 있음%s → 그 로봇에서 떨어진 자리부터 들어가 본다: y = %s" % (
                    value, "".join(" (%.2f, %.2f)" % q for q in inside), ", ".join("%.2f" % y for y in good + rest)))
            return good + rest
        except Exception as error:  # noqa: BLE001
            self._say(f, "거점 안의 로봇을 피할 자리를 고르다 예외 (%s: %s) → 전의 차례대로" % (type(error).__name__, error))
            return spots

    # ================================================================== 저장 · 복구
    def snapshot(self):
        """파일에 적어 둘 것 (checkpoint.py). 면 기록 · M1 해석 · 서버가 받은 보고 · 쏜 횟수."""
        return {"m1": self.m1, "sent": {str(k): v for k, v in self._sent.items() if k in (1, 3)}, "target_shots": self.target_shots,
                "target_done": self.target_done, "ugvs": self.ugvs, "book": self.book.snapshot() if self.book is not None else self._restored_book}

    def restore(self, data):
        """다시 켜졌을 때 snapshot() 으로 적어 둔 것을 되살린다. 면 기록은 지도가 온 뒤에 들어간다."""
        if data.get("m1"):
            self.m1 = dict(data["m1"])
            self.designated = self.m1.get("target") or None
        self._sent.update({int(k): v for k, v in (data.get("sent") or {}).items()})
        self.target_shots = int(data.get("target_shots") or 0)
        self.target_done = bool(data.get("target_done"))
        self.ugvs = dict(data.get("ugvs") or {})
        self._restored_book = data.get("book") or None
