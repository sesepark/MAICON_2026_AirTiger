"""목표 하나를 실행하는 상태 기계. 제어 주기마다 ``step()`` 을 부르면 (v, w) 를 돌려준다.

모드
  NORMAL    A* 경로 → 앞 lookahead_m 의 경로 점을 따라감(방향 오차가 크면 서서 돎) → 감속 → 정지
            → settle_s 기다려 새 위치로 확인 → (방향을 쓰면) 닫힌 루프 회전 → 다시 확인 → REACHED
  STRAIGHT  회전 없이 시작할 때의 방향으로만 앞뒤로 간다 (M5 다가서기). 시작 때 목표 방향과 차체 방향이
            straight_max_heading_deg 넘게 다르면 INVALID_GOAL
  ASSAULT   NORMAL 과 같은 경로지만 assault_v 를 유지하다가 assault_decel 제동 곡선으로만 선다 (M5 진입).
            목표가 벽 가까이(부풀린 구역 안)여도 된다 — 가장 가까운 빈 칸까지 경로, 그다음 곧게

막힘 (D10, 2026-11-18 팀 테스트 뒤 고침)
  진행 띠(지금 도는 호를 따라 굽힌 차체 폭 띠) 앞에 지도에 없는 물체:
    slow_dist 안이면 obstacle_decel 로 설 수 있게 감속, 「stop_dist + 속도 × stop_reaction_s」 안이면 정지
    → wait_before_replan_s 기다림 → 임시 장애물을 넣고 다시 짬.
  닿았거나 바로 앞(backoff_dist, 서서 돌 때는 차체가 쓸고 갈 원)에 backoff_after_s 넘게 있으면 뒤가 빈 경우 뒤로 뺀다.
  목표에 blocked_report_s 동안 가까워지지 못하면 피드백 blocked=true 를 내되 계속 시도한다. 다만 임시 장애물을 넣으면 길이 없는
  (그 물체를 지나야만 갈 수 있는) 목표에서 obstacle_give_up_s 동안 못 나아가면 시한 전에 BLOCKED 로 끝낸다.
  제한 시간이 지났을 때 여전히 막혀 있으면 BLOCKED, 아니면 TIMED_OUT.

서서 돌기와 돌 자리 (2026-11-19)
  서서 돌기 전에 도는 쪽으로 차체가 쓸고 갈 자리를 본다 — 지도에 없는 점(sweep_margin)과 지도 구조물(벽 · 박스 · 고정 기둥,
  structure_turn_clear). 막히면 반대쪽으로 길게 돈다. 돌라고 rot_stuck_s 냈는데 방향이 안 바뀌어도(지도와 실제가 어긋남) 막힌 것으로 본다.
  구조물 · 안 도는 회전으로 막히면
    자리에는 왔는데 도착 방향으로 못 돎 → 비키지 않고 바로 REACHED, detail 에 NO_ROOM_AT_GOAL (미션을 이어 가게 — 방향은 final_pose 로)
    목표 쪽으로 출발할 수 없음         → 구조물에서 멀어지는 쪽(앞뒤 곧게 · 완만한 호)으로 no_room_moves 번까지 비켜 보고, 안 되면
                                          BLOCKED, detail 에 NO_ROOM_TO_LEAVE
  (detail 은 목표를 옮겼으면 「목표를 … 로 옮김; 」 뒤에 붙는다). 도착 방향 맞추기는 align_max_s 가 지나면 남은 각도와 상관없이 REACHED.

되짚기 · 뒤로 갈 때의 진행 확인 (2026-11-19 밤)
  시한 전에 BLOCKED 로 끝내려는 참에(OBSTACLE · NO_PROGRESS · NO_ROOM_TO_LEAVE) 그 자리가 빠듯하면(통로 안 · 차체가 돌며 쓸 원 곁에
  구조물이나 지도에 없는 점) 끝내기 전에 지나온 길(drive/trail.py, 로컬 궤적)을 돌지 않고 뒤로 retrace_m 까지 되짚어 물러난다 (phase retracing).
  다 물러나면 처음에 내려던 결과로 끝낸다 — detail 끝에 「지나온 길로 0.60 m 물러난 뒤 끝냄」. 돌 자리 없음(출발)이었으면 물러난 뒤에는
  그 말 · 코드 대신 [NO_PROGRESS] 를 싣는다 (돌 자리가 생겼다).
  뒤로 가라는 명령(뒤로 빼기 · 비키기 · 되짚기 · MODE_STRAIGHT 뒤로)이 reverse_check_s 이어졌는데 차체가 거의 안 갔으면 뒤가 막힌 것으로 보고
  그만두며, rear_blocked_s 동안(목표가 바뀌어도) 다시 뒤로 가지 않는다. MODE_STRAIGHT 로 뒤로 가던 목표는 BLOCKED [OBSTACLE] 로 끝낸다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from . import pose_tracker as loc
from .geometry import path_length, wrap
from .trail import Trail
from .turn_rate import SpinRate

MODE_NORMAL, MODE_STRAIGHT, MODE_ASSAULT = 0, 1, 2
MODE_NAMES = {MODE_NORMAL: "normal", MODE_STRAIGHT: "straight", MODE_ASSAULT: "assault"}

REACHED, CANCELED, BLOCKED, TIMED_OUT, CONTROL_LOST, INVALID_GOAL, LOCALIZATION_LOST = 1, 2, 3, 4, 5, 6, 7
OUTCOME_NAMES = {
    REACHED: "REACHED", CANCELED: "CANCELED", BLOCKED: "BLOCKED", TIMED_OUT: "TIMED_OUT",
    CONTROL_LOST: "CONTROL_LOST", INVALID_GOAL: "INVALID_GOAL", LOCALIZATION_LOST: "LOCALIZATION_LOST",
}

YAW_IGNORE_TOL = math.pi  # yaw_tolerance_rad 가 이 이상이면 방향을 보지 않는다 (supervisor 요청서: 3.15)
WAYPOINT_REACHED_M = 0.18
STOP_FRACTION = 0.3  # 허용 오차의 이만큼 + 미끄러질 거리 앞에서 정지 명령 → 허용 오차 가운데 쪽에 선다
ALIGN_MOVE_AFTER_S = 0.3  # 도착 방향 맞추기가 서 있는 물체에 막힌 채 이만큼 이어지면 비켜 선다 (스캔 한두 장의 튄 점에는 비키지 않는다)

# 돌 자리가 없어 끝낸 BLOCKED 의 detail 에 들어가는 말. supervisor 는 이 말이 detail 에 있는지로 구별한다
# (목표를 옮겼으면 그 말 뒤에 붙으므로 「들어 있는지」 로 본다) — 바꾸면 interfaces.yaml navigate 와 supervisor 를 함께 (2026-11-19)
NO_ROOM = "돌 자리 없음"
NO_ROOM_AT_GOAL = NO_ROOM + "(도착 방향)"  # 위치는 허용 안에 왔는데 도착 방향으로 서서 돌 자리가 없다
NO_ROOM_TO_LEAVE = NO_ROOM + "(출발)"  # 목표 쪽으로 가려면 서서 돌아야 하는데 돌 자리가 없다 (자리에는 못 갔다)
# 결과 코드 (2026-11-19). detail 끝에 「 [코드]」 로 붙는다 (여럿이면 「 [A] [B]」). 한글 문장은 사람이 읽는 것이고 바뀔 수 있다 —
# supervisor 는 "[YAW_NOT_MET]" in detail 처럼 코드가 들어 있는지로 가른다. 바꾸면 interfaces.yaml navigate 와 supervisor 를 함께.
# REACHED 의 두 코드는 끝낼 때의 자세로 finish() 가 스스로 붙인다 (어느 길로 끝났든 빠지지 않게).
CODE_YAW_NOT_MET = "YAW_NOT_MET"  # REACHED 인데 방향이 요청한 허용 밖 (실제 방향은 final_pose)
CODE_POS_NOT_MET = "POS_NOT_MET"  # REACHED 인데 위치가 요청한 허용 밖 (돌진으로 벽 옆에 선 경우)
CODE_NO_ROOM_AT_GOAL = "NO_ROOM_AT_GOAL"  # 자리에는 왔는데 도착 방향으로 서서 돌 수 없다 (REACHED 와 함께)
CODE_NO_ROOM_TO_LEAVE = "NO_ROOM_TO_LEAVE"  # BLOCKED: 목표 쪽으로 서서 돌 수 없다 (비켜 봐도 안 됨)
CODE_OBSTACLE = "OBSTACLE"  # 지도에 없는 물체(다른 UGV · 빠진 상자) 때문에 나아갈 방법이 없다 — 나중에는 열릴 수 있다
CODE_NO_PROGRESS = "NO_PROGRESS"  # BLOCKED: 가는 중인데 no_progress_give_up_s 동안 가까워지지 못했다 (길이 자꾸 바뀜 · 끼임)
CODE_BLOCKED_AT_TIMEOUT = "BLOCKED_AT_TIMEOUT"  # BLOCKED: 시한이 됐을 때 막혀 있었다
CODE_INTERNAL = "INTERNAL_ERROR"  # BLOCKED: 주행 노드 내부 오류
TURN_STEP = math.radians(3.0)  # 서서 돌기 검사의 각도 간격 (반 대각선 0.17 m 에서 모서리가 9 mm 씩 움직인다)


def _rect_gap(dx, dy, a, b):
    """축에 나란한 사각형(반 크기 a, b, 가운데 원점)과 점 (dx, dy) 사이의 틈. 안이면 음수(가장 가까운 변까지)."""
    ox, oy = abs(dx) - a, abs(dy) - b
    if ox <= 0.0 and oy <= 0.0:
        return max(ox, oy)
    return math.hypot(max(ox, 0.0), max(oy, 0.0))


def body_clearance(spec, x, y, yaw, half_length, half_width):
    """차체 사각형(가운데 x, y · 방향 yaw)과 지도 구조물(경기장 벽 · 박스 · 고정 기둥) 사이의 가장 짧은 틈 (m). 겹치면 0 이하.

    두 볼록 사각형의 틈은 한쪽의 꼭짓점과 다른 쪽 사이에서 나온다. 박스 · 기둥(0.40)이 차체(0.25 × 0.23)보다 커서
    꼭짓점 없이 「+」 꼴로 겹칠 수도 없다. 그래서 꼭짓점 8개씩만 보면 정확하다.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    corners = [(x + c * u - s * v, y + s * u + c * v)
               for u, v in ((half_length, half_width), (half_length, -half_width), (-half_length, -half_width), (-half_length, half_width))]
    gap = min(min(px, spec.width - px, py, spec.height - py) for px, py in corners)
    half = spec.box_size / 2.0
    reach = math.hypot(half_length, half_width) + half
    for bx, by in spec.solids:
        if max(abs(bx - x), abs(by - y)) - reach >= gap:
            continue  # 이 박스는 지금 찾은 틈보다 멀다
        for px, py in corners:
            gap = min(gap, _rect_gap(px - bx, py - by, half, half))
        for qx, qy in ((bx - half, by - half), (bx - half, by + half), (bx + half, by - half), (bx + half, by + half)):
            dx, dy = qx - x, qy - y
            gap = min(gap, _rect_gap(c * dx + s * dy, -s * dx + c * dy, half_length, half_width))
    return gap


@dataclass
class GoalSpec:
    x: float
    y: float
    yaw: float | None  # None = 방향 상관없음
    mode: int
    pos_tol: float
    yaw_tol: float
    timeout_s: float
    frame_id: str = "arena"
    run_id: str = ""


@dataclass
class Command:
    v: float = 0.0
    w: float = 0.0


STOP = Command()


def _clamp(value, limit):
    return max(-limit, min(limit, value))


class TurnModel:
    """서서 돌기에서 실제로 잰 것 — 목표가 바뀌어도 이어 쓴다 (주행 노드가 하나를 갖고 목표마다 넘긴다).

    2026-11-19 실차 기록(연습 경기 다섯 판)으로 잰 회전: 서서 도는 명령 0.8 · 1.2 · 1.4 rad/s 는 표본의 52 ~ 100 % 가 거의 안 돌았고,
    1.6 rad/s 는 실제 중앙 0.39 rad/s, 멈춘 뒤 더 돈 각도는 중앙 −1° · 최대 10° 였다. 그런데 목표마다 이것을 처음부터 다시 배워
    (정지 여유 |w| × coast_s = 18° 에서 시작, 최소 0.8 rad/s 에서 돌기 시작) 방향 맞추기 한 번이 3 ~ 6 s 걸렸다.
    lead    정지 명령 뒤에도 더 도는 각도 (rad). None = 아직 못 쟀다 (그때는 |w| × coast_s)
    start_w 서 있다가 실제로 돌기 시작한 명령 각속도 (rad/s). 방향 맞추기는 이보다 작게 내지 않는다. None = 아직 못 쟀다 (min_turn_w)
    """

    def __init__(self):
        self.lead = None
        self.start_w = None
        self.rate = SpinRate()  # 실제로 도는 빠르기 (drive/turn_rate.py — 주행 노드가 SLAM 방향 · IMU 각속도를 넣는다)
        self.boost = 0.0        # 바라는 빠르기로 돌게 한 「올린 양」 (rad/s, _spin_boost). 다음 회전은 여기서 시작한다
        self.last_rate = None   # 마지막으로 잰 실제 빠르기 · 그때 낸 명령 (기록용)
        self.last_cmd = None

    def summary(self):
        out = {'lead_deg': None if self.lead is None else round(math.degrees(self.lead), 1),
               'start_w': None if self.start_w is None else round(self.start_w, 2),
               'boost': round(self.boost, 2),
               'rate': None if self.last_rate is None else round(self.last_rate, 2),
               'cmd': None if self.last_cmd is None else round(self.last_cmd, 2)}
        out.update(self.rate.summary())
        return out


def _wall_gap(grid, x, y):
    """(x, y) 에서 가장 가까운 외벽까지의 거리 [m] — 거점은 외벽에 붙어 있으므로 돌진 목표의 「깊이」 다."""
    return min(x, grid.width - x, y, grid.height - y)


class GoalExecutor:
    def __init__(self, goal: GoalSpec, params, start_t: float, turn_model: TurnModel | None = None, trail: Trail | None = None):
        self.goal = goal
        self.trail = trail if trail is not None else Trail()  # 지나온 길 · 「뒤가 막혔다」 는 기억 (주행 노드가 목표 사이에 이어 준다)
        self._retrace = None  # 되짚어 물러나는 중: {since, start, outcome, detail, code} — 다 물러나면 그 결과로 끝낸다
        self._retraced = False  # 이 목표에서 이미 한 번 되짚었다
        self._rev_ref = None  # 뒤로 가라는 명령이 이어진 동안: [시작 시각, 그때 자세, 그 뒤로 가라고 한 거리, 마지막 주기 시각]
        self._rev_stalled = False  # 뒤로 가라고 했는데 안 움직였다 (이번 되짚기를 끝낸다)
        self._t = start_t
        self._static_since = None  # 막고 있는 것이 서 있는 물체임을 처음 안 때 (_stall)
        self.blocker_kind = None  # 막은 것을 드론으로 본 결과 (obstacles.blocker_ugv) — 마지막 _stall 때의 값 (기록 · detail)
        self.drone_slips = 0  # 드론으로 보아 헛돈 것으로 판정한 횟수 (_watch_slip)
        self._world = (None, None)  # 이번 주기의 (grid, obstacles) — _stall 이 되짚기를 시작할 때 쓴다
        self.turn = turn_model if turn_model is not None else TurnModel()  # 서서 돌기에서 잰 것 (주행 노드가 목표 사이에 이어 준다)
        self._turn_cmds = []  # 이 회전 시도에서 낸 (시각, |w|) — 돌기 시작한 때의 명령을 되짚어 start_w 를 잰다
        self._boost = 0.0  # 지금 서서 도는 명령에 더하는 양 (rad/s, _spin_boost)
        self._boost_t = None  # 마지막으로 _spin_boost 를 부른 때 (0.3 s 넘게 비면 새 회전)
        self._boost_since = start_t  # 지금 회전이 시작된 때
        self._spin_hist = []  # 이 회전에서 낸 (시각, |w|)
        self._moving_w = None  # 이 회전에서 실제로 돌던 가장 작은 명령 (rad/s)
        self.p = params
        self.start_t = start_t
        self.outcome = None
        self.detail = ""
        self.final_pose = None
        self.phase = "accepted"
        self.remaining = None  # 남은 경로 길이 (모르면 None)
        self.blocked = False
        self.blocked_since = None
        self.adjusted = False  # 목표 칸이 막혀 가까운 빈 칸으로 옮겼나
        self.target = (goal.x, goal.y)  # 실제로 찾아갈 점 (NORMAL 은 옮겨질 수 있다)
        self._entry = None  # ASSAULT: 경로의 끝(빈 칸), 그다음 target 까지 곧게
        self._state = "drive"
        self._settle_since = None
        self._path = None
        self._plan_t = -1e9
        self._plan_version = -1
        self._waiting_since = None
        self._best = None
        self._progress_t = start_t
        self._credit = 0.0  # 가까워지지 못한 동안 실제로 돌거나 통로를 간 시간 (s) — NO_PROGRESS 한도를 이만큼 늘린다 (no_progress_extra_s 까지)
        self._credit_ref = None  # (시각, 자세) — 이 뒤로 얼마나 돌았나 · 갔나를 0.5 s 마다 잰다
        self._detour_credit = 0.0  # 가까워지지 못한 동안 「돌아가는 길」 을 실제로 간 시간 (s) — 위와 합쳐 no_progress_extra_s 까지 (params.no_progress_detour_m)
        self._detour_jumps = 0  # 마지막 진행 뒤로 길이 no_progress_detour_m 넘게 길어진 횟수 (한 번 = 돌아가는 길, 두 번부터 = 길이 자꾸 바뀜)
        self._last_remaining = None  # 앞 주기의 남은 경로 길이
        self._stuck_ref = None
        self._backup_until = None
        self._backup_w = 0.0
        self._straight_heading = None
        self._rot_since = None  # 지금 회전 시도가 시작된 때 (서 있다가 돌라는 명령이 처음 나간 때)
        self._near_enough = False  # 허용을 조금 넘었지만 이 자리에서 끝내기로 했다 (_check_arrival)
        self._accept_here = False  # 서 있는 물체가 목표 곁을 막아, 목표 바로 앞인 이 자리에서 도착으로 끝내기로 했다 (_accept_near_goal)
        self._shift_home = None  # 서 있는 물체가 목표 칸을 덮어 target 을 곁의 빈 칸으로 옮겨 두었다: 원래의 목표 자리 (_standing_shift)
        self._align_moves = 0  # 도착 방향으로 돌 자리를 내려고 서 있는 물체에서 비켜 선 횟수 (params.align_room_moves)
        self._reapproaches = 0  # 자리에 섰는데 허용 밖이거나 목표 곁을 지나쳐서, 다시 다가간 횟수 (params.reapproach_slack_m)
        self._min_dist = None  # 이 목표(target)에 가장 가까이 갔던 거리 (m) — 지나쳤는지를 본다
        self._hop_turning = False  # 지나친 목표 쪽으로 서서 도는 중 (한 번의 되돌아가기를 한 번만 세려고)
        self._near_slack = 0.0  # 「이 자리에서 끝낸다」 로 정했을 때의 여유 (m) — 방향을 맞춘 뒤 다시 볼 때 같은 값을 쓴다 (_check_arrival)
        self._rot_ref = (start_t, 0.0)  # 이 시도에서 마지막으로 방향이 turn_moved_rad 바뀐 때 · 그때 방향 (그 뒤 오래 안 바뀌면 각속도를 올린다)
        self._rot_rate = None  # 이 시도에서 실제로 도는 빠르기 (rad/s) — 첫 정지 여유에 쓴다
        self._rot_prev = None  # 빠르기를 재는 앞 주기의 (시각, 방향)
        self._rot_yaw0 = 0.0  # 그 시도가 시작될 때의 방향
        self._rot_err0 = 0.0
        self._rot_moved = False  # 이 시도에서 차체가 실제로 돌기 시작했나 (방향이 turn_moved_rad 바뀜)
        self._stop_ref = None  # 방향 맞추기 정지 명령을 낸 때의 방향
        self._rot_attempts = 0  # 이 목표에서 서서 돌기 시작한 횟수 (도착 방향 맞추기)
        self._close_since = None  # 지도에 없는 점이 차체 바로 앞(backoff_dist)에 있기 시작한 때
        self._through_obstacle = False  # 지금 경로는 임시 장애물을 빼고 짠 것 (그것을 지나야만 갈 수 있다)
        self._tight_path = False  # 지금 경로는 서 있는 물체를 tight_radius 로 줄여 짠 것 (좁은 틈 — tight_v 로 간다)
        self.narrow_path = False  # 지금 경로는 지도 구조물을 narrow_grow_m 만 부풀려 짠 것 (지도상 0.40 m 보다 좁은 틈 — narrow_v 로 간다)
        self.in_corridor = False  # 지금 통로 안인가 (양옆 구조물 사이가 corridor_width_m 보다 좁다)
        self._fresh_path = False  # 지금 경로는 기억으로만 남은 임시 장애물을 빼고 짠 것 (memory_detour_ratio)
        self._fresh_keeps_rear = False  # 그때 차체 뒤쪽의 기억은 남겼나
        self._stalled_since = None  # 지도에 없는 물체 때문에 나아갈 방법이 없기 시작한 때 (그것을 지나야만 감 · 붙었는데 뒤도 막힘 · 돌 자리 없음)
        self._best_direct = None  # 목표까지 곧은 거리의 최소 (경로를 다시 짜도 이어 간다)
        self._align_since = None  # 이 목표에서 도착 방향 맞추기를 처음 시작한 때 (align_max_s)
        self._turn_since = None  # 서서 돌라는 명령이 이어진 때 · 그때 방향 — 돌라고 해도 안 도는지 본다 (rot_stuck_s)
        self._turn_yaw0 = 0.0  # 마지막으로 본 방향 (창 안에서 돈 각을 이어 더한다)
        self._turn_sum = 0.0  # 이번 창에서 돈 각의 합 (rad)
        self._turn_block = set()  # 마지막 _free_turn 에서 회전을 막은 것: "obstacle"(지도에 없는 점) · "structure"(벽 · 박스 · 기둥)
        self._room_moves = 0  # 돌 자리가 없어 앞뒤로 비켜 본 횟수
        self._room_move_until = None
        self._room_move = (0.0, 0.0)  # 비키는 명령 (v, w)
        self._room_by_structure = False  # 그 비키기는 구조물 때문으로 굴려 보고 고른 것 (가는 동안 지도에 없는 점만 본다)
        self.codes = []  # 끝난 결과의 코드 (CODE_*) — detail 끝에도 붙는다
        self._reverses = 0  # 목표에 가까워지지 못한 채 뒤(라이다가 못 보는 쪽)로 뺀 횟수 (reverse_max)
        self._reverse_ref = None  # 마지막으로 뒤로 뺄 때 목표까지의 곧은 거리
        self._last = STOP

    # ------------------------------------------------------------------ 시작
    def validate(self, spec, grid, pose):
        """목표를 받을 수 있는지 본다. 안 되면 INVALID_GOAL 로 끝내고 False."""
        g, p = self.goal, self.p
        if g.frame_id != p.expected_frame:
            return self._invalid("frame_id 가 %r 이어야 함 (받은 값 %r)" % (p.expected_frame, g.frame_id))
        if g.mode not in MODE_NAMES:
            return self._invalid("모르는 mode %r" % g.mode)
        if not (g.pos_tol > 0.0 and g.yaw_tol > 0.0 and g.timeout_s > 0.0):
            return self._invalid("허용 오차·제한 시간은 양수여야 함")
        if not all(math.isfinite(v) for v in (g.x, g.y)):
            return self._invalid("좌표가 숫자가 아님")
        if spec is None or grid is None:
            return self._invalid("지도(/team/navigation/map)를 아직 받지 못함")
        if not spec.inside(g.x, g.y):
            return self._invalid("목표 (%.2f, %.2f) 가 경기장 %.2f × %.2f 밖" % (g.x, g.y, spec.width, spec.height))
        if g.mode == MODE_STRAIGHT:
            if pose is None:
                return self._invalid("자기 위치를 몰라 직진 방향을 정할 수 없음")
            dist = math.hypot(g.x - pose.x, g.y - pose.y)
            if dist > g.pos_tol:
                off = abs(wrap(math.atan2(g.y - pose.y, g.x - pose.x) - pose.yaw))
                off = min(off, math.pi - off)  # 뒤로 곧게 가는 것도 직진
                if off > math.radians(p.straight_max_heading_deg):
                    return self._invalid("직진 목표가 차체 방향에서 %.0f° 벗어남 (한도 %.0f°)" % (math.degrees(off), p.straight_max_heading_deg))
            self._straight_heading = pose.yaw
            return True
        if spec.inside_box(g.x, g.y):
            return self._invalid("목표가 박스 안")
        if g.mode == MODE_ASSAULT:
            entry = grid.nearest_free(g.x, g.y, max_dist=0.6)
            if entry is None:
                return self._invalid("돌진 목표 둘레 0.6 m 안에 들어갈 빈 칸이 없음")
            self._entry = entry
            return True
        if not grid.free(g.x, g.y):
            snapped = grid.nearest_free(g.x, g.y, max_dist=p.snap_max_m)
            if snapped is None:
                return self._invalid("목표 칸이 벽·박스에 막혔고 %.2f m 안에 빈 칸이 없음" % p.snap_max_m)
            self.target = snapped
            self.adjusted = True
            self.detail = "목표를 (%.2f, %.2f) 로 옮김 (벽·박스에 너무 가까움)" % snapped
        return True

    def _invalid(self, why):
        self.finish(INVALID_GOAL, why)
        return False

    # ------------------------------------------------------------------ 끝
    @property
    def done(self):
        return self.outcome is not None

    def finish(self, outcome, detail="", pose=None, code=""):
        """목표를 끝낸다. code(CODE_*)와, REACHED 인데 끝낸 자세가 요청한 허용 밖이면 그 코드를 detail 끝에 「 [코드]」 로 붙인다."""
        if self.outcome is not None:
            return
        self.outcome = outcome
        codes = [code] if code else []
        if outcome == REACHED and pose is not None and self.goal.mode != MODE_STRAIGHT:
            # 「도착」 이라고 하되 무엇이 요청과 다른지는 늘 적는다 — 어느 길(회전 3번 · 시간 상한 · 돌 자리 없음 · 돌진 끝)로 끝났든 같다
            g = self.goal
            home = self.target
            if self._shift_home is not None:
                # 서 있는 물체가 목표 칸을 덮어 곁의 빈 칸으로 옮겨 왔다 (_standing_shift) — 허용은 원래 목표 자리로 잰다
                home = self._shift_home
                note = "서 있는 물체가 목표 자리를 덮어 %.2f m 옆 (%.2f, %.2f) 로 옮겨 감" % (
                    math.hypot(self.target[0] - home[0], self.target[1] - home[1]), self.target[0], self.target[1])
                detail = note + ("; " + detail if detail else "")
            if math.hypot(home[0] - pose.x, home[1] - pose.y) > g.pos_tol:
                codes.append(CODE_POS_NOT_MET)
            if self.use_yaw and abs(wrap(g.yaw - pose.yaw)) > g.yaw_tol:
                codes.append(CODE_YAW_NOT_MET)
        self.codes = codes
        if codes:
            detail = (detail + " " if detail else "") + " ".join("[%s]" % c for c in codes)
        if detail:
            self.detail = (self.detail + "; " if self.detail and outcome != INVALID_GOAL else "") + detail
        self.final_pose = pose
        self.phase = "done"
        self._last = STOP

    @property
    def use_yaw(self):
        return self.goal.yaw is not None and self.goal.yaw_tol < YAW_IGNORE_TOL

    # ------------------------------------------------------------------ 한 주기
    def step(self, t, pose, level, grid, obstacles, local=None):
        """local = 로컬 궤적의 지금 자세 (pose_tracker.local — 지나온 길을 적는 좌표계). 없으면 arena 자세로 적는다 (시험)."""
        if self.outcome is not None:
            return STOP
        p, g = self.p, self.goal
        self._t = t
        self._world = (grid, obstacles)
        self.trail.update(local if local is not None else pose, rewinding=self._retrace is not None)
        if t - self.start_t > g.timeout_s:
            if (pose is not None and g.mode != MODE_STRAIGHT and self._state in ("align", "settle")
                    and (self._accept_here or math.hypot(self.target[0] - pose.x, self.target[1] - pose.y) <= g.pos_tol + p.timeout_reach_slack_m)):
                # 자리에는 와 있고 방향을 맞추던 중에 시한이 됐다 → 도착으로 끝낸다 (방향은 final_pose, finish 가 YAW_NOT_MET 을 붙인다).
                # supervisor 의 시한이 거리에 맞춰 짧아져(2026-11-19 통합 4번) 가까운 목표에서 방향 맞추기가 시한에 걸릴 수 있다 — 미션을 이어 가게.
                # 거리는 허용 + timeout_reach_slack_m: 허용 안에서 돌기 시작해도 서서 도는 동안 몇 cm 밀린다 (2026-11-19 14:13 실차: 0.117 m 에서
                # 돌기 시작해 0.128 m 에서 시한 → 허용 0.12 m 를 8 mm 넘어 TIMED_OUT). 허용 밖이면 finish 가 POS_NOT_MET 을 붙인다
                self.finish(REACHED, "제한 시간 %.1f s — 자리에서 방향을 맞추던 중에 끝냄" % g.timeout_s, pose)
                return STOP
            if self.blocked:
                self.finish(BLOCKED, "제한 시간 %.1f s 동안 막힘 (%.1f s 째)" % (g.timeout_s, t - self.blocked_since), pose, CODE_BLOCKED_AT_TIMEOUT)
            else:
                self.finish(TIMED_OUT, "제한 시간 %.1f s 초과" % g.timeout_s, pose)
            return STOP
        if pose is not None:
            self.turn.rate.on_pose_yaw(t, pose.yaw)
        self._credit_motion(t, pose)
        if (self.blocked and self._state == "drive" and p.no_progress_give_up_s > 0.0 and self.blocked_since is not None
                and self._retrace is None
                and t - self.blocked_since >= p.no_progress_give_up_s + min(p.no_progress_extra_s, self._credit + self._detour_credit)):
            # 가는 중인데 오랫동안 목표에 조금도 가까워지지 못했다 (길이 열렸다 닫혔다 해서 이쪽 저쪽으로 돌기만 하는 때 등).
            # 2026-11-19 실차: 임시 장애물 때문에 길이 번갈아 바뀌어 한 자리에서 40 s 를 돌다가 시한(47.7 s)을 다 썼다.
            # 시한까지 기다리지 않고 BLOCKED 로 끝내 supervisor 가 다른 자리를 고르게 한다
            # 목표 바로 앞에서 서 있는 물체에 막혀 못 나아가는 것이면 그 자리에서 도착으로 끝낸다 (_accept_near_goal — 방향을 맞춰야 하면 이어서 맞춘다)
            standing = obstacles is not None and (
                (getattr(obstacles, "blocker_ugv", lambda _t: None)(t) or ("", 0.0))[0] == "parked"
                or obstacles.blocker_static(t, grid, p.tight_still_s))
            if self._accept_near_goal(pose, standing):
                if self.outcome is not None:
                    return STOP
            elif not self._give_up(t, pose, grid, obstacles, BLOCKED,
                                   "%.1f s 동안 목표에 가까워지지 못해 시한 전에 끝냄 (길이 막혔거나 자꾸 바뀜)" % (t - self.blocked_since), CODE_NO_PROGRESS):
                return STOP
        if level == loc.LOST:
            if t - self.start_t < p.pose_lost_s:  # 목표가 위치보다 먼저 왔을 때(시작 직후 등) 잠깐은 기다린다
                self.phase = "holding_localization"
                self._pause(t)
                return self._out(STOP)
            self.finish(LOCALIZATION_LOST, "자기 위치가 %.1f s 넘게 오지 않음" % p.pose_lost_s, pose)
            return STOP
        if pose is None or level == loc.HOLD:
            self.phase = "holding_localization"
            self._pause(t)
            return self._out(STOP)
        cap = p.degraded_v if level == loc.DEGRADED else float("inf")
        self.in_corridor = g.mode != MODE_ASSAULT and self._corridor(pose, grid, obstacles, t)
        if obstacles is not None:
            obstacles.side_margin = p.corridor_side_margin if self.in_corridor else None
        if self.in_corridor and g.mode == MODE_STRAIGHT:
            cap = min(cap, max(p.corridor_v, p.min_v))  # 통로는 천천히 · 옆 여유는 작게 (경로를 따라갈 때는 _step_path 가 정한다 — _corridor_speed)
        if g.mode == MODE_STRAIGHT:
            cmd = self._step_straight(t, pose, cap, obstacles)
        else:
            cmd = self._step_path(t, pose, cap, grid, obstacles)
        self._watch_reverse(t, pose, cmd)
        self._watch_slip(t, pose, cmd, grid, obstacles)
        if self.outcome is not None:
            return STOP
        if level == loc.DEGRADED and self.phase not in ("settling",):
            self.phase += "+degraded_localization"
        return self._out(cmd)

    def _out(self, cmd):
        self._last = cmd
        return cmd

    def _pause(self, t):
        """위치를 기다리는 동안: 막힘 · 끼임 시간을 세지 않는다."""
        self._progress_t = t
        self._stuck_ref = None
        self._waiting_since = None

    # ------------------------------------------------------------------ 정지 · 확인 · 회전
    def _settle(self, t):
        if self._settle_since is None:
            self._settle_since = t
        self.phase = "settling"
        return t - self._settle_since >= self.p.settle_s

    def _align(self, t, pose, obstacles=None, grid=None):
        p = self.p
        err = wrap(self.goal.yaw - pose.yaw)
        self.phase = "aligning"
        self._pause(t)
        self._learn_lead(pose)
        if abs(err) <= self.goal.yaw_tol:
            self._stop_ref = None
            self._state, self._settle_since = "settle", None
            return STOP
        if self._align_since is None:
            self._align_since = t
        if p.align_max_s > 0.0 and t - self._align_since >= p.align_max_s:
            # 몇 번을 해도 방향이 허용 안에 안 든다 (넘어감 · 미끄러짐 · 위치 흔들림) — 자리는 맞으니 도착으로 끝낸다
            self.finish(REACHED, "방향 맞추기 %.0f s 넘게 허용 %.1f° 안에 못 듦 — 방향 %.1f° 남김"
                        % (p.align_max_s, math.degrees(self.goal.yaw_tol), math.degrees(abs(err))), pose)
            return STOP
        # 돌면 차체가 지도에 없는 것(다른 UGV 등) · 지도 구조물에 닿는가 — 닿으면 반대쪽으로 길게.
        # 양쪽 다 구조물에 닿으면(기다려도 비키지 않는다) 비켜 보고 그래도 안 되면, 지도에 없는 것이면 서서 기다려 보고 —
        # 자리는 맞으니 REACHED 로 끝낸다 (detail 에 NO_ROOM_AT_GOAL). 시한까지 서 있지 않는다
        turn = self._free_turn(t, err, abs(err) - 0.5 * self.goal.yaw_tol, obstacles, pose, grid)
        if turn is None:
            if "obstacle" not in self._turn_block:
                return self._no_room(t, pose, grid, obstacles, NO_ROOM_AT_GOAL, "structure", err)
            self.phase = "waiting_obstacle"
            self._rot_since = self._turn_since = None
            # 막은 것이 서 있는 물체면 기다려도 비키지 않는다 — 한 번 비켜 서서 돌아 보고, 안 되면 곧 끝낸다 (params.align_room_moves)
            reach = math.hypot(p.robot_half_length, p.robot_half_width) + p.sweep_margin + 0.05
            kind = getattr(obstacles, "near_standing", lambda *_a: None)(t, grid, reach) if p.align_room_moves > 0 else None
            if kind is None:
                self._stall(t, pose, NO_ROOM_AT_GOAL + ": 지도에 없는 물체가 막음 (방향 %.0f° 남김)" % math.degrees(abs(err)), REACHED,
                            CODE_NO_ROOM_AT_GOAL)
                return STOP
            what = "드론: %.0f s 째 서 있는 UGV" % kind[1] if kind[0] == "parked" else "한자리에 서 있는 물체"
            limit = p.obstacle_give_up_parked_s if kind[0] == "parked" else p.obstacle_give_up_static_s
            if self._stall(t, pose, NO_ROOM_AT_GOAL + ": 서 있는 물체가 막음 (%s, 방향 %.0f° 남김, 비켜 선 횟수 %d)"
                           % (what, math.degrees(abs(err)), self._align_moves), REACHED, CODE_NO_ROOM_AT_GOAL, limit=limit if limit > 0.0 else None):
                return STOP
            if self._align_moves < p.align_room_moves and t - self._stalled_since >= ALIGN_MOVE_AFTER_S:
                move = self._room_move_plan(t, pose, grid, obstacles, "obstacle")
                if move is not None:
                    v, w, duration = move
                    if v < 0:
                        self._note_reverse(pose)
                    self._align_moves += 1
                    self._room_move_until, self._room_move = t + duration, (v, w)
                    self._room_by_structure = False
                    if not self._accept_here:
                        # 비킨 자리가 도착 자리다 — 방향을 맞춘 뒤 자리로 되돌아가지 않는다 (_check_arrival). 허용 밖이면 finish 가 [POS_NOT_MET] 을 붙인다
                        self._accept_here = True
                        self.detail = (self.detail + "; " if self.detail else "") + "도착 방향으로 돌 자리를 내려고 서 있는 물체(%s)에서 비켜 섬" % what
                    self.phase = "backing_up" if v < 0 else "moving_off"
                    return Command(v, w)
            return STOP
        self._stalled_since = None
        if self._turn_stuck(t, pose):
            return self._no_room(t, pose, grid, obstacles, NO_ROOM_AT_GOAL, "stuck", err)
        if turn != math.copysign(1.0, err):  # 반대쪽으로 길게 돈다
            self._rot_since = None
            return Command(0.0, self._spin_boost(t, pose, turn * p.rotate_max_w, math.pi, grid, obstacles))

        # 새 회전 시도: 서 있다가 처음 돌라는 명령이 나가는 때
        if abs(self._last.w) < 1e-6 or self._rot_since is None:
            if (p.align_give_up_attempts > 0 and self._rot_attempts >= p.align_give_up_attempts
                    and abs(err) <= p.align_accept_rad):
                self.finish(REACHED, "방향이 허용 %.1f° 밖 %.1f° 에서 끝냄 — 회전 %d번 시도했으나 지연 · 정지 마찰로 더 못 맞춤"
                            % (math.degrees(self.goal.yaw_tol), math.degrees(abs(err)), self._rot_attempts), pose)
                return STOP
            self._rot_attempts += 1
            self._rot_since, self._rot_yaw0, self._rot_err0, self._rot_moved = t, pose.yaw, abs(err), False
            self._rot_ref = (t, pose.yaw)
            self._turn_cmds = []
            self._rot_rate = self._rot_prev = None
        # 「돌고 있다」 는 시간이 아니라 방향이 실제로 바뀐 것으로 본다 — 시도의 처음뿐 아니라 도는 내내.
        # (전: 명령이 latency_s 이어지면 차체가 안 돌았어도 정지 여유를 적용해, 짧은 명령 → 정지 → 재확인을 시한까지 되풀이했다)
        # 2026-11-19 실차: 한 번 돌기 시작한 뒤 남은 각이 작아져 명령이 1.4 rad/s 밑으로 내려가면 다시 서 버려, 남은 10 ~ 20° 에서
        # 0.8 rad/s 를 내며 방향 맞추기 상한(align_max_s)까지 섰다 (가상 REAL_ROBOT_1119 로 재현: 64번 중 37번이 허용 밖)
        moved_rad = min(p.turn_moved_rad, 0.5 * self._rot_err0)
        ref_t, ref_yaw = self._rot_ref
        if abs(wrap(pose.yaw - ref_yaw)) >= moved_rad:
            stalled = t - ref_t > p.breakaway_hold_s
            if stalled or not self._rot_moved:
                self._learn_start_w(t, stalled=stalled)
            self._rot_moved = True
            self._rot_ref = ref_t, ref_yaw = (t, pose.yaw)
        prev_t, prev_yaw = self._rot_prev if self._rot_prev is not None else (t, pose.yaw)
        if self._rot_moved and t > prev_t:
            rate = abs(wrap(pose.yaw - prev_yaw)) / (t - prev_t)  # 실제로 도는 빠르기 (실차: 1.6 rad/s 명령에 0.4 ~ 0.6)
            self._rot_rate = rate if self._rot_rate is None else 0.5 * (self._rot_rate + rate)
        self._rot_prev = (t, pose.yaw)

        floor = p.min_turn_w if self.turn.start_w is None else max(p.min_turn_w, min(self.turn.start_w, p.rotate_max_w))
        w = _clamp(p.rotate_gain * err, p.rotate_max_w)
        base = abs(w)  # 남은 각만으로 낼 각속도 — 이보다 크게 냈을 때(바닥값 · 올리기)만 그 명령이 「돌기 시작한 각속도」 를 알려 준다
        if abs(w) < floor:
            w = math.copysign(floor, err)
        if self._rot_moved:
            # 정지 명령 뒤에도 더 돈다 — 처음에는 실제로 도는 빠르기 × coast_s (모르면 |w|), 한 번 멈춘 뒤에는 실제로 더 돈 각도
            # (_learn_lead, 목표 사이에 이어 쓴다). 전: |w| × coast_s = 1.6 × 0.2 → 18° 앞에서 섰는데, 실차는 그 1/3 빠르기로 돌고 멈춘 뒤 거의 안 돈다
            rate = abs(w) if self._rot_rate is None else min(abs(w), self._rot_rate)
            lead = self.turn.lead if self.turn.lead is not None else rate * p.coast_s
            if abs(err) <= self.goal.yaw_tol + lead:
                self._stop_ref = pose.yaw
                self._state, self._settle_since = "settle", None
                return STOP
        # 지금 안 돈다(처음의 정지 마찰 · 돌다가 명령이 작아져 섬): 멈추지 않고 명령을 이어 가다가, breakaway_hold_s 가 지나도
        # 방향이 안 바뀌면 각속도를 올린다
        stuck = t - ref_t - p.breakaway_hold_s
        # 구조물(벽 · 상자 · 기둥) 곁에서 돌다 선 것은 걸렸을 수 있다 — 다시 올려 밀지 않는다 (안 도는 회전 감지 rot_stuck_s 가 받는다).
        # 가상(실측 회전 모델): 상자에 닿은 채 올린 각속도로 2.5 s 를 밀었다
        tight = grid is not None and grid.spec.distance_to_structure(pose.x, pose.y) < math.hypot(p.robot_half_length, p.robot_half_width) + 0.05
        if stuck > 0.0 and not (tight and self._rot_moved):
            w = math.copysign(min(p.rotate_max_w, abs(w) + p.breakaway_ramp * stuck), err)
        self._turn_cmds.append((t, abs(w), base))  # 「돌기 시작한 각속도」(start_w) 학습에는 올리기 전의 명령을 남긴다 (아래에서 올린 양이 섞이면
        if len(self._turn_cmds) > 64:              # start_w 가 곧 rotate_max_w 가 되어, 상자 곁의 작은 각도 늘 그 세기로 돌았다 — 가상: 닿은 주기 311 → 491)
            del self._turn_cmds[:-64]
        # 실제로 도는 빠르기가 모자라면 rotate_max_w 위로도 올린다 (곁이 비고 남은 각이 클 때만)
        return Command(0.0, self._spin_boost(t, pose, w, err, grid, obstacles))

    def _learn_start_w(self, t, stalled):
        """서 있다가 방향이 바뀌기 시작했다 → 그 명령 크기를 TurnModel.start_w 에 반영한다 (다음 시도 · 다음 목표는 이보다 작게 내지 않는다).

        stalled(각속도를 올려야 돌았다): 돌기 시작하게 한 명령은 반응 지연(latency_s) 앞의 것이다 — 그때 낸 |w|.
        곧바로 돌았다: 지금 바닥값으로도 돈다 → 조금씩(0.05) 내려 본다 (min_turn_w 밑으로는 안 간다)."""
        p, a = self.p, self.p.lead_learn
        if a <= 0.0 or not self._turn_cmds:
            return
        if stalled:
            then = t - p.latency_s
            seen = [c for c in self._turn_cmds if c[0] <= then]
            _, w, base = seen[-1] if seen else self._turn_cmds[0]
        else:
            _, w, base = self._turn_cmds[0]
        if base >= w - 0.02:
            return  # 남은 각이 커서 원래 그만큼 낸 명령이다 — 돌기 시작한 각속도에 대해 알려 주는 것이 없다 (가상: 1.6 을 배워 작은 각에서 40° 넘어감)
        observed = w if stalled else w - 0.05
        observed = max(p.min_turn_w, min(p.rotate_max_w, observed))
        model = self.turn
        model.start_w = observed if model.start_w is None else (1.0 - a) * model.start_w + a * observed

    def _learn_lead(self, pose):
        """정지 명령을 낸 뒤 settle 이 끝나 멈춘 자리까지 실제로 더 돈 각도를 재서 다음 정지 여유에 반영한다."""
        if self._stop_ref is None or abs(self._last.w) > 1e-6:
            return
        extra = abs(wrap(pose.yaw - self._stop_ref))
        self._stop_ref = None
        a = self.p.lead_learn
        if a <= 0.0:
            return
        extra = min(extra, math.radians(30.0))
        self.turn.lead = extra if self.turn.lead is None else (1.0 - a) * self.turn.lead + a * extra

    def _check_arrival(self, t, pose, dist, grid=None):
        """settle 이 끝난 뒤: 도착 · 방향을 확인하고 다음 상태를 정한다."""
        if self._accept_here or (self._near_enough and dist <= self.goal.pos_tol + max(self._near_slack, self.p.timeout_reach_slack_m)):
            dist = min(dist, self.goal.pos_tol)  # 위에서 「이 자리에서 끝낸다」 로 정했다 — 방향을 맞춘 뒤에도 다시 돌아 고치지 않는다
        if dist > self.goal.pos_tol:
            p = self.p
            off = abs(wrap(math.atan2(self.target[1] - pose.y, self.target[0] - pose.x) - pose.yaw))
            if (self.goal.mode == MODE_ASSAULT and dist <= p.assault_accept_m and off > math.radians(p.rotate_in_place_deg)
                    and grid is not None and grid.spec.distance_to_structure(pose.x, pose.y) <= p.assault_wall_m):
                # 돌진으로 벽 옆(거점 안)까지 들어와 섰는데 목표가 옆 · 뒤로 조금 벗어났다. 다시 맞추려면 서서 돌아야 하는데 돌 자리가 없다 —
                # 차체 모서리가 벽에 걸린 채 시한까지 간다 (2026-11-19 가상: 22번 중 7번). 들어온 것으로 끝낸다. 목표가 앞에 있으면 그대로 더 간다
                self.finish(REACHED, "돌진 끝: 목표에서 %.2f m (벽 옆이라 서서 돌아 다시 맞추지 않음)" % dist, pose)
                return True
            # 허용을 몇 cm 넘어 섰는데 고치려면 목표 쪽으로 크게 돌아야 한다 → 돌아서 고치지 않고 이 자리에서 끝낸다 (방향만 맞춘다).
            # 가상: 허용 0.10 m 바로 밖에서 「목표 쪽으로 180° 돌기 → 1 cm 감 → 도착 방향으로 180° 돌기」 를 시한까지 되풀이했다 (2026-11-19).
            # 한 번 다시 다가가고도 또 허용 밖이면(2026-11-20, params.reapproach_slack_m) 더 넓은 여유로 같은 일을 한다 — 두 번째부터는 고치지 않는다.
            # finish 가 [POS_NOT_MET] 을 붙인다
            slack = max(p.timeout_reach_slack_m, p.arrive_slack_m) if off > math.radians(p.rotate_in_place_deg) else 0.0
            again = self._reapproaches >= 1 and p.reapproach_slack_m > 0.0
            if again:
                slack = max(slack, p.reapproach_slack_m)
            if self.goal.mode == MODE_NORMAL and slack > 0.0 and dist <= self.goal.pos_tol + slack:
                if self.use_yaw and abs(wrap(self.goal.yaw - pose.yaw)) > self.goal.yaw_tol:
                    self._state = "align"
                    self._near_enough, self._near_slack = True, slack
                    return False
                if again and not off > math.radians(p.rotate_in_place_deg):
                    self.finish(REACHED, "목표에서 %.2f m — 한 번 다시 다가가고도 허용 밖이라 이 자리에서 끝냄" % dist, pose)
                else:
                    self.finish(REACHED, "목표에서 %.2f m — 허용을 조금 넘었지만 고치려면 크게 돌아야 해서 이 자리에서 끝냄" % dist, pose)
                return True
            self._reapproaches += 1
            self._state, self._path = "drive", None
            self._reset_progress(t)
            return False
        if self.use_yaw and abs(wrap(self.goal.yaw - pose.yaw)) > self.goal.yaw_tol:
            self._state = "align"
            return False
        self.finish(REACHED, "", pose)
        return True

    # ------------------------------------------------------------------ 진행 감시
    def _reset_progress(self, t):
        self._best = None
        self._best_direct = None
        self._progress_t = t
        self._credit = 0.0
        self._detour_credit, self._detour_jumps, self._last_remaining = 0.0, 0, None

    def _credit_motion(self, t, pose):
        """가까워지지는 못했어도 실제로 돌고 있거나(서서 도는 단계) 통로 · 좁은 틈을 실제로 가고 있으면 그 시간을 적는다 (params.no_progress_extra_s).
        자세는 제어 주기보다 드물게 오므로 0.5 s 마다 그 사이에 바뀐 양으로 본다. 단계(phase)는 바로 앞 주기의 것이다."""
        p = self.p
        if pose is None or p.no_progress_extra_s <= 0.0:
            self._credit_ref = None
            return
        if self._credit_ref is None:
            self._credit_ref = (t, pose)
            return
        since, was = self._credit_ref
        dt = t - since
        if dt < 0.5:
            return
        self._credit_ref = (t, pose)
        if dt > 1.5:
            return
        turning = self.phase.startswith("rotating") and abs(wrap(pose.yaw - was.yaw)) >= p.no_progress_turn_rate * dt
        creeping = (self.in_corridor or self._tight_path or self.narrow_path) and math.hypot(pose.x - was.x, pose.y - was.y) >= p.no_progress_creep_v * dt
        if turning or creeping:
            self._credit = min(p.no_progress_extra_s, self._credit + dt)
        # 돌아가는 길을 실제로 가고 있다: 길이 한 번 길어졌고(막힌 길 대신 돌아가는 길), 그 길이 아직 이 목표에서 본 가장 짧은 길보다 길며, 차체가 가고 있다.
        # 길이 두 번째로 길어지면(이쪽 저쪽으로 번갈아 바뀜) _watch_progress 가 이 시간을 0 으로 되돌린다
        if (p.no_progress_detour_m > 0.0 and self._detour_jumps == 1 and self._best is not None and self.remaining is not None
                and self.remaining > self._best + p.progress_m and self._state == "drive"
                and math.hypot(pose.x - was.x, pose.y - was.y) >= p.no_progress_creep_v * dt):
            self._detour_credit = min(p.no_progress_extra_s, self._detour_credit + dt)

    def replan(self):
        """위치가 크게 튀었을 때 등: 다음 주기에 경로를 다시 짠다."""
        self._path = None
        self._plan_t = -1e9

    def on_grid_changed(self, grid):
        """가는 중에 지도(상자)가 바뀌었다: 목표 칸이 새 지도에서 막혔으면 validate 가 하듯 곁의 빈 칸으로 옮기고, 경로를 다시 짠다.

        전에는 경로만 다시 짰다. 목표 칸이 막히면 grid.plan 이 바로 None 이라 그 자리에 서서 no_progress_give_up_s 뒤 BLOCKED 였다
        (2026-11-19 가상: 1 s 에 상자가 목표 곁 5 cm 로 옮겨지자 8.9 s 에 NO_PROGRESS, 목표까지 3.05 m 남김). map_fusion 이 상자를 옮기거나
        되살릴 때마다 생길 수 있다 — 본선 경기장 8판의 지도 바뀜 21번에서는 새 상자가 목표에서 0.69 m 보다 가까웠던 적이 없어 드러나지 않았다.
        옮길 빈 칸이 없으면 그대로 둔다 (전과 같이 NO_PROGRESS 로 끝난다)."""
        self.replan()
        g, p = self.goal, self.p
        if self.outcome is not None or grid is None or g.mode == MODE_STRAIGHT:
            return
        self._min_dist = None
        if self._shift_home is not None:   # 서 있는 물체 때문에 옮겨 둔 목표는 원래 자리로 되돌려 놓고 본다 (다음 _plan 이 다시 정한다)
            self.target, self._shift_home = self._shift_home, None
        if g.mode == MODE_ASSAULT:
            entry = grid.nearest_free(g.x, g.y, max_dist=0.6)
            if entry is not None:
                self._entry = entry
            return
        if grid.free(*self.target):
            return
        snapped = (g.x, g.y) if grid.free(g.x, g.y) else grid.nearest_free(g.x, g.y, max_dist=p.snap_max_m)
        if snapped is not None:
            self.target = snapped
            self.adjusted = math.hypot(snapped[0] - g.x, snapped[1] - g.y) > 1e-6
            self._reset_progress(self._progress_t)   # 목표 자리가 바뀌었다: 남은 거리의 최소를 새로 잡는다 (진행 시계는 그대로)

    def _watch_progress(self, t, remaining, direct=None):
        """목표에 progress_m 넘게 가까워질 때만 「진행」 — 남은 경로 길이로, 또는 목표까지 곧은 거리(direct)로.
        두 기준(그 목표에서 지금까지의 최소)은 경로를 다시 짜도 이어 간다. 곧은 거리: 0.3 초마다 다시 짜며 천천히 틈을 지나갈 때도 진행으로 센다.
        경로 길이: 전에는 다시 짤 때마다 새로 잡아, 길이 두 갈래로 번갈아 바뀌면 바뀐 길을 따라 달리는 동안을 「진행」 으로 셌다 —
        2026-11-19 실차 11:19 판 첫 목표: 출발 앞 지도에 없는 물체 때문에 북쪽 · 남쪽 우회로를 3 ~ 4 s 씩 오가며 41 s 시한을 다 썼고
        no_progress_give_up_s 가 한 번도 켜지지 않았다 (기록 재생에서 같은 자리에서 끝남). 이제 더 짧은 길이 열리거나 실제로 가까워질 때만 진행이다."""
        progressed = False
        if self._last_remaining is not None and remaining > self._last_remaining + self.p.no_progress_detour_m > self._last_remaining:
            self._detour_jumps += 1   # 길이 훌쩍 길어졌다 (다시 짠 길이 돌아가는 길이다)
            if self._detour_jumps >= 2:
                self._detour_credit = 0.0   # 두 번째부터는 「길이 자꾸 바뀜」 — 돌아가는 길로 쳐 준 시간을 거둔다
        self._last_remaining = remaining
        if self._best is not None and remaining <= self._best + self.p.progress_m:
            self._detour_jumps = 0   # 지금 길은 이 목표에서 본 가장 짧은 길만큼 짧다 — 돌아가는 길이 아니다
        if self._best is None:
            self._best = remaining
        elif remaining < self._best - self.p.progress_m:
            self._best = remaining
            progressed = True
        if direct is not None:
            if self._best_direct is None:
                self._best_direct = direct
            elif direct < self._best_direct - self.p.progress_m:
                self._best_direct = direct
                progressed = True
        if progressed:
            self._progress_t = t
            self._credit = 0.0
            self._detour_credit = 0.0   # 진행했다: 한도를 새로 센다 (돌아가는 길 위라면 _detour_jumps 는 그대로 — 아래에서 길이 짧아지면 푼다)
            self._stalled_since = None
            self.blocked = False
            self.blocked_since = None
            return
        self._mark_blocked(t)

    def _mark_blocked(self, t):
        if not self.blocked and t - self._progress_t >= self.p.blocked_report_s:
            self.blocked = True
            self.blocked_since = self._progress_t

    def _watch_stuck(self, t, pose, v, grid=None, obstacles=None):
        """움직이라고 했는데 안 움직이면(바퀴가 걸림 · 벽에 닿음) 잠깐 뒤로 뺀다 (뒤가 비었을 때)."""
        p = self.p
        if abs(v) <= 0.05:
            self._stuck_ref = None
            return
        if self._stuck_ref is None:
            self._stuck_ref = (t, pose.x, pose.y)
            return
        t0, x0, y0 = self._stuck_ref
        if t - t0 >= p.stuck_s:
            if (math.hypot(pose.x - x0, pose.y - y0) < p.progress_m and self._rear_clear(t, pose, grid, obstacles)
                    and self._reverse_ok(pose)):
                self._note_reverse(pose)
                self._backup_until = t + p.backup_s
                self._backup_w = 0.6 if int(t * 10) % 2 else -0.6
                self._path = None
            self._stuck_ref = (t, pose.x, pose.y)

    def _corridor(self, pose, grid, obstacles=None, t=0.0):
        """차체 양옆의 막힌 것 사이가 corridor_width_m 보다 좁은가 (params.corridor_width_m 의 설명).

        막힌 것 = 지도 구조물(벽 · 상자 · 기둥)과, 그 곁에 찍힌 지도에 없는 점(지도보다 안쪽에 있는 진짜 상자 면 · 벽).
        지금 자리와 앞 0.3 · 0.5 m 를 본다 — 통로 어귀에서부터 좁은 띠로 봐야 들어설 수 있다 (어귀에서는 아직 한쪽이 트여 있다)."""
        width = self.p.corridor_width_m
        if width <= 0.0 or grid is None or pose is None:
            return False
        spec = grid.spec
        band = self.p.corridor_wall_band_m
        if band > 0.0 and min(pose.y, spec.height - pose.y) < band:
            return True  # 위 · 아래 벽 쪽 띠 (지도상의 자리만으로)
        fx, fy = math.cos(pose.yaw), math.sin(pose.yaw)
        sx, sy = -fy, fx  # 차체 왼쪽
        # 지도에 없는 점은 지도 구조물 곁(displaced_m 안)의 것만 센다 = 지도보다 조금 어긋나 있는 벽 · 상자의 면.
        # 구조물에서 떨어져 선 다른 UGV 는 통로의 벽으로 치지 않는다 (그 곁에서는 옆 여유를 줄이지 않는다).
        # 구조물 곁에 선 UGV 도 마찬가지다 — 드론이 본 UGV 곁(ugv_match_m)의 점은 벽으로 세지 않는다 (2026-11-20, obstacles.on_scan 과 같은 기준)
        points = []
        centres = getattr(obstacles, "ugv_centres", lambda _t: ())(t) if obstacles is not None else ()
        for bx, by in (self._near_points(t, obstacles, reach=width + 0.6) if obstacles is not None else ()):
            wx, wy = pose.x + fx * bx - fy * by, pose.y + fy * bx + fx * by
            if (spec.distance_to_structure(wx, wy) <= self.p.displaced_m
                    and not any(math.hypot(wx - ux, wy - uy) <= self.p.ugv_match_m for ux, uy in centres)):
                points.append((bx, by))
        for ahead in (0.0, 0.3, 0.5):
            cx, cy = pose.x + fx * ahead, pose.y + fy * ahead
            if not spec.inside(cx, cy) or spec.distance_to_structure(cx, cy) <= 0.0:
                continue
            total = 0.0
            for side in (1.0, -1.0):
                reach = 0.0
                while reach < width:
                    x, y = cx + side * sx * (reach + 0.05), cy + side * sy * (reach + 0.05)
                    if not spec.inside(x, y) or spec.distance_to_structure(x, y) <= 0.0:
                        break
                    reach += 0.05
                for bx, by in points:  # 그 자리 곁(앞뒤 0.15 m)의 지도에 없는 점
                    if abs(bx - ahead) <= 0.15 and by * side > 0.0:
                        reach = min(reach, abs(by))
                total += reach
            if total < width:
                return True
        return False

    def _corridor_speed(self, t, err, remaining, obstacles=None):
        """통로에서 낼 속도 상한. 경로와 나란히 곧게 가고 있고(방향 오차 corridor_fast_deg 안) 목표가 아직 멀고(corridor_fast_remaining_m 넘게)
        차체 곁(corridor_fast_clear_m 안)에 지도에 없는 점이 없으면 corridor_fast_v, 아니면 corridor_v (params 의 설명).
        통로에 들어서며 도는 동안 · 목표에 다가가 서는 동안 · 통로가 지도보다 좁거나 무엇이 곁에 있는 동안은 전과 같은 속도다."""
        p = self.p
        if (p.corridor_fast_v > p.corridor_v and abs(err) <= math.radians(p.corridor_fast_deg)
                and remaining is not None and remaining >= p.corridor_fast_remaining_m
                and not self._near_points(t, obstacles, reach=p.corridor_fast_clear_m)):
            return p.corridor_fast_v
        return p.corridor_v

    # ------------------------------------------------------------------ 되짚기 · 뒤로 갈 때의 진행 확인
    def _give_up(self, t, pose, grid, obstacles, outcome, detail, code):
        """시한 전에 끝내려는 참이다. 돌 자리가 빠듯한 곳이면 끝내기 전에 지나온 길을 돌지 않고 뒤로 되짚어 물러난다 (params.retrace_m 의 설명).
        되짚기를 시작했으면 True (다 물러난 뒤 _retrace_step 이 같은 결과로 끝낸다), 아니면 바로 끝내고 False."""
        if self._retrace is None and not self._retraced and self._can_retrace(t, pose, grid, obstacles):
            here = self.trail.here
            self._retraced = True
            self._retrace = {"since": t, "start": (here[0], here[1]), "outcome": outcome, "detail": detail, "code": code}
            self._backup_until = self._room_move_until = None
            self._rev_stalled = False
            self._rev_ref = None
            return True
        self.finish(outcome, detail, pose, code)
        return False

    @staticmethod
    def _rear_err(target):
        """차체 기준 점 target 이 차체 뒤쪽 축에서 벗어난 각 (rad, 왼쪽 +). 뒤로 가며 이 부호로 돌면 뒤가 그 점을 향한다."""
        return wrap(math.atan2(target[1], target[0]) - math.pi)

    def _can_retrace(self, t, pose, grid, obstacles):
        p = self.p
        if p.retrace_m <= 0.0 or self.goal.mode == MODE_STRAIGHT or pose is None or self.trail.rear_blocked(t):
            return False
        target = self.trail.back_target(p.retrace_reach_m)
        if target is None or self.trail.back_room() < p.retrace_min_m:
            return False
        if abs(self._rear_err(target)) > math.radians(p.retrace_max_deg):
            return False  # 온 뒤에 서서 돌았다 — 뒤로는 그 길을 따라갈 수 없다
        if obstacles is not None and obstacles.blocking_ahead(t, -1.0, speed=p.retrace_v):
            return False
        return self._needs_room(t, pose, grid, obstacles)

    def _needs_room(self, t, pose, grid, obstacles):
        """이 자리는 다음 목표로 출발하기에 빠듯한가: 통로 안이거나, 서서 돌 때 차체가 쓸고 갈 원 곁에 구조물 · 지도에 없는 점이 있다.
        넓은 곳에서 막혔으면 물러나지 않는다 (그 자리에서 돌아 다른 데로 가면 된다 — 물러나는 4 s 가 아깝다)."""
        p = self.p
        if self.in_corridor:
            return True
        reach = math.hypot(p.robot_half_length, p.robot_half_width) + p.sweep_margin + 0.05
        if grid is not None and grid.spec.distance_to_structure(pose.x, pose.y) <= reach:
            return True
        return bool(self._near_points(t, obstacles, reach=reach))

    def _retrace_step(self, t, pose, grid, obstacles):
        """지나온 길을 뒤로 되짚는다. retrace_m 만큼 물러났거나 더 갈 수 없으면 처음에 내려던 결과로 끝낸다."""
        p, r = self.p, self._retrace
        here = self.trail.here
        moved = math.hypot(here[0] - r["start"][0], here[1] - r["start"][1])
        target = self.trail.back_target(p.retrace_reach_m)
        err = self._rear_err(target) if target is not None else 0.0
        if (moved >= p.retrace_m or self._rev_stalled or target is None or abs(err) > math.radians(p.retrace_max_deg)
                or t - r["since"] > p.retrace_m / max(p.retrace_v, 0.05) + 3.0
                or (obstacles is not None and obstacles.blocking_ahead(t, -1.0, speed=abs(self._last.v)))):
            self._retrace = None
            detail, code = r["detail"], r["code"]
            if moved >= 0.15:
                if code == CODE_NO_ROOM_TO_LEAVE:
                    # 물러나 돌 자리가 생겼다 — supervisor 가 「돌 자리 없음(출발)」 으로 보고 또 곧게 빼지 않게 그 말 · 코드를 싣지 않는다
                    detail, code = "출발할 자리가 좁아 서서 돌지 못함", CODE_NO_PROGRESS
                detail += " — 지나온 길로 %.2f m 물러난 뒤 끝냄" % moved
            self.finish(r["outcome"], detail, pose, code)
            return STOP
        self.phase = "retracing"
        self._pause(t)
        return Command(-p.retrace_v, _clamp(p.retrace_gain * err, p.retrace_max_w))

    def _watch_reverse(self, t, pose, cmd):
        """뒤로 가라는 명령이 reverse_check_s 이어졌는데 차체가 그 방향으로 거의 안 갔으면(가라고 한 거리의 reverse_check_ratio 미만) 뒤가 막힌 것이다
        (라이다는 뒤 90° 를 못 본다). 뒤로 가는 동작(뒤로 빼기 · 비키기 · 되짚기)을 그만두고 rear_blocked_s 동안 다시 뒤로 가지 않는다.
        MODE_STRAIGHT 로 뒤로 가던 목표는 BLOCKED 로 끝낸다. 자세는 SLAM 으로 고친 것을 쓴다 — 바퀴가 헛돌면 /odom 은 간 것으로 적는다."""
        p = self.p
        if p.reverse_check_s <= 0.0 or pose is None or cmd.v >= -0.02:
            self._rev_ref = None
            return
        ref = self._rev_ref
        if ref is None:
            self._rev_ref = [t, pose, 0.0, t, t]
            return
        if t - ref[4] < p.reverse_check_grace_s:
            # 명령이 듣기까지(반응 지연)는 세지 않는다 — 2026-11-19 22:39 실차: 되짚기를 시작한 지 0.7 s 에 「안 움직임」 으로 잘못 끝냈다
            # (그 0.7 s 의 대부분이 반응 지연이었고 차체는 그 뒤 0.08 m 를 물러났다)
            self._rev_ref = [t, pose, 0.0, t, ref[4]]
            return
        ref[2] += abs(cmd.v) * (t - ref[3])
        ref[3] = t
        if t - ref[0] < p.reverse_check_s:
            return
        was = ref[1]
        back = -((pose.x - was.x) * math.cos(was.yaw) + (pose.y - was.y) * math.sin(was.yaw))
        asked, spent = ref[2], t - ref[4]
        self._rev_ref = [t, pose, 0.0, t, ref[4]]
        if back >= p.reverse_check_ratio * asked:
            return
        self.trail.rear_blocked_until = t + p.rear_blocked_s
        self._rev_stalled = True
        self._backup_until = None
        if self._room_move_until is not None and self._room_move[0] < 0.0:
            self._room_move_until = None
        self._path = None
        if self.goal.mode == MODE_STRAIGHT:
            self.finish(BLOCKED, "뒤로 가라고 %.1f s 냈는데 %.2f m 밖에 못 감 — 뒤가 막혀 시한 전에 끝냄" % (spent, max(0.0, back)), pose, CODE_OBSTACLE)

    def _watch_slip(self, t, pose, cmd, grid, obstacles):
        """드론으로 본 헛돎 (drive/ugvs.UgvTracks.slip): 바퀴는 갔다는데 드론이 본 우리 차는 그 자리다 = 무엇에 걸려 헛돈다.
        _watch_stuck 은 자세(SLAM + /odom)가 안 움직일 때만 켜진다 — 닿아서 헛도는 동안에는 /odom 이 간 것으로 적어 자세도 따라 움직일 수 있다.
        앞으로 가던 중이면 _watch_stuck 과 같이 뒤로 뺀다 (뒤가 비었고 더 빼도 될 때). 뒤로 가던 것은 _watch_reverse 가 0.7 s 안에 본다."""
        ugvs = getattr(obstacles, "ugvs", None)
        if ugvs is None or pose is None or self.goal.mode == MODE_STRAIGHT:
            return
        slip = ugvs.slip(t)
        if slip is None:
            return
        self.drone_slips += 1
        if (cmd.v > 0.05 and self._state == "drive" and self._backup_until is None and self._room_move_until is None and self._retrace is None
                and self._rear_clear(t, pose, grid, obstacles) and self._reverse_ok(pose)):
            self._note_reverse(pose)
            self._backup_until = t + self.p.backup_s
            self._backup_w = 0.0
            self._path = None
            self._stuck_ref = None

    def _reverse_ok(self, pose):
        """뒤로 빼도 되는가. 라이다는 뒤쪽 90° 를 못 본다 — 뒤는 「빈 곳」 이 아니라 「모르는 곳」 이다. 그래서 목표에 가까워지지 못한 채
        잇달아 reverse_max 번 뺐으면 더 빼지 않는다 (뒤에 선 다른 UGV 를 되풀이해 미는 것을 막는다 — 2026-11-19 가상: 5번 뺐다).
        그 뒤 목표에 reverse_refill_m 넘게 가까워졌으면 다시 센다."""
        p = self.p
        if self.trail.rear_blocked(self._t):
            return False  # 방금 뒤로 가라고 했는데 안 움직였다 (_watch_reverse)
        if p.reverse_max <= 0 or self._reverses < p.reverse_max:
            return True
        dist = math.hypot(self.target[0] - pose.x, self.target[1] - pose.y)
        if self._reverse_ref is not None and dist <= self._reverse_ref - p.reverse_refill_m:
            self._reverses = 0
            return True
        return False

    def _note_reverse(self, pose):
        self._reverses += 1
        self._reverse_ref = math.hypot(self.target[0] - pose.x, self.target[1] - pose.y)

    def _room_clear(self, t, obstacles, v):
        """구조물 때문에 굴려 보고 고른 비키기: 가는 쪽(앞 · 뒤)에 지도에 없는 점만 없으면 된다 (구조물과의 틈은 굴려 볼 때 봤다)."""
        return obstacles is None or not obstacles.blocking_ahead(t, math.copysign(1.0, v))

    def _rear_clear(self, t, pose, grid, obstacles):
        """뒤로 backup_v × backup_s 만큼 빼도 지도의 벽 · 박스(차체 뒷면 세 점)와 지도에 없는 점에 닿지 않는가."""
        return self._move_clear(t, pose, grid, obstacles, -1.0)

    def _move_clear(self, t, pose, grid, obstacles, direction):
        """direction(-1 뒤, +1 앞)으로 backup_v × backup_s 만큼 가도 지도의 벽 · 박스 · 기둥(차체 그 면의 세 점)과 지도에 없는 점에 닿지 않는가."""
        p = self.p
        if obstacles is not None and obstacles.blocking_ahead(t, direction):
            return False
        if grid is None:
            return True
        reach = p.robot_half_length + p.backup_v * p.backup_s
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        for side in (-p.robot_half_width, 0.0, p.robot_half_width):
            x, y = pose.x + direction * c * reach - s * side, pose.y + direction * s * reach + c * side
            if not grid.spec.inside(x, y) or grid.spec.distance_to_structure(x, y) < 0.02:
                return False
        return True

    # ------------------------------------------------------------------ STRAIGHT
    def _step_straight(self, t, pose, cap, obstacles):
        p, g = self.p, self.goal
        h = self._straight_heading if self._straight_heading is not None else pose.yaw
        self._straight_heading = h
        ux, uy = math.cos(h), math.sin(h)
        remaining = (g.x - pose.x) * ux + (g.y - pose.y) * uy  # 진행 방향 성분 (뒤면 음수)
        self.remaining = abs(remaining)
        if self._state == "settle":
            if not self._settle(t):
                return STOP
            self._settle_since = None
            if abs(remaining) <= g.pos_tol:
                self.finish(REACHED, "", pose)
                return STOP
            self._state = "drive"
            self._reset_progress(t)
        if abs(remaining) <= g.pos_tol * STOP_FRACTION + abs(self._last.v) * p.coast_s:
            self._state = "settle"
            self._settle(t)
            return STOP
        direction = 1.0 if remaining > 0 else -1.0
        v = min(p.straight_v, 0.08 + 1.5 * abs(remaining), cap)
        v = max(v, min(p.min_v, cap))
        w = _clamp(p.heading_hold_gain * wrap(h - pose.yaw), 0.5)
        self.phase = "straight"
        self._watch_progress(t, abs(remaining))
        if obstacles is not None:
            speed, extra = self._stop_speed()
            if obstacles.blocking_ahead(t, direction, speed=speed, extra=extra):
                if self._waiting_since is None:
                    self._waiting_since = t
                self.phase = "waiting_obstacle"
                return Command(0.0, 0.0)
            v = self._slow_for(obstacles.gap_ahead(t, direction), v, cap)
        self._waiting_since = None
        return Command(direction * v, w)

    def _stop_speed(self):
        """정지 판단에 쓸 (속도, 덧붙일 거리). 장애물 앞에 서 있는 동안(_waiting_since)은 다시 나가면 낼 최저 속도와 stop_hysteresis 를 넣는다 —
        서면 속도 항이 0 이 되어 판단 거리가 줄고, 다시 min_v 로 나가면 또 서는 「섰다 갔다」를 막는다 (2026-11-19 검토)."""
        p = self.p
        if self._waiting_since is not None:
            return max(abs(self._last.v), p.min_v), p.stop_hysteresis
        return abs(self._last.v), 0.0

    def _slow_for(self, gap, v, cap):
        """앞의 지도에 없는 점까지 gap(m) 남았을 때, 반응하는 동안 가는 거리와 obstacle_decel 제동 거리를 더해도
        stop_dist 앞에서 설 수 있는 속도로 줄인다: stop_dist + v·R + v²/(2a) = gap 의 v (R = stop_reaction_s + 제어 두 주기).
        정지 판단(stop_dist + v·stop_reaction_s)보다 늘 안쪽이라, 줄이며 다가가다 min_v 근처에서 선다."""
        p = self.p
        if not math.isfinite(gap):
            return v
        a, r = p.obstacle_decel, p.stop_reaction_s + 2.0 / max(1.0, p.control_hz)
        room = max(0.0, gap - p.stop_dist)
        limit = -a * r + math.sqrt((a * r) ** 2 + 2.0 * a * room)
        return min(v, max(limit, min(p.min_v, cap)))

    # ------------------------------------------------------------------ NORMAL · ASSAULT
    def _standing_shift(self, t, pose, grid, extra, obstacles):
        """서 있는 물체가 목표 칸을 덮고 있으면 목표 곁의 빈 칸으로 target 을 옮기고 그리로 가는 길을 돌려준다 (params.goal_shift_m 의 설명).
        옮기지 않으면 None — 옮겨 두었던 target 은 원래 자리로 되돌린다 (덮은 것이 사라졌다 · 옮길 빈 칸이 없다)."""
        p, g = self.p, self.goal
        assault = g.mode == MODE_ASSAULT
        home = self._shift_home if self._shift_home is not None else self.target
        reach = min(p.goal_shift_m, p.assault_accept_m) if assault else p.goal_shift_m
        path = alt = None
        if reach > 0.0 and extra and obstacles is not None and g.mode != MODE_STRAIGHT:
            spot = self._entry if assault and self._entry is not None else home
            cell = grid.cell_of(*spot)
            covered = grid.blocked(cell, extra) and not grid.blocked(cell)
            # 이미 옮겨 둔 동안에는 덮은 것이 「서 있는가」 를 다시 묻지 않는다 (드론 · 라이다의 판단이 한 장씩 흔들려도 목표가 오가지 않게)
            if covered and (self._shift_home is not None or getattr(obstacles, "standing_cover", lambda *_a: False)(t, grid, cell)):
                if self._shift_home is not None and grid.free(self.target[0], self.target[1], extra):
                    alt = self.target   # 옮겨 둔 자리가 아직 비어 있다 — 그대로 (칸이 바뀔 때마다 옮겨 다니지 않게)
                else:
                    alt = grid.nearest_free(home[0], home[1], extra=extra, max_dist=reach)
                    if assault and alt is not None and p.assault_shift_out_m >= 0.0:
                        # 돌진은 깊이가 전부다: 벽에서 멀어지는 칸으로는 옮기지 않는다 (params.assault_shift_out_m 의 설명 — 04:48 실차)
                        centre = grid.center(*grid.cell_of(home[0], home[1]))   # 목표가 든 칸의 줄은 늘 받는다 (칸 금이 어디에 놓이든)
                        limit = max(_wall_gap(grid, home[0], home[1]), _wall_gap(grid, centre[0], centre[1])) + p.assault_shift_out_m
                        if _wall_gap(grid, alt[0], alt[1]) > limit:
                            alt = grid.nearest_free(home[0], home[1], extra=extra, max_dist=reach,
                                                    accept=lambda x, y: _wall_gap(grid, x, y) <= limit)
                if alt is not None:
                    path = grid.plan((pose.x, pose.y), alt, extra)
        if path is None:
            if self._shift_home is not None:
                self.target, self._shift_home = self._shift_home, None
                self._reset_progress(self._progress_t)   # 목표 자리가 바뀌었다: 남은 거리의 최소를 새로 잡는다 (진행 시계는 그대로)
            return None
        if self._shift_home is None:
            self._shift_home = home
        if alt != self.target:
            self.target = alt
            self._reset_progress(self._progress_t)
        return path

    def _plan(self, t, pose, grid, extra, obstacles=None):
        shifted = self._standing_shift(t, pose, grid, extra, obstacles)
        if shifted is not None:   # 서 있는 물체가 목표 칸을 덮었다 → 곁의 빈 칸으로 (보통 길 — 그 물체를 지나는 길이 아니다)
            self._through_obstacle = self._tight_path = self.narrow_path = self._fresh_path = False
            self._path = list(shifted)
            self._plan_t = t
            return
        target = self._entry if self.goal.mode == MODE_ASSAULT else self.target
        path = grid.plan((pose.x, pose.y), target, extra)
        self._through_obstacle = False
        self._tight_path = False
        self.narrow_path = False
        self._fresh_path = False
        ratio = self.p.memory_detour_ratio
        if extra and obstacles is not None and (path is None or ratio > 0.0):
            # 길이 없으면 기억을 다 빼고, 길은 있는데 크게 돌아가는 경우(ratio)에는 차체 뒤쪽의 기억은 남기고 뺀다 (memory_rear_deg)
            fresh = obstacles.fresh_cells(t, grid, pose if path is not None else None)
            if fresh is not extra:  # 기억으로만 남은 칸이 있다 → 그것 때문에 길이 없으면(켰으면: 크게 돌아가도) 기억만 빼고 짠 길로
                alt = grid.plan((pose.x, pose.y), target, fresh)
                start = (pose.x, pose.y)
                if alt is not None and (path is None or path_length(start, path) > ratio * path_length(start, alt) + self.p.memory_detour_m):
                    self._fresh_keeps_rear = path is not None
                    path, self._fresh_path = alt, True
        if path is None and extra and obstacles is not None:
            # 좁은 틈 2단계: 서 있는 물체만 작게 보고 한 번 더 (params.tight_radius). 길이 나오면 천천히 그 곁을 지난다
            tight = obstacles.tight_cells(t, grid)
            if tight is not None:
                path = grid.plan((pose.x, pose.y), target, tight)
                self._tight_path = path is not None
        if path is None and extra:
            path = grid.plan((pose.x, pose.y), target)  # 임시 장애물을 빼면 길이 있다 → 가다가 앞에서 멈추고 기다린다
            self._through_obstacle = path is not None
        if path is None and self.p.narrow_grow_m > 0.0 and self.goal.mode == MODE_NORMAL:
            # 좁은 틈 3단계: 지도만으로도 길이 없다 — 지도에 0.40 m(규정의 최소 이격)보다 좁게 그려진 틈이 막고 있을 수 있다 (드론 상자 오차).
            # 구조물을 narrow_grow_m 만 부풀린 격자로 짠다 (칸 번호가 같아 임시 장애물 칸은 그대로 쓴다). 임시 장애물 때문에 안 되면 그것을 빼고 —
            # 가다가 앞에서 본다 (위의 _through_obstacle 과 같다)
            narrow = grid.narrowed(self.p.narrow_grow_m)
            if narrow is not grid:
                path = narrow.plan((pose.x, pose.y), target, extra)
                if path is None and extra:
                    path = narrow.plan((pose.x, pose.y), target)
                    self._through_obstacle = path is not None
                self.narrow_path = path is not None
        if path is not None and self.goal.mode == MODE_ASSAULT and math.hypot(self.target[0] - target[0], self.target[1] - target[1]) > 1e-3:
            path = list(path) + [self.target]
        self._path = list(path) if path is not None else None
        self._plan_t = t
        # 남은 경로 길이의 최소(_best)는 다시 짜도 이어 간다 (_watch_progress)


    def _step_path(self, t, pose, cap, grid, obstacles):
        p, g = self.p, self.goal
        tx, ty = self.target
        dist = math.hypot(tx - pose.x, ty - pose.y)
        if self._state == "drive":
            self._min_dist = dist if self._min_dist is None else min(self._min_dist, dist)
        if self._retrace is not None:  # 끝내기 전에 지나온 길로 물러나는 중 (_give_up)
            return self._retrace_step(t, pose, grid, obstacles)
        if self._room_move_until is not None:  # 돌 자리가 없어 비키는 중 (_no_room)
            v, w = self._room_move
            clear = self._room_clear(t, obstacles, v) if self._room_by_structure else self._move_clear(t, pose, grid, obstacles, math.copysign(1.0, v))
            if t < self._room_move_until and clear:
                self.phase = "backing_up" if v < 0 else "moving_off"
                self._pause(t)
                return Command(v, w)
            self._room_move_until = None
            self._path = None
            self._rot_since = self._turn_since = None
        if self._state == "settle":
            self._pause(t)
            if not self._settle(t):
                return STOP
            self._settle_since = None
            if self._check_arrival(t, pose, dist, grid):
                return STOP
        if self._state == "align":
            return self._align(t, pose, obstacles, grid)
        if dist <= g.pos_tol * STOP_FRACTION:
            return self._stop_at_goal(t, pose, grid, obstacles)

        # 뒤로 빼는 중
        if self._backup_until is not None:
            if t < self._backup_until and self._rear_clear(t, pose, grid, obstacles):
                self.phase = "backing_up"
                return Command(-p.backup_v, self._backup_w)
            self._backup_until = None
            self._path = None

        # 경로
        extra = obstacles.extra_cells(t, grid) if obstacles is not None else set()
        version = obstacles.version if obstacles is not None else 0
        # replan_s(1 s)마다 무조건 다시 짜는 것은 그대로 둔다 — 2026-11-19 통합 C12 에서 빼 보았더니 가상 30판 닿은 주기 6 → 184,
        # 오전 실차 10:50 끝 장면 기록 재생에서 58 s 를 서 있으며 965 주기를 닿았다 (낡은 경로를 따라 붙은 채 밀었다). 계산량은 다른 데서 줄인다
        need = self._path is None and t - self._plan_t >= p.replan_min_s
        need = need or (self._path is not None and t - self._plan_t >= p.replan_s)
        if self._path is not None and version != self._plan_version and t - self._plan_t >= p.replan_min_s:
            watch = (obstacles.tight_cells(t, grid) or extra) if self._tight_path else extra
            if self._fresh_path:
                watch = obstacles.fresh_cells(t, grid, pose if self._fresh_keeps_rear else None)
            need = need or grid.path_blocked((pose.x, pose.y), self._path, watch)
        if need:
            self._plan(t, pose, grid, extra, obstacles)
            self._plan_version = version
            if (tx, ty) != tuple(self.target):   # _plan 이 목표를 곁의 빈 칸으로 옮겼다 · 되돌렸다 (_standing_shift)
                tx, ty = self.target
                dist = math.hypot(tx - pose.x, ty - pose.y)
                self._min_dist = dist
        if not self._path:
            self.phase = "no_path"
            self.remaining = None
            self._mark_blocked(t)
            return STOP
        while len(self._path) > 1 and math.hypot(self._path[0][0] - pose.x, self._path[0][1] - pose.y) < WAYPOINT_REACHED_M:
            self._path.pop(0)
        remaining = path_length((pose.x, pose.y), self._path)
        self.remaining = remaining
        self._watch_progress(t, remaining, dist)

        assault = g.mode == MODE_ASSAULT
        if remaining - abs(self._last.v) * p.coast_s <= g.pos_tol * STOP_FRACTION:
            return self._stop_at_goal(t, pose, grid, obstacles)

        lx, ly = self._lookahead(pose)
        err = wrap(math.atan2(ly - pose.y, lx - pose.x) - pose.yaw)
        # 좁은 틈 길(narrow_path)에서 구조물 바로 곁 구간(보통 격자로는 막힌 칸)에 있거나 곧 들어선다 → 천천히, 그리고 경로와 나란히 맞춘 뒤에 들어간다
        narrow_here = self.narrow_path and (grid.blocked(grid.cell_of(pose.x, pose.y)) or grid.blocked(grid.cell_of(lx, ly)))
        turn_limit = math.radians(min(p.narrow_align_deg, p.rotate_in_place_deg) if narrow_here else p.rotate_in_place_deg)
        if abs(err) > turn_limit:
            if dist <= g.pos_tol:
                # 이미 도착 허용 안이다. 몇 cm 더 다가가려고 서서 돌지 않고 여기서 선다 (2026-11-19 가상: 거점 안으로 비스듬히 돌진해
                # 목표 7 cm 옆에 왔는데, 3 cm 안으로 가려고 벽에 붙은 채 서서 돌다 모서리가 걸려 시한까지 갔다)
                return self._stop_at_goal(t, pose, grid, obstacles)
            if (g.mode == MODE_NORMAL and (p.reapproach_slack_m > 0.0 or p.arrive_slack_m > 0.0)
                    and self._min_dist is not None and self._min_dist < dist - 0.03 and remaining <= dist + 0.20
                    and self._min_dist <= g.pos_tol + max(p.reapproach_slack_m, p.arrive_slack_m, p.timeout_reach_slack_m)):
                # 목표 곁을 서지 못하고 지나쳤다 (가장 가까웠던 때보다 멀어졌고, 목표가 옆 · 뒤이고, 길이 곧장 그리로 간다) — 서서 돌아 다시 다가가려는 참이다.
                # 자리에 선 뒤의 규칙(_check_arrival)과 같게: 허용을 조금 넘은 정도면 돌아서 고치지 않고 여기서 끝내고, 아니면 한 번 다시 다가간다
                # (2026-11-20 가상, 명령이 0.45 s 늦게 듣는 차: 0.16 m 지나침 → 돌아서 0.16 m 가다 또 지나침 을 16 s 되풀이하다 NO_PROGRESS)
                slack = max(p.timeout_reach_slack_m, p.arrive_slack_m)
                if self._reapproaches >= 1 and p.reapproach_slack_m > 0.0:
                    slack = max(slack, p.reapproach_slack_m)
                if dist <= g.pos_tol + slack:
                    self._near_enough, self._near_slack = True, slack
                    return self._stop_at_goal(t, pose, grid, obstacles)
                if not self._hop_turning:
                    self._hop_turning = True
                    self._reapproaches += 1
            # 서서 돌면 차체 모서리가 쓸고 간다. 다시 달릴 수 있을 때(오차 rotate_in_place_deg)까지 돌면 지도에 없는 것에 닿는가 —
            # 닿으면 반대쪽으로 길게, 그것도 닿으면 돌지 않고 물러나거나 기다린다 (2026-11-18 팀 테스트 2차: 다른 UGV 에 닿은 채 시한까지)
            turn = self._free_turn(t, err, abs(err) - turn_limit + math.radians(5.0 if narrow_here else 10.0), obstacles, pose, grid)
            if turn is None:
                if "obstacle" not in self._turn_block:  # 벽 · 박스 · 기둥만 막는다 — 기다려도 비키지 않는다
                    return self._no_room(t, pose, grid, obstacles, NO_ROOM_TO_LEAVE, "structure", err)
                # 지도에 없는 것이 바짝 붙어 못 돈다 → 기다리다 backoff_after_s 뒤 비키기 계획(_room_move_plan "obstacle")이
                # 그 물체에서 멀어지는 쪽(보이는 앞 곧게 · 앞쪽 호, 아니면 뒤 곧게)으로 비킨다. 전의 「곧게 조금 나아가기」(_creep_ahead,
                # cfb804c)는 이 계획의 앞 곧게 후보와 같은 일이라 2026-11-19 통합 때 뺐다 (test_moves_off_when_there_is_no_room_to_turn)
                return self._wait_obstacle(t, pose, grid, obstacles, close=True)
            if self._turn_stuck(t, pose):
                return self._no_room(t, pose, grid, obstacles, NO_ROOM_TO_LEAVE, "stuck", err)
            self.phase = "rotating"
            mag = max(p.min_turn_w, min(p.rotate_max_w, p.rotate_gain * abs(err)))
            if turn != math.copysign(1.0, err):
                mag = p.rotate_max_w  # 반대쪽으로 길게
            # 돌라고 breakaway_hold_s 넘게 냈는데 아직 안 돈다(정지 마찰) → rotate_kick_w 까지 올린다. 돌기 시작하면 바로 내린다 (params 의 설명)
            waited = t - self._turn_since - p.breakaway_hold_s if self._turn_since is not None else 0.0
            if p.rotate_kick_w > mag and waited > 0.0 and abs(self._turn_sum) < p.rotate_kick_moved_rad:
                mag = min(p.rotate_kick_w, mag + p.breakaway_ramp * waited)
            self._stuck_ref = None
            self._waiting_since = self._close_since = None
            far = self.remaining is None or self.remaining > p.spin_far_m  # 목표 바로 앞의 회전은 전과 같게 (params.spin_far_m)
            return Command(0.0, self._spin_boost(t, pose, math.copysign(mag, turn), err, grid, obstacles, allow=far))
        self._turn_since = None
        self._hop_turning = False
        vmax = p.assault_v if assault else p.max_v
        decel = p.assault_decel if assault else p.approach_decel
        v = vmax if assault else vmax * math.cos(err) ** 2
        if not assault and 0.0 < p.approach_v < vmax:
            # 목표 approach_m 앞부터는 approach_v 이하 — 도착 직전의 속도 · 서는 자리가 max_v 를 올리기 전과 같게 (params.max_v 의 설명)
            v = min(v, math.sqrt(p.approach_v ** 2 + 2.0 * decel * max(0.0, remaining - p.approach_m)))
        v = min(v, math.sqrt(2.0 * decel * max(0.0, remaining - g.pos_tol * STOP_FRACTION)))
        if not assault and p.approach_lag_s > 0.0 and (self._reapproaches > 0 or abs(err) > math.radians(p.approach_curve_deg)):
            # 다시 다가가는 짧은 거리 · 굽어 들어오는 마지막 구간: 명령이 듣기까지 가는 거리가 남은 거리를 넘지 않게 (params.approach_lag_s)
            v = min(v, max(0.0, remaining - g.pos_tol * STOP_FRACTION) / p.approach_lag_s)
        v = max(v, p.min_v)
        v = min(v, cap)
        if self.in_corridor:
            v = min(v, max(self._corridor_speed(t, err, remaining, obstacles), p.min_v))
        if self._tight_path:
            v = min(v, max(p.tight_v, p.min_v))  # 서 있는 물체 곁의 좁은 틈은 천천히
        if narrow_here:
            v = min(v, max(p.narrow_v, p.min_v))  # 지도상 0.40 m 보다 좁은 틈 속과 그 앞뒤는 더 천천히
        w = _clamp(p.turn_gain * err, p.max_w)
        self.phase = "assault" if assault else "driving"

        if obstacles is not None:
            curve = _clamp(w / max(v, 0.05), 5.0)  # 지금 도는 호 (1/m) — 진행 띠를 그 호를 따라 굽혀 본다
            speed, extra = self._stop_speed()
            hold = self._standoff(t, obstacles, curve, remaining)
            extra += hold
            if obstacles.blocking_ahead(t, 1.0, speed=speed, curvature=curve, extra=extra):
                # 그 호로 돌면 닿는다 → 덜 돌거나 곧게 가면 비켜 가는지 본다 (출발하며 급히 돌 때 옆의 것을 쓸지 않게)
                for softer in (0.5 * curve, 0.0):
                    if curve and not obstacles.blocking_ahead(t, 1.0, speed=speed, curvature=softer, extra=extra):
                        curve, w = softer, v * softer
                        break
                else:
                    if dist <= g.pos_tol:
                        # 이미 도착 허용 안인데 앞의 물체 때문에 더 못 간다 → 여기서 선다. 기다려 봐야 몇 cm 를 더 가려는 것뿐이다
                        # (2026-11-20: 서 있는 물체 곁으로 옮긴 목표 0.09 m 앞에서 정지 띠에 걸려 8 s 뒤 NO_PROGRESS 였다 — 가상)
                        return self._stop_at_goal(t, pose, grid, obstacles)
                    close = obstacles.gap_ahead(t, 1.0, reach=p.backoff_dist) <= p.backoff_dist
                    return self._wait_obstacle(t, pose, grid, obstacles, close=close)
            slowed = self._slow_for(obstacles.gap_ahead(t, 1.0, curvature=curve, reach=p.slow_dist + hold) - hold, v, cap)
            if slowed < v:
                w *= slowed / v  # 같은 호로 (본 띠와 실제로 가는 길이 같게)
                v = slowed
            if self._through_obstacle and p.through_pass_m > 0.0:
                # 임시 장애물을 빼고 짠 길은 그 물체 곁을 바짝 지날 수 있다 — 띠 바로 밖의 것은 서지도 줄이지도 않았다. 옆 틈만큼만 줄인다:
                # 틈이 through_pass_m 넘으면 그대로, 띠의 가장자리(stop_side_margin)면 tight_v. 뚫고 가는 것은 그대로다 (params.through_pass_m)
                floor = max(p.min_v, min(p.tight_v, v))
                reach = v * p.stop_reaction_s + max(0.0, v * v - floor * floor) / (2.0 * p.obstacle_decel) + 0.05
                side = obstacles.pass_gap(t, curve, reach, p.through_pass_m)
                if side < p.through_pass_m:
                    edge = p.stop_side_margin if obstacles.side_margin is None else obstacles.side_margin
                    share = max(0.0, min(1.0, (side - edge) / max(1e-6, p.through_pass_m - edge)))
                    limit = floor + (v - floor) * share
                    if limit < v:
                        w *= limit / v
                        v = limit
        self._waiting_since = self._close_since = None
        self._watch_stuck(t, pose, v, grid, obstacles)
        return Command(v, w)

    def _standoff(self, t, obstacles, curve, remaining):
        """지금 경로가 임시 장애물을 빼고 짠 것(그 물체를 지나야만 갈 수 있다)이면 그 물체 앞에서 더 일찍 선다 → 정지 거리에 더할 값 (m).
        stop_dist(0.30 m)까지 다가가 서면 돌 자리가 없고 상대의 길도 막는다 (params.through_standoff_m 의 설명).
        목표가 그 물체보다 앞에 있으면(물체 곁의 자리로 가는 중) 더하지 않는다 — 전처럼 목표까지 간다."""
        p = self.p
        hold = p.through_standoff_m - p.stop_dist
        if not self._through_obstacle or hold <= 0.0:
            return 0.0
        gap = obstacles.gap_ahead(t, 1.0, curvature=curve, reach=p.through_standoff_m + 1.0)
        if not math.isfinite(gap) or gap + p.robot_half_length >= remaining:
            return 0.0
        return hold

    def _stop_at_goal(self, t, pose, grid, obstacles):
        """목표 앞에서 선다. 보통은 settle_s 뒤 새 위치로 확인(_check_arrival)한다.
        방향을 맞춰야 하는 NORMAL 목표는 그 첫 확인을 건너뛰고 곧장 방향 맞추기로 간다 — 회전 뒤의 확인(settle_s)이 위치와 방향을
        한 번에 본다 (위치가 허용 밖이면 그때 다시 간다). 정지 → 0.3 s → 회전 → 0.3 s 가 정지 → 회전 → 0.3 s 로, 목표당 settle_s 가 준다
        (2026-11-19 통합 검토 5번, align_without_settle). 방향을 안 보는 목표 · 이미 방향이 맞은 목표 · 돌진은 전과 같다.
        어느 쪽으로 돌아도 닿을 것이 없을 때만 건너뛴다 — 지도 구조물이 차체 반 대각선 + align_skip_clear_m 밖이고 그 안에 지도에 없는 점이
        없을 때. 정지 직후에는 아직 앞으로 미끄러지는데 서서 돌기 전 확인은 지금 자세로 보기 때문이다 (가상: 상자 면 0.19 ~ 0.22 m 자리에서
        건너뛰면 미끄러지며 돌다 모서리가 걸려 못 돈 주기 23 → 601. 그런 자리는 전처럼 멈춘 뒤 확인하고 돈다)."""
        g, p = self.goal, self.p
        room = math.hypot(p.robot_half_length, p.robot_half_width) + p.align_skip_clear_m
        if (p.align_without_settle and g.mode == MODE_NORMAL and self.use_yaw
                and abs(wrap(g.yaw - pose.yaw)) > g.yaw_tol
                and (grid is None or grid.spec.distance_to_structure(pose.x, pose.y) > room)
                and not any(math.hypot(x, y) <= room for x, y in self._near_points(t, obstacles, reach=room))):
            self._state, self._settle_since = "align", None
            return self._align(t, pose, obstacles, grid)
        self._state = "settle"
        self._settle(t)
        return STOP

    def _wait_obstacle(self, t, pose, grid, obstacles, close):
        """진행 방향에 지도에 없는 물체. 서서 기다리며 wait_before_replan_s 마다 임시 장애물을 넣고 다시 짠다.

        - close(닿았거나 바로 앞)가 backoff_after_s 넘게 이어지면 뒤가 빈 경우 backup_s 동안 뒤로 뺀다
        - 그 물체를 지나야만 갈 수 있는 목표(임시 장애물을 넣으면 길이 없다)에서 obstacle_give_up_s 동안 못 나아가면 시한 전에 BLOCKED
        """
        p = self.p
        if self._waiting_since is None:
            self._waiting_since = t
        self.phase = "waiting_obstacle"
        self._stuck_ref = None
        rear = self._rear_clear(t, pose, grid, obstacles) and self._reverse_ok(pose)
        # 서 있는 물체 곁으로 옮긴 목표(_standing_shift)로 가다 그 물체 앞에 선 것도 같다 — 기다려도 비키지 않는다
        through = self._through_obstacle or self._shift_home is not None
        if through or (close and not rear):
            why = "지도에 없는 물체가 길을 막음" if through else "지도에 없는 물체에 붙었는데 뒤도 막힘"
            if self._stall(t, pose, why):
                return STOP
        else:
            self._stalled_since = None
        if close:
            if self._close_since is None:
                self._close_since = t
            if t - self._close_since >= p.backoff_after_s:
                # 어디로 비킬지는 비키기 계획 하나가 고른다 — 지도와 지금 라이다 점을 함께 보아, 그 물체에서 가장 멀어지는 쪽
                # (보이는 앞쪽 호가 되면 그쪽, 아니면 뒤). 고르지 못했는데 뒤가 비어 있으면 전처럼 곧게 뒤로
                move = self._room_move_plan(t, pose, grid, obstacles, "obstacle")
                if move is not None:
                    v, w, duration = move
                    if v < 0:
                        self._note_reverse(pose)
                    self._close_since = None
                    self._room_move_until, self._room_move = t + duration, (v, w)
                    self._room_by_structure = False
                    self._path = None
                    self.phase = "backing_up" if v < 0 else "moving_off"
                    return Command(v, w)
            if t - self._close_since >= p.backoff_after_s and rear:
                self._note_reverse(pose)
                self._close_since = None
                self._backup_until = t + p.backup_s
                self._backup_w = 0.0
                self._path = None
                self.phase = "backing_up"
                return Command(-p.backup_v, 0.0)
        else:
            self._close_since = None
        if t - self._waiting_since >= p.wait_before_replan_s and t - self._plan_t >= p.replan_min_s:
            self._path = None  # 다음 주기에 임시 장애물을 넣고 다시 짠다
            self._plan_t = t - p.replan_min_s
            self._waiting_since = t
        return Command(0.0, 0.0)

    def _stall(self, t, pose, why, outcome=BLOCKED, code=CODE_OBSTACLE, limit=None):
        """나아갈 방법이 없는 채로 obstacle_give_up_s 가 지나면 시한 전에 끝낸다 (끝냈으면 True). 보통 BLOCKED + OBSTACLE,
        자리에는 왔는데 도착 방향만 못 맞출 때는 REACHED + NO_ROOM_AT_GOAL (미션을 이어 가게 — detail 의 말과 코드가 같은 뜻이게).
        limit = 부르는 쪽이 막은 것이 서 있는 물체임을 알 때의 더 짧은 기다림 (s — obstacle_give_up_s 보다 길게는 안 된다)."""
        if self._stalled_since is None:
            self._stalled_since = t
        p = self.p
        limit = p.obstacle_give_up_s if limit is None else min(limit, p.obstacle_give_up_s)
        grid, obstacles = self._world
        # 서 있는 물체 앞에서 일찍 끝내는 것은 「그 물체를 지나야만 갈 수 있을 때」 만 (돌아갈 길이 아예 없다). 길은 있는데 붙어서 못 나아가는 때는
        # 다시 짜다가 풀리기도 한다 — 2026-11-19 21:53 실차: 0.07 m 앞의 서 있는 물체에 붙어 3.0 s 를 서 있다가 다시 짠 길로 돌아 나가 도착했다
        through = self._through_obstacle or self._shift_home is not None   # 옮긴 목표 앞에서 그 서 있는 물체에 막힌 것도 같다
        if through and 0.0 < p.obstacle_give_up_static_s < limit and obstacles is not None:
            # 드론으로 보아 서 있는 UGV 면 라이다로 tight_still_s 를 지켜보지 않아도 「서 있는 물체」 다 (obstacles.blocker_ugv). 드론이 「움직인다」 ·
            # 「곁에 UGV 없음」 이라 해도 라이다의 판단은 그대로 둔다 (드론은 늦고, 화면 밖 · 가림이 있다)
            self.blocker_kind = getattr(obstacles, "blocker_ugv", lambda _t: None)(t)
            parked = self.blocker_kind is not None and self.blocker_kind[0] == "parked"
            if not (parked or obstacles.blocker_static(t, grid, p.tight_still_s)):
                self._static_since = None
            elif self._static_since is None:
                self._static_since = t
            # 막은 것은 서 있는 물체다 — 기다려도 비키지 않는다 (params 의 설명). 다만 「서 있다」 를 안 뒤에 한 번은 다시 짜 본 다음이어야 한다:
            # 서 있는 물체는 작게 보고 짜는 좁은 틈 길(tight_radius)이 그때 처음 열린다 (가상: 문에 비켜 선 UGV 곁을 지나가던 것이 2 s 에 포기했다)
            if self._static_since is not None and self._plan_t > self._static_since:
                limit = p.obstacle_give_up_static_s
                if parked and 0.0 < p.obstacle_give_up_parked_s < limit:
                    limit = p.obstacle_give_up_parked_s
        if t - self._stalled_since >= limit:
            detail = "%s — %.1f s 동안 못 나아가 시한 전에 끝냄" % (why, t - self._stalled_since)
            if through and self.blocker_kind is not None:
                kind, still = self.blocker_kind
                detail += {"parked": " (드론: %.0f s 째 서 있는 UGV)" % still, "moving": " (드론: 움직이던 UGV)"}.get(kind, " (드론: 그 곁에 UGV 안 보임)")
            if outcome == BLOCKED:
                if self._accept_near_goal(pose, through and self._static_since is not None):
                    return True
                return not self._give_up(t, pose, self._world[0], self._world[1], outcome, detail, code)
            self.finish(outcome, detail, pose, code)
            return True
        return False

    def _accept_near_goal(self, pose, standing):
        """서 있는 물체가 목표 곁을 막아 못 나아가는데 목표 바로 앞(near_goal_accept_m 안)까지 와 있으면, BLOCKED 로 끝내는 대신 이 자리에서 도착으로 끝낸다
        (params.near_goal_accept_m 의 설명). 그렇게 했으면 True. 방향을 맞춰야 하면 맞춘 뒤에 끝난다 (못 돌면 전처럼 NO_ROOM_AT_GOAL).
        되짚어 물러나지 않는다 — 그 자리가 도착 자리다. finish 가 [POS_NOT_MET] 을 붙인다.
        standing = 막은 것이 서 있는 물체인가 (드론으로 서 있는 UGV · 라이다로 한자리에 찍혀 온 것) — 부르는 쪽이 정한다."""
        p, g = self.p, self.goal
        if p.near_goal_accept_m <= 0.0 or g.mode != MODE_NORMAL or pose is None or not standing or self._accept_here:
            return False
        dist = math.hypot(self.target[0] - pose.x, self.target[1] - pose.y)
        if dist > p.near_goal_accept_m:
            return False
        note = "서 있는 물체가 목표 곁을 막아 %.2f m 앞에서 도착으로 끝냄" % dist
        if self.blocker_kind is not None and self.blocker_kind[0] == "parked":
            note += " (드론: %.0f s 째 서 있는 UGV)" % self.blocker_kind[1]
        self._accept_here = True
        self._stalled_since = self._waiting_since = self._close_since = None
        self._backup_until = self._room_move_until = None
        if self.use_yaw and abs(wrap(g.yaw - pose.yaw)) > g.yaw_tol:
            self.detail = (self.detail + "; " if self.detail else "") + note
            self._state, self._settle_since, self._path = "align", None, None
            return True
        self.finish(REACHED, note, pose)
        return True

    def _free_turn(self, t, err, need, obstacles, pose=None, grid=None):
        """제자리에서 err 쪽으로 need(rad)만큼 돌아도 지도에 없는 것 · 지도 구조물에 닿지 않으면 그 쪽(±1), 닿으면 반대쪽으로 같은 자리까지
        길게 돌 수 있는지 보고 그 쪽, 둘 다 닿으면 None. 막은 것은 self._turn_block 에 남긴다 ("obstacle" · "structure")."""
        side = math.copysign(1.0, err)
        self._turn_block = set()
        margin = self.p.sweep_margin
        need = max(need, math.radians(5.0))

        def hits(angle):
            why = set()
            if obstacles is not None and obstacles.turn_hits(t, angle, margin):
                why.add("obstacle")
            if self._structure_turn_hits(pose, angle, grid):
                why.add("structure")
            self._turn_block |= why
            return bool(why)

        if not hits(side * need):
            return side
        other = 2.0 * math.pi - abs(err) - (abs(err) - need)  # 반대쪽으로 같은 자리까지
        if other > 0.0 and not hits(-side * max(other, math.radians(5.0))):
            return -side
        return None

    def _structure_turn_hits(self, pose, angle, grid):
        """제자리에서 angle(rad, 왼쪽 +)만큼 돌면 차체가 지도 구조물(벽 · 박스 · 고정 기둥)에 structure_turn_clear 보다 붙는가.
        이미 그보다 붙어 있으면 더 붙는 회전만 막는다 (멀어지는 쪽으로는 돈다)."""
        if grid is None or pose is None or abs(angle) < 1e-6:
            return False
        p, spec = self.p, grid.spec
        hl, hw, clear = p.robot_half_length, p.robot_half_width, p.structure_turn_clear
        if spec.distance_to_structure(pose.x, pose.y) > math.hypot(hl, hw) + clear:
            return False  # 어느 방향으로 돌아도 차체가 쓸고 가는 원 밖이다
        g0 = body_clearance(spec, pose.x, pose.y, pose.yaw, hl, hw)
        if g0 < -p.structure_trust_overlap:
            # 지도로는 차체가 구조물 속에 들어가 있다 — 실제로는 그럴 수 없으니 이 자리에서는 지도가 실제와 다르다 (상자 위치 오차 등).
            # 지도로 막지 않고, 정말 걸리면 안 도는 회전 감지(_turn_stuck)가 받는다 (2026-11-19 실차 bag 1118_202820: 지도상 9 cm 겹친 곳에서 실제로는 106° 돌았다)
            return False
        n = max(1, int(math.ceil(abs(angle) / TURN_STEP)))
        for k in range(1, n + 1):
            g = body_clearance(spec, pose.x, pose.y, pose.yaw + angle * k / n, hl, hw)
            if g < clear and g < g0 - 0.003:
                return True
        return False

    def _turn_stuck(self, t, pose):
        """서서 돌라는 명령이 rot_stuck_s 이어지는 동안 방향이 모두 합쳐 rot_stuck_rad 도 안 바뀌었나 (모서리가 걸려 돌라고 해도 안 돈다).

        창(rot_stuck_s)의 처음부터 끝까지 돈 각을 이어 더해 견준다. 위치 추정의 방향은 SLAM 위치 사이를 바퀴로 이어 붙여, 걸린 채 바퀴만
        헛돌 때도 0.1 s 마다 몇 도씩 돌았다가 SLAM 이 오면 되돌아온다 — 전의 「0.05 rad 바뀔 때마다 기준을 옮긴다」 는 그 흔들림 때문에
        걸린 채 15 s 를 돌아도 한 번도 켜지지 않았다 (2026-11-19 통합 검증, 가상 경기장 · 상자 곁 방향 맞추기)."""
        if self._turn_since is None or abs(self._last.w) < 1e-6 or abs(self._last.v) > 1e-6:
            self._turn_since, self._turn_yaw0, self._turn_sum = t, pose.yaw, 0.0
            return False
        self._turn_sum += wrap(pose.yaw - self._turn_yaw0)
        self._turn_yaw0 = pose.yaw
        if t - self._turn_since < self.p.rot_stuck_s:
            return False
        if abs(self._turn_sum) < self.p.rot_stuck_rad:
            return True
        self._turn_since, self._turn_sum = t, 0.0  # 돌고 있다 — 다음 창
        return False

    def _no_room(self, t, pose, grid, obstacles, prefix, why, err):
        """돌 자리가 없다: 구조물에서 멀어지는 쪽으로 no_room_moves 번까지 비켜 보고, 비킬 곳이 없거나 다 써 봤으면 BLOCKED.
        detail 에 prefix(NO_ROOM_AT_GOAL · NO_ROOM_TO_LEAVE)가 들어간다 — supervisor 가 이것으로 구별한다."""
        p = self.p
        self._rot_since = self._turn_since = None
        # 비켜 보기는 출발할 때만. 자리에 와 있으면 비키면 자리를 벗어나 「돌아와서 또 못 돎 → 또 비킴」 을 시한까지 되풀이한다 (가상 확인) —
        # 그 자리에서 바로 도착으로 끝낸다 (방향은 supervisor 가 팬으로)
        leaving = prefix == NO_ROOM_TO_LEAVE
        plan_why = why
        if why == "stuck" and grid is not None and body_clearance(grid.spec, pose.x, pose.y, pose.yaw, p.robot_half_length, p.robot_half_width) < 0.01:
            # 안 도는 까닭을 지도가 안다: 차체가 벽 · 상자 · 기둥에 닿아 있다. 「까닭 모름」 의 앞뒤 곧게만 보면, 두 상자가 만나는 모서리에
            # 비스듬히 닿았을 때 앞뒤가 다 막혀 비킬 곳이 없다 → 목표마다 곧바로 BLOCKED 를 끝없이 되풀이했다 (가상, 2026-11-19).
            # 구조물 때문으로 보고 완만한 호까지 굴려 본다
            plan_why = "structure"
        move = self._room_move_plan(t, pose, grid, obstacles, plan_why) if leaving and self._room_moves < p.no_room_moves else None
        if move is not None:
            v, w, duration = move
            self._room_moves += 1
            self._room_by_structure = plan_why == "structure" and grid is not None
            self._room_move_until, self._room_move = t + duration, (v, w)
            self.phase = "backing_up" if v < 0 else "moving_off"
            return Command(v, w)
        cause = {"structure": "벽 · 상자 · 기둥에 차체 모서리가 걸림",
                 "stuck": "서서 돌라고 %.1f s 냈는데 방향이 안 바뀜" % p.rot_stuck_s}[why]
        tx, ty = self.target
        # 자리에는 왔고 방향만 못 맞췄다 → 도착으로 끝내 미션을 이어 가게 한다 (supervisor 는 팬으로 겨눈다). 출발을 못 했으면 BLOCKED
        outcome = REACHED if prefix == NO_ROOM_AT_GOAL else BLOCKED
        detail = "%s: %s — 목표까지 %.2f m, 돌아야 할 방향 %.0f° 남김, 앞뒤로 비켜 본 횟수 %d" % (
            prefix, cause, math.hypot(tx - pose.x, ty - pose.y), math.degrees(abs(err)), self._room_moves)
        if outcome == BLOCKED:
            self._give_up(t, pose, grid, obstacles, outcome, detail, CODE_NO_ROOM_TO_LEAVE)
        else:
            self.finish(outcome, detail, pose, CODE_NO_ROOM_AT_GOAL)
        return STOP

    def _room_move_plan(self, t, pose, grid, obstacles, why):
        """돌 자리를 만들려면 어떻게 비킬까 → (v, w, 시간) 또는 None.

        후보: 앞 · 뒤로 곧게 backup_v × backup_s, 그리고 앞 · 뒤로 완만한 호(반지름 room_arc_r1 · r2, room_arc_m 만큼) 좌우.
        구조물 때문이면 미리 굴려 보아 차체가 구조물에 더 붙지 않으면서(structure_turn_clear 또는 지금 틈 중 작은 것) 차체 가운데가
        구조물에서 1 cm 넘게 멀어지는 것 중 가장 멀어지는 것. 벽과 나란히 붙어 서면 곧게는 안 멀어지므로 호가 고른다.
        안 도는 회전(stuck)이면 지도로는 까닭을 모르니 뒤로 곧게(안 되면 앞으로).

        비키는 동작은 이것 하나가 고른다 (2026-11-19) — 지도 구조물과 지금 라이다의 지도에 없는 점을 함께 본다:
        - 어느 까닭이든 굴려 보는 동안 차체가 지도에 없는 점에 지금보다(또는 sweep_margin 보다) 1 cm 넘게 더 붙는 후보는 버린다
        - why "obstacle"(지도에 없는 물체가 바짝 붙음): 차체 가운데가 그 점들에서 가장 멀어지는 후보
        - 뒤로 가는 후보는 라이다가 못 보는 쪽이다. 같은 값이면 보이는 앞쪽을 고르고(blind_penalty), 구조물 때문이 아니면 _reverse_ok 일 때만
        """
        p = self.p
        v0 = p.backup_v
        hl, hw = p.robot_half_length, p.robot_half_width
        spec = grid.spec if grid is not None else None
        pts = self._near_points(t, obstacles)
        # 드론이 본, 곁에 서 있는 UGV 도 점으로 넣는다 — 라이다가 못 보는 뒤쪽 · 몸 바로 곁에 있으면 그쪽으로 비키지 않게 (obstacles.ugv_body_points)
        pts = pts + list(getattr(obstacles, "ugv_body_points", lambda _t, _pose: ())(t, pose))

        def point_gap(rx, ry, ryaw):  # 출발 자세 기준 (rx, ry, ryaw)에 선 차체와 지도에 없는 점 사이의 가장 좁은 틈
            c, s = math.cos(ryaw), math.sin(ryaw)
            return min((_rect_gap(c * (px - rx) + s * (py - ry), -s * (px - rx) + c * (py - ry), hl, hw) for px, py in pts), default=math.inf)

        point_floor = min(p.sweep_margin, point_gap(0.0, 0.0, 0.0)) - 0.01
        reverse_ok = why == "structure" or self._reverse_ok(pose)
        straight = [(d * v0, 0.0, p.backup_s) for d in (-1.0, 1.0)]
        arcs = [(d * v0, k * d * v0 / r, p.room_arc_m / v0) for d in (-1.0, 1.0) for r in (p.room_arc_r1, p.room_arc_r2) for k in (-1.0, 1.0)]
        by_structure = why == "structure" and spec is not None
        if by_structure:
            # 짧은 후보도 본다 (길이 40 %): 두 상자가 만나는 오목한 모서리에 몇 mm 닿으면 긴 비키기는 모두 다른 쪽 상자에 걸렸다 (가상 2026-11-19)
            straight = straight + [(v, w, d * 0.4) for v, w, d in straight]
            arcs = arcs + [(v, w, d * 0.4) for v, w, d in arcs]
        if by_structure:
            body_now = body_clearance(spec, pose.x, pose.y, pose.yaw, hl, hw)
            floor = min(p.structure_turn_clear, body_now) - 0.003
            now = spec.distance_to_structure(pose.x, pose.y)
        elif why == "obstacle":
            if not pts:
                return None
            floor = -math.inf  # 지도 구조물은 _move_clear 가 본다 (전의 곧게 뒤로 빼기와 같은 기준)
            now = min(math.hypot(px, py) for px, py in pts)
            # 뒤쪽 호는 쓰지 않는다: 라이다가 못 보는 쪽으로는 곧게 · 짧게만(backup_s) 뺀다. 호로 길게(room_arc_m) 돌아 들어가면
            # 못 본 것(뒤의 UGV · 지도와 어긋난 상자)을 옆으로 쓸게 된다 (가상: 지도가 13 cm 어긋난 틈에서 접촉 3 → 62 주기)
            arcs = [a for a in arcs if a[0] > 0.0]
        else:
            floor = -math.inf
            arcs = []
        best = None
        for v, w, duration in straight + arcs:
            if v < 0 and (not reverse_ok or self.trail.rear_blocked(t)):
                continue
            # 구조물 때문이면 아래에서 차체를 굴려 보며 구조물과의 틈을 본다 — 곧게 0.1 m 끝의 세 점만 보는 _move_clear 는 두 상자가 만나는
            # 모서리에 비스듬히 닿았을 때 앞뒤를 다 막아 비킬 곳을 못 찾게 했다 (2026-11-19 가상). 지도에 없는 점은 그대로 본다
            if not (self._room_clear(t, obstacles, v) if by_structure else self._move_clear(t, pose, grid, obstacles, math.copysign(1.0, v))):
                continue
            x, y, yaw, ok = pose.x, pose.y, pose.yaw, True
            rx = ry = ryaw = 0.0
            n = max(1, int(math.ceil(abs(v) * duration / 0.02)))
            dt = duration / n
            for _ in range(n):
                yaw += w * dt
                x += v * dt * math.cos(yaw)
                y += v * dt * math.sin(yaw)
                ryaw += w * dt
                rx += v * dt * math.cos(ryaw)
                ry += v * dt * math.sin(ryaw)
                if point_gap(rx, ry, ryaw) < point_floor:
                    ok = False
                    break
                if spec is not None and (not spec.inside(x, y) or body_clearance(spec, x, y, yaw, hl, hw) < floor):
                    ok = False
                    break
            if not ok:
                continue
            if by_structure:
                # 차체 가운데가 구조물에서 멀어진 만큼, 또는 차체 가장자리의 틈이 늘어난 만큼 (두 상자가 만나는 모서리에 닿았을 때는 한쪽에서
                # 멀어지면 다른 쪽에 다가가 가운데 거리는 거의 그대로다 — 그래도 닿은 모서리를 떼면 돌 수 있다)
                gain = max(spec.distance_to_structure(x, y) - now, body_clearance(spec, x, y, yaw, hl, hw) - body_now)
            elif why == "obstacle":
                gain = min(math.hypot(px - rx, py - ry) for px, py in pts) - now
            else:
                return v, w, duration  # 까닭을 모름: 되는 것 중 첫째 (뒤로 곧게, 안 되면 앞으로)
            score = gain - (p.blind_penalty if v < 0 else 0.0)
            need_gain = 0.005 if by_structure and body_now < 0.01 else 0.01  # 이미 닿아 있으면 몇 mm 떼는 것도 다음 비키기의 발판이 된다
            if gain > need_gain and (best is None or score > best[0] + 1e-6):
                best = (score, (v, w, duration))
        return best[1] if best else None

    def _spin_boost(self, t, pose, w, err, grid, obstacles, allow=True):
        """서서 도는 명령 w (rad/s) 를 실제 회전에 맞춰 올린 것. err = 남은 각 (rad) — 바라는 빠르기를 정한다.

        실제로 도는 빠르기(TurnModel.rate — SLAM 방향 차이, 확인되면 IMU 각속도)가 바라는 빠르기보다 느리면 더하는 양을 spin_ramp 로 늘리고,
        빠르면 줄인다 (0 밑으로는 안 간다 — 원래 명령보다 작게 내지는 않는다). 상한은 spin_max_w. 남은 각이 작아지면 바라는 빠르기도 작아져
        더한 양이 빠진다. 곁(반 대각선 + spin_clear_m)에 지도 구조물 · 지도에 없는 점이 있으면 더하지 않는다 — 걸린 채 더 세게 밀지 않는다
        (안 도는 회전 감지 _turn_stuck 이 받는다). 설명 · 값: params.spin_max_w"""
        p = self.p
        if abs(w) < 1e-6 or pose is None:
            return w
        model = self.turn
        if p.spin_max_w <= p.rotate_max_w:  # 끔: 명령은 그대로. 실제 빠르기와 낸 명령만 기록에 남긴다 (켤지 정할 자료)
            rate = model.rate.rate(t)
            if rate is not None:
                model.last_rate = rate
            model.last_cmd = abs(w)
            return w
        fresh = self._boost_t is None or t - self._boost_t > 0.3
        if fresh:  # 서 있다가(또는 가다가) 새로 돌기 시작한다 — 지난 회전에서 배운 양의 일부에서 시작
            self._boost = p.spin_boost_carry * model.boost
            self._boost_since = t
            self._spin_hist, self._moving_w = [], None
        dt = 0.0 if fresh else min(0.2, t - self._boost_t)
        self._boost_t = t
        room = math.hypot(p.robot_half_length, p.robot_half_width) + p.spin_clear_m
        if (not allow or (grid is not None and grid.spec.distance_to_structure(pose.x, pose.y) < room)
                or any(math.hypot(x, y) <= room for x, y in self._near_points(t, obstacles, reach=room))):
            self._boost = 0.0
            model.last_cmd = abs(w)
            return w
        ref = max(p.spin_rate_min, min(p.spin_rate_max, p.spin_rate_gain * abs(err)))
        rate = model.rate.rate(t)
        # 돌기 시작한 직후는 누구나 느리다 (반응 지연 + 정지 마찰을 넘는 시간) — 그 뒤에도 느릴 때부터 올린다
        if rate is not None and t - self._boost_since >= p.latency_s + p.breakaway_hold_s:
            model.last_rate = rate
            # 그 빠르기를 낸 명령 = 잰 구간(반응 지연만큼 앞) 동안 낸 것 중 가장 큰 것 (작아지는 중이면 큰 쪽이 돌린 것이다)
            lo, hi = t - model.rate.window_s - p.latency_s, t - p.latency_s
            then = [c for ts, c in self._spin_hist if lo <= ts <= hi]
            if then and rate >= p.spin_moving_rate:
                self._moving_w = max(then) if self._moving_w is None else min(self._moving_w, max(then))
            elif then and rate < p.spin_stall_rate and self._moving_w is not None and max(then) < self._moving_w - 0.02:
                floor = min(p.rotate_max_w, self._moving_w)  # 돌다가 명령이 작아져 섰다 → 돌던 명령으로 (params.spin_moving_rate)
                if p.lead_learn > 0.0:
                    model.start_w = floor if model.start_w is None else max(model.start_w, floor)
                w = math.copysign(max(abs(w), floor), w)
            if rate < p.spin_rate_low * ref:
                self._boost += p.spin_ramp * dt
            elif rate > p.spin_rate_high * ref:
                self._boost -= p.spin_ramp * dt
            elif p.lead_learn > 0.0:  # 바라는 빠르기로 돌고 있다 → 이 양을 기억한다
                model.boost = (1.0 - p.lead_learn) * model.boost + p.lead_learn * self._boost
        self._boost = max(0.0, min(max(0.0, p.spin_max_w - abs(w)), self._boost))
        if self._boost >= p.spin_max_w - abs(w) - 1e-6 and p.lead_learn > 0.0:
            model.boost = max(model.boost, self._boost)  # 상한까지 올려야 했다 → 다음에도 높게 시작한다
        full, none = math.radians(p.spin_taper_full_deg), math.radians(p.spin_taper_none_deg)
        share = 1.0 if full <= none else max(0.0, min(1.0, (abs(err) - none) / (full - none)))  # 남은 각이 작으면 올리지 않는다 (params)
        out = math.copysign(max(abs(w), min(p.spin_max_w, abs(w) + share * self._boost)), w)  # 받은 명령보다 작게 내지는 않는다
        model.last_cmd = abs(out)
        self._spin_hist.append((t, abs(out)))
        if len(self._spin_hist) > 64:
            del self._spin_hist[:-64]
        return out

    def _near_points(self, t, obstacles, reach=1.0):
        """지금 스캔의 지도에 없는 점 가운데 차체 가까이 있는 것 (차체 좌표). 스캔이 없거나 낡았으면 빈 목록."""
        if obstacles is None or not getattr(obstacles, "_fresh", lambda _t: False)(t):
            return []
        return [(x, y) for x, y in getattr(obstacles, "unexplained_body", ()) if math.hypot(x, y) <= reach]

    def _lookahead(self, pose):
        """경로를 따라 lookahead_m 앞의 점."""
        left = self.p.lookahead_m
        prev = (pose.x, pose.y)
        for pt in self._path:
            seg = math.hypot(pt[0] - prev[0], pt[1] - prev[1])
            if seg >= left and seg > 1e-9:
                k = left / seg
                return prev[0] + (pt[0] - prev[0]) * k, prev[1] + (pt[1] - prev[1]) * k
            left -= seg
            prev = pt
        return self._path[-1]
