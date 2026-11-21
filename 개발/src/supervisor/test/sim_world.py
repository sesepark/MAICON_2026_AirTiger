"""시험용 가짜 경기장: 상자 배치(면마다 정답 포함)와, 사진을 보고 답하는 가짜 perception.

배치 파일은 가상 경기장(mock)이 만든 것(test/fixtures/layout13.json). 실제 경기장에는 빈 면이 없으므로 blank 는 non_object 로 바꾼다.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from supervisor import arena as A
from supervisor.faces import CameraParams, expected_faces

FIXTURES = Path(__file__).parent / "fixtures"


def load_layout(name="layout13.json"):
    data = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    boxes = [(b["x"], b["y"]) for b in data["boxes"]]
    fmap = A.field_map(boxes, A.arena_from(data["arena"]), data["bases"])
    truth = {}
    for i, b in enumerate(data["boxes"]):
        for side, text in b["faces"].items():
            if text.startswith("object:"):
                kind = text.split(":", 1)[1]
            elif text.startswith("target:"):
                kind = text.split(":", 1)[1].split("_", 1)[1].lower()  # BLUE_OUTPOST → outpost
            else:
                kind = A.NON_OBJECT
            truth[(i, side)] = kind
    return fmap, truth


class FakePerception:
    """참 위치 · 참 카메라로 찍힌 면을 정답대로 답한다. 가장자리 · 많이 가린 면 · 아주 비스듬한 면은 모름이거나 빠진다."""

    def __init__(self, fmap, truth, cam=None, seed=0, x_noise=0.01, wrong=0.0, rich=False):
        self.fmap, self.truth = fmap, truth
        # rich: 실차의 답 모양 (종류, 가로 자리, 잘림, 가로 크기, 세로 크기, 확신) 으로 답한다. 크기 · 확신이 있어야 「본 것」 (scout.pick_votes) 이 생긴다
        self.rich = rich
        self.cam = cam or CameraParams(hfov_deg=128.0)
        self.rng = random.Random(seed)
        self.x_noise = x_noise
        self.wrong = wrong  # 종류를 틀리게 답할 확률

    def look(self, pose, pan_deg=0.0):
        """돌려주는 값: ([(kind, x_center, cut_off)], [그 그림이 실제로 붙은 면 key])."""
        pics, keys = [], []
        for e in expected_faces(self.fmap, pose, pan_deg, self.cam):
            clear = e.fraction >= 0.6 and e.incidence_deg <= 70.0
            if clear:
                kind = self.truth[e.face.key]
                if self.rng.random() < self.wrong:
                    kind = self.rng.choice([k for k in A.OBJECT_KINDS if k != kind])
                cut = e.fraction < 0.95
            else:
                if self.rng.random() < 0.5:
                    continue  # 아예 못 봄
                kind, cut = A.UNKNOWN, True
            x = e.x + self.rng.gauss(0.0, self.x_noise)
            if self.rich:
                conf = 0.4 if kind == A.UNKNOWN else 0.9
                pics.append((kind, x, cut, e.width * self.rng.uniform(0.92, 1.08), e.height * self.rng.uniform(0.92, 1.08), conf))
            else:
                pics.append((kind, x, cut))
            keys.append(e.face.key)
        order = sorted(range(len(pics)), key=lambda k: pics[k][1])
        return [pics[k] for k in order], [keys[k] for k in order]


def free_poses(fmap, n, seed=0, margin=0.25):
    """상자와 벽에서 margin 넘게 떨어진 임의의 자리 · 방향 n 개."""
    rng = random.Random(seed)
    half = fmap.box_size / 2.0 + margin
    out = []
    while len(out) < n:
        x = rng.uniform(margin, fmap.width - margin)
        y = rng.uniform(margin, fmap.height - margin)
        if any(abs(x - bx) < half and abs(y - by) < half for bx, by in fmap.boxes):
            continue
        out.append((x, y, rng.uniform(-math.pi, math.pi)))
    return out


def scout_grid(fmap, radius=0.15, keepout=0.45):
    """정찰에 쓰는 격자: 거점 앞은 막힌 곳으로 둔다 (sim-strategy navigator.py 와 같은 값)."""
    from supervisor.grid import Grid

    grid = Grid(fmap, radius=radius)
    for value, (_x0, y0, _x1, y1) in fmap.bases.items():
        rect = (0.0, y0 - 0.15, keepout, y1 + 0.15) if fmap.base_camp(value) == "BLUE" else (fmap.width - keepout, y0 - 0.15, fmap.width, y1 + 0.15)
        grid.add_zone(value, rect, closed=True)
    return grid


def scout_tour(fmap, truth, camp, start, seed=0, pose_xy=0.03, pose_yaw_deg=2.0, assumed_hfov=120.0, true_hfov=128.0,
               wrong=0.0, looks=1, view=None, max_stops=80):
    """정찰 한 번을 흉내 낸다: 설 자리를 고르고 → 그 자리에서 사진 → 짝짓기 → 면 기록. 주행은 순간 이동으로 친다.

    돌려주는 값: {"sectors_ok": 맞은 구역 수, "faces_wrong": 틀리게 적힌 면 수, "left": 못 본 면 수, "stops", "path_m"}
    """
    from supervisor.coverage import CoveragePlanner, ViewParams
    from supervisor.faces import FaceBook, match
    from supervisor.grid import path_length

    rng = random.Random(seed)
    eyes = FakePerception(fmap, truth, CameraParams(hfov_deg=true_hfov), seed=seed + 7, wrong=wrong)
    cam = CameraParams(hfov_deg=assumed_hfov)
    grid = scout_grid(fmap)
    book = FaceBook(fmap, camp)
    view = view or ViewParams(range_m=1.5, max_incidence_deg=50.0)
    cover = CoveragePlanner(fmap, grid, book.goal, view)
    pose = start
    visited, stops, path_m, rounds = set(), 0, 0.0, 0
    while stops < max_stops:
        remaining = book.remaining() - cover.unreachable
        if not remaining:
            break
        pick = cover.next_view(pose, remaining, banned=visited)
        if pick is None:
            if rounds >= 2:
                break
            rounds += 1
            visited = set()
            continue
        k, vx, vy, heading, _keys = pick
        route = grid.plan(pose[:2], (vx, vy))
        path_m += path_length(pose[:2], route) if route else 0.0
        true_pose = (vx + rng.gauss(0, 0.02), vy + rng.gauss(0, 0.02), heading + math.radians(rng.gauss(0, 2.0)))  # 도착 오차
        for _ in range(looks):
            said = (true_pose[0] + rng.gauss(0, pose_xy), true_pose[1] + rng.gauss(0, pose_xy), true_pose[2] + math.radians(rng.gauss(0, pose_yaw_deg)))
            pics, _keys2 = eyes.look(true_pose)
            book.add_all(match(expected_faces(fmap, said, 0.0, cam), pics))
        visited.add(k)
        stops += 1
        pose = true_pose
    want = {sector: {} for sector in book.scored}
    for f in book.goal:
        kind = truth[f.key]
        if kind in A.OBJECT_KINDS:
            want[f.sector][kind] = want[f.sector].get(kind, 0) + 1
    got = book.counts()
    wrong_faces = sum(1 for f in book.goal if book.kind(f.key) is not None and book.kind(f.key) != truth[f.key])
    return {"sectors_ok": sum(1 for s_ in book.scored if got[s_] == want[s_]), "faces_wrong": wrong_faces,
            "left": len(book.remaining()), "unreachable": len(cover.unreachable), "stops": stops, "path_m": round(path_m, 1)}


PARKED_KEEP = 0.50  # 서 있는 다른 UGV 의 가운데에서 이 안으로는 우리 가운데가 못 들어간다 (차체 두 대의 반 + 정지 거리)


class SimRobot:
    """시험용 로봇: 목표 좌표를 받으면 격자 길을 따라 돌고 · 가고 · 마지막에 방향을 맞춘다 (navigation 대신)."""

    def __init__(self, grid, pose, speed=0.4, turn_rate=1.5, seed=0, stop_xy=0.02, stop_yaw_deg=2.0, stall_yaw_deg=0.0, parked=(), give_up_s=8.0,
                 fmap=None, turn_room_m=0.0):
        self.grid, self.pose = grid, pose
        # turn_room_m: 자리에 와서 끝의 방향을 맞출 때, 차체 가운데에서 벽 · 상자 · 기둥까지가 이보다 가까우면 서서 돌지 못한다 (fmap 이 있어야 한다).
        # 그 방향 그대로 「도착」 으로 끝내고 no_turn 에 그 목표를 적는다 — navigation 의 REACHED + 「돌 자리 없음(도착 방향)」 흉내.
        # 2026-11-19 14:52 실차: 벽에서 0.175 m · 0.225 m 인 정찰 자리 넷에서 그랬다 (차체 반 대각선 0.17 m + 선 자리의 오차)
        self.fmap, self.turn_room_m = fmap, turn_room_m
        self.no_turn = None  # 이번 틱에 「돌 자리 없음(도착 방향)」 으로 끝낸 목표 (x, y) — SimMatch 가 가져간다
        self.no_turns = 0
        # parked: 경기장에 서 있는 다른 UGV 의 자리 [(x, y)]. 그 둘레 PARKED_KEEP 안으로는 못 간다 — 목표가 그 안이면 가장자리까지 가서 서 있다가
        # give_up_s 뒤에 「못 갔다」(BLOCKED) 로 끝낸다 (navigation 의 시한 전 BLOCKED 흉내. 2026-11-18 3차 팀 테스트: 정찰 자리에 다른 팀 로봇)
        self.parked = [(float(px), float(py)) for px, py in parked]
        self.give_up_s = give_up_s
        self.failed = None  # 이번 틱에 못 갔다고 알릴 것 (x, y, 까닭 번호) — SimMatch 가 가져간다
        self.blocked_goals = 0
        self._short, self._waited = False, 0.0
        self._extra = set()
        for px, py in self.parked:
            for i in range(int((px - PARKED_KEEP) / grid.cell), int((px + PARKED_KEEP) / grid.cell) + 1):
                for j in range(int((py - PARKED_KEEP) / grid.cell), int((py + PARKED_KEEP) / grid.cell) + 1):
                    cx, cy = grid.center(i, j)
                    if math.hypot(cx - px, cy - py) <= PARKED_KEEP:
                        self._extra.add((i, j))
        # stall_yaw_deg: 끝의 방향 맞추기에서 남은 각도가 이보다 작으면 더 돌지 못하고 「도착」 도 알리지 않는다
        # (실제 로봇에서 본 일 — 2026-11-18 팀 테스트: 11 ~ 16도를 남기고 3분을 서 있었다)
        self.stall = math.radians(stall_yaw_deg)
        self.speed, self.turn_rate = speed, turn_rate
        self.rng = random.Random(seed)
        self.stop_xy, self.stop_yaw = stop_xy, math.radians(stop_yaw_deg)
        self.goal, self.route, self.moving = None, [], False
        self.path_m = self.turn_rad = 0.0
        self.goals = 0

    def structure_gap(self, x, y):
        """(x, y) 에서 가장 가까운 벽 · 상자 · 고정 기둥의 면까지 (m)."""
        m = self.fmap
        half = m.box_size / 2.0
        gap = min(x, m.width - x, y, m.height - y)
        for bx, by in list(m.boxes) + list(m.pillars):
            gap = min(gap, math.hypot(max(abs(x - bx) - half, 0.0), max(abs(y - by) - half, 0.0)))
        return gap

    def go(self, x, y, yaw, yaw_tol=None, pos_tol=None):
        goal = (round(x, 3), round(y, 3), round(yaw, 3))
        if goal == self.goal:
            return
        if self.goal is None or (goal[0], goal[1]) != (self.goal[0], self.goal[1]):
            self.goals += 1
            if pos_tol is not None and math.hypot(x - self.pose[0], y - self.pose[1]) <= pos_tol:
                # 이미 위치 허용 안에 서 있다 → 몇 cm 더 가려고 그쪽으로 돌지 않고 그 자리에서 방향만 맞춘다
                # (navigation executor 의 「이미 도착 허용 안이다 … 여기서 선다」 와 같다. 선 자리에서 방향만 바꾸라는 목표가 그렇다)
                self.route, self._short = [], False
            else:
                self.route = list(self.grid.plan(self.pose[:2], (x, y), self._extra) or [(x, y)])
                end = self.route[-1]
                self._short = bool(self._extra) and math.hypot(end[0] - x, end[1] - y) > 0.05  # 목표가 서 있는 UGV 둘레 안 → 가장자리까지만 간다
            self._waited = 0.0
        self.goal, self.moving = goal, True
        self.yaw_tol = max(math.radians(4.0), yaw_tol or 0.0)  # 끝의 방향은 이 안이면 더 돌지 않는다 (navigation 의 yaw_tolerance_rad)

    def tick(self, dt):
        if not self.moving or self.goal is None:
            return
        x, y, yaw = self.pose
        while self.route and math.hypot(self.route[0][0] - x, self.route[0][1] - y) < 0.02:
            self.route.pop(0)
        want = math.atan2(self.route[0][1] - y, self.route[0][0] - x) if self.route else self.goal[2]
        err = math.atan2(math.sin(want - yaw), math.cos(want - yaw))
        if abs(err) > (math.radians(4.0) if self.route else getattr(self, "yaw_tol", math.radians(4.0))):
            if not self.route and abs(err) <= self.stall:
                return  # 조금 남은 각도를 돌지 못한다. 목표는 그대로 살아 있다 (도착 아님)
            if not self.route and self.turn_room_m > 0.0 and self.fmap is not None and self.structure_gap(x, y) < self.turn_room_m:
                self.no_turn = (round(self.goal[0], 2), round(self.goal[1], 2))  # 돌 자리가 없다 → 방향은 그대로 두고 도착으로 끝낸다
                self.no_turns += 1
                self.moving = False
                return
            step = max(-self.turn_rate * dt, min(self.turn_rate * dt, err))
            self.turn_rad += abs(step)
            self.pose = (x, y, yaw + step)
            return
        if self.route:
            wx, wy = self.route[0]
            seg = math.hypot(wx - x, wy - y)
            step = min(self.speed * dt, seg)
            self.path_m += step
            self.pose = (x + math.cos(want) * step, y + math.sin(want) * step, want)
            return
        if self._short:  # 서 있는 UGV 앞에서 기다리다 포기한다
            self._waited += dt
            if self._waited >= self.give_up_s:
                self.failed = (round(self.goal[0], 2), round(self.goal[1], 2), 3)
                self.blocked_goals += 1
                self.goal, self.moving, self._short = None, False, False
            return
        # 도착: 멈춘 자리에는 조금 오차가 있다
        self.pose = (x + self.rng.gauss(0, self.stop_xy), y + self.rng.gauss(0, self.stop_xy), yaw + self.rng.gauss(0, self.stop_yaw))
        self.moving = False


def run_scout(mode, camp="BLUE", start=(0.2, 2.5, 0.0), seed=0, pose_xy=0.03, pose_yaw_deg=2.0, assumed_hfov=120.0, true_hfov=128.0,
              latency_s=1.0, limit_s=400.0, dt=0.1, params=None, layout="layout13.json"):
    """정찰 한 번을 시간에 따라 돌린다. 주행은 SimRobot, 인식은 FakePerception(답이 latency_s 뒤에 온다).

    돌려주는 값: {"sectors_ok", "faces_wrong", "left", "secs", "goals", "shots", "path_m", "turn_deg", "log"}
    """
    from supervisor.faces import FaceBook
    from supervisor.scout import ScoutParams, make_scout

    fmap, truth = load_layout(layout)
    rng = random.Random(seed + 11)
    eyes = FakePerception(fmap, truth, CameraParams(hfov_deg=true_hfov), seed=seed + 7)
    grid = scout_grid(fmap)
    book = FaceBook(fmap, camp)
    params = params or ScoutParams(mode=mode)
    scout = make_scout(fmap, grid, book, CameraParams(hfov_deg=assumed_hfov), params, start=start)
    robot = SimRobot(grid, start, seed=seed)
    mail = []  # (도착 시각, tag, 그때 알고 있던 위치, 팬, 답)
    now, done_at = 0.0, None

    def said(pose):
        return (pose[0] + rng.gauss(0, pose_xy), pose[1] + rng.gauss(0, pose_xy), pose[2] + math.radians(rng.gauss(0, pose_yaw_deg)))

    while now < limit_s:
        for item in [m for m in mail if m[0] <= now]:
            mail.remove(item)
            scout.on_answer(item[1], item[2], item[3], item[4])
        act = scout.step(said(robot.pose), now)
        if act[0] == "done":
            done_at = now
            break
        if act[0] == "go":
            robot.go(act[1], act[2], act[3], yaw_tol=None if len(act) < 5 else max(0.10, act[4] - math.radians(4.0)))
        elif act[0] == "shoot" and not robot.moving:
            pics, _keys = eyes.look(robot.pose, act[1])
            mail.append((now + latency_s, act[2], said(robot.pose), act[1], pics))
            scout.shot_taken(act[2], now)
        robot.tick(dt)
        now += dt
    want = {s: {} for s in book.scored}
    for f in book.goal:
        kind = truth[f.key]
        if kind in A.OBJECT_KINDS:
            want[f.sector][kind] = want[f.sector].get(kind, 0) + 1
    got = book.counts()
    return {"sectors_ok": sum(1 for s in book.scored if got[s] == want[s]),
            "faces_wrong": sum(1 for f in book.goal if book.kind(f.key) not in (None, truth[f.key])),
            "left": len(book.remaining()), "secs": round(done_at if done_at is not None else now, 1), "finished": done_at is not None,
            "goals": robot.goals, "shots": scout.shots, "rejected": scout.rejected, "path_m": round(robot.path_m, 1),
            "turn_deg": round(math.degrees(robot.turn_rad)), "log": scout.log}


class SimPorts:
    """Mission 이 시키는 것을 가상 세계에서 받아 준다 (navigation · perception · server_bridge · llm_worker 대신)."""

    def __init__(self, world):
        self.w = world

    def go(self, x, y, yaw, yaw_free=False, mode="normal", yaw_tol=None, pos_tol=None):
        self.w.robot.go(x, y, yaw, yaw_tol=math.pi if yaw_free else yaw_tol, pos_tol=pos_tol)

    def stop(self):
        self.w.robot.goal, self.w.robot.moving, self.w.robot.route = None, False, []

    def aim(self, pan_deg, tilt_deg):
        self.w.gimbal = (pan_deg, tilt_deg)
        self.w.gimbal_at = self.w.now
        return True

    def pan_now(self):
        return self.w.gimbal[0]

    def aim_ready(self):
        return self.w.now >= self.w.gimbal_at + self.w.gimbal_s

    def fire(self):
        w = self.w
        w.fires.append((w.now, w.robot.pose, w.gimbal))
        w.beam_off.append(w.beam_miss())  # 레이저가 수광부 가운데에서 벗어난 거리 (m). 표적 면을 모르면 None
        w.later(0.1, "fire", ("fired", 10000))
        if getattr(w, "watching", None) is not None:
            w.later(1.5, "spot", (w.watching, w.spot_pixel() if w.spot_seen else None))
            w.watching = None
        return True

    def watch_laser(self, tag, near=None):
        """레이저 점 확인 흉내: 다음 발사 1.5초 뒤, 그때의 레이저 점이 표적 면에 찍힌 픽셀을 답한다 (w.spot_seen 이 False 면 못 찾은 것으로)."""
        if not self.w.can_watch:
            return False
        self.w.watching = tag
        return True

    def shoot_aim(self, tag):
        """조준 확인 사진: 지금 팬틸트 그대로 찍는다. 답은 {"pictures", "sensors"} (실제 node.py 가 넘기는 모양)."""
        w = self.w
        if w.robot.moving or not self.aim_ready():
            return False
        w.aim_photos.append((w.now, tag))
        w.later(w.latency_s, "answer", (tag, w.said(), w.gimbal[0], w.aim_view()))
        return True

    def shoot(self, tag, pan_deg):
        w = self.w
        if w.robot.moving:
            return False
        w.gimbal = (pan_deg, w.gimbal[1])  # 실제 창구처럼: 찍을 때 팬을 그 각도로 맞춘다
        pics, _keys = w.eyes.look(w.robot.pose, pan_deg)
        w.later(w.latency_s, "answer", (tag, w.said(), pan_deg, pics))
        return True

    def shoot_pass(self, tag):
        """지나가며 찍기 (SimMatch(pass_shots=True) 일 때만 — 아니면 창구가 없는 것과 같다). 팬틸트는 건드리지 않는다.

        찍은 순간의 위치 값은 서 있을 때보다 크게 틀린 것으로 친다 (pass_noise = (m, °) — 움직일 때 위치 값이 늦고 흔들리는 몫).
        """
        w = self.w
        if not w.pass_shots or not self.aim_ready():
            return False
        pics, _keys = w.eyes.look(w.robot.pose, w.gimbal[0])
        x, y, yaw = w.said()
        xy, deg = w.pass_noise
        pose = (x + w.rng.gauss(0, xy), y + w.rng.gauss(0, xy), yaw + math.radians(w.rng.gauss(0, deg)))
        w.pass_photos += 1
        w.later(w.latency_s, "answer", (tag, pose, w.gimbal[0], pics))
        return True

    def look_ugvs(self):
        self.w.later(self.w.latency_s, "ugvs", list(self.w.ugvs_in_view))
        return True

    def analyze_order(self, text):
        self.w.later(2.0, "parsed", dict(self.w.order_truth))
        return True

    def set_ready(self):
        self.w.later(0.2, "result", ("ready", True))
        return True

    def report(self, kind, payload):
        w = self.w
        w.reports.append((round(w.t, 1) if w.t is not None else None, kind, payload))
        w.later(0.2, "result", (kind, True))
        return True


class SimMatch:
    """가상 경기 한 판: 대기 → 시작 → 5분 → 보고. Mission 을 0.1초마다 돌린다."""

    def __init__(self, mode="place", slot="B-1", seed=0, pose_xy=0.03, pose_yaw_deg=2.0, assumed_hfov=120.0, true_hfov=128.0,
                 latency_s=1.0, layout="layout13.json", objective_row=None, stall_yaw_deg=0.0, parked=(), sensor_row=None,
                 receiver_h=0.32, pan_scale=1.0, tilt_scale=1.0, sensor_rate=1.0, aim_pictures=True, aim_answers=True, aim_px_noise=0.7,
                 unmapped=(), lidar_from_s=0.0, params=None, can_watch=True, spot_seen=True, yaw_bias_deg=0.0, turn_room_m=0.0, scout=None,
                 rich=False, pass_shots=False, pass_noise=(0.03, 3.0), wrong=0.0):
        from supervisor.mission import Mission, MissionParams
        from supervisor.scout import ScoutParams

        self.fmap, self.truth = load_layout(layout)
        data = json.loads((FIXTURES / layout).read_text(encoding="utf-8"))
        # unmapped: 경기장에는 있는데 지도(드론 사진)에는 없는 상자 [(x, y, {면: 종류})]. 경기장(self.fmap)의 상자 목록 맨 뒤에 붙인다 — 지도의 상자 번호는 그대로다.
        # 가짜 주행은 그 상자를 비켜 가고, 가짜 인식은 그 면도 답한다. mission 에는 앞의 self.mapped 개만 지도로 주고, 지도에 없는 상자는
        # navigation 이 라이다로 찾은 것으로 쳐서 경기 lidar_from_s 초부터 Facts.extra_boxes 로 알려 준다 (self.lidar_boxes 를 바꾸면 다른 목록을 줄 수 있다).
        # unmapped 가 없으면 Facts.extra_boxes 는 None 이다 (전과 같다)
        self.mapped = len(self.fmap.boxes)
        if unmapped:
            self.fmap = A.field_map(list(self.fmap.boxes) + [(u[0], u[1]) for u in unmapped], A.arena_from(data["arena"]), data["bases"])
            for k, u in enumerate(unmapped):
                for side in "ENWS":
                    self.truth[(self.mapped + k, side)] = (u[2] if len(u) > 2 else {}).get(side, A.NON_OBJECT)
        self.lidar_boxes = (lambda t: [(u[0], u[1]) for u in unmapped] if t is not None and t >= lidar_from_s else []) if unmapped else None
        self.slot, self.camp = slot, A.camp_of(slot)
        self.rng = random.Random(seed + 11)
        self.pose_xy, self.pose_yaw = pose_xy, math.radians(pose_yaw_deg)
        self.latency_s = latency_s
        self.eyes = FakePerception(self.fmap, self.truth, CameraParams(hfov_deg=true_hfov), seed=seed + 7, rich=rich, wrong=wrong)
        # pass_shots: 지나가며 찍기 창구를 준다 (기본은 없음 — 전의 시험과 같다). pass_noise: 그 사진의 위치 값에 더 얹는 오차 (m, °)
        self.pass_shots, self.pass_noise, self.pass_photos = pass_shots, pass_noise, 0
        grid = scout_grid(self.fmap)
        x0, y0, x1, y1 = self.fmap.bases[self.fmap.base_in_row(self.camp, A.row_of(slot))]
        start = ((x0 + x1) / 2.0, (y0 + y1) / 2.0, 0.0 if self.camp == "BLUE" else math.pi)
        for name in list(grid.zones):
            grid.set_zone(name, False)  # 가짜 주행은 거점도 지나다닌다 (출발 · 진입)
        # turn_room_m: 벽 · 상자에 이보다 가까운 자리에서는 끝의 방향을 맞추지 못하는 로봇 (SimRobot). scout: ScoutParams 의 값을 바꿀 때 (예 {"roomy_first": False})
        self.robot = SimRobot(grid, start, seed=seed, stall_yaw_deg=stall_yaw_deg, parked=parked, fmap=self.fmap, turn_room_m=turn_room_m)
        # 가상 경기장(시험용 배치)의 치수를 mission 에도 준다 — 기본값(포털의 실제 치수)과 이 배치의 치수는 다르다
        # params: MissionParams 의 값을 바꿀 때 (예 {"extra_boxes": False})
        self.mission = Mission(MissionParams(scout=ScoutParams(mode=mode, **(scout or {})), **(params or {})), arena=A.arena_from(data["arena"]), base_kinds=data["bases"], cam=CameraParams(hfov_deg=assumed_hfov))
        self.ports = SimPorts(self)
        foe = A.opponent(self.camp)
        kind = next(k for (key, k) in sorted(self.truth.items()) if k in A.TARGET_KINDS and A.side_of(self.fmap, next(f for f in self.fmap.faces if f.key == key).cx) == foe)
        self.order_truth = {"target": "%s_%s" % (foe, kind.upper()), "search_sectors": ["CM"], "base": self.fmap.base_in_row(foe, 2)}
        self.objective = self.fmap.base_in_row(foe, objective_row or A.row_of(slot))
        # 점령: 게시(4:00) 뒤 목표 거점의 센서가 처음 가려지는 순간 경기가 끝난다 (보고 단계로). sensor_row 를 주면 「서버가 말한 목표」 의 센서가
        # 실제로는 그 줄의 거점에 있다 — 거점 종류의 배치가 지도(연습 포털의 배치)와 다른 경기 (2026-11-19 공지)
        self.sensor_base = self.fmap.base_in_row(foe, sensor_row) if sensor_row else self.objective
        self.captured_at = None
        self.entered = []  # (경기 시각, 거점) — 게시 뒤 로봇이 깊숙이 들어간 거점들
        self.ugvs_in_view = []
        self.gimbal = (0.0, 0.0)
        self.fires, self.reports, self.mail, self.tasks = [], [], [], []
        # ---- 사진 조준 흉내의 참값 (mission 은 모른다)
        self.receiver_h = receiver_h  # 수광부 가운데의 높이 (m)
        self.pan_scale, self.tilt_scale = pan_scale, tilt_scale  # 보낸 각도 1도에 실제로 도는 각도 (음수 = 방향이 반대)
        self.sensor_rate = sensor_rate  # 조준 사진에서 수광부가 탐지될 확률
        self.aim_pictures = aim_pictures  # 조준 사진의 답에 표적 그림 상자가 있는가
        self.aim_answers = aim_answers  # False 면 조준 사진의 판독이 실패한다 (답 None)
        self.aim_px_noise = aim_px_noise  # 탐지 자리의 떨림 (픽셀)
        self.aim_ppd = 640.0 / true_hfov  # 1도에 픽셀
        self.gimbal_at, self.gimbal_s = -1e9, 0.4  # 팬틸트를 보낸 때 · 도는 데 걸리는 시간
        self.aim_photos, self.beam_off = [], []
        self.can_watch, self.spot_seen, self.watching = can_watch, spot_seen, None  # 레이저 점 확인 창구가 있는가 · 켜진 사진에서 레이저 점이 보이는가
        self.yaw_bias = math.radians(yaw_bias_deg)
        self.now, self.t = 0.0, None

    def later(self, delay, kind, value):
        self.mail.append((self.now + delay, kind, value))

    def said(self):
        x, y, yaw = self.robot.pose
        # yaw_bias: 위치 값의 방향이 늘 이만큼 틀리다 (rad) — 계산한 각도로 겨눈 사진에 표적이 안 들어오는 경우를 만든다
        return (x + self.rng.gauss(0, self.pose_xy), y + self.rng.gauss(0, self.pose_xy), yaw + self.yaw_bias + self.rng.gauss(0, self.pose_yaw))

    def run(self, standby_s=3.0, running_s=300.0, report_s=5.0, dt=0.1):
        from supervisor.mission import Facts

        end = standby_s + running_s + report_s
        parsed = None
        while self.now < end:
            if self.now < standby_s:
                phase, self.t = "standby", None
            elif self.now < standby_s + running_s and self.captured_at is None:
                phase, self.t = "running", self.now - standby_s
            else:
                phase, self.t = "report", (self.captured_at if self.captured_at is not None else running_s)
                if self.captured_at is not None and self.now > standby_s + self.captured_at + report_s:
                    break
            due = [m for m in self.mail if m[0] <= self.now]
            self.mail = [m for m in self.mail if m[0] > self.now]
            f = Facts(now=self.now, t=self.t, phase=phase, control=phase == "running", slot=self.slot, pose=self.said(), arrived=not self.robot.moving,
                      boxes=list(self.fmap.boxes[:self.mapped]) if self.t is not None and self.t >= 0.5 else None,
                      extra_boxes=self.lidar_boxes(self.t) if self.lidar_boxes is not None else None,
                      order_text="가상 명령문" if phase == "running" else None, parsed=parsed,
                      objective=self.objective if self.t is not None and self.t >= A.TARGET_AT_S else None, hp=4,
                      go_failed=self.robot.failed, no_turn=self.robot.no_turn)
            self.robot.failed = self.robot.no_turn = None
            for _at, kind, value in due:
                if kind == "answer":
                    f.answers.append(value)
                elif kind == "ugvs":
                    f.ugv_answers += value
                elif kind == "parsed":
                    parsed = f.parsed = value
                elif kind == "spot":
                    f.laser_spots.append(value)
                elif kind == "fire":
                    f.fire_reply = value
                elif kind == "result":
                    f.results.append(value)
            task = self.mission.tick(f, self.ports)
            if not self.tasks or self.tasks[-1][1] != task.split(" ")[0]:
                self.tasks.append((round(self.t, 1) if self.t is not None else None, task.split(" ")[0]))
            if phase == "running":
                self.robot.tick(dt)
                if self.t >= A.TARGET_AT_S:
                    x, y, _yaw = self.robot.pose
                    for value in self.fmap.bases_of(A.opponent(self.camp)):
                        x0, y0, x1, y1 = self.fmap.bases[value]
                        wall = 0.0 if self.fmap.inward(value) > 0 else self.fmap.width
                        if y0 <= y <= y1 and abs(x - wall) <= 0.30:  # 차체가 센서 빔(벽에서 0.20 m [추정])을 가릴 만큼 깊이
                            if not self.entered or self.entered[-1][1] != value:
                                self.entered.append((round(self.t, 1), value))
                            if value == self.sensor_base and self.captured_at is None:
                                self.captured_at = self.t
            self.now += dt
        return self

    # ---- 사진 조준 흉내: 카메라 · 레이저의 참 모형. 잰 값 둘에 맞춘 것이다.
    #   · 레이저 점은 렌즈 기준 식(측정 기록의 둘째 식 u = 332.4 + 1.1/Zc, v = 250.9 − 15.9/Zc)대로 찍힌다 —
    #     mission 은 첫째 식(Z 기준)과 제 위치 값으로 계산하므로 둘은 조금 다르다 (실제와 같은 종류의 어긋남)
    #   · 틸트 0 에서 카메라는 수평을 보고, 렌즈 앞 1.07 m 의 수광부(높이 0.32 m)가 사진 가운데보다 30픽셀 위에 찍힌다 (2026-11-18 측정) → 렌즈 높이 0.21 m.
    #     레이저는 카메라가 보는 방향보다 (250.9 − 240)픽셀만큼 아래를 향한다
    #   가로는 팬 방향 = 레이저 방향으로 둔다 (사진 가운데와 12픽셀 어긋난 것이 렌즈 중심 때문인지 레이저 방향 때문인지는 재지 않았다)
    CAM_H, LENS_FORWARD = 0.21, 0.05  # 렌즈의 높이 · 차체 가운데보다 앞
    LASER_UP, LASER_RIGHT = 0.0553, 0.0038  # 발광부가 렌즈보다 위 · 오른쪽에 있는 거리
    BEAM_U, BEAM_V = 332.4, 250.9  # 아주 먼 곳의 레이저 점이 찍히는 픽셀 (= 레이저가 향하는 방향)
    DOME_M = 0.05  # 수광 돔의 지름

    def _head(self):
        """렌즈 자리 (x, y, z) · 카메라가 보는 방향(rad) · 올려본 각(rad) — 참값."""
        x, y, yaw = self.robot.pose
        heading = yaw - math.radians(self.gimbal[0] * self.pan_scale)
        return (x + self.LENS_FORWARD * math.cos(heading), y + self.LENS_FORWARD * math.sin(heading), self.CAM_H), heading, math.radians(self.gimbal[1] * self.tilt_scale)

    def _pixel(self, point):
        """경기장의 점 (x, y, z) 이 지금 사진에 찍히는 픽셀 (각도에 비례하는 렌즈)."""
        (lx, ly, lz), heading, up = self._head()
        dx, dy, dz = point[0] - lx, point[1] - ly, point[2] - lz
        right = -math.degrees(math.atan2(math.sin(math.atan2(dy, dx) - heading), math.cos(math.atan2(dy, dx) - heading)))
        above = math.degrees(math.atan2(dz, math.hypot(dx, dy)) - up)
        return self.BEAM_U + right * self.aim_ppd, 240.0 - above * self.aim_ppd

    def receiver_point(self):
        """mission 이 쏘려는 면의 수광부 자리 (x, y, z). 그 면이 실제로 표적이 아니면 None."""
        face = self.mission.target_face
        if face is None or self.truth.get(face.key) not in A.TARGET_KINDS:
            return None
        return face.cx, face.cy, self.receiver_h

    def beam_miss(self):
        """지금 쏘면 레이저 선이 수광부 가운데에서 얼마나 떨어져 지나가는가 (m)."""
        target = self.receiver_point()
        if target is None:
            return None
        (lx, ly, lz), heading, up = self._head()
        up -= math.radians((self.BEAM_V - 240.0) / self.aim_ppd)  # 레이저는 카메라가 보는 방향보다 조금 아래를 향한다
        ex, ey, ez = lx + self.LASER_RIGHT * math.sin(heading), ly - self.LASER_RIGHT * math.cos(heading), lz + self.LASER_UP
        d = (math.cos(heading) * math.cos(up), math.sin(heading) * math.cos(up), math.sin(up))
        r = (target[0] - ex, target[1] - ey, target[2] - ez)
        along = sum(r[k] * d[k] for k in range(3))
        return math.sqrt(max(0.0, sum(v * v for v in r) - along * along))

    def spot_pixel(self):
        """지금 쏘면 레이저 점이 표적 면에 찍히는 자리의 픽셀 {"u", "v"} (참값). 표적 면을 모르면 None."""
        target = self.receiver_point()
        if target is None:
            return None
        (lx, ly, lz), heading, up = self._head()
        beam_up = up - math.radians((self.BEAM_V - 240.0) / self.aim_ppd)
        ex, ey, ez = lx + self.LASER_RIGHT * math.sin(heading), ly - self.LASER_RIGHT * math.cos(heading), lz + self.LASER_UP
        d = (math.cos(heading) * math.cos(beam_up), math.sin(heading) * math.cos(beam_up), math.sin(beam_up))
        along = sum((target[k] - (ex, ey, ez)[k]) * d[k] for k in range(3))
        u, v = self._pixel((ex + d[0] * along, ey + d[1] * along, ez + d[2] * along))
        return {"u": u + self.rng.gauss(0.0, 0.5), "v": v + self.rng.gauss(0.0, 0.5), "peak": 80.0, "area": 30}

    def aim_view(self):
        """지금 팬틸트로 찍은 사진의 판독 답: 표적 그림 상자 하나와 그 위쪽의 수광부. 자리 · 크기는 사진에 대한 비율."""
        if not self.aim_answers:
            return None
        face = self.mission.target_face
        pictures, sensors = [], []
        if face is not None and self.aim_pictures:
            half = self.fmap.box_size / 2.0
            corners = [self._pixel((px, py, z)) for px, py in (face.point(-half), face.point(half)) for z in (0.0, self.fmap.box_size)]
            u0, u1 = min(c[0] for c in corners), max(c[0] for c in corners)
            v0, v1 = min(c[1] for c in corners), max(c[1] for c in corners)
            cut = u0 <= 8 or v0 <= 8 or u1 >= 632 or v1 >= 472
            u0, u1, v0, v1 = max(0.0, u0), min(640.0, u1), max(0.0, v0), min(480.0, v1)
            if u1 > u0 and v1 > v0:
                pictures.append({"kind": self.truth[face.key], "x": (u0 + u1) / 1280.0, "y": (v0 + v1) / 960.0, "w": (u1 - u0) / 640.0, "h": (v1 - v0) / 480.0, "conf": 0.9, "cut": cut})
        target = self.receiver_point()
        if target is not None and self.rng.random() < self.sensor_rate:
            u, v = self._pixel(target)
            lens = self._head()[0]
            size = math.degrees(self.DOME_M / max(0.2, math.hypot(target[0] - lens[0], target[1] - lens[1]))) * self.aim_ppd
            u, v = u + self.rng.gauss(0.0, self.aim_px_noise), v + self.rng.gauss(0.0, self.aim_px_noise)
            if 0.0 <= u <= 640.0 and 0.0 <= v <= 480.0:
                sensors.append({"x": u / 640.0, "y": v / 480.0, "w": size / 640.0, "h": size / 480.0, "conf": 0.85})
        return {"pictures": pictures, "sensors": sensors}

    def m3_truth(self):
        want = {s: {} for s in A.scored_sectors(self.camp)}
        for f in self.fmap.faces:
            kind = self.truth[f.key]
            if f.sector in want and kind in A.OBJECT_KINDS:
                want[f.sector][kind] = want[f.sector].get(kind, 0) + 1
        return want
