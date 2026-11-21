"""SLAM 위치(/team/navigation/pose)를 받아 제어에 쓸 「지금 자세」 를 만들고, 위치 상태 단계를 판정한다.

사이 메우기 (D4 ①)
  /odom 위치 차이의 크기(이동 거리)와 방향 변화(기본: /odom/odom_raw 바퀴 거리 차이, 선택: /imu/data yaw)로 「로컬 궤적」 을 계속 적분한다.
  /odom 의 orientation 은 상수라 쓰지 않는다 [실측 2026-11-18]. 위치 x, y 는 내부 방향으로 적분돼 거리는 맞다.
  SLAM 위치가 오면 그 위치가 맞는 시각(header.stamp)의 로컬 궤적과 비교해 보정 변환 T 를 구한다.
  지금 자세 = T ⊕ 지금 로컬 궤적. 그래서 SLAM 이 늦게 와도(지연) 그 사이 움직임이 빠지지 않는다.

위치 상태 단계 (D7)
  OK        → 정상
  DEGRADED  → 마지막 위치가 pose_slow_s 넘게 오래됨 → 속도 제한
  HOLD      → pose_hold_s 넘게 오래됨 → 정지하고 기다림 (목표는 유지)
  LOST      → HOLD 가 pose_lost_s 이어짐 → 목표를 LOCALIZATION_LOST 로 끝낸다
  돌아올 때는 제때(pose_slow_s 안) 온 위치가 pose_recover_count 번 이어져야 OK (경계에서 왔다 갔다 하지 않게).
"""
from __future__ import annotations

import math
from collections import deque

from .geometry import Pose2D, compose, relative, wrap

OK, DEGRADED, HOLD, LOST = 0, 1, 2, 3
LEVEL_NAMES = {OK: "ok", DEGRADED: "degraded", HOLD: "hold", LOST: "lost"}


class PoseTracker:
    def __init__(self, params):
        self.p = params
        self._local = Pose2D(0.0, 0.0, 0.0)  # 로컬 궤적 (원점 아무 데나)
        self._history = deque(maxlen=400)  # (t, Pose2D 로컬) — 약 20 초 (20 Hz)
        self._history.append((-1e9, self._local))
        self._last_odom = None  # (x, y)
        self._last_imu = None  # yaw
        self._last_wheels = None  # (왼쪽, 오른쪽) 누적 거리
        self._correction = None  # 로컬 → arena
        self.last_pose_recv = None  # 마지막 SLAM 위치를 받은 시각 (노드 시계)
        self.last_pose_stamp = None
        self.level = LOST
        self._hold_since = None
        self._good_streak = 0
        self.jumped = False  # 최근 위치가 크게 튐 (부르는 쪽이 경로를 다시 짜고 지운다)
        self.motion_sign = 1.0  # 마지막으로 움직이라고 한 방향 (/odom 거리에는 부호가 없다)
        # 바퀴로 잰 방향 변화에 곱하는 값 (params.yaw_scale_learn 의 설명). SLAM 위치가 올 때마다 그 사이 SLAM 이 본 회전과 견줘 맞춘다
        self.yaw_scale = 1.0
        self.yaw_scale_updates = 0
        self._raw_yaw = 0.0  # 바퀴로 잰 방향 변화를 곱하지 않고 이어 더한 것 (rad)
        self._raw_history = deque(maxlen=400)  # (t, _raw_yaw)
        self._scale_ref = None  # 앞 SLAM 위치: (stamp, 그때의 _raw_yaw, 방향, x, y)
        self._scale_raw = 0.0  # 이번 창에 쌓인 바퀴 회전의 크기 (rad)
        self._scale_seen = 0.0  # 그동안 SLAM 이 본 회전 (바퀴가 돈 쪽을 + 로)

    # ------------------------------------------------------------------ 입력
    def on_command(self, v):
        if abs(v) > 1e-3:
            self.motion_sign = 1.0 if v > 0 else -1.0

    def on_odom(self, t, x, y):
        if not self.p.use_odom:
            return
        if self._last_odom is None:
            self._last_odom = (x, y)
            return
        dist = math.hypot(x - self._last_odom[0], y - self._last_odom[1])
        self._last_odom = (x, y)
        if dist > 0.5:  # 0.1 초에 50 cm 는 오류 (노드 재시작 등)
            return
        self._integrate(t, self.motion_sign * dist, 0.0)

    def on_wheels(self, t, left, right):
        """/odom/odom_raw: 좌·우 바퀴 누적 이동 거리(m). 방향 변화 = (오른쪽 − 왼쪽) / 바퀴 간격 (좌회전이 +).

        로봇 odom 노드(base_node.cpp)가 /odom x·y 를 적분하는 내부 방향과 같은 식이다.
        """
        if not self.p.use_imu or self.p.yaw_source != "wheel":
            return
        if not (math.isfinite(left) and math.isfinite(right)):
            return
        last = self._last_wheels
        self._last_wheels = (left, right)
        if last is None:
            return
        dl, dr = left - last[0], right - last[1]
        if abs(dl) > 1.0 or abs(dr) > 1.0:  # 한 번에 1 m 넘게 바뀜 = 엔코더 초기화 등. 이 한 번은 버린다
            return
        raw = (dr - dl) / self.p.wheel_base_m
        self._raw_yaw += raw
        self._raw_history.append((t, self._raw_yaw))
        self._integrate(t, 0.0, raw * self.yaw_scale)

    def on_imu(self, t, yaw):
        if not self.p.use_imu or self.p.yaw_source != "imu":
            return
        if self._last_imu is None:
            self._last_imu = yaw
            return
        dyaw = wrap(yaw - self._last_imu)
        self._last_imu = yaw
        self._integrate(t, 0.0, dyaw)

    def _integrate(self, t, forward, dyaw):
        l = self._local
        mid = l.yaw + dyaw / 2.0
        self._local = Pose2D(l.x + forward * math.cos(mid), l.y + forward * math.sin(mid), wrap(l.yaw + dyaw))
        self._history.append((t, self._local))

    def _local_at(self, t):
        """시각 t 의 로컬 궤적 (가장 가까운 앞뒤 기록 사이를 보간)."""
        hist = self._history
        if t >= hist[-1][0]:
            return hist[-1][1]
        prev = hist[0]
        for item in hist:
            if item[0] >= t:
                t0, a = prev
                t1, b = item
                if t1 - t0 <= 1e-9:
                    return b
                k = (t - t0) / (t1 - t0)
                return Pose2D(a.x + (b.x - a.x) * k, a.y + (b.y - a.y) * k, wrap(a.yaw + wrap(b.yaw - a.yaw) * k))
            prev = item
        return hist[-1][1]

    def _raw_at(self, t):
        """시각 t 까지 바퀴로 잰 방향 변화의 합 (곱하지 않은 것). 기록이 없으면 None."""
        if not self._raw_history:
            return None
        value = self._raw_history[0][1] if t < self._raw_history[0][0] else None
        if value is not None:
            return value
        for when, raw in reversed(self._raw_history):
            if when <= t:
                return raw
        return self._raw_history[0][1]

    def _learn_yaw_scale(self, stamp, pose):
        """SLAM 위치 둘 사이에 바퀴가 잰 회전과 SLAM 이 본 회전을 쌓아, 바퀴 회전이 yaw_scale_window 만큼 모이면 그 비로 yaw_scale 을 고친다."""
        p = self.p
        if not getattr(p, "yaw_scale_learn", False) or not p.use_imu or p.yaw_source != "wheel":
            return
        raw = self._raw_at(stamp)
        ref, self._scale_ref = self._scale_ref, (stamp, raw, pose.yaw, pose.x, pose.y)
        if ref is None or raw is None or ref[1] is None:
            return
        dt = stamp - ref[0]
        if not (0.0 < dt <= 0.5) or math.hypot(pose.x - ref[3], pose.y - ref[4]) > p.pose_jump_m:
            self._scale_raw = self._scale_seen = 0.0  # 위치가 끊겼거나 뛰었다 — 그 사이는 견주지 않는다
            return
        d_raw = raw - ref[1]
        if abs(d_raw) < 1e-6:
            return
        self._scale_raw += abs(d_raw)
        self._scale_seen += wrap(pose.yaw - ref[2]) * math.copysign(1.0, d_raw)
        if self._scale_raw < p.yaw_scale_window_rad:
            return
        ratio = self._scale_seen / self._scale_raw
        self._scale_raw = self._scale_seen = 0.0
        if 0.2 <= ratio <= 1.5:  # 그 밖은 SLAM 이 그 사이에 틀렸다고 본다
            a = p.yaw_scale_gain
            self.yaw_scale = max(p.yaw_scale_min, min(p.yaw_scale_max, (1.0 - a) * self.yaw_scale + a * ratio))
            self.yaw_scale_updates += 1

    def on_pose(self, t_recv, stamp, pose: Pose2D):
        """SLAM 위치. stamp = 그 위치가 맞는 시각 (노드 시계와 같은 기준이어야 한다)."""
        if stamp is None or stamp > t_recv + 0.5 or stamp < t_recv - 5.0:
            stamp = t_recv  # 시계가 어긋난 stamp 는 받은 시각으로 본다
        self._learn_yaw_scale(stamp, pose)
        before = self.estimate()
        local_then = self._local_at(stamp)
        self._correction = compose(pose, relative(local_then, Pose2D(0.0, 0.0, 0.0)))
        after = self.estimate()
        if before is not None and math.hypot(after.x - before.x, after.y - before.y) > self.p.pose_jump_m:
            self.jumped = True
        on_time = self.last_pose_recv is not None and (t_recv - self.last_pose_recv) <= self.p.pose_slow_s
        self._good_streak = self._good_streak + 1 if on_time else 1
        self.last_pose_recv = t_recv
        self.last_pose_stamp = stamp

    # ------------------------------------------------------------------ 출력
    def local(self):
        """로컬 궤적의 지금 자세 (원점 아무 데나). SLAM 위치가 뛰어도 이어진다 — 지나온 길을 적는 데 쓴다 (drive/trail.py)."""
        return self._local

    def estimate_at(self, t):
        """시각 t 의 자세 (arena) — 로컬 궤적의 그때 기록에 지금의 보정을 씌운 것. SLAM 위치를 한 번도 못 받았으면 None."""
        if self._correction is None:
            return None
        return compose(self._correction, self._local_at(t))

    def sweep(self, t_start, t_end):
        """t_start 때의 차체 자세를 t_end 때의 차체에서 본 것 (dx, dy) — 스캔 한 바퀴 동안 간 만큼 (obstacles.scan_to_body 의 sweep)."""
        rel = relative(self._local_at(t_end), self._local_at(t_start))
        return rel.x, rel.y

    def yaw_rate(self, t, window=0.3):
        """시각 t 앞 window 동안 돈 빠르기 (rad/s, 왼쪽 +). 바퀴 거리는 1 cm 단위라 방향이 3.3° 씩 뛴다 — 그래서 창을 두고 고르게 본다."""
        a, b = self._local_at(t - window), self._local_at(t)
        return wrap(b.yaw - a.yaw) / window

    def estimate(self):
        """지금 자세 (arena). SLAM 위치를 한 번도 못 받았으면 None."""
        if self._correction is None:
            return None
        return compose(self._correction, self._local)

    def update_level(self, t):
        """위치 상태 단계를 다시 매기고 돌려준다. 제어 주기마다 부른다."""
        p = self.p
        if self.last_pose_recv is None:
            self.level = LOST
            return self.level
        age = t - self.last_pose_recv
        if age > p.pose_hold_s:
            if self._hold_since is None:
                self._hold_since = t
            self._good_streak = 0
            self.level = LOST if t - self._hold_since >= p.pose_lost_s else HOLD
        elif age > p.pose_slow_s:
            self._hold_since = None
            self._good_streak = 0
            self.level = max(DEGRADED, self.level) if self.level != LOST else HOLD
        else:
            self._hold_since = None
            if self.level == OK or self._good_streak >= p.pose_recover_count:
                self.level = OK
            elif self.level == LOST:
                self.level = HOLD
        return self.level
