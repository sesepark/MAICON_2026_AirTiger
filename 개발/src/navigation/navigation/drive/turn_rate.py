"""서서 돌 때 차체가 실제로 도는 빠르기 (rad/s) 를 잰다 — 명령을 실제 회전에 맞춰 올리고 내리는 데 쓴다 (executor._spin_boost, 2026-11-19 밤).

왜: 실차 기록(2026-11-19 본선 경기장 10판, navigation/debug.jsonl 0.5 s 위치)에서 서서 도는 명령 1.6 rad/s 는 실제 가운데값 0.42 ~ 0.47 rad/s 이고
표본의 17 ~ 18 % 는 0.1 rad/s 도 못 돌았다 (1.4 rad/s 밑은 거의 안 돈다 — 정지 마찰). 21:53 판은 목표를 가는 시간의 45 % (97 s)가 서서 돌기 · 방향 맞추기였다.
명령이 얼마일 때 얼마나 도는지는 바닥 · 배터리 · 붙은 물체에 따라 달라, 재면서 맞춘다.

무엇으로 재나 (좋은 것부터)
  gyro  /imu/data 의 angular_velocity.z. **실차에서 회전을 따라가는지 확인된 적이 없다** (2026-11-18 시험기록: 정지 때 값은 살아 있으나 launch 주석은
        ESP IMU 값이 고정된다고 적었다). 그래서 스스로 확인한 뒤에만 쓴다: SLAM 방향이 gyro_check_rad 넘게 바뀐 구간마다 gyro 를 적분한 각과 견주어
        비(크기)가 gyro_ratio 범위 안이면 「맞음」, 아니면 「틀림」. 맞음이 gyro_min_ok 번 이상이고 틀림의 두 배를 넘을 때만 gyro 를 쓴다.
  slam  SLAM 위치(/team/navigation/pose)의 방향 차이. 스캔마다 한 번(10 Hz) 오므로 window_s 동안의 처음 · 끝으로 잰다.
  pose  위의 둘이 없을 때(시험 · SLAM 위치를 따로 넣지 않는 곳): 제어 주기의 자세 방향 차이.
"""
from __future__ import annotations

import math
from collections import deque

from .geometry import wrap


class SpinRate:
    def __init__(self, window_s=0.5, gyro_check_rad=0.35, gyro_ratio=(0.6, 1.6), gyro_min_ok=3):
        self.window_s = window_s
        self.gyro_check_rad = gyro_check_rad
        self.gyro_ratio = gyro_ratio
        self.gyro_min_ok = gyro_min_ok
        self._slam = deque(maxlen=40)   # (시각, 방향) — SLAM 위치
        self._pose = deque(maxlen=40)   # (시각, 방향) — 제어 주기의 자세
        self._gyro = deque(maxlen=80)   # (시각, 각속도 z)
        self._check = None              # gyro 확인 구간: [시작 시각, 시작 SLAM 방향, 누적 SLAM 각, 누적 gyro 각, 마지막 gyro 시각]
        self.gyro_ok = 0                # gyro 가 SLAM 과 맞은 구간 수
        self.gyro_bad = 0
        self.gyro_last_ratio = None     # 마지막 확인 구간의 |gyro 각| / |SLAM 각|
        self.source = None              # 마지막 rate() 가 쓴 것: 'gyro' · 'slam' · 'pose'

    # ------------------------------------------------------------------ 입력
    def on_slam_yaw(self, t, yaw):
        if self._slam and t <= self._slam[-1][0]:
            return
        if self._slam and self._check is not None:
            self._check[2] += wrap(yaw - self._slam[-1][1])
            self._close_check(t)
        self._slam.append((t, yaw))

    def on_pose_yaw(self, t, yaw):
        if not self._pose or t > self._pose[-1][0]:
            self._pose.append((t, yaw))

    def on_gyro(self, t, wz):
        if not math.isfinite(wz):
            return
        if self._gyro and t <= self._gyro[-1][0]:
            return
        if self._check is None:
            self._check = [t, None, 0.0, 0.0, t]
        else:
            dt = t - self._check[4]
            if 0.0 < dt < 0.5:
                self._check[3] += wz * dt
            else:   # 끊겼다 — 구간을 새로
                self._check = [t, None, 0.0, 0.0, t]
            self._check[4] = t
        self._gyro.append((t, wz))

    def _close_check(self, t):
        """SLAM 방향이 gyro_check_rad 넘게 바뀌었으면 그 구간의 gyro 적분과 견주고 구간을 새로 연다. 오래 안 돌면(10 s) 버리고 새로 (gyro 치우침이 쌓이지 않게)."""
        _t0, _y0, slam_angle, gyro_angle, last = self._check
        if abs(slam_angle) >= self.gyro_check_rad:
            ratio = abs(gyro_angle) / abs(slam_angle)
            self.gyro_last_ratio = round(ratio, 2)
            if self.gyro_ratio[0] <= ratio <= self.gyro_ratio[1]:
                self.gyro_ok += 1
            else:
                self.gyro_bad += 1
            self._check = [t, None, 0.0, 0.0, last]
        elif t - self._check[0] > 10.0:
            self._check = [t, None, 0.0, 0.0, last]

    # ------------------------------------------------------------------ 출력
    @property
    def gyro_trusted(self):
        return self.gyro_ok >= self.gyro_min_ok and self.gyro_ok > 2 * self.gyro_bad

    def rate(self, t):
        """지금 실제로 도는 빠르기 |rad/s| (최근 window_s). 잴 것이 모자라면 None."""
        if self.gyro_trusted:
            recent = [abs(w) for ts, w in self._gyro if t - ts <= self.window_s]
            if len(recent) >= 3:
                self.source = 'gyro'
                return sum(recent) / len(recent)
        for name, samples in (('slam', self._slam), ('pose', self._pose)):
            if not samples or t - samples[-1][0] > 0.35:
                continue   # 끊겼다 (다음 것으로)
            inside = [s for s in samples if t - s[0] <= self.window_s + 0.06]
            if len(inside) >= 3 and inside[-1][0] - inside[0][0] >= 0.6 * self.window_s:
                total = sum(wrap(b[1] - a[1]) for a, b in zip(inside, inside[1:]))
                self.source = name
                return abs(total) / (inside[-1][0] - inside[0][0])
        return None

    def summary(self):
        return {'src': self.source, 'gyro_ok': self.gyro_ok, 'gyro_bad': self.gyro_bad, 'gyro_ratio': self.gyro_last_ratio}
