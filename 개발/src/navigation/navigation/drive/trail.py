"""지나온 길 — 차체가 실제로 지나온 자리를 적어 두었다가, 막혔을 때 돌지 않고 뒤로 되짚어 물러난다 (executor 의 「되짚기」).

왜 따로 적나 (2026-11-19 본선 경기장 8판의 기록):
  supervisor 의 「지나온 길로 되돌아가기」 는 지나온 점 하나를 보통 목표로 보냈다 — 28번 가운데 2번만 닿았고 274 s 를 썼다.
  그 점은 거의 늘 뒤쪽(24번이 차체 방향에서 100° 넘게)이라 막힌 자리에서 먼저 150° 쯤 서서 돌아야 했고(21번은 60° 도 못 돌았다),
  navigation 은 그 길을 「따라가는」 것이 아니라 둘레의 지도에 없는 점을 넣고 새로 짰다. 위치가 0.5 m 뛴 판(18:06)에서는 적어 둔 길이
  실제로 지나온 곳이 아니었다.

그래서
  - 길은 **로컬 궤적**(바퀴 · /odom 을 이어 붙인 자리, pose_tracker)으로 적는다. SLAM 위치가 뛰거나 옮겨져도 「방금 지나온 곳」 은 그대로다.
    (로컬 궤적을 주지 않는 시험에서는 arena 자세로 적는다.)
  - 되짚을 때는 차체 기준의 점만 내준다 — 지도 · arena 좌표를 거치지 않는다.

주행 노드가 하나를 갖고 목표마다 넘긴다 (TurnModel 과 같다). 「뒤가 막혔다」 는 기억(rear_blocked_until)도 여기에 둔다 — 목표가 바뀌어도 이어진다.
"""
from __future__ import annotations

import math


class Trail:
    def __init__(self, step_m=0.05, keep=160):
        self.step = step_m
        self.keep = keep  # 점 수 (0.05 m 간격 160개 = 8 m)
        self._pts = []  # [(x, y)] — 오래된 것부터
        self.here = None  # 지금 자세 (x, y, yaw) — 길을 적는 좌표계
        self.rear_blocked_until = -1e9  # 이때까지는 뒤로 가지 않는다 (뒤로 가라고 했는데 안 움직였다)

    def __len__(self):
        return len(self._pts)

    def update(self, pose, rewinding=False):
        """제어 주기마다. rewinding(되짚는 중)이면 적지 않고, 지나쳐 온 끝 점들을 지운다."""
        if pose is None:
            return
        self.here = (pose.x, pose.y, pose.yaw)
        x, y = pose.x, pose.y
        if rewinding:
            # 지금 자리에 가장 가까운 지나온 점(끝의 40개 = 2 m 안에서)까지 되짚어 왔다 — 그 뒤의 점과, 바로 곁(1.5 칸)의 점을 지운다
            tail = self._pts[-40:]
            if tail:
                near = min(range(len(tail)), key=lambda k: math.hypot(tail[k][0] - x, tail[k][1] - y))
                del self._pts[len(self._pts) - len(tail) + near + 1:]
            while self._pts and math.hypot(self._pts[-1][0] - x, self._pts[-1][1] - y) < 1.5 * self.step:
                self._pts.pop()
            return
        if not self._pts or math.hypot(self._pts[-1][0] - x, self._pts[-1][1] - y) >= self.step - 1e-9:
            self._pts.append((x, y))
            if len(self._pts) > self.keep:
                del self._pts[:-self.keep]

    def back_target(self, reach):
        """되짚어 갈 다음 점: 길의 끝에서부터 거슬러, 지금 자리에서 reach 넘게 떨어진 첫 점 → 차체 기준 (bx, by).
        그런 점이 없으면 가장 오래된 점(0.10 m 넘게 떨어져 있을 때). 길이 없으면 None."""
        if self.here is None or not self._pts:
            return None
        x, y, yaw = self.here
        pick = None
        for px, py in reversed(self._pts):
            if math.hypot(px - x, py - y) >= reach:
                pick = (px, py)
                break
        if pick is None:
            px, py = self._pts[0]
            if math.hypot(px - x, py - y) < 0.10:
                return None
            pick = (px, py)
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = pick[0] - x, pick[1] - y
        return c * dx + s * dy, -s * dx + c * dy

    def back_room(self):
        """지금 자리에서 길을 거슬러 갈 수 있는 거리 (가장 멀리 떨어진 지나온 점까지의 곧은 거리, m)."""
        if self.here is None or not self._pts:
            return 0.0
        x, y = self.here[0], self.here[1]
        return max(math.hypot(px - x, py - y) for px, py in self._pts)

    def rear_blocked(self, t):
        return t < self.rear_blocked_until

    def summary(self, t):
        return {'points': len(self._pts), 'rear_blocked_s': round(max(0.0, self.rear_blocked_until - t), 1)}
