"""위에서 본 UGV 자리들(/team/perception/ugvs — 드론 사진에서 찾은 것) → 「한자리에 서 있는 다른 로봇」. ROS 를 쓰지 않는다.

정찰의 실시간 순서(scout.FaceScout._live_pick)가 「그 자리 곁에 로봇이 서 있다 · 얼마나 서 있었다」 를 알게 하려는 것이다.
navigation 의 drive/ugvs.py (UgvTracks) 와 같은 생각 · 같은 값을 쓴다 — 그 가운데 여기에 필요한 것만 다시 적었다 (헛돎 · 기억 칸 정리는 없다):
  · 관측에는 우리 차도 들어 있다 (모델은 진영 · 호출부호를 가리지 못한다). 그 무렵 우리가 있던 자리들에서 self_m 안의 가장 가까운 것이 우리 차다.
    우리 자리를 모르면 그 관측은 통째로 버린다. 우리 차를 못 골랐으면 우리 곁(doubt_m)의 것은 그 관측에서는 쓰지 않는다 (우리 차일 수 있다)
  · 앞 관측의 로봇과 gate_m 안이면 같은 로봇이다 (가까운 짝부터). 관측이 뜸하면(2 s · 5 s — 우리 시설물이 파괴될수록 사진이 준다) 그만큼 넓게 잇는다
  · 처음 선 곳(anchor)에서 still_m 안에만 있었으면 서 있는 것이다. 벗어나면 그 자리에서 다시 센다
  · ttl 동안 안 보이면 잊는다. 관측이 끊기면(feed) 아무것도 답하지 않는다 — 「안 보인다」 는 「없다」 가 아니고(화면 밖 · 가림), 오래된 자리는 믿을 수 없다
"""
from __future__ import annotations

import math


def add_sight(sights, at, poses, keep=12):
    """요즘 관측 목록에 하나를 더한 새 목록 (가장 새 keep 개만). poses = PoseArray 의 poses (position.x · y = 가운데 m, position.z = 점수) —
    숫자가 아닌 자리는 뺀다. 돌려주는 값: [(때, [(x, y, 점수)])] 오래된 것부터."""
    points = [(float(q.position.x), float(q.position.y), float(q.position.z)) for q in poses
              if math.isfinite(q.position.x) and math.isfinite(q.position.y)]
    return (list(sights or []) + [(float(at), points)])[-keep:]


class Standing:
    def __init__(self, still_m=0.15, self_m=0.40, doubt_m=0.80, gate_m=0.60, ttl_s=3.5, feed_s=3.0):
        self.still_m, self.self_m, self.doubt_m, self.gate_m, self.ttl_s, self.feed_s = still_m, self_m, doubt_m, gate_m, ttl_s, feed_s
        self.tracks = []  # 로봇마다 [x, y, 마지막으로 본 때, 선 곳 x, 선 곳 y, 거기 서기 시작한 때, 본 횟수]
        self.stamp = None  # 마지막으로 쓴 관측의 때
        self.interval = 1.0  # 요즘 관측 간격 (s)
        self._gaps = []
        self.used = self.dropped = 0

    def _ttl(self):
        return max(self.ttl_s, 2.5 * self.interval + 1.0)  # 관측 두 번을 놓쳐도 이어 가게 (1 s 면 3.5 s, 5 s 면 13.5 s)

    def alive(self, now):
        """관측이 오고 있는가 (마지막 관측이 feed_s 안 — 간격이 늘었으면 그만큼 늘려 본다)."""
        return self.stamp is not None and now - self.stamp <= max(self.feed_s, 2.2 * self.interval + 1.0)

    def add(self, at, points, recent):
        """관측 한 번을 넣는다. 썼으면 True.

        at = 그 관측의 때 (초 — robots() 의 now 와 같은 시계), points = [(x, y) 또는 (x, y, 점수)] 우리 차 포함,
        recent = 그 무렵 우리가 있던 자리들 [(x, y)] (비었으면 = 우리 자리를 모른다 → 버린다). 앞에 쓴 것보다 새롭지 않은 관측도 버린다.
        """
        if self.stamp is not None and at <= self.stamp:
            return False
        if not recent:
            self.dropped += 1
            return False
        pts = [(float(q[0]), float(q[1])) for q in points if math.isfinite(q[0]) and math.isfinite(q[1])]

        def from_us(q):
            return min(math.hypot(q[0] - x, q[1] - y) for x, y in recent)

        me = min((q for q in pts if from_us(q) <= self.self_m), key=from_us, default=None)
        others = [q for q in pts if q is not me]
        if me is None:
            others = [q for q in others if from_us(q) > self.doubt_m]  # 우리 차를 못 골랐다 → 우리 곁의 것은 우리 차일 수 있다
        if self.stamp is not None:
            self._gaps = (self._gaps + [at - self.stamp])[-2:]
            self.interval = min(6.0, max(0.5, max(self._gaps)))  # 늘어난 것은 바로, 줄어든 것은 두 번 뒤에 따라간다
        gate = min(1.5, self.gate_m * max(1.0, self.interval))
        self.stamp = at
        self.used += 1
        pairs = sorted((math.hypot(t[0] - q[0], t[1] - q[1]), i, k) for i, t in enumerate(self.tracks) for k, q in enumerate(others))
        used_t, used_q = set(), set()
        for d, i, k in pairs:
            if d > gate:
                break
            if i in used_t or k in used_q:
                continue
            used_t.add(i)
            used_q.add(k)
            t = self.tracks[i]
            t[0], t[1], t[2], t[6] = others[k][0], others[k][1], at, t[6] + 1
            if math.hypot(t[0] - t[3], t[1] - t[4]) > self.still_m:
                t[3], t[4], t[5] = t[0], t[1], at  # 선 곳을 벗어났다 → 여기서부터 다시 센다
        self.tracks += [[q[0], q[1], at, q[0], q[1], at, 1] for k, q in enumerate(others) if k not in used_q]
        ttl = self._ttl()
        self.tracks = [t for t in self.tracks if at - t[2] <= ttl]
        return True

    def robots(self, now, min_s=4.0):
        """지금 서 있는 다른 로봇들 [(선 곳 x, y, 서 있은 시간 s)] — 한자리에 min_s 넘게 있었고 요즘도 보인 것. 관측이 끊겼으면 빈 목록."""
        if not self.alive(now):
            return []
        ttl = self._ttl()
        return [(t[3], t[4], t[2] - t[5]) for t in self.tracks if t[6] >= 2 and now - t[2] <= ttl and t[2] - t[5] >= min_s]
