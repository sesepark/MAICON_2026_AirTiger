"""드론이 본 UGV 자리(/team/perception/ugvs, 약 1 Hz) → 우리 차 골라내기 · 다른 UGV 의 궤적(서 있었나 · 얼마나).

드론 자리는 0.3 ~ 1 s 늦고(사진 간격 1 s + 받는 시간 + 판독) 화면 위아래가 잘려 벽 쪽은 안 보인다.
사진 간격은 우리 진영 시설물이 파괴될수록 1 → 2 → 5 s 로 늘고, 셋이 파괴되면 아예 안 온다 (공식 규정 — 경기서버연동API 4단계).
그래서 잊는 시간 · 잇는 거리는 요즘 간격에 맞춰 늘리고(_ttl · _feed), 헛돎 확인은 1 s 간격일 때만 하며, 안 오면 아래 것이 모두 쉰다 (전과 같이 라이다로만)
(UGV 높이에서 보이는 세로 범위 — 4.80 지도: 본선 카메라 0.33 ~ 4.17 m (2026-11-20 다시 잼 — 맞게 찾은 경계 · UGV 높이 0.10. 공식 테스트 경기 여섯 판에서 우리 차가 잡힌 비율: y 0.6 ~ 3.9 에서 99 %, 3.9 ~ 4.15 에서 82 %, 4.15 위에서 15 %, 0.35 아래에서 18 %),
 연습 test1 0.23 ~ 4.43 · test2 0.33 ~ 4.46 [2026-11-19 계산] — 실차 기록의 값은 0.24 ~ 4.55).
그래서 **정지 · 감속에는 쓰지 않는다** (그것은 지금 라이다 스캔만 본다). 쓰는 곳은 이것뿐이다 (2026-11-19 사용자 결정):
  2) 오래 서 있는 UGV(ugv_plan_parked_s) 는 길 고르기에 미리 넣는다 — 라이다로 보기 전부터 그 자리를 피한다 (obstacles.sync_ugvs · _ugv_plan_cells)
  1) 길을 막은 것이 「서 있는 UGV」 인지 — 그러면 좁은 틈 길을 바로 열고, 포기를 앞당긴다 (obstacles.blocker_ugv · executor._stall)
  3) 끼임 · 헛돎 — 바퀴(/odom)는 갔다는데 드론이 본 우리 차는 안 움직였다 (slip), 붙어 선 UGV 를 비키기 계획에 넣는다 (obstacles.ugv_body_points)
  6) 기억 칸 정리 — UGV 때문에 생긴 임시 장애물 칸은 그 UGV 가 떠나면 지우고, 그대로 있으면 붙든다 (obstacles.sync_ugvs)
  8) 지도 고치기에서 UGV 에 맞은 라이다 점을 뺀다 (slam_node — 이 모듈의 split_self 만 쓴다)
「드론에 안 보인다」 는 「없다」 가 아니다 (화면 밖 · 가림 · 놓침) — 없다는 판단은 「앞서 그 자리에서 본 UGV 가 딴 데로 갔다」 일 때만 한다.

사진이 끊긴 뒤 (2026-11-20, 붙들기 — ugv_hold_s): 끊길 때 서 있던 UGV(ugv_parked_s 넘게 한자리)는 바로 잊지 않고 그 자리에 있는 것으로 둔다.
  끊기기 전에는 5 s 쯤 뒤 통째로 잊어, 260 s 째 서 있던 UGV 자리로 다시 길을 골랐다 (00:00 실차의 (3.05, 2.57) — 그 판은 끊기지 않았다).
  그 뒤로는 라이다가 진실이다 (obstacles._check_held): 그 자리에 지도에 없는 점이 찍히면 「아직 있다」 로 붙드는 시간을 늘리고,
  광선이 그 자리 너머까지 닿으면(ugv_hold_clear_scans 번 잇달아) 「떠났다」 로 지운다. 라이다가 못 보는 자리의 것은 마지막으로 확인한 뒤 ugv_hold_s 까지.
  움직이던 UGV 는 붙들지 않는다 (몇 초면 딴 데 있다). 헛돎 확인 · 기억 칸 정리는 사진이 올 때만 한다 (전과 같다).

우리 차 고르기: 사진은 header.stamp 보다 조금 앞서 찍혔다. 그 사이 우리가 지나온 자리들(recent — stamp 앞 self_window_s 동안의 자세) 가운데
어느 하나에서 self_match_m 안인 것 중 가장 가까운 것이 우리 차다. 자세를 모르면 그 사진은 통째로 버린다 (우리 차를 다른 UGV 로 볼 수 있다).
"""
from __future__ import annotations

import math


def split_self(points, recent, gate):
    """points [(x, y, …)] 를 (우리 차 또는 None, 나머지) 로 나눈다. recent = 그 무렵 우리가 있던 자리들 [(x, y)]."""
    best = None
    for k, pt in enumerate(points):
        d = min((math.hypot(pt[0] - x, pt[1] - y) for x, y in recent), default=math.inf)
        if d <= gate and (best is None or d < best[0]):
            best = (d, k)
    if best is None:
        return None, list(points)
    return points[best[1]], [pt for k, pt in enumerate(points) if k != best[1]]


def without_points_near(body, pose, centres, radius):
    """차체 좌표 점 배열 body (N, 2) 에서, 경기장 좌표 centres [(x, y)] 의 radius 안에 찍힌 점을 뺀 것과 뺀 수. pose = (x, y, yaw)."""
    import numpy as np

    body = np.asarray(body, dtype=float).reshape(-1, 2)
    if not len(body) or not centres:
        return body, 0
    c, s = math.cos(pose[2]), math.sin(pose[2])
    wx, wy = pose[0] + c * body[:, 0] - s * body[:, 1], pose[1] + s * body[:, 0] + c * body[:, 1]
    keep = np.ones(len(body), dtype=bool)
    for x, y in centres:
        keep &= np.hypot(wx - x, wy - y) > radius
    return body[keep], int((~keep).sum())


class Track:
    __slots__ = ("x", "y", "born", "seen", "anchor", "still_since", "hits", "doubt", "lidar", "clear", "lost", "n", "spot")

    def __init__(self, x, y, stamp):
        self.x, self.y = x, y
        self.born = self.seen = self.still_since = stamp
        self.anchor = (x, y)
        self.hits = 1
        self.doubt = None  # 우리 차일지도 모른다고 본 마지막 사진의 시각 (UgvTracks.on_frame)
        self.lidar = None  # 사진이 끊긴 뒤 라이다가 그 자리의 물체를 마지막으로 확인한 시각 (UgvTracks.confirm)
        self.clear = 0  # 사진이 끊긴 뒤 라이다 광선이 그 자리를 잇달아 지나간 스캔 수 (UgvTracks.refute)
        self.lost = None  # 서 있다가 드론에서 사라진 것을 안 때 (UgvTracks.lost 에 든 동안)
        self.n = 1  # 지금 서 있는 자리에서 본 사진 수 — (x, y) 는 그 사진들의 평균이다 (UgvTracks._place)
        self.spot = (round(x, 2), round(y, 2))  # 길 고르기에 넣는 자리: 평균이 3 cm 넘게 옮겨 갈 때만 따라간다 (사진마다 mm 씩 달라져 다시 짜지 않게)

    def last(self):
        """마지막으로 그 자리에 있다고 안 시각 (드론 사진, 끊긴 뒤에는 라이다 확인)."""
        return self.seen if self.lidar is None else max(self.seen, self.lidar)

    def still_s(self):
        """마지막으로 본 때까지 한자리(ugv_still_m 안)에 있던 시간 (s)."""
        return self.last() - self.still_since


class UgvTracks:
    def __init__(self, params):
        self.p = params
        self.tracks = []
        # 서 있다가(tight_still_s 넘게) 드론에서 사라진 궤적들 — ugv_lost_keep_s 동안 든다. 「안 보인다」 는 「떠났다」 가 아니다:
        # 2026-11-20 00:00 실차에서 오른쪽 거점에 서 있던 UGV 를 드론이 50 s 동안 31장 중 1장만 잡았다 (라이다는 그동안 계속 봤다).
        # 떠나는 UGV 는 떠나는 것이 보인다 (사진 사이에 ugv_track_gate_m 안에서 옮겨 간다 → 움직이는 궤적으로 끝난다 → 여기 들지 않는다).
        #   - 그 자리(ugv_still_m)에 다시 보이면 같은 UGV 로 잇는다 — 「서 있던 시간」 을 0 부터 다시 세지 않는다 (on_frame)
        #   - 그 곁의 기억 칸은 드론으로 지우지 않는다 (obstacles.sync_ugvs — 전처럼 라이다 규칙으로만 사라진다)
        # 길 고르기(ugv_plan) · 막은 것 판단에는 넣지 않는다 (지금 보이는 궤적만)
        self.lost = []
        self.stamp = None  # 마지막으로 쓴 사진의 시각
        self.version = 0
        self.frames = self.skipped = 0
        self.self_obs = []  # [(stamp, 드론이 본 우리 x, y, 그때 로컬 궤적 x, y)] — 헛돎 확인용
        self.residual = None  # 마지막 사진: 드론이 본 우리 자리 − 우리 자세 (m) — 드론 높이 보정(ugv_watch_height)을 맞추는 데 쓴다 (기록)
        self._slip_told = None
        self.refuted = 0  # 사진이 끊긴 뒤 라이다가 「떠났다」 고 지운 UGV 수 (기록용)
        self.interval = 1.0  # 요즘 사진 간격 (s) — 우리 진영 시설물이 파괴되면 1 → 2 → 5 s 로 늘고 셋이면 끊긴다 (경기서버연동API 4단계)
        self._gaps = []

    def _ttl(self):
        """UGV 를 잊는 시간: 사진 두 장을 놓쳐도 이어 가게 간격에 맞춰 늘린다 (1 s 면 ugv_ttl_s 그대로, 5 s 면 13.5 s)."""
        return max(self.p.ugv_ttl_s, 2.5 * self.interval + 1.0)

    def _feed(self):
        """드론 자리가 「오고 있다」 고 보는 시간 (1 s 면 3.2 s, 5 s 면 12 s)."""
        return max(self.p.ugv_feed_s, 2.2 * self.interval + 1.0)

    def clear(self):
        self.tracks, self.self_obs, self.stamp, self.residual = [], [], None, None
        self.lost = []
        self.version += 1

    # ------------------------------------------------------------------ 입력
    def on_frame(self, stamp, points, recent, local=None, pose=None):
        """사진 한 장. points [(x, y) 또는 (x, y, 점수)] — 우리 차 포함. recent = stamp 앞 self_window_s 동안 우리가 있던 자리 [(x, y)]
        (비었으면 = 자세를 모른다 → 버린다). local = stamp 때의 로컬 궤적 자세 (x, y), pose = stamp 때의 자세 (x, y)."""
        p = self.p
        if self.stamp is not None and stamp <= self.stamp:
            return False
        if not recent:
            self.skipped += 1
            return False
        if self.stamp is not None:
            self._gaps = (self._gaps + [stamp - self.stamp])[-2:]
            self.interval = min(6.0, max(0.5, max(self._gaps)))  # 늘어난 것은 바로, 줄어든 것은 두 장 뒤에 따라간다
        gate = min(1.5, p.ugv_track_gate_m * max(1.0, self.interval))  # 사진이 뜸하면 그 사이 더 멀리 갔을 수 있다
        me, others = self._pick_self([(float(q[0]), float(q[1])) for q in points], recent, gate)
        self.stamp = stamp
        self.frames += 1
        self.version += 1
        if me is not None:
            if local is not None:
                self.self_obs.append((stamp, me[0], me[1], float(local[0]), float(local[1])))
                self.self_obs = [o for o in self.self_obs if stamp - o[0] <= p.ugv_slip_window_s + 2.0]
            if pose is not None:
                self.residual = (round(me[0] - pose[0], 3), round(me[1] - pose[1], 3))
        # 이 사진에서 우리 차를 못 골랐으면 (자세가 틀렸거나 드론 자리가 밀렸다 · 우리가 화면 밖이다 — 공식 테스트 경기 6판: 화면 안에 있던
        # 500장 중 2 %, 2026-11-20 00:00 실차: 196장 중 10장 · 그중 7장은 우리가 벽 쪽 통로에 있을 때) 우리 곁(ugv_self_doubt_m)의 검출은
        # 우리 차일 수 있다 → 이 사진으로는 믿지 않는다 (fresh 가 뺀다). 다음 사진에서 우리를 고르면 풀린다.
        # 다만 **이미 믿던 · 서 있는 UGV 가 서 있던 그 자리에서 다시 보인 것** 은 우리 차일 수 없다 (우리가 그 자리에 올라설 수는 없다) → 그대로 믿는다.
        # 전에는 그것도 한 장 동안 뺐다 — 그러면 obstacles.sync_ugvs 가 「그 UGV 가 떠났다」 로 보고 그 곁의 기억 칸을 통째로 지우고,
        # 길 고르기에 넣은 자리(_ugv_plan)도 한 장 동안 빠졌다 (가상: 뒤 0.6 m 에 선 UGV 의 기억 칸 20개가 사진 한 장에 0). 반대로 그런 궤적을
        # 우리 차일 수 있는 검출이 **다른 자리로 옮기지도 못한다** (옮기면 「서 있던 시간」 이 0 부터 다시다) — 그 검출은 새 궤적(믿지 않음)이 된다
        doubtful = set()
        if me is None:
            doubtful = {k for k, (x, y) in enumerate(others) if min(math.hypot(x - rx, y - ry) for rx, ry in recent) <= p.ugv_self_doubt_m}
        # 가까운 것끼리 잇는다 (가까운 짝부터)
        pairs = sorted((math.hypot(t.x - x, t.y - y), i, k) for i, t in enumerate(self.tracks) for k, (x, y) in enumerate(others))
        used_t, used_o = set(), set()
        for d, i, k in pairs:
            if d > gate:
                break
            if i in used_t or k in used_o:
                continue
            t = self.tracks[i]
            keep_trust = False
            if k in doubtful:
                keep_trust = t.hits >= 2 and t.doubt != t.seen and t.still_s() >= p.ugv_parked_s   # 믿던 · 서 있는 UGV
                if keep_trust and math.hypot(others[k][0] - t.anchor[0], others[k][1] - t.anchor[1]) > p.ugv_still_m:
                    continue   # 서 있던 자리가 아니다 — 이 검출로 그 궤적을 옮기지 않는다
            used_t.add(i)
            used_o.add(k)
            if self._place(t, *others[k]):
                t.still_since = stamp
            t.seen = stamp
            t.lidar, t.clear = None, 0
            t.hits += 1
            if k in doubtful and not keep_trust:
                t.doubt = stamp
        for k, (x, y) in enumerate(others):
            if k not in used_o:
                # 서 있다가 사라졌던 UGV 가 그 자리에 다시 보였으면 그 궤적을 잇는다. 우리 차일 수 있는 검출로는 잇지 않는다 (그 사이 우리가 그 자리에 왔을 수 있다)
                track = None if k in doubtful else self._revive(x, y)
                if track is not None:
                    self._place(track, x, y)
                    track.seen, track.lost = stamp, None
                    track.lidar, track.clear = None, 0
                    track.hits += 1
                else:
                    track = Track(x, y, stamp)
                    if k in doubtful:
                        track.doubt = stamp
                self.tracks.append(track)
        ttl, kept = self._ttl(), []
        for t in self.tracks:
            if stamp - t.seen <= ttl:
                kept.append(t)
            elif self._was_parked(t):
                t.lost = stamp   # 서 있다가 사라졌다 (떠나는 것을 본 적이 없다)
                self.lost.append(t)
        self.tracks = kept
        self.lost = [t for t in self.lost if stamp - t.lost <= p.ugv_lost_keep_s]
        return True

    def _place(self, track, x, y):
        """궤적에 이번 사진의 자리 (x, y) 를 넣는다. 서 있던 자리(anchor)에서 ugv_still_m 넘게 옮겨 갔으면 그 자리로 옮기고 True (서 있기 시작한 때를 다시 센다).
        그 안이면 서 있는 것이다 — (x, y) 와 anchor 를 그 자리에서 본 사진들의 평균으로 둔다 (2026-11-20). 전에는 (x, y) 가 마지막 사진 한 장의 값,
        anchor 가 서 있기 시작한 첫 장의 값이었다 (길 고르기 · 「옮겨 갔나」 가 그 한 장에 묶였다) — 00:00 실차에서 서 있는 UGV 의 드론 좌표는
        사진마다 3 cm 안팎으로 흔들렸다 (x 3.02 ~ 3.08 · y 2.55 ~ 2.59). 평균은 요즘 20장 쪽으로 따라간다 (조금씩 밀려 가는 것을 따라가게)."""
        if math.hypot(x - track.anchor[0], y - track.anchor[1]) > self.p.ugv_still_m:
            track.x, track.y, track.anchor, track.n = x, y, (x, y), 1
            track.spot = (round(x, 2), round(y, 2))
            return True
        track.n = min(track.n + 1, 20)
        track.x += (x - track.x) / track.n
        track.y += (y - track.y) / track.n
        track.anchor = (track.x, track.y)
        if math.hypot(track.x - track.spot[0], track.y - track.spot[1]) > 0.03:
            track.spot = (round(track.x, 2), round(track.y, 2))
        return False

    def _pick_self(self, points, recent, gate):
        """points 를 (우리 차 또는 None, 나머지) 로 나눈다 — split_self 와 같되, 우리 차일 수 있는 것(지나온 길에서 ugv_self_match_m 안)이 **둘 이상** 이면
        이미 믿는 다른 UGV 궤적이 제 것으로 가져갈 검출(그 궤적에서 가장 가까운 것)은 뺀 뒤에 고른다 (2026-11-20).
        사진은 header.stamp 보다 0.3 ~ 1.7 s 앞서 찍혀서 지나온 길 전체와 견주는데, 바로 뒤를 따라오는 UGV 는 그 길 위에 있다 — 길까지의 거리만 보면
        우리 차와 구별이 안 된다 (가상: 0.30 ~ 0.45 m 뒤를 따라오는 UGV 를 39 % 우리 차로 골랐다). 따라오는 UGV 는 앞 사진들에서 이미 궤적이 있다.
        서 있는 궤적은 서 있던 자리(ugv_still_m)의 검출만, 움직이는 궤적은 gate 안의 가장 가까운 검출을 제 것으로 친다."""
        p = self.p
        near = []
        for k, (x, y) in enumerate(points):
            d = min(math.hypot(x - rx, y - ry) for rx, ry in recent)
            if d <= p.ugv_self_match_m:
                near.append((d, k))
        if not near:
            return None, list(points)
        if len(near) > 1:
            claimed = set()
            for t in self.tracks:
                if t.hits < 2 or t.doubt == t.seen:
                    continue
                parked = t.still_s() >= p.ugv_parked_s
                ref, reach = (t.anchor, p.ugv_still_m) if parked else ((t.x, t.y), gate)
                d, k = min((math.hypot(x - ref[0], y - ref[1]), k) for k, (x, y) in enumerate(points))
                if d <= reach:
                    claimed.add(k)
            left = [c for c in near if c[1] not in claimed]
            near = left or near
        mine = min(near)[1]
        return points[mine], [q for k, q in enumerate(points) if k != mine]

    def _was_parked(self, u):
        """끊기면 lost 에 넣을 궤적인가: 믿던 것이고 tight_still_s 넘게 서 있었다."""
        p = self.p
        return p.ugv_lost_keep_s > 0.0 and u.hits >= 2 and u.doubt != u.seen and u.still_s() >= p.tight_still_s

    def _revive(self, x, y):
        """(x, y) 가 서 있다가 사라진 궤적의 서 있던 자리(ugv_still_m 안)면 그 궤적을 lost 에서 꺼내 돌려준다. 없으면 None."""
        best = None
        for t in self.lost:
            d = math.hypot(t.anchor[0] - x, t.anchor[1] - y)
            if d <= self.p.ugv_still_m and (best is None or d < best[0]):
                best = (d, t)
        if best is None:
            return None
        self.lost.remove(best[1])
        return best[1]

    def lost_parked(self, t):
        """서 있다가 드론에서 사라진 UGV 들의 서 있던 자리 [(x, y)] — 아직 거기 있을 수 있다 (ugv_lost_keep_s 동안)."""
        out = [u.anchor for u in self.lost if t - u.lost <= self.p.ugv_lost_keep_s]
        ttl = self._ttl()   # 방금 끊겨 다음 사진에 lost 로 옮겨질 것도 (fresh 는 이미 뺐다)
        out += [u.anchor for u in self.tracks if t - u.seen > ttl and self._was_parked(u)]
        return out

    # ------------------------------------------------------------------ 물음
    def alive(self, t):
        """드론 자리가 오고 있는가 (마지막 사진이 ugv_feed_s 안 — 사진 간격이 늘었으면 그만큼 늘려 본다). 끊기면(403 · 통신) False → 쓰는 쪽이 모두 쉰다."""
        return self.stamp is not None and t - self.stamp <= self._feed()

    def _held(self, t):
        """사진이 끊긴 뒤에도 그 자리에 있는 것으로 두는 UGV 들: 마지막 사진 때 믿던 것 가운데 ugv_parked_s 넘게 서 있던 것,
        마지막으로 확인한(사진 · 라이다) 뒤 ugv_hold_s 안. ugv_hold_s 0 이하 = 끔 (끊기면 전처럼 모두 잊는다)."""
        p = self.p
        if p.ugv_hold_s <= 0.0 or self.stamp is None:
            return []
        ttl = self._ttl()
        return [u for u in self.tracks if u.hits >= 2 and self.stamp - u.seen <= ttl and u.doubt != u.seen
                and u.still_s() >= p.ugv_parked_s and t - u.last() <= p.ugv_hold_s]

    def holding(self, t):
        """사진은 끊겼는데 붙들고 있는 UGV 가 있는가."""
        return self.stamp is not None and not self.alive(t) and bool(self._held(t))

    def confirm(self, u, t):
        """붙들고 있는 UGV 의 자리에 라이다로 지도에 없는 점이 찍혔다 — 아직 있다."""
        u.lidar, u.clear = t, 0

    def refute(self, u, clear=True):
        """붙들고 있는 UGV 의 자리를 라이다 광선이 지나갔다(clear) — ugv_hold_clear_scans 번 잇달아면 떠난 것으로 지운다 (지웠으면 True).
        clear=False (그 자리가 가렸다 · 무엇이 있다) 면 세던 것을 0 으로."""
        if not clear:
            u.clear = 0
            return False
        u.clear += 1
        if u.clear < max(1, int(self.p.ugv_hold_clear_scans)):
            return False
        self.tracks = [k for k in self.tracks if k is not u]
        self.refuted += 1
        self.version += 1
        return True

    def fresh(self, t):
        """지금 믿을 다른 UGV 들 (두 번 넘게 보였고, 마지막으로 본 지 ugv_ttl_s 안 — 사진 간격에 맞춰 늘린 값).
        사진이 끊겼으면 붙들고 있는 것(_held — 서 있던 UGV 만)을 돌려준다."""
        if not self.alive(t):
            return self._held(t)
        ttl = self._ttl()
        return [u for u in self.tracks if u.hits >= 2 and t - u.seen <= ttl and u.doubt != u.seen]

    def near(self, t, x, y, radius):
        """(x, y) 에서 radius 안의 가장 가까운 UGV 궤적, 없으면 None."""
        best = None
        for u in self.fresh(t):
            d = math.hypot(u.x - x, u.y - y)
            if d <= radius and (best is None or d < best[0]):
                best = (d, u)
        return best[1] if best else None

    def parked(self, t):
        """ugv_parked_s 넘게 한자리에 서 있는 UGV 들."""
        return [u for u in self.fresh(t) if u.still_s() >= self.p.ugv_parked_s]

    def slip(self, t, lag=None):
        """바퀴(로컬 궤적)는 ugv_slip_asked_m 넘게 갔다는데 드론이 본 우리 차는 ugv_slip_moved_m 도 안 움직였으면 {"asked", "moved", "at"}, 아니면 None.
        같은 구간은 한 번만 알린다. 드론 사진이 늦는 만큼(ugv_latency_s) 방금 출발한 것을 헛돎으로 보지 않게, 창의 앞 · 뒤 로컬 자세를
        그대로 견준 것과 한 장씩 앞당겨 견준 것 가운데 작은 쪽을 「간 거리」 로 친다."""
        p = self.p
        if p.ugv_slip_window_s <= 0.0 or self.interval > 1.6 or not self.alive(t) or len(self.self_obs) < 4:  # 사진이 뜸하면(2 s · 5 s) 보지 않는다
            return None
        last = self.self_obs[-1]
        window = [o for o in self.self_obs if last[0] - o[0] <= p.ugv_slip_window_s]
        if len(window) < 4 or last[0] - window[0][0] < 0.6 * p.ugv_slip_window_s or self._slip_told == last[0]:
            return None
        first = window[0]
        moved = max(math.hypot(o[1] - first[1], o[2] - first[2]) for o in window[1:])
        asked = min(math.hypot(last[3] - first[3], last[4] - first[4]), math.hypot(window[-2][3] - first[3], window[-2][4] - first[4]))
        if asked >= p.ugv_slip_asked_m and moved <= p.ugv_slip_moved_m:
            self._slip_told = last[0]
            return {"asked": round(asked, 3), "moved": round(moved, 3), "at": last[0]}
        return None

    def summary(self, t):
        """기록용."""
        out = {"age": None if self.stamp is None else round(t - self.stamp, 2), "interval": round(self.interval, 1), "frames": self.frames, "skipped": self.skipped,
               "others": [[round(u.x, 2), round(u.y, 2), round(u.still_s(), 1)] for u in self.fresh(t)],
               # 서 있다가 사라진 것 [x, y, 사라진 지 s] (2026-11-20)
               "lost": [[round(u.anchor[0], 2), round(u.anchor[1], 2), round(t - u.lost, 1)] for u in self.lost]}
        if self.stamp is not None and not self.alive(t):
            out["held"] = len(out["others"])  # 사진이 끊겼다 — others 는 붙들고 있는 것 (끊길 때 서 있던 UGV)
        if self.refuted:
            out["refuted"] = self.refuted
        if self.residual is not None:
            out["self_off"] = list(self.residual)
        return out
