"""라이다 스캔으로 위아래 벽 사이 재기 (arena_height). 가짜 스캔: 네모난 경기장 + 기둥 + 상자, 출발 칸에 선 로봇 (뒤쪽 90° 는 안 보임)."""
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from perception import arena_height  # noqa: E402


def square(cx, cy, half):
    return [(cx - half, cy - half, cx + half, cy - half), (cx + half, cy - half, cx + half, cy + half),
            (cx + half, cy + half, cx - half, cy + half), (cx - half, cy + half, cx - half, cy - half)]


def room(width, height, boxes=(), pillars=True):
    """벽 4개 + 고정 기둥(양쪽 벽의 1/3 · 2/3 높이, 위아래 가운데) + 상자의 변들 [(x1, y1, x2, y2)]."""
    segs = [(0, 0, width, 0), (width, 0, width, height), (width, height, 0, height), (0, height, 0, 0)]
    if pillars:
        for cx, cy in ((0.2, height / 3), (0.2, 2 * height / 3), (width - 0.2, height / 3), (width - 0.2, 2 * height / 3), (width / 2, 0.2), (width / 2, height - 0.2)):
            segs += square(cx, cy, 0.2)
    for cx, cy in boxes:
        segs += square(cx, cy, 0.2)
    return np.array(segs, dtype=float)


def scan(segs, pose, blind_deg=90.0, noise=0.008, seed=1, count=504):
    """pose(x, y, yaw) 에서 본 스캔 (ranges, angle_min, angle_increment). 차체 뒤쪽 blind_deg 는 0 (안 보임)."""
    rng = np.random.default_rng(seed)
    x, y, yaw = pose
    inc = 2 * math.pi / count
    out = np.zeros(count)
    x1, y1, x2, y2 = segs.T
    for i in range(count):
        a = i * inc  # 라이다 각도 = 차체 기준
        if abs(math.atan2(math.sin(a - math.pi), math.cos(a - math.pi))) < math.radians(blind_deg / 2):
            continue
        dx, dy = math.cos(yaw + a), math.sin(yaw + a)
        ex, ey = x2 - x1, y2 - y1
        den = dx * ey - dy * ex
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((x1 - x) * ey - (y1 - y) * ex) / den
            u = ((x1 - x) * dy - (y1 - y) * dx) / den
        hit = (np.abs(den) > 1e-12) & (t > 0.05) & (u >= 0) & (u <= 1)
        if hit.any():
            out[i] = float(t[hit].min()) + float(rng.normal(0, noise))
    return out, 0.0, inc


BOXES = [(1.46, 3.9), (1.86, 3.9), (4.3, 4.0), (4.7, 4.0), (3.06, 2.4), (1.86, 2.4), (4.3, 2.4), (2.86, 1.3), (3.26, 1.3), (1.4, 0.8), (4.2, 0.8), (4.6, 0.8)]


class WallSpanTests(unittest.TestCase):
    def measure(self, height, pose, boxes=BOXES, seeds=(1, 2, 3, 4, 5)):
        vote = arena_height.HeightVote()
        segs = room(6.12, height, [(x, y * height / 4.8) for x, y in boxes])
        for seed in seeds:
            vote.add(arena_height.wall_spans(*scan(segs, pose, seed=seed)))
        return vote

    def test_from_every_start_base_the_answer_is_right_or_missing_never_wrong(self):
        decided = 0
        for height in (4.80, 5.00):
            for row in (1, 2, 3):
                for side in ("left", "right"):
                    y = height * (2 * (3 - row) + 1) / 6.0
                    pose = (0.22, y, 0.03) if side == "left" else (6.12 - 0.22, y, math.pi - 0.04)
                    vote = self.measure(height, pose)
                    answer = vote.decide()
                    self.assertIn(answer, (None, height), "높이 %.2f · %d번 줄 · %s: 잰 값 %s" % (height, row, side, vote.values))
                    if answer is not None:
                        decided += 1
                        self.assertLess(abs(vote.median() - height), 0.03)
        self.assertGreaterEqual(decided, 4)  # 건너편 벽이 상자에 다 가려진 자리에서는 답이 없을 수 있다 (그때는 카메라로 고른 값)

    def test_random_layouts_never_decide_the_wrong_size(self):
        import random

        rng = random.Random(3)
        right = wrong = 0
        for trial in range(40):
            height = rng.choice((4.80, 5.00))
            boxes = set()
            while len(boxes) < rng.randint(8, 16):  # 상자는 벽에서 0.4 m 넘게 떨어져 있다 (2026-11-19 주최측)
                boxes.add((round(rng.randrange(6, 26) * 0.2 + 0.06, 2), round(rng.randrange(3, int(height / 0.2) - 3) * 0.2 + (0.0 if height == 4.8 else 0.1), 2)))
            segs = room(6.12, height, sorted(b for b in boxes if 0.9 < b[0] < 5.3))
            row, left = rng.choice((1, 2, 3)), rng.random() < 0.5
            y = height * (2 * (3 - row) + 1) / 6.0 + rng.uniform(-0.15, 0.15)
            pose = (0.2 + rng.uniform(-0.05, 0.08), y, rng.uniform(-0.2, 0.2)) if left else (6.12 - 0.2 - rng.uniform(-0.05, 0.08), y, math.pi + rng.uniform(-0.2, 0.2))
            vote = arena_height.HeightVote()
            for seed in range(5):
                vote.add(arena_height.wall_spans(*scan(segs, pose, seed=seed + trial * 10)))
            answer = vote.decide()
            right += answer == height
            wrong += answer is not None and answer != height
        self.assertEqual(wrong, 0)
        self.assertGreaterEqual(right, 30)

    def test_no_answer_between_the_two_sizes_or_when_a_wall_is_hidden(self):
        vote = self.measure(4.90, (0.22, 2.45, 0.0))
        self.assertIsNone(vote.decide(), vote.values)  # 두 값의 중간 → 정하지 않는다
        # 위쪽 벽 앞을 상자 줄이 다 가린다 → 위쪽 벽이 안 보인다 → 답 없음 (상자 줄을 벽으로 잘못 보지 않는다)
        wall_of_boxes = [(0.6 + 0.4 * i, 4.2) for i in range(14)]
        vote = self.measure(4.80, (0.22, 2.4, 0.0), boxes=wall_of_boxes)
        self.assertNotEqual(vote.decide(), 5.00)
        self.assertTrue(vote.decide() is None or vote.decide() == 4.80)

    def test_robot_turned_at_an_angle_still_measures(self):
        vote = self.measure(4.80, (0.25, 2.4, math.radians(25.0)))
        self.assertEqual(vote.decide(), 4.80, vote.values)

    def test_empty_or_short_scans_give_nothing(self):
        self.assertEqual(arena_height.wall_spans([0.0] * 504, 0.0, 2 * math.pi / 504), [])
        self.assertEqual(arena_height.wall_spans([], 0.0, 0.01), [])
        vote = arena_height.HeightVote()
        vote.add([])
        self.assertIsNone(vote.decide())


if __name__ == "__main__":
    unittest.main()
