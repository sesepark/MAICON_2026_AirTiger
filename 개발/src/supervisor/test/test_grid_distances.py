"""Grid.distances 의 빠른 길(미리 만든 이웃 목록)이 원래 계산과 같은 값을 내는지 본다. ROS 없이 돈다.

경기장 밖의 칸 하나를 임시 장애물(extra)로 주면 아무것도 막지 않으면서 원래 계산(칸마다 _neighbors)으로 돈다 → 둘을 견준다.
비용뿐 아니라 칸이 적히는 순서까지 같아야 한다 (정찰 계획이 이 값으로 자리와 순서를 정한다).
"""
import random
import unittest

from sim_world import load_layout, scout_grid
from supervisor.grid import Grid

OUTSIDE = {(-9, -9)}  # 경기장 밖의 칸 — 막는 것은 없고, 원래 계산으로 돌게만 한다


class GridDistancesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fmap, _truth = load_layout()

    def same(self, grid, start, limit=None):
        fast = grid.distances(start, limit=limit)
        slow = grid.distances(start, extra=OUTSIDE, limit=limit)
        self.assertEqual(list(fast.items()), list(slow.items()), (start, limit))
        return fast

    def test_fast_path_equals_the_original_everywhere(self):
        rng = random.Random(5)
        for grid in (Grid(self.fmap, radius=0.15), scout_grid(self.fmap)):
            for k in range(8):
                start = (rng.uniform(0.0, self.fmap.width), rng.uniform(0.0, self.fmap.height))
                reach = self.same(grid, start, limit=None if k % 2 else rng.uniform(0.5, 3.0))
                self.assertTrue(reach)
                self.assertTrue(all(isinstance(c, tuple) and len(c) == 2 for c in reach))

    def test_opening_and_closing_a_zone_rebuilds_the_table(self):
        grid = scout_grid(self.fmap)
        name = sorted(grid.zones)[0]
        start = (self.fmap.width / 2.0, self.fmap.height / 2.0)
        closed = self.same(grid, start)
        grid.set_zone(name, False)
        opened = self.same(grid, start)
        self.assertGreater(len(opened), len(closed))  # 구역을 열면 닿는 칸이 늘어난다
        grid.set_zone(name, True)
        self.assertEqual(list(self.same(grid, start).items()), list(closed.items()))


if __name__ == "__main__":
    unittest.main()
