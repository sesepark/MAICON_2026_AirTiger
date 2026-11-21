"""고정 기둥 옆면을 믿는 구조로 쓰기 (2026-11-19 밤, S3). ROS 없이 돈다.

본선 경기장 9판의 스캔: 기둥이 덮은 벽 구간에 찍힌 점 4개(기둥은 늘 있다), 옆면(붙은 벽에 수직인 면)은 가정한 자리에서 +0.2 ~ +2.3 cm,
경기장 쪽 면은 +2 ~ +7 cm. 그래서 옆면만 벽으로 다시 보기(slam_node.wall_candidate)와 지도 고치기의 믿는 점(map_fusion)에 넣는다.
real_scans_1119_pillars.json = 16:55 판(201 s 부터 끝까지 x 가 약 0.45 m 틀림) 스캔 3개, 21:53 판(위치 맞음) 스캔 4개.
"""
import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

from navigation.drive.fieldmap import fixed_pillars, pillar_side_faces
from navigation.drive.geometry import wrap
from navigation.map_fusion import FusionParams, MapFusion, _face_distance
from test_node_split import load_slam

DATA = Path(__file__).resolve().parent / 'real_scans_1119_pillars.json'


def as_scan(r):
    ranges = [x if x else float('nan') for x in r['ranges']]
    return NS(header=NS(frame_id='base_lidar_link'), ranges=ranges, range_min=0.02, range_max=12.0,
              angle_min=r['angle_min'], angle_increment=r['angle_increment'])


def body_points(world, pose):
    """경기장 점 → 차체 좌표 (시험용 가짜 스캔)."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.asarray(world, dtype=float) - (x, y)
    return np.stack((c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]), axis=1)


class FacesTest(unittest.TestCase):
    def test_side_faces_of_the_six_wall_pillars(self):
        faces = pillar_side_faces(6.12, 4.8, fixed_pillars(6.12, 4.8))
        self.assertEqual(len(faces), 12)   # 옆 벽 4개 + 위아래 가운데 2개, 면 둘씩. 모서리 4개는 뺀다
        ys = sorted(round(at, 2) for axis, at, lo, hi in faces if axis == 'y' and lo < 1.0)
        self.assertEqual(ys, [1.39, 1.81, 2.99, 3.41])   # 왼쪽 벽 기둥 (1.6 · 3.2) ± 0.21
        xs = sorted(round(at, 2) for axis, at, lo, hi in faces if axis == 'x' and lo < 1.0)
        self.assertEqual(xs, [2.85, 3.27])                # 아래 벽 가운데 기둥 3.06 ± 0.21
        self.assertEqual(pillar_side_faces(3.0, 2.0, fixed_pillars(3.0, 2.0)), ())

    def test_face_distance_only_along_the_segment(self):
        faces = (('y', 1.39, 0.05, 0.46),)
        d = _face_distance(np.array([[0.30, 1.40], [0.70, 1.39], [0.02, 1.39]]), faces)
        self.assertAlmostEqual(d[0], 0.01)
        self.assertTrue(np.isinf(d[1]) and np.isinf(d[2]))   # 선분 밖 · 벽 바로 곁(외벽 점)


class WallCandidateTest(unittest.TestCase):
    """왼쪽 벽 앞, 두 기둥 사이(거점 자리)에서 벽만 보일 때: 외벽만으로는 벽을 따라 미끄러진 것을 모르고, 기둥 옆면이 있으면 안다."""

    @classmethod
    def setUpClass(cls):
        cls.slam, _ = load_slam()

    def scene(self):
        W, H = 6.12, 4.8
        pts = [(0.0, y) for y in np.arange(1.85, 2.95, 0.02)]                  # 두 기둥(1.6 · 3.2) 사이의 왼쪽 벽
        pts += [(x, 1.81) for x in np.arange(0.08, 0.40, 0.02)]               # 아래 기둥의 위 옆면
        pts += [(x, 2.99) for x in np.arange(0.08, 0.40, 0.02)]               # 위 기둥의 아래 옆면
        return W, H, np.array(pts)

    def test_pillar_faces_pin_the_position_along_the_wall(self):
        s = self.slam
        W, H, world = self.scene()
        truth = (0.85, 2.40, math.pi)
        points = body_points(world, truth)
        wrong = (0.85, 2.65, math.pi)        # 벽을 따라 0.25 m 미끄러진 자세
        _here, _there, where = s['wall_candidate'](points, wrong, W, H)
        self.assertAlmostEqual(where[1], truth[1], delta=0.06)
        self.assertAlmostEqual(where[0], truth[0], delta=0.06)
        # 외벽만(faces=())으로는 y 를 모른다 — 벽을 따라 옮겨도 점수가 거의 같아 의심 기준(WALL_GAIN 배)에 한참 못 미친다
        here0, there0, _where0 = s['wall_candidate'](points, wrong, W, H, faces=())
        self.assertLess(there0, s['WALL_GAIN'] * here0)
        # 외벽 + 옆면 점수: 진짜 자리가 확실히 낫다 (옆면 32점이 더해진다 — 의심 기준(1.5 배)을 넘으려면 벽 점 대비 옆면 점이 충분해야 한다)
        here, there, _ = s['wall_candidate'](points, wrong, W, H)
        self.assertGreater(there, here + 25)

    def test_switch_off(self):
        s = self.slam
        W, H, _world = self.scene()
        names = s['trusted_faces'].__globals__   # 노드 파라미터 trust_pillars:=false 가 바꾸는 모듈 값
        names['TRUST_PILLARS'] = False
        try:
            self.assertEqual(s['trusted_faces'](W, H), ())
        finally:
            names['TRUST_PILLARS'] = True
        self.assertEqual(len(s['trusted_faces'](W, H)), 12)


class RealScanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.slam, _ = load_slam()
        cls.scans = json.loads(DATA.read_text(encoding='utf-8'))['scans']

    def check(self, r):
        s = self.slam
        f = r['field']
        field = NS(width=f['width'], height=f['height'], box_size=f['box_size'], boxes=[NS(x=x, y=y) for x, y in f['boxes']])
        scan, pose = as_scan(r), tuple(r['pose'])
        suspect = s['wall_suspect'](np.array(s['scan_points'](scan, 2), dtype=float), pose, field.width, field.height)
        if suspect is None:
            return None
        second = s['confirm_wall_candidate'](field, scan, pose, suspect[2], boxes=[tuple(b) for b in f['boxes']])
        return None if second is None else (second[0] - pose[0], second[1] - pose[1], wrap(second[2] - pose[2]))

    def test_wrong_pose_at_1655_points_to_the_same_place_every_time(self):
        """280 · 290 · 298 s: 외벽만으로는 290 · 298 s 에 y 를 +0.09 ~ +0.12 로 잘못 가리켜 이어진 판정이 끊겼다 (그 구간 옮김 판정 0).
        기둥 옆면을 넣으면 세 스캔 모두 (+0.45, −0.05) 쪽 — 204 ~ 308 s 내내 같은 쪽을 가리킨다 (기록 재생: 옮김 판정 70 → 100번)."""
        shifts = [self.check(r) for r in self.scans if r['kind'] == 'x_off']
        for shift in shifts:
            self.assertIsNotNone(shift)
            self.assertAlmostEqual(shift[0], 0.45, delta=0.05)
            self.assertAlmostEqual(shift[1], -0.05, delta=0.05)

    def test_correct_pose_at_2153_is_left_alone(self):
        for r in self.scans:
            if r['kind'] == 'good':
                self.assertIsNone(self.check(r), r['t'])


class FusionTrustTest(unittest.TestCase):
    def test_pillar_face_points_count_as_trusted(self):
        """벽이 가려지고 확인 못 한 드론 상자와 기둥 옆면만 보일 때: 옆면을 믿으면 지도를 고치고, 끄면 전처럼 건너뛴다."""
        W, H = 6.12, 4.8
        pose = (0.85, 2.40, math.pi)
        world = [(x, 1.81) for x in np.arange(0.10, 0.40, 0.025)] + [(x, 2.99) for x in np.arange(0.10, 0.40, 0.025)]
        world += [(0.70, y) for y in np.arange(2.25, 2.55, 0.02)]    # 확인 전 드론 상자 (0.5, 2.4) 의 오른쪽 면
        points = body_points(np.array(world), pose)
        results = {}
        for trust in (True, False):
            fusion = MapFusion(FusionParams(observe_every=1, trust_pillar_faces=trust))
            fusion.set_map([(0.5, 2.4)], 0.4, W, H, fixed=fixed_pillars(W, H))
            fusion.observe(points, pose, int(1e9))
            results[trust] = fusion.last.get('skip', '')
        self.assertEqual(results[True], '')
        self.assertIn('믿기 어려움', results[False])


if __name__ == '__main__':
    unittest.main()
