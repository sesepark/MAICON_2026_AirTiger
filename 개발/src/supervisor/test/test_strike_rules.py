"""지정표적(M2) 발사 규칙 시험 (2026-11-19): 맞았는지 사진으로 확인 · 사진에서 표적 찾기 · 못 찾으면 쏘지 않고 정찰로.

조준 계산 자체(aim.py)는 test_aim.py 가 본다. 가상 카메라 · 레이저는 sim_world.SimMatch 의 참 모형.
"""
import unittest

import numpy as np

from sim_world import SimMatch
from supervisor import arena as A
from supervisor import laser_spot

HIT_M = 0.025  # 수광 돔의 반지름


def begin(m):
    """지도와 M1 해석이 온 뒤, 정찰을 기다리지 않고 표적 면을 알려 준다."""
    m.run(running_s=3.0, report_s=0.0)
    camp, kind = A.target_kind(m.order_truth["target"])
    m.mission.target_face = next(f for f in m.fmap.faces if m.truth[f.key] == kind and A.side_of(m.fmap, f.cx) == camp)
    return m


def strike(seconds=100.0, **world):
    return begin(SimMatch(mode="face", slot="B-1", **world)).run(running_s=seconds, report_s=0.0)


def notes(m):
    return [text for _t, text in m.mission.log]


class HitCheckTests(unittest.TestCase):
    def test_stops_after_the_first_shot_seen_on_the_receiver(self):
        m = strike()
        self.assertEqual(len(m.fires), 1, notes(m)[-6:])
        self.assertLess(m.beam_off[0], HIT_M)
        self.assertEqual(m.mission.target_hits, 1)
        self.assertTrue(m.mission.target_done)
        self.assertTrue(any("명중으로 보임" in text for text in notes(m)))

    def test_stays_still_while_the_laser_is_on_after_a_hit(self):
        """맞은 것으로 보여도 레이저가 켜진 5초 동안은 그 자리에 있다 (유효타격은 2초 연속 조사)."""
        m = strike(seconds=30.0)
        fired = m.fires[0][0]
        after = [(t, task) for t, task in m.tasks if t is not None and task != "strike"]
        left = min((t for t, _task in after if t + 3.0 > fired), default=None)   # m.tasks 의 시각은 경기 시각 (가상 시계 − 대기 3초)
        self.assertTrue(left is None or left + 3.0 >= fired + m.mission.p.laser_on_s - 0.2, (fired, m.tasks))

    def test_two_shots_when_the_spot_cannot_be_seen(self):
        """레이저 점이 사진에 안 보이면(돔에 가려짐 등) 맞았는지 모른다 → 조준이 맞은 채 2발 쏘고 그만둔다 (전에는 늘 3발)."""
        for world in ({"spot_seen": False}, {"can_watch": False}):
            m = strike(**world)
            self.assertEqual(len(m.fires), 2, (world, notes(m)[-6:]))
            self.assertTrue(all(off < HIT_M for off in m.beam_off), m.beam_off)
            self.assertEqual(m.mission.target_hits, 0)
            self.assertTrue(m.mission.target_done)

    def test_a_seen_miss_is_corrected_and_shot_again(self):
        """조준은 「맞음」 인데 켜진 사진의 레이저 점이 수광부에서 14픽셀 벗어나 있다 → 맞은 것으로 치지 않고, 잰 레이저 점으로 고쳐 다시 쏜다."""
        m = begin(SimMatch(mode="face", slot="B-1"))
        true_spot, calls = m.spot_pixel, []

        def spot():
            got = true_spot()
            calls.append(1)
            if len(calls) == 1 and got is not None:
                got["u"] += 14.0   # 첫 발: 레이저 점이 식의 자리보다 오른쪽에 찍혔다
            return got

        m.spot_pixel = spot
        m.run(running_s=100.0, report_s=0.0)
        self.assertGreaterEqual(len(m.fires), 2, notes(m)[-8:])
        self.assertLessEqual(len(m.fires), m.mission.p.target_shots)
        self.assertTrue(any("빗나간 것으로 보임" in text for text in notes(m)), notes(m)[-8:])
        self.assertFalse(any("명중으로 보임" in text and "(발사 1)" in text for text in notes(m)))


class SightSearchTests(unittest.TestCase):
    def test_finds_the_target_by_sight_when_the_pose_heading_is_wrong(self):
        """위치 값의 방향이 48° 틀리다 → 계산한 각도로 겨눈 사진에는 표적이 없다 (40° 까지는 aim.find_receiver 가 그 사진에서 찾는다). 팬을 돌려 가며 사진에서 표적 그림을 찾아 맞힌다."""
        for bias in (48.0, -48.0):
            m = strike(yaw_bias_deg=bias)
            self.assertTrue(any("사진에서 표적 그림을 찾음" in text for text in notes(m)), (bias, notes(m)[-8:]))
            self.assertGreaterEqual(len(m.fires), 1, (bias, notes(m)[-8:]))
            self.assertLess(m.beam_off[-1], HIT_M, (bias, m.beam_off, notes(m)[-8:]))
            self.assertEqual(m.mission.target_hits, 1)

    def test_does_not_search_when_the_first_photo_already_shows_the_target(self):
        m = strike()
        self.assertFalse(any("팬을 돌려 가며" in text for text in notes(m)))

    def test_no_shot_and_back_to_scouting_when_the_target_is_not_in_any_photo(self):
        """판독은 되는데 어느 사진에도 표적이 없다 (그 면이 표적이 아니거나 가려졌다) → 쏘지 않고 정찰로 돌아간다. 탄을 버리지 않는다."""
        m = strike(sensor_rate=0.0, aim_pictures=False, seconds=80.0)
        self.assertEqual(m.fires, [], notes(m)[-8:])
        self.assertTrue(any("정찰로 돌아간다" in text for text in notes(m)), notes(m)[-8:])
        self.assertFalse(m.mission.target_done)          # 정찰을 마친 뒤 한 번 더 해 본다
        self.assertTrue(m.mission._strike_wait)
        self.assertEqual(m.tasks[-1][1], "scout", m.tasks[-5:])

    def test_gives_up_for_good_after_the_second_try(self):
        m = begin(SimMatch(mode="face", slot="B-1", sensor_rate=0.0, aim_pictures=False))
        m.run(running_s=60.0, report_s=0.0)
        self.assertTrue(m.mission._strike_wait)
        first_spot = list(m.mission._strike_bad)
        m.mission.scout_done_at = 60.0                  # 정찰이 끝났다고 치면 다시 해 본다 (다른 자리에서)
        m.mission.scout.step = lambda pose, now: ("done",)
        m.run(running_s=140.0, report_s=0.0)
        self.assertEqual(m.fires, [])
        self.assertTrue(m.mission.target_done, notes(m)[-6:])
        self.assertEqual(len(m.mission._strike_bad), 2)
        self.assertNotEqual(m.mission._strike_bad[1], first_spot[0])

    def test_still_fires_by_the_computed_angle_when_perception_is_dead(self):
        """판독 답이 아예 없으면(perception 이 죽음) 사진으로 찾을 수도 없다 → 전처럼 계산한 각도로 쏜다."""
        m = strike(aim_answers=False, can_watch=False)
        self.assertEqual(len(m.fires), m.mission.p.target_shots)   # 사진으로 확인하지 못한 조준이므로 상한(3발)까지 쏜다
        self.assertTrue(any("계산한 각도로 쏜다" in text for text in notes(m)))


class LaserSpotTests(unittest.TestCase):
    @staticmethod
    def scene(spot=None, seed=0):
        rng = np.random.default_rng(seed)
        img = np.random.default_rng(0).integers(60, 120, size=(480, 640, 3)).astype(np.int16)
        img = np.clip(img + rng.integers(-4, 5, size=img.shape), 0, 255).astype(np.uint8)   # 사진마다 조금씩 다른 잡음
        img[100:300, 200:420] = (235, 235, 235)   # 밝은 면 (그림)
        if spot is not None:
            yy, xx = np.mgrid[0:480, 0:640]
            glow = np.exp(-((xx - spot[0]) ** 2 + (yy - spot[1]) ** 2) / (2 * 4.0 ** 2))
            img = img.astype(np.float32)
            img[:, :, 0] = np.clip(img[:, :, 0] + 200 * glow, 0, 255)
            img[:, :, 1] = np.clip(img[:, :, 1] - 60 * glow, 0, 255)
            img[:, :, 2] = np.clip(img[:, :, 2] - 60 * glow, 0, 255)
            img = img.astype(np.uint8)
        return img

    def test_finds_the_spot_on_a_dark_and_on_a_bright_surface(self):
        for where in ((333.0, 232.0), (300.0, 180.0)):   # 어두운 곳 · 밝은 면 위
            off = self.scene()
            got = laser_spot.find_spot([off], [self.scene(where, seed=1), self.scene(where, seed=2)])
            self.assertIsNotNone(got, where)
            self.assertAlmostEqual(got["u"], where[0], delta=1.5)
            self.assertAlmostEqual(got["v"], where[1], delta=1.5)

    def test_something_red_moving_elsewhere_is_not_the_spot(self):
        """2026-11-19 실차: 레이저가 사진에 안 찍힌 발에서, 상자 위로 보이는 사람 머리(28 픽셀 위)가 레이저 점으로 잡혔다 → 계산한 자리의 둘레만 본다."""
        off, laser, head = self.scene(), (333.0, 232.0), (328.0, 204.0)
        moved = self.scene(head, seed=1)                      # 레이저는 안 찍히고, 뒤의 붉은 것이 움직였다
        self.assertIsNotNone(laser_spot.find_spot([off], [moved]))                  # 사진 전체에서 찾으면 그것이 잡힌다
        self.assertIsNone(laser_spot.find_spot([off], [moved], near=laser))         # 계산한 자리의 둘레만 보면 「못 찾음」
        both = self.scene(laser, seed=2)
        both[190:216, 318:340] = moved[190:216, 318:340]      # 레이저도 찍히고 뒤의 것도 움직였다
        got = laser_spot.find_spot([off], [both], near=laser)
        self.assertIsNotNone(got)
        self.assertAlmostEqual(got["u"], laser[0], delta=1.5)
        self.assertAlmostEqual(got["v"], laser[1], delta=1.5)

    def test_no_spot_when_nothing_changed(self):
        self.assertIsNone(laser_spot.find_spot([self.scene()], [self.scene(seed=3)]))
        self.assertIsNone(laser_spot.find_spot([], [self.scene()]))


if __name__ == "__main__":
    unittest.main()
