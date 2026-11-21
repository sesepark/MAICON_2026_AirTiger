"""SLAM 노드와 주행 노드를 같이 돌리는 닫힌 루프 시험 (ROS 없이, 가상 경기장).

가상 로봇의 /odom · /odom/odom_raw · /scan → navigation_slam → /team/navigation/pose · map → navigation → /cmd_vel → 가상 로봇.
경기와 같은 순서(ready → running → 첫 지도 → 출발 칸에서 위치 찾기)로 시작해 Navigate 목표를 보낸다.
경기장은 실측 수치(612 × 500 cm, 고정 기둥 10개 [추정])에 임무 상자 8개를 놓은 것이다. 도구는 시험기록/web_drive/sim_world.py.
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '시험기록' / 'web_drive'))
import sim_world as W  # noqa: E402

REACHED, CONTROL_LOST, LOCALIZATION_LOST = 1, 5, 7


PHANTOM = (1.6, 2.5)   # B-2 출발 자리(0.2, 2.5) 바로 앞 1.4 m. 드론 지도에만 있는 상자 (실제로는 빈 바닥)


def make(slot='B-2', hidden=(), slip=0.85, lidar=True, odom_offset=1.0, boxes=W.PILLARS + W.SAMPLE_BOXES, map_boxes=None,
         slam_params=None):
    world = W.World(boxes=boxes, hidden=hidden, **W.ARENA)
    robot = W.SimRobot(world, W.slot_start_pose(slot), odom_offset=odom_offset, slip=slip)
    loop = W.ClosedLoop(world, robot, lidar=lidar, slam_params=slam_params)
    loop.start_from_slot(slot, W.base_region(slot), map_boxes=map_boxes)
    loop.run(1.5)
    return loop


def last_map_boxes(loop):
    """SLAM 노드가 마지막으로 내보낸 지도(/team/navigation/map)의 상자 [(x, y)]."""
    maps = []
    loop.listeners['map'] = maps.append
    loop.run(0.3)
    loop.listeners.pop('map')
    return [(b.x, b.y) for b in maps[-1].boxes]


def near(boxes, target, tolerance=0.06):
    return [b for b in boxes if math.hypot(b[0] - target[0], b[1] - target[1]) < tolerance]


class ClosedLoopTest(unittest.TestCase):
    def assert_arrived(self, loop, result, x, y, tolerance=0.10):
        self.assertIsNotNone(result, '목표가 제한 시간 안에 끝나지 않음')
        self.assertEqual(result.outcome, REACHED, result.detail)
        robot = loop.robot
        # 실제 자리는 허용 오차 + SLAM 오차 안
        self.assertLess(math.hypot(robot.x - x, robot.y - y), tolerance + 0.05)
        self.assertEqual(robot.collisions, 0)
        position_error, heading_error = loop.error()
        self.assertLess(position_error, 0.05)
        self.assertLess(heading_error, 0.06)

    def test_starts_from_the_slot_and_reaches_goals_across_the_arena(self):
        """출발 칸에서 위치를 잡고, 상자 사이를 지나 경기장 반대편까지 갔다가 돌아온다 (바퀴가 15% 헛도는 로봇)."""
        loop = make()
        position_error, heading_error = loop.error()
        self.assertLess(position_error, 0.03)
        self.assertLess(heading_error, 0.03)
        for x, y in ((2.2, 1.6), (5.2, 4.0), (0.8, 0.8)):
            self.assert_arrived(loop, loop.drive_to(x, y), x, y)
        self.assertGreater(loop.robot.min_clearance, W.ROBOT_RADIUS)
        # 가는 동안 위치가 계속 나왔다: /odom(10 Hz) 마다 한 번 이상
        duration = loop.poses[-1][0] - loop.poses[0][0]
        self.assertGreater(len(loop.poses) / duration, 9.5)
        self.assertLess(max(math.hypot(p[1] - p[4], p[2] - p[5]) for p in loop.poses), 0.08)

    def test_starts_from_a_red_slot_facing_the_other_way(self):
        loop = make(slot='R-3', odom_offset=-2.0)
        self.assert_arrived(loop, loop.drive_to(3.6, 1.6), 3.6, 1.6)

    def test_goes_around_a_box_that_is_not_on_the_map(self):
        """지도에 없는 상자가 곧은 길을 막고 있으면 라이다로 보고 돌아간다."""
        loop = make(hidden=((2.2, 3.0),), boxes=W.PILLARS)
        result = loop.drive_to(4.4, 3.0)
        self.assert_arrived(loop, result, 4.4, 3.0)
        self.assertGreater(loop.robot.min_clearance, W.ROBOT_RADIUS)

    def test_box_missing_from_the_map_is_used_for_matching_only(self):
        """지도에 없던 상자를 지나가며 여러 번 보면 SLAM 이 정합에 쓰는 상자로 더한다. 내보내는 지도에는 넣지 않는다 —
        주행 노드는 그것을 /scan 으로 직접 보고 지도에 없는 물체로 다룬다 (서 있는 다른 UGV 일 수도 있다)."""
        loop = make(hidden=((2.2, 3.0),), boxes=W.PILLARS)
        self.assert_arrived(loop, loop.drive_to(4.4, 3.0), 4.4, 3.0)
        self.assertEqual(len(near(loop.slam._fusion.added(), (2.2, 3.0))), 1, loop.slam._fusion.added())
        boxes = last_map_boxes(loop)
        self.assertEqual(near(boxes, (2.2, 3.0)), [])
        self.assertEqual(len(boxes), len(W.PILLARS))
        self.assertGreater(loop.robot.min_clearance, W.ROBOT_RADIUS)

    def test_removes_a_phantom_box_and_drives_straight_through(self):
        """드론 지도에만 있는 상자가 앞길에 있으면, 라이다가 그 속을 지나가는 것을 보고 지도에서 뺀다 → 돌아가지 않고 곧게 간다."""
        loop = make(boxes=W.PILLARS, map_boxes=W.PILLARS + (PHANTOM,))
        self.assertEqual(near(last_map_boxes(loop), PHANTOM), [])
        self.assert_arrived(loop, loop.drive_to(3.0, 2.5), 3.0, 2.5)
        self.assertLess(max(abs(p[5] - 2.5) for p in loop.poses), 0.12)   # 상자를 돌아갔다면 0.3 m 넘게 벗어난다

    def test_map_fusion_can_be_turned_off(self):
        """map_fusion:=false 이면 드론 지도를 그대로 낸다 (지도 고치기 전과 같다)."""
        loop = make(boxes=W.PILLARS, map_boxes=W.PILLARS + (PHANTOM,), slam_params={'map_fusion': False})
        position_error, _ = loop.error()
        self.assertLess(position_error, 0.03)
        self.assertEqual(len(near(last_map_boxes(loop), PHANTOM, 1e-6)), 1)

    def test_turns_to_the_requested_heading_on_arrival(self):
        loop = make(boxes=W.PILLARS)
        result = loop.drive_to(1.6, 3.0, yaw=math.pi / 2, yaw_tol=0.10)
        self.assert_arrived(loop, result, 1.6, 3.0)
        self.assertLess(abs(math.atan2(math.sin(loop.robot.yaw - math.pi / 2), math.cos(loop.robot.yaw - math.pi / 2))), 0.16)

    def test_stops_when_the_motion_allowance_is_withdrawn(self):
        loop = make(boxes=W.PILLARS)
        loop.allowed = True
        handle, thread, box = loop.send_goal(5.0, 2.4)
        start_x = loop.robot.x
        while loop.robot.x < start_x + 0.6 and loop.t < 1100:
            loop.step()
        self.assertGreater(loop.robot.x, start_x + 0.6)
        loop.allowed = False            # 시험 화면의 「정지」 · 화면 연결 끊김과 같다
        loop.run(1.5)
        thread.join(1.0)
        self.assertEqual(box['result'].outcome, CONTROL_LOST)
        self.assertEqual(loop.cmd, (0.0, 0.0))
        stopped_x = loop.robot.x
        loop.run(1.0)
        self.assertAlmostEqual(loop.robot.x, stopped_x, places=3)

    def test_does_not_move_without_a_position(self):
        """라이다가 없으면 위치를 못 잡는다 → 목표는 LOCALIZATION_LOST 로 끝나고 로봇은 움직이지 않는다."""
        loop = make(lidar=False)
        self.assertEqual(loop.poses, [])
        start = (loop.robot.x, loop.robot.y)
        result = loop.drive_to(3.0, 2.4, limit_s=10)
        self.assertEqual(result.outcome, LOCALIZATION_LOST)
        self.assertEqual((loop.robot.x, loop.robot.y), start)


if __name__ == '__main__':
    unittest.main()
