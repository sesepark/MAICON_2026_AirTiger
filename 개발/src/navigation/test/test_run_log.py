"""경기 기록(navigation/run_log.py)과 두 노드의 기록 연결 시험. rclpy 없이 돈다.

경기 중에는 로봇 안에서 /team/navigation/debug 를 볼 수 없어서(로그_파일/1118_팀테스트_경기장1_R2/상황.md 5절),
TEAM_LOG_DIR 이 있으면 두 노드가 <기록 폴더>/navigation/debug.jsonl · navigation_slam/debug.jsonl 에 내부 상태를 쓴다 (2026-11-19).
"""
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from navigation.run_log import RunLog  # noqa: E402
import test_node_split as fake  # noqa: E402  (이름을 바로 가져오면 unittest 가 DrivingNodeTest 를 여기서 한 번 더 돌린다)


def read_lines(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f]


class RunLogTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_off_without_session_or_when_disk_is_full(self):
        self.assertFalse(RunLog('navigation', environ={}).enabled)
        off = RunLog('navigation', environ={'TEAM_LOG_DIR': self.tmp.name, 'TEAM_LOG_OFF': '1'})
        self.assertFalse(off.enabled)
        self.assertFalse(off.write({'a': 1}))
        self.assertTrue(off.flush())
        off.close()
        self.assertEqual(os.listdir(self.tmp.name), [])

    def test_writes_one_json_line_per_record_under_the_program_folder(self):
        log = RunLog('navigation', environ={'TEAM_LOG_DIR': self.tmp.name})
        self.addCleanup(log.close)
        self.assertTrue(log.enabled)
        self.assertEqual(Path(log.path), Path(self.tmp.name) / 'navigation' / 'debug.jsonl')
        for k in range(5):
            self.assertTrue(log.write({'event': 'state', 'k': k, 'text': '한글 그대로'}))
        self.assertTrue(log.flush())
        lines = read_lines(log.path)
        self.assertEqual([r['k'] for r in lines], list(range(5)))
        self.assertEqual(lines[0]['text'], '한글 그대로')
        self.assertEqual(log.lines, 5)

    def test_stops_at_the_size_limit_with_a_marker(self):
        log = RunLog('navigation_slam', environ={'TEAM_LOG_DIR': self.tmp.name}, max_bytes=200)
        self.addCleanup(log.close)
        for k in range(50):
            log.write({'k': k, 'pad': 'x' * 20})
        log.flush()
        lines = read_lines(log.path)
        self.assertEqual(lines[-1]['log'], 'truncated')
        self.assertLessEqual(lines[-1]['bytes'], 200)
        self.assertTrue(log.truncated)
        self.assertFalse(log.enabled)
        self.assertFalse(log.write({'k': 99}))

    def test_light_mode_uses_the_smaller_limit(self):
        log = RunLog('navigation', environ={'TEAM_LOG_DIR': self.tmp.name, 'TEAM_LOG_LIGHT': '1'})
        self.addCleanup(log.close)
        self.assertTrue(log.light)
        self.assertLess(log.max_bytes, RunLog('x', environ={}).max_bytes)

    def test_unwritable_folder_is_quietly_off(self):
        blocker = Path(self.tmp.name) / 'file'
        blocker.write_text('')
        log = RunLog('navigation', environ={'TEAM_LOG_DIR': str(blocker)})   # 폴더 자리에 파일 → 만들 수 없음
        self.assertFalse(log.enabled)
        self.assertFalse(log.write({'k': 1}))


class DrivingNodeLogTest(unittest.TestCase):
    """team_node.py 가 켰을 때(TEAM_LOG_DIR 있음) 주행 노드가 목표 · 단계 · 결과 · 상태를 남긴다."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = patch.dict(os.environ, {'TEAM_LOG_DIR': self.tmp.name})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop('TEAM_LOG_OFF', None)
        self.drive = fake.DrivingNodeTest('test_wiring')
        self.drive.setUp()
        self.addCleanup(self.drive.doCleanups)
        self.addCleanup(self.drive.node.run_log.close)

    def test_goal_phases_result_and_state_are_recorded(self):
        node = self.drive.node
        self.assertTrue(node.run_log.enabled)
        result, _handle, _pose, _sent = self.drive.run_goal(
            fake.request(5.0, 2.5, yaw=math.pi / 2), during=lambda k, h: node._write_log())
        self.assertEqual(result.outcome, 1, result.detail)   # REACHED
        node._write_log()   # 쉬는 동안: log_idle_s 마다만
        self.assertTrue(node.run_log.flush())
        lines = read_lines(Path(self.tmp.name) / 'navigation' / 'debug.jsonl')
        events = [r['event'] for r in lines]
        self.assertEqual(events[0], 'start')
        self.assertIn('max_v', lines[0]['params'])
        self.assertEqual(lines[1]['event'], 'map')
        self.assertEqual(lines[1]['boxes'], [[3.06, 2.5]])
        goal = next(r for r in lines if r['event'] == 'goal')
        self.assertEqual(goal['goal'][:2], [5.0, 2.5])
        self.assertTrue(goal['accepted'])
        phases = [r['phase'] for r in lines if r['event'] == 'phase']
        self.assertIn('driving', phases)
        self.assertEqual(phases[-1], 'done')
        done = next(r for r in lines if r['event'] == 'result')
        self.assertEqual((done['no'], done['outcome']), (goal['no'], 'REACHED'))
        states = [r for r in lines if r['event'] == 'state']
        busy = [r for r in states if 'job' in r]
        self.assertTrue(busy)
        self.assertIn('path', busy[0]['job'])
        self.assertNotIn('obstacles', busy[0])   # 큰 칸 목록 대신 칸 수만
        self.assertIn('obstacle_cells', busy[0])

    def test_idle_state_is_written_every_log_idle_s(self):
        node = self.drive.node
        node._receive_map(SimpleNamespace(header=SimpleNamespace(frame_id='arena'), width=6.12, height=5.0, box_size=0.4,
                                          boxes=[fake.point(3.06, 2.5)], vehicles=[]))
        for _ in range(16):   # 2 Hz 로 8 s
            node._write_log()
        node.run_log.flush()
        states = [r for r in read_lines(node.run_log.path) if r['event'] == 'state']
        self.assertEqual(len(states), 4)   # log_idle_s 2 s 마다


class SlamNodeLogTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = patch.dict(os.environ, {'TEAM_LOG_DIR': self.tmp.name})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop('TEAM_LOG_OFF', None)

    def test_state_and_map_are_recorded(self):
        module, _ = fake.load_slam()
        slam = module['SlamNode']()
        self.addCleanup(slam.run_log.close)
        self.assertTrue(slam.run_log.enabled)
        field = fake.FieldMap(5)
        field.boxes = [fake.point(1.0, 1.0), fake.point(2.0, 2.0)]
        slam._receive_perception_map(field)
        slam._write_log()
        self.assertTrue(slam.run_log.flush())
        lines = read_lines(Path(self.tmp.name) / 'navigation_slam' / 'debug.jsonl')
        self.assertEqual([r['event'] for r in lines], ['start', 'map', 'state'])
        self.assertEqual(lines[1]['boxes'], 2)
        self.assertEqual(lines[2]['node'], 'navigation_slam')
        self.assertTrue(lines[2]['map'])

    @staticmethod
    def scan(stamp, ranges=(1.23456, float('inf'), float('nan'), 0.5)):
        sec = int(stamp)
        return SimpleNamespace(header=SimpleNamespace(frame_id='base_lidar_link', stamp=SimpleNamespace(sec=sec, nanosec=int(round((stamp - sec) * 1e9)))),
                               angle_min=-3.14159265, angle_increment=0.01244444, range_min=0.05, range_max=12.0, ranges=list(ranges))

    def test_raw_scan_is_recorded_once_per_second(self):
        """스캔 원본(event scan_raw)을 log_raw_scan_s(기본 1초)마다 한 줄. inf · NaN 은 0, 거리는 mm 까지. 지도가 없어 위치를 못 낼 때 pose 는 null."""
        module, _ = fake.load_slam()
        slam = module['SlamNode']()
        self.addCleanup(slam.run_log.close)
        self.assertEqual(slam.log_raw_scan_s, 1.0)
        for k in range(25):                       # 10 Hz 로 2.5초
            slam._receive_scan(self.scan(100.0 + 0.1 * k))
            slam._clock.t += 0.1
        self.assertTrue(slam.run_log.flush())
        raw = [r for r in read_lines(slam.run_log.path) if r['event'] == 'scan_raw']
        self.assertEqual([r['stamp'] for r in raw], [100.0, 101.0, 102.0])
        self.assertEqual(raw[0]['ranges'], [1.235, 0, 0, 0.5])
        self.assertEqual((raw[0]['angle_min'], raw[0]['angle_increment']), (-3.141593, 0.0124444))
        self.assertIsNone(raw[0]['pose'])
        self.assertEqual(set(raw[0]), {'event', 't', 'stamp', 'angle_min', 'angle_increment', 'ranges', 'pose'})

    def test_raw_scan_carries_the_pose_being_published(self):
        module, _ = fake.load_slam()
        slam = module['SlamNode']()
        self.addCleanup(slam.run_log.close)

        def pose(x):
            return SimpleNamespace(position=SimpleNamespace(x=x, y=0.0), orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0))

        def odometry(sec, x):
            return SimpleNamespace(header=SimpleNamespace(frame_id='odom', stamp=fake.FieldMap(sec).header.stamp),
                                   child_frame_id='base_footprint', pose=SimpleNamespace(pose=pose(x)))

        slam._receive_odometry(odometry(1, 4.0))
        slam._delta = 0.0
        slam._receive_odometry(odometry(2, 5.0))
        field = fake.FieldMap(1)
        field.boxes = []
        field.vehicles = [SimpleNamespace(role=1, SELF=1, pose=pose(10.0))]
        slam._receive_perception_map(field)       # 우리 차가 (10, 0) 에 있던 때부터 /odom 으로 1 m 더 갔다 → 지금 (11, 0)
        slam._receive_scan(self.scan(50.0))       # 이 스캔 시각의 /odom 은 없다 → 기록만 하고 정합은 하지 않는다
        self.assertTrue(slam.run_log.flush())
        raw = [r for r in read_lines(slam.run_log.path) if r['event'] == 'scan_raw']
        self.assertEqual(len(raw), 1)
        self.assertEqual(raw[0]['pose'], [11.0, 0.0, 0.0])

    def test_raw_scan_can_be_turned_off_and_never_breaks_the_scan_callback(self):
        module, _ = fake.load_slam()
        fake.FakeNode.parameters = {'log_raw_scan_s': 0.0}
        self.addCleanup(setattr, fake.FakeNode, 'parameters', {})
        slam = module['SlamNode']()
        self.addCleanup(slam.run_log.close)
        slam._receive_scan(self.scan(100.0))
        slam.log_raw_scan_s = 1.0
        broken = self.scan(101.0)
        broken.ranges = None                      # 기록하다 예외가 난다 (스캔 형식이 이상함)
        slam._receive_scan(broken)                # 그래도 스캔 처리는 평소대로 끝난다 (지도가 없어 여기서 돌아온다)
        self.assertEqual(slam._scan_note['result'], '지도 없음')
        slam.get_logger().warning.assert_called_once()
        self.assertTrue(slam.run_log.flush())
        self.assertEqual([r['event'] for r in read_lines(slam.run_log.path)], ['start'])

    def test_lidar_boxes_are_recorded_when_the_published_list_changes(self):
        module, _ = fake.load_slam()
        slam = module['SlamNode']()
        self.addCleanup(slam.run_log.close)
        slam._publish_lidar_boxes()               # 처음 낸 목록 (빈 것)
        slam._publish_lidar_boxes()               # 그대로 → 다시 적지 않는다
        slam._fusion.added = lambda: [(1.0, 4.8)]
        slam._publish_lidar_boxes(changed_only=True)
        slam._publish_lidar_boxes()
        slam._fusion.added = lambda: []
        slam._publish_lidar_boxes()
        self.assertTrue(slam.run_log.flush())
        events = [r for r in read_lines(slam.run_log.path) if r['event'] == 'lidar_boxes']
        self.assertEqual([r['boxes'] for r in events], [[], [[1.0, 4.8]], []])

    def test_nothing_is_written_without_a_session(self):
        os.environ.pop('TEAM_LOG_DIR', None)
        module, _ = fake.load_slam()
        slam = module['SlamNode']()
        self.assertFalse(slam.run_log.enabled)
        slam._write_log()
        self.assertEqual(os.listdir(self.tmp.name), [])


if __name__ == '__main__':
    unittest.main()
