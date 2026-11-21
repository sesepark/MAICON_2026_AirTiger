"""두 실행 노드(navigation_slam · navigation)의 ROS 연결 시험. rclpy 없이 가짜 모듈로 돈다.

- SLAM 노드: 지도 발행자 · perception 지도 구독자 (지도 QoS 는 transient_local)
- 주행 노드: 구독 · 발행 토픽, 그리고 가짜 시계로 액션 실행 전체(도착 · 교체 · 제어 허용 끊김 · 잘못된 목표)
"""
import importlib
import math
import runpy
import sys
import threading
import time
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

importlib.import_module('numpy')  # 가짜 모듈로 sys.modules 를 바꾸기 전에 불러 둔다 (patch.dict 가 끝나며 지우면 두 번 불러와 깨진다)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NODE_DIRECTORY = PACKAGE_ROOT / 'navigation'
MAP_TOPIC = '/team/navigation/map'
ROS_MODULES = (
    'rclpy', 'rclpy.node', 'rclpy.qos', 'rclpy.action', 'rclpy.callback_groups', 'rclpy.executors',
    'geometry_msgs', 'geometry_msgs.msg', 'nav_msgs', 'nav_msgs.msg', 'sensor_msgs', 'sensor_msgs.msg',
    'std_msgs', 'std_msgs.msg', 'team_interfaces', 'team_interfaces.msg', 'team_interfaces.action',
)


class FieldMap:
    def __init__(self, sec=0):
        self.vehicles = []
        self.header = SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=0))


class FieldVehicle(SimpleNamespace):
    """team_interfaces/FieldVehicle 흉내: 상수와 기본 pose."""
    UNKNOWN, SELF, OTHER = 0, 1, 2

    def __init__(self, role=0, pose=None):
        super().__init__(role=role, pose=pose or SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)))


class AutoModule(ModuleType):
    """없는 이름을 물으면 Mock 을 만들어 주는 가짜 모듈."""

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        value = Mock(name='%s.%s' % (self.__name__, name))
        setattr(self, name, value)
        return value


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def now(self):
        return SimpleNamespace(nanoseconds=int(self.t * 1e9), to_msg=Mock)


class FakeNode:
    clock = None
    distinct_publishers = False  # 주행 노드 시험: 토픽마다 다른 발행자
    parameters = {}  # 시험에서 바꿀 ROS 파라미터 (이름 → 값). 없으면 기본값

    def __init__(self, name):
        self.name = name
        self.create_publisher = Mock()
        if FakeNode.distinct_publishers:
            self.create_publisher.side_effect = lambda *a, **k: Mock(name='pub %s' % a[1])
        self.create_subscription = Mock()
        self.create_timer = Mock()
        self.destroy_node = Mock()
        self._logger = Mock()
        self._clock = FakeNode.clock or FakeClock()

    def declare_parameter(self, name, default):
        return SimpleNamespace(value=FakeNode.parameters.get(name, default))

    def get_logger(self):
        return self._logger

    def get_clock(self):
        return self._clock


def fake_ros():
    modules = {name: AutoModule(name) for name in ROS_MODULES}
    ros = modules['rclpy']
    ros.init = Mock()
    ros.spin = Mock()
    ros.try_shutdown = Mock()
    ros.ok = Mock(return_value=True)
    modules['rclpy.node'].Node = FakeNode
    modules['rclpy.qos'].qos_profile_sensor_data = object()
    modules['nav_msgs.msg'].Odometry = type('Odometry', (), {})
    modules['sensor_msgs.msg'].LaserScan = type('LaserScan', (), {})
    modules['team_interfaces.msg'].ServerState = type('ServerState', (), {})
    modules['team_interfaces.msg'].FieldVehicle = FieldVehicle
    modules['team_interfaces.msg'].FieldMap = FieldMap
    modules['rclpy.executors'].ExternalShutdownException = type('ExternalShutdownException', (Exception,), {})
    # 메시지는 만들 때마다 새 객체여야 한다 (Mock() 의 반환값은 하나를 같이 쓴다 → 두 목표의 결과가 섞인다)
    modules['team_interfaces.action'].Navigate = SimpleNamespace(Result=Mock, Feedback=Mock)
    for name in ('PoseStamped', 'Twist'):
        setattr(modules['geometry_msgs.msg'], name, Mock)
    # 지도를 고쳐 내보낼 때 새로 만드는 메시지 (만들 때마다 새 객체, 넣은 값을 그대로 읽는다)
    modules['geometry_msgs.msg'].Point = SimpleNamespace
    # 라이다로 찾은 상자 목록(/team/navigation/lidar_boxes)의 메시지 — 만들 때마다 새 객체
    modules['geometry_msgs.msg'].PoseArray = lambda: SimpleNamespace(header=SimpleNamespace(frame_id='', stamp=None), poses=[])
    modules['geometry_msgs.msg'].Pose = lambda: SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                                                                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0))
    modules['team_interfaces.msg'].FieldFace = SimpleNamespace
    modules['std_msgs.msg'].String = Mock   # String(data=...) 의 data 를 그대로 읽을 수 있게
    modules['team_interfaces.msg'].NodeStatus = Mock(INITIALIZING=0, READY=1, DEGRADED=2, ERROR=3, side_effect=lambda: Mock())
    return modules


def load_slam():
    modules = fake_ros()
    sys.path.insert(0, str(PACKAGE_ROOT))   # slam_node 가 navigation.run_log · navigation.map_fusion 을 부른다 (ROS 와 무관한 모듈)
    try:
        with patch.dict(sys.modules, modules):
            return runpy.run_path(str(NODE_DIRECTORY / 'slam_node.py')), modules
    finally:
        sys.path.remove(str(PACKAGE_ROOT))


def load_driving():
    modules = fake_ros()
    sys.path.insert(0, str(PACKAGE_ROOT))
    try:
        with patch.dict(sys.modules, modules):
            for name in [n for n in sys.modules if n == 'navigation.driving_node']:
                del sys.modules[name]
            import importlib
            module = importlib.import_module('navigation.driving_node')
            sys.modules.pop('navigation.driving_node', None)
            return module, modules
    finally:
        sys.path.remove(str(PACKAGE_ROOT))


def point(x, y):
    return SimpleNamespace(x=x, y=y, z=0.0)


def quat(yaw):
    return SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))


def field_map(frame='arena', boxes=((3.06, 2.5),)):
    return SimpleNamespace(header=SimpleNamespace(frame_id=frame), width=6.12, height=5.0, box_size=0.4,
                           boxes=[point(*b) for b in boxes], vehicles=[])


def pose_msg(x, y, yaw, stamp):
    sec = int(stamp)
    return SimpleNamespace(header=SimpleNamespace(frame_id='arena', stamp=SimpleNamespace(sec=sec, nanosec=int((stamp - sec) * 1e9))),
                           pose=SimpleNamespace(position=point(x, y), orientation=quat(yaw)))


def control(allowed=True, run_id='run-1'):
    return SimpleNamespace(motion_allowed=allowed, lease_ms=1000, run_id=run_id, reason='')


def request(x, y, yaw=0.0, mode=0, yaw_tol=0.1, run_id='run-1', timeout_ms=60000):
    return SimpleNamespace(run_id=run_id, mode=mode, ignore_yaw=False, position_tolerance_m=0.1, yaw_tolerance_rad=yaw_tol,
                           execution_timeout_ms=timeout_ms,
                           goal=SimpleNamespace(header=SimpleNamespace(frame_id='arena'), pose=SimpleNamespace(position=point(x, y), orientation=quat(yaw))))


class FakeGoalHandle:
    def __init__(self, req):
        self.request = req
        self.is_cancel_requested = False
        self.succeed = Mock()
        self.abort = Mock()
        self.canceled = Mock()
        self.publish_feedback = Mock()


class NodeSplitTest(unittest.TestCase):
    def setUp(self):
        FakeNode.distinct_publishers = True
        self.addCleanup(setattr, FakeNode, 'distinct_publishers', False)

    def test_slam_owns_map_publisher(self):
        module, modules = load_slam()
        slam = module['SlamNode']()

        self.assertEqual(slam.name, 'navigation_slam')
        args = slam.create_publisher.call_args_list[0].args
        self.assertEqual(args[:2], (FieldMap, MAP_TOPIC))
        self.assertIs(args[2], module['MAP_QOS'])
        qos_kwargs = modules['rclpy.qos'].QoSProfile.call_args.kwargs
        self.assertIs(qos_kwargs['durability'], modules['rclpy.qos'].DurabilityPolicy.TRANSIENT_LOCAL)
        self.assertIs(qos_kwargs['reliability'], modules['rclpy.qos'].ReliabilityPolicy.RELIABLE)
        self.assertIsNot(slam.map_publisher, slam.pose_publisher)
        self.assertIsNone(slam.latest_perception_map)
        slam.create_subscription.assert_any_call(
            FieldMap, '/team/perception/field_map', slam._receive_perception_map, 1
        )
        callback = slam._receive_perception_map
        samples = (FieldMap(1), FieldMap(2))
        for sample in samples:
            callback(sample)
            self.assertIs(slam.latest_perception_map, sample)
        self.assertEqual(slam.map_publisher.publish.call_count, 2)
        self.assertEqual(
            [call.args[0] for call in slam.map_publisher.publish.call_args_list],
            list(samples),
        )

    def test_slam_entry_point_releases_node_after_spin_error(self):
        module, modules = load_slam()
        ros = modules['rclpy']
        ros.spin.side_effect = RuntimeError('spin failed')
        with self.assertRaisesRegex(RuntimeError, 'spin failed'):
            module['main']()
        ros.init.assert_called_once_with()
        node = ros.spin.call_args.args[0]
        node.destroy_node.assert_called_once_with()
        ros.try_shutdown.assert_called_once_with()

    def test_slam_uses_observation_time_and_bounds_odometry(self):
        module, _ = load_slam()
        slam = module['SlamNode']()

        def pose(x):
            return SimpleNamespace(
                position=SimpleNamespace(x=x, y=0.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )

        def odometry(sec, x):
            return SimpleNamespace(
                header=SimpleNamespace(frame_id='odom', stamp=FieldMap(sec).header.stamp),
                child_frame_id='base_footprint',
                pose=SimpleNamespace(pose=pose(x)),
            )

        slam._receive_odometry(odometry(1, 4.0))
        # odom 축 → 경기장 축 각도(delta)를 이미 맞춘 상태. 모르면 5 cm 넘게 이어 붙이지 않는다 (test_slam_tracking)
        # (첫 /odom 은 odom 좌표계가 바뀐 것으로 보고 delta 를 지우므로 그 뒤에 넣는다)
        slam._delta = 0.0
        slam._receive_odometry(odometry(2, 5.0))
        field = FieldMap(1)
        field.vehicles = [SimpleNamespace(role=1, SELF=1, pose=pose(10.0))]
        slam._receive_perception_map(field)
        output = slam.map_publisher.publish.call_args.args[0]
        self.assertEqual(output.header.stamp.sec, 2)
        self.assertEqual(output.vehicles[0].pose.position.x, 11.0)
        self.assertEqual(field.vehicles[0].pose.position.x, 10.0)
        slam._receive_odometry(odometry(4, 7.0))
        slam.map_publisher.publish.assert_called_once()

    def test_fused_map_renumbers_boxes_and_faces(self):
        """라이다로 뺀 상자는 상자 · 면에서 빠지고, 남은 상자의 면은 새 번호를 쓴다 (더한 상자는 내보내지 않는다)."""
        module, _ = load_slam()
        field = FieldMap(1)
        field.box_size = 0.4
        field.boxes = [point(1.0, 1.0), point(2.0, 2.0), point(3.0, 3.0)]
        field.faces = [SimpleNamespace(box=i, side=side, cx=0.0, cy=0.0, nx=0.0, ny=0.0, sector='S%d' % i, image_ref='img%d%s' % (i, side))
                       for i in range(3) for side in ('E', 'W')]
        kept = [(1.0, 1.0, 0), (3.0, 3.0, 2)]   # 1번(2.0, 2.0)은 없는 상자
        output = module['fused_map'](field, kept)
        self.assertIsNot(output, field)
        self.assertEqual([(b.x, b.y) for b in output.boxes], [(1.0, 1.0), (3.0, 3.0)])
        self.assertEqual([(f.box, f.side, f.image_ref) for f in output.faces],
                         [(0, 'E', 'img0E'), (0, 'W', 'img0W'), (1, 'E', 'img2E'), (1, 'W', 'img2W')])
        self.assertEqual(len(field.boxes), 3)          # 받은 지도는 건드리지 않는다
        self.assertEqual(len(field.faces), 6)

    def test_slam_without_map_fusion_publishes_the_drone_boxes(self):
        """map_fusion:=false 이면 지도 고치기 대신 그대로 넘기는 객체가 들어간다."""
        module, _ = load_slam()
        FakeNode.parameters = {'map_fusion': False}
        self.addCleanup(setattr, FakeNode, 'parameters', {})
        slam = module['SlamNode']()
        self.assertFalse(slam._fusion.wants_scan())
        field = FieldMap(1)
        field.box_size, field.width, field.height = 0.4, 6.12, 5.0
        field.boxes = [point(1.0, 1.0)]
        slam._receive_perception_map(field)
        boxes, weights = slam._fusion.structure()
        self.assertEqual(boxes.tolist(), [[1.0, 1.0]])
        self.assertIsNone(weights)
        self.assertIs(slam._fused(field), field)

    def test_lidar_boxes_go_out_on_their_own_topic_and_not_into_the_map(self):
        """라이다로 찾은 상자(map_fusion 의 added)는 /team/navigation/lidar_boxes 로만 나간다 — 1초 타이머와, 목록이 바뀐 때."""
        module, modules = load_slam()
        slam = module['SlamNode']()
        made = {c.args[1]: c.args for c in slam.create_publisher.call_args_list}
        self.assertEqual(made['/team/navigation/lidar_boxes'][0], modules['geometry_msgs.msg'].PoseArray)
        self.assertEqual(made['/team/navigation/lidar_boxes'][2], 1)
        self.assertIn(((1.0, slam._publish_lidar_boxes), {}), [(c.args, c.kwargs) for c in slam.create_timer.call_args_list])
        publish = slam.lidar_boxes_publisher.publish
        self.assertEqual(module['lidar_boxes'](slam._fusion), [])
        slam._publish_lidar_boxes()                      # 타이머: 빈 목록도 낸다
        self.assertEqual(publish.call_args.args[0].poses, [])
        self.assertEqual(publish.call_args.args[0].header.frame_id, 'arena')
        slam._fusion.added = lambda: [(1.23456, 4.8)]    # 더한 상자 하나
        slam._publish_lidar_boxes(changed_only=True)     # 목록이 바뀌었다 → 바로 낸다
        self.assertEqual(publish.call_count, 2)
        message = publish.call_args.args[0]
        self.assertEqual([(q.position.x, q.position.y, q.orientation.w) for q in message.poses], [(1.235, 4.8, 1.0)])
        slam._publish_lidar_boxes(changed_only=True)     # 그대로다 → 내지 않는다
        self.assertEqual(publish.call_count, 2)
        slam._publish_lidar_boxes()                      # 타이머는 그대로여도 낸다
        self.assertEqual(publish.call_count, 3)
        slam.map_publisher.publish.assert_not_called()   # 지도에는 아무것도 하지 않는다
        publish.side_effect = RuntimeError('publish failed')
        slam._publish_lidar_boxes()                      # 발행이 실패해도 예외가 밖으로 나가지 않는다 (타이머 콜백에서 노드가 죽지 않게)
        slam.get_logger().warning.assert_called_once()

    def test_lidar_boxes_are_not_published_without_map_fusion(self):
        module, _ = load_slam()
        FakeNode.parameters = {'map_fusion': False}
        self.addCleanup(setattr, FakeNode, 'parameters', {})
        slam = module['SlamNode']()
        self.assertIsNone(module['lidar_boxes'](slam._fusion))
        self.assertNotIn(slam._publish_lidar_boxes, [c.args[1] for c in slam.create_timer.call_args_list])
        slam._publish_lidar_boxes()
        slam._publish_lidar_boxes(changed_only=True)
        slam.lidar_boxes_publisher.publish.assert_not_called()

    def test_slot_region_is_the_base_named_after_our_slot(self):
        """출발 칸은 지도 bases 중 name 이 우리 slot 인 것 하나. 항공 차량 검출은 필요 없다."""
        module, _ = load_slam()
        slam = module['SlamNode']()
        field = FieldMap(1)
        field.bases = [SimpleNamespace(name='B-2', min_x=0, max_x=0.4, min_y=1.9, max_y=3.1)]
        slam._receive_server_state(SimpleNamespace(
            run_id='r', team_assigned=True, phase='running', camp='BLUE', slot='B-2',
            header=field.header))
        self.assertEqual(slam._slot_region(field, 1_000_000_000), (0, 1.9, 0.4, 3.1))
        self.assertIsNone(slam._slot_region(field, 4_000_000_000))   # 단계 정보가 2 s 넘게 묵음
        field.bases.append(field.bases[0])
        self.assertIsNone(slam._slot_region(field, 1_000_000_000))   # 같은 이름이 둘
        field.bases = [SimpleNamespace(name='R-2', min_x=0, max_x=0.4, min_y=1.9, max_y=3.1)]
        self.assertIsNone(slam._slot_region(field, 1_000_000_000))

    def test_still_means_not_moved_since_the_robot_was_locked(self):
        """waiting·ready 동안의 마지막 /odom 에서 5 cm 안이어야 한다. 잠긴 것을 못 봤으면 아니다."""
        module, _ = load_slam()
        slam = module['SlamNode']()
        header = SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0))

        def odometry(sec, x):
            return SimpleNamespace(
                header=SimpleNamespace(frame_id='odom', stamp=FieldMap(sec).header.stamp),
                child_frame_id='base_footprint',
                pose=SimpleNamespace(pose=SimpleNamespace(
                    position=SimpleNamespace(x=x, y=0.0),
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0))),
            )

        slam._receive_server_state(SimpleNamespace(run_id='r', phase='running', header=header))
        slam._receive_odometry(odometry(1, 2.0))
        self.assertFalse(slam._still((2.0, 0.0)))     # 경기 중에 켜짐: 출발 칸이라고 볼 수 없다
        slam._receive_server_state(SimpleNamespace(run_id='r', phase='ready', header=header))
        slam._receive_odometry(odometry(2, 2.0))
        slam._receive_server_state(SimpleNamespace(run_id='r', phase='running', header=header))
        slam._receive_odometry(odometry(3, 2.03))
        self.assertTrue(slam._still((2.03, 0.0)))
        self.assertEqual(slam._locked[:2], (2.0, 0.0))   # running 의 /odom 은 기준을 바꾸지 않는다
        self.assertFalse(slam._still((2.08, 0.0)))
        slam._receive_server_state(SimpleNamespace(run_id='next', phase='waiting', header=header))
        self.assertIsNone(slam._locked)                   # 다음 경기

    def test_scan_rejects_insufficient_data(self):
        module, _ = load_slam()
        scan = SimpleNamespace(header=SimpleNamespace(frame_id='base_lidar_link'),
                               ranges=[])
        field = SimpleNamespace(box_size=0.4, boxes=[SimpleNamespace(x=1, y=1)])
        self.assertIsNone(module['scan_match'](field, scan, (0, 0, 0), True))



class DrivingNodeTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        FakeNode.clock = self.clock
        FakeNode.distinct_publishers = True
        self.addCleanup(setattr, FakeNode, 'clock', None)
        self.addCleanup(setattr, FakeNode, 'distinct_publishers', False)
        self.module, self.modules = load_driving()
        self.node = self.module.DrivingNode()

    def topics(self, method):
        return {c.args[1]: c for c in getattr(self.node, method).call_args_list}

    def test_wiring(self):
        node = self.node
        self.assertEqual(node.name, 'navigation')
        subs = self.topics('create_subscription')
        self.assertEqual(set(subs), {MAP_TOPIC, '/team/navigation/pose', '/scan', '/odom', '/imu/data', '/odom/odom_raw', '/team/control/state', '/team/mission/state',
                                     '/team/perception/ugvs'})
        self.assertIs(subs[MAP_TOPIC].args[0], FieldMap)
        self.assertIs(subs[MAP_TOPIC].args[3], self.module.MAP_QOS)
        self.assertEqual(set(self.topics('create_publisher')), {'/cmd_vel', '/team/node_status', '/team/navigation/debug'})
        self.assertIsNone(node.latest_map)
        action_args = self.modules['rclpy.action'].ActionServer.call_args
        self.assertEqual(action_args.args[2], '/team/navigation/navigate')

    def test_map_is_kept_and_wrong_frame_is_not_used(self):
        node = self.node
        wrong = field_map(frame='map')
        node._receive_map(wrong)
        self.assertIs(node.latest_map, wrong)
        self.assertIsNone(node.grid)
        good = field_map()
        node._receive_map(good)
        self.assertIs(node.latest_map, good)
        self.assertIsNotNone(node.grid)
        grid = node.grid
        node._receive_map(field_map())  # 같은 기하 → 격자를 다시 만들지 않는다
        self.assertIs(node.grid, grid)

    def run_goal(self, req, pose=(1.0, 2.5, 0.0), max_s=40.0, during=None, allowed=True):
        """가짜 시계로 20 Hz 를 돌리며 _execute 를 다른 스레드에서 실행한다. 차체는 보낸 /cmd_vel 대로 움직인다."""
        node = self.node
        node._receive_map(field_map())
        x, y, yaw = pose
        handle = FakeGoalHandle(req)
        out = {}
        sent = []
        thread = None
        steps = int(max_s * 20)
        for k in range(steps):
            t = self.clock.t
            node._receive_pose(pose_msg(x, y, yaw, t))
            node._receive_control(control(allowed))
            if k == 2:
                thread = threading.Thread(target=lambda: out.setdefault('result', node._execute(handle)))
                thread.start()
                deadline = time.monotonic() + 2.0
                while node._job is None and thread.is_alive() and time.monotonic() < deadline:
                    time.sleep(0.001)
            if during:
                during(k, handle)
            node.cmd_publisher.publish.reset_mock()
            node._on_control_tick()
            if node.cmd_publisher.publish.called:
                msg = node.cmd_publisher.publish.call_args.args[0]
                v, w = msg.linear.x, msg.angular.z
                sent.append((v, w))
                yaw = math.atan2(math.sin(yaw + w * 0.05), math.cos(yaw + w * 0.05))
                x += v * 0.05 * math.cos(yaw)
                y += v * 0.05 * math.sin(yaw)
            self.clock.t += 0.05
            if thread is not None and not thread.is_alive():
                break
            if node._job is None and thread is not None:
                thread.join(timeout=1.0)
                if not thread.is_alive():
                    break
        if thread is not None:
            thread.join(timeout=2.0)
        return out.get('result'), handle, (x, y, yaw), sent

    def test_drives_to_goal_and_succeeds(self):
        result, handle, (x, y, yaw), sent = self.run_goal(request(5.0, 2.5, yaw=math.pi / 2))
        self.assertIsNotNone(result)
        self.assertEqual(result.outcome, 1, result.detail)  # REACHED
        handle.succeed.assert_called_once_with()
        self.assertLess(math.hypot(x - 5.0, y - 2.5), 0.10)
        self.assertTrue(handle.publish_feedback.called)
        self.assertTrue(all(abs(v) <= 0.6 and abs(w) <= 3.0 for v, w in sent))
        self.assertEqual(sent[-1], (0.0, 0.0))  # 끝나면 0 을 보낸다

    def test_control_lost_stops(self):
        def drop(k, handle):
            if k >= 20:
                self.node._receive_control(control(False))

        result, handle, _pose, sent = self.run_goal(request(5.0, 4.0), during=drop)
        self.assertEqual(result.outcome, 5, result.detail)  # CONTROL_LOST
        handle.abort.assert_called_once_with()
        self.assertEqual(sent[-1], (0.0, 0.0))

    def test_invalid_run_id_and_empty_orientation(self):
        self.node._receive_mission(SimpleNamespace(run_id='run-1'))
        result, handle, _pose, sent = self.run_goal(request(5.0, 4.0, run_id='old-run'))
        self.assertEqual(result.outcome, 6)  # INVALID_GOAL
        handle.abort.assert_called_once_with()
        self.assertFalse(any(v or w for v, w in sent))
        req = request(5.0, 4.0)
        req.goal.pose.orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0)
        result, _h, _p, _s = self.run_goal(req)
        self.assertEqual(result.outcome, 6)
        req.yaw_tolerance_rad = 3.15  # 방향 상관없음 → 빈 orientation 도 된다
        result, _h, _p, _s = self.run_goal(req)
        self.assertEqual(result.outcome, 1, result.detail)

    def test_cancel(self):
        def cancel(k, handle):
            if k == 30:
                handle.is_cancel_requested = True  # rclpy: 취소를 받아들이면 CANCELING 상태가 된다
                self.assertIs(self.node._on_cancel(handle), self.modules['rclpy.action'].CancelResponse.ACCEPT)

        result, handle, _pose, sent = self.run_goal(request(5.0, 4.0), during=cancel)
        self.assertEqual(result.outcome, 2)  # CANCELED
        handle.canceled.assert_called_once_with()
        self.assertEqual(sent[-1], (0.0, 0.0))

    def test_new_goal_replaces_running_goal_without_stopping(self):
        node = self.node
        holder = {}

        def second(k, handle):
            if k == 30:
                h2 = FakeGoalHandle(request(1.5, 4.0, yaw_tol=3.15))
                holder['handle'] = h2
                holder['thread'] = threading.Thread(target=lambda: holder.setdefault('result', node._execute(h2)))
                holder['thread'].start()
                deadline = time.monotonic() + 2.0
                while (node._job is None or node._job.goal.y != 4.0 or node._job.goal.x != 1.5) and time.monotonic() < deadline:
                    time.sleep(0.001)

        result, first, _pose, sent = self.run_goal(request(5.0, 2.5, yaw_tol=3.15), during=second, max_s=60.0)
        self.assertEqual(result.outcome, 2, result.detail)  # CANCELED (교체)
        self.assertIn('교체', result.detail)
        first.abort.assert_called_once_with()  # 취소 요청이 아니라서 abort 로 끝낸다
        # 두 번째 목표는 남은 시간 동안 계속 돈다
        for _ in range(1200):
            if not holder['thread'].is_alive():
                break
            est = node.tracker.estimate()
            node._receive_pose(pose_msg(est.x, est.y, est.yaw, self.clock.t))
            node._receive_control(control())
            node.cmd_publisher.publish.reset_mock()
            node._on_control_tick()
            if node.cmd_publisher.publish.called:
                msg = node.cmd_publisher.publish.call_args.args[0]
                yaw = est.yaw + msg.angular.z * 0.05
                node._receive_pose(pose_msg(est.x + msg.linear.x * 0.05 * math.cos(yaw), est.y + msg.linear.x * 0.05 * math.sin(yaw), yaw, self.clock.t))
            self.clock.t += 0.05
            time.sleep(0.001)
        holder['thread'].join(timeout=2.0)
        self.assertEqual(holder['result'].outcome, 1, holder['result'].detail)

    def test_entry_point_stops_robot_and_releases_node(self):
        module, modules = self.module, self.modules
        executor = modules['rclpy.executors'].MultiThreadedExecutor.return_value
        executor.spin.side_effect = KeyboardInterrupt()
        with patch.object(module, 'DrivingNode', return_value=Mock()) as make:
            module.main()
        node = make.return_value
        node.stop_robot.assert_called_once_with()
        node.destroy_node.assert_called_once_with()
        modules['rclpy'].try_shutdown.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
