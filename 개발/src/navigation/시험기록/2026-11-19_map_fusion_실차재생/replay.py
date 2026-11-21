"""실차 rosbag 을 SLAM 노드에 받은 순서대로 다시 넣고, 낸 위치 · 내보낸 지도 · 지도 고치기(map_fusion) 상태를 본다.

ROS 없이 돈다 (가짜 rclpy + bagread.py 의 CDR 읽기). 로봇을 움직이지 않는다 — /cmd_vel 은 쓰지도 내지도 않는다.

    python -I replay.py <bag_0.db3> <결과.json> [true|false] [navigation 패키지 폴더] [이름=값 …]

  세 번째 = map_fusion 파라미터 (기본 true). 네 번째를 주면 다른 사본(예: git archive 로 꺼낸 예전 커밋)의 slam_node 를 돌린다.
  이름=값 은 SLAM 노드의 다른 ROS 파라미터 (예: fixed_pillars=false — 모의 경기장에는 규칙의 고정 기둥이 없다).
  결과 json 은 report.py 로 여러 개를 나란히 본다.

bag 은 웹 시험기(시험기록/web_drive, run.sh 의 ros2 bag record)가 로봇 ~/team/nav_web/runs/<시각>/bag 에 남긴 것.
"""
import importlib
import json
import math
import runpy
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock, patch

sys.stdout.reconfigure(encoding='utf-8')
importlib.import_module('numpy')            # 가짜 모듈을 끼우기 전에 불러 둔다
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bagread  # noqa: E402

db, out = sys.argv[1:3]
extra = dict(arg.split('=', 1) for arg in sys.argv[3:] if '=' in arg)      # 이름=값 → ROS 파라미터
plain = [arg for arg in sys.argv[3:] if '=' not in arg]
fusion_on = (plain[0] if plain else 'true') == 'true'
package = Path(plain[1]) if len(plain) > 1 else HERE.parents[1]


def parameter(name, default):
    if name == 'map_fusion':
        return fusion_on
    if name not in extra:
        return default
    text = extra[name]
    return text.lower() == 'true' if isinstance(default, bool) else type(default)(text)

ROS_MODULES = ('rclpy', 'rclpy.node', 'rclpy.qos', 'rclpy.executors', 'geometry_msgs', 'geometry_msgs.msg',
               'nav_msgs', 'nav_msgs.msg', 'sensor_msgs', 'sensor_msgs.msg', 'std_msgs', 'std_msgs.msg',
               'team_interfaces', 'team_interfaces.msg')


class Auto(ModuleType):
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        value = Mock(name=name)
        setattr(self, name, value)
        return value


class Clock:
    t = 0       # 지금 넣고 있는 메시지를 bag 이 받은 시각 (ns) — odom_raw 처럼 header 가 없는 것의 시각

    def now(self):
        return NS(nanoseconds=Clock.t)


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class Node:
    def __init__(self, name):
        self.publishers, self.logs = {}, []
        self._logger = Mock()
        self._logger.info.side_effect = lambda text: self.logs.append((Clock.t / 1e9, text))

    def create_publisher(self, kind, topic, qos):
        return self.publishers.setdefault(topic, Publisher())

    def create_subscription(self, *args, **kwargs):
        return Mock()

    def create_timer(self, *args, **kwargs):
        return Mock()

    def declare_parameter(self, name, default):
        return NS(value=parameter(name, default))

    def get_logger(self):
        return self._logger

    def get_clock(self):
        return Clock()


modules = {name: Auto(name) for name in ROS_MODULES}
modules['rclpy.node'].Node = Node
modules['rclpy.qos'].qos_profile_sensor_data = object()
modules['rclpy.executors'].ExternalShutdownException = type('ExternalShutdownException', (Exception,), {})
modules['team_interfaces.msg'].FieldVehicle = bagread.Vehicle
modules['team_interfaces.msg'].FieldFace = NS
modules['geometry_msgs.msg'].Point = NS
modules['geometry_msgs.msg'].PoseStamped = NS
modules['std_msgs.msg'].String = NS
sys.path.insert(0, str(package))
with patch.dict(sys.modules, modules):
    node = runpy.run_path(str(package / 'navigation' / 'slam_node.py'))['SlamNode']()


def stamp(header):
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def boxes_now():
    """지도 고치기 상태: 드론 상자 [x, y, 점수, 뺐나], 정합에만 쓰는 더한 상자 [x, y, 점수]."""
    return {'drone': [[round(t.x, 3), round(t.y, 3), round(t.score, 1), t.rejected] for t in fusion._drone],
            'added': [[round(t.x, 3), round(t.y, 3), round(t.score, 1)] for t in fusion._added if t.promoted]}


fusion = node._fusion if hasattr(getattr(node, '_fusion', None), '_drone') else None
skips, events, scan_counts, fusion_ms = {}, [], {}, []
if fusion is not None:
    inner = fusion.observe

    def observe(*args, **kwargs):
        changed = inner(*args, **kwargs)
        key = (fusion.last.get('skip') or '고침').split(' (')[0]
        skips[key] = skips.get(key, 0) + 1
        if changed:
            events.append(dict(boxes_now(), t=round(Clock.t / 1e9, 2)))
        return changed
    fusion.observe = observe

recorded = []       # 그때 로봇이 실제로 낸 위치
handlers = {'/odom': node._receive_odometry, '/odom/odom_raw': node._receive_wheels, '/imu/data': node._receive_imu,
            '/team/perception/field_map': node._receive_perception_map, '/team/server/state': node._receive_server_state}
started = time.perf_counter()
for received, topic, message in bagread.read(db, set(handlers) | {'/scan', '/team/navigation/pose'}):
    Clock.t = received
    if topic == '/team/navigation/pose':
        recorded.append((stamp(message.header), message.pose.position.x, message.pose.position.y, yaw_of(message.pose.orientation)))
    elif topic == '/scan':
        node._scan_note = {}
        node._receive_scan(message)
        result = node._scan_note.get('result')
        if result:
            scan_counts[result] = scan_counts.get(result, 0) + 1
        if 'fusion_ms' in node._scan_note:
            fusion_ms.append(node._scan_note['fusion_ms'])
    else:
        handlers[topic](message)

poses = [(stamp(m.header), m.pose.position.x, m.pose.position.y, yaw_of(m.pose.orientation))
         for m in node.publishers['/team/navigation/pose'].messages]
maps = node.publishers['/team/navigation/map'].messages
result = {
    'bag': db, 'package': str(package), 'map_fusion': fusion_on, 'parameters': extra, 'seconds': round(time.perf_counter() - started, 1),
    'poses': poses, 'recorded': recorded, 'logs': node.logs, 'scan_counts': scan_counts, 'skips': skips, 'events': events,
    'published_box_counts': sorted({len(m.boxes) for m in maps}),
    'fusion_ms': {'n': len(fusion_ms), 'mean': sum(fusion_ms) / max(1, len(fusion_ms)), 'max': max(fusion_ms or [0.0])},
}
json.dump(result, open(out, 'w', encoding='utf-8'), ensure_ascii=False)
print('낸 위치 %d개 (그때 로봇이 낸 것 %d개), 스캔 %s' % (len(poses), len(recorded), scan_counts))
print('내보낸 지도의 상자 수', result['published_box_counts'], '· 지도 고치기 관측', skips, '· 한 번 %.2f ms (최대 %.1f)' % (result['fusion_ms']['mean'], result['fusion_ms']['max']))
for t, text in node.logs[:12]:
    print('  %.1f %s' % (t, text))
