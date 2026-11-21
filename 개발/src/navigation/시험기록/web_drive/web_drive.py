#!/usr/bin/env python3
"""웹 주행 시험기: 경기와 같은 순서(① 지도 → ② 처음 위치 → ③ 주행)로 navigation 두 노드를 시험한다.

  ① 지도      드론 사진 → perception 의 지도 함수(drone_map.py) → /team/perception/field_map. 드론 사진이 없으면 라이다 · 직접 편집
  ② 처음 위치  출발 칸: 단계 ready(잠김) → running → 지도 → SLAM 이 그 칸 안을 라이다로 찾는다 (경기와 같은 길)
  ③ 주행      지도를 클릭하면 그 좌표로 간다

supervisor · server_bridge · perception 이 하는 일을 사람이 대신한다 (시험할 때만 쓴다. 경기에서는 켜지 않는다):
  /team/perception/field_map   경기장 지도 (상자 + 출발 칸 6개, 이름 B-1 … R-3)
  /team/server/state           단계(ready → running) · 진영 · slot  — SLAM 의 출발 칸 초기화에 필요
  /team/control/state          주행 허용 임대 (10 Hz). 화면의 「정지」 또는 화면 연결이 끊기면 꺼진다
  /team/mission/state          run_id
  /team/navigation/navigate    목표 (액션)
받는 것: /team/navigation/pose · map · debug, /scan, /cmd_vel, /team/node_status, /odom, /team/server/uav_image (있으면)

  python3 web_drive.py                # 로봇 (ROS 2 환경에서). http://<로봇 IP>:5960
  python3 web_drive.py --sim          # PC, ROS 없이: 가상 경기장에서 실제 노드 코드를 돌린다. http://localhost:5960
"""
import argparse
import base64
import json
import math
import os
import signal
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import drone_map
from arena_default import ARENA, bases as default_bases, default_config, pillars

try:
    import lidar_map   # numpy 가 있어야 한다 (로봇에는 있다)
except ImportError:
    lidar_map = None

HERE = Path(__file__).resolve().parent
MAP_SOURCES = ('기본값', '드론 사진', '라이다', '직접 편집')
MODES = {'normal': 0, 'straight': 1, 'assault': 2}
OUTCOMES = {1: 'REACHED', 2: 'CANCELED', 3: 'BLOCKED', 4: 'TIMED_OUT', 5: 'CONTROL_LOST', 6: 'INVALID_GOAL', 7: 'LOCALIZATION_LOST'}
STATUS = {0: 'INITIALIZING', 1: 'READY', 2: 'DEGRADED', 3: 'ERROR'}
LIDAR_X = 0.0398          # base_link → base_lidar_link (노드와 같은 값)
READY_S = 1.2             # 처음 위치를 잡을 때 ready 로 두는 시간 (SLAM 이 「잠긴 자리」 를 기억한다)
MAP_AFTER_RUNNING_S = 0.5


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def clean_config(raw):
    """화면에서 온 지도 설정을 검사해 돌려준다. 잘못된 값이면 ValueError."""
    width, height, box = float(raw['width']), float(raw['height']), float(raw['box_size'])
    if not (0.5 <= width <= 30 and 0.5 <= height <= 30 and 0.05 <= box <= 2):
        raise ValueError('경기장 크기·상자 크기가 범위를 벗어남')
    boxes = sorted({(round(float(x), 3), round(float(y), 3)) for x, y in raw.get('boxes', [])})
    bases = []
    for base in raw.get('bases', []):
        bases.append({'name': str(base['name']), **{k: float(base[k]) for k in ('min_x', 'min_y', 'max_x', 'max_y')}})
    source = str(raw.get('source') or '직접 편집')
    return {'width': width, 'height': height, 'box_size': box, 'boxes': [list(b) for b in boxes], 'bases': bases,
            'source': source if source in MAP_SOURCES else '직접 편집', 'made': str(raw.get('made') or '')}


def with_pillars(boxes, box_size, width, height):
    """고정 기둥 10개 [추정] 를 더한다 (이미 그 자리 가까이 상자가 있으면 빼고). 드론 사진은 끝 띠 안 물체를 내지 않아 기둥이 빠진다."""
    out = [tuple(b) for b in boxes]
    for px, py in pillars(width, height):
        if all(max(abs(px - x), abs(py - y)) > box_size * 0.75 for x, y in out):
            out.append((px, py))
    return out


class Bus:
    """시험기의 공통 로직. ROS 로 내보내고 받는 부분은 RosBus, 가상 경기장은 SimBus 가 채운다."""

    sim = False

    def __init__(self, config_path, deadman_s=3.0, secrets=()):
        self.lock = threading.RLock()
        self.config_path = Path(config_path)
        self.config = clean_config(dict(default_config(), source='기본값'))
        if self.config_path.exists():
            try:
                saved = json.loads(self.config_path.read_text(encoding='utf-8'))
                if 'source' in saved:
                    self.config = clean_config(saved)
                else:   # 2026-11-18 앞의 시험기가 저장한 것 (경기장 600 × 480 시절) — 새 기본값(612 × 500)으로 시작한다
                    print('예전 형식의 지도 설정이라 기본값으로 시작함 (경기장 %.2f × %.2f m)' % (ARENA['width'], ARENA['height']), flush=True)
            except (ValueError, KeyError, TypeError) as error:
                print('지도 설정을 읽지 못해 기본값을 씀: %r' % (error,), flush=True)
        self.run_id = ''
        self.new_run()
        self.phase, self.camp, self.slot = 'waiting', '', ''
        self.allowed = False
        self.stop_reason = '시작 전 (목표를 보내면 켜진다)'
        self.deadman_s = deadman_s
        self._last_client = time.monotonic()
        self._schedule = []       # [(실행할 시각 monotonic, 함수)]
        self._ticks = 0
        self.pose = None          # (x, y, yaw, 받은 시각 monotonic)
        self.trail = []           # [(x, y)] 3 cm 넘게 움직일 때마다
        self.scan = None          # {'points': [[x, y] 차체 기준], 'pose': 그때의 위치, 't': 받은 시각}
        self.scan_body = None     # 마지막 스캔의 모든 점 (numpy, 차체 기준) — 라이다로 지도를 만들 때
        self.cmd = (0.0, 0.0, None)
        self.odom = None
        self.status = {}          # 노드 이름 → {'state', 'detail', 't'}
        self.debug = {}           # 노드 이름 → (dict, 받은 시각)
        self.nav_map = None
        self.goal = None          # 마지막으로 보낸 목표
        self.init = None          # 마지막 처음 위치 잡기 {'slot', 'region', 'how', 'started', 'map_sent'}
        self.feedback = None
        self.log = deque(maxlen=80)
        # 드론 사진 → 지도
        self.secrets = [Path(p) for p in secrets]
        self.mapper = drone_map.DroneMapper()
        self.drone = {'state': 'idle', 'detail': '', 'version': 0, 'result': None, 'source': '', 'started': None}
        self.drone_images = {}    # 'photo' · 'overlay' · 'top' → JPEG
        self.uav = None           # server_bridge 가 낸 마지막 사진 (JPEG, 촬영 시각 글자, 받은 시각)
        self.say('시험기 시작 (run_id %s)' % self.run_id)
        if not self.mapper.ready:
            self.say('드론 사진 → 지도: 쓸 수 없음 — ' + self.mapper.error)

    # ------------------------------------------------------------------ 하위 클래스가 채우는 것
    def _pub_server(self): raise NotImplementedError
    def _pub_control(self): raise NotImplementedError
    def _pub_mission(self): raise NotImplementedError
    def _pub_field_map(self, bases, self_pose): raise NotImplementedError
    def _send_goal(self, goal): raise NotImplementedError
    def _cancel(self): raise NotImplementedError
    def links(self): return {}
    def warnings(self): return []
    def extra_state(self): return {}
    def uav_publishers(self): return 0

    # ------------------------------------------------------------------ 공통
    def say(self, text):
        with self.lock:
            self.log.append({'t': time.strftime('%H:%M:%S'), 'text': text})
        try:
            print('[web_drive] ' + text, flush=True)
        except (UnicodeError, OSError):   # 콘솔 글자판이 한글·기호를 못 찍어도 동작은 이어 간다
            pass

    def new_run(self):
        self.run_id = 'web-' + time.strftime('%H%M%S') + '-%03d' % (int(time.time() * 1000) % 1000)

    def tick(self):
        """10 Hz. 허용 임대를 내보내고 예약된 일을 한다."""
        now = time.monotonic()
        with self.lock:
            if self.allowed and self.deadman_s > 0 and now - self._last_client > self.deadman_s:
                self.stop('웹 화면이 %.0f s 동안 연결되지 않음' % self.deadman_s)
            due = [item for item in self._schedule if item[0] <= now]
            self._schedule = [item for item in self._schedule if item[0] > now]
            self._ticks += 1
            slow = self._ticks % 5 == 0
        for _, action in sorted(due, key=lambda item: item[0]):
            action()
        self._pub_control()
        if slow:
            self._pub_server()
            self._pub_mission()

    def client_seen(self):
        self._last_client = time.monotonic()

    # ------------------------------------------------------------------ 화면에서 부르는 것
    def set_config(self, raw, source=None):
        config = clean_config(dict(raw, source=source or raw.get('source') or '직접 편집', made=time.strftime('%H:%M:%S')))
        with self.lock:
            self.config = config
        self._save_config()
        self._pub_field_map(self._bases_for_slam(), None)
        self.say('지도 보냄 (%s): %.2f × %.2f m, 상자 %d개' % (config['source'], config['width'], config['height'], len(config['boxes'])))

    def _bases_for_slam(self, override=None):
        """출발 칸 6개를 이름(B-1 … R-3)과 함께 넣는다 — perception 이 base_naming="slot" 으로 내는 것과 같다.

        SLAM 은 그중 name = slot 인 하나를 찾는다. override 를 주면 그 이름의 칸을 바꿔 넣는다 (「주변 탐색」).
        """
        bases = [dict(base) for base in self.config['bases']]
        if override is not None:
            bases = [base for base in bases if base['name'] != override['name']] + [override]
        return bases

    def init_region(self, slot, region=None):
        """처음 위치 잡기: ready(로봇이 서 있는 자리를 SLAM 이 기억) → running → 지도. SLAM 이 출발 칸 안에서 라이다로 찾는다.

        region 을 주면 그 사각형(min_x, min_y, max_x, max_y)을 출발 칸으로 삼는다 — 경기장 아무 데서나 시작할 때.
        """
        if slot not in ('B-1', 'B-2', 'B-3', 'R-1', 'R-2', 'R-3'):
            raise ValueError('slot 은 B-1 ~ R-3')
        with self.lock:
            if self.allowed:
                self.stop('처음 위치를 다시 잡음')
            if region is None:
                named = [base for base in self.config['bases'] if base['name'] == slot]
                if len(named) != 1:
                    raise ValueError('지도 설정에 %s 출발 칸이 하나가 아님' % slot)
                b = named[0]
                bases = self._bases_for_slam()
            else:
                x0, y0, x1, y1 = (float(v) for v in region)
                b = {'name': slot, 'min_x': min(x0, x1), 'min_y': min(y0, y1), 'max_x': max(x0, x1), 'max_y': max(y0, y1)}
                bases = self._bases_for_slam(override=b)
            self.new_run()   # SLAM 은 run_id 가 바뀌면 위치·출발 칸·보정을 모두 지운다
            self.camp, self.slot, self.phase = ('BLUE' if slot.startswith('B') else 'RED'), slot, 'ready'
            self.trail = []
            self.init = {'slot': slot, 'region': [b['min_x'], b['min_y'], b['max_x'], b['max_y']], 'how': 'slot' if region is None else 'region',
                         'started': time.monotonic(), 'map_sent': None}
            now = time.monotonic()
            self._schedule = [(now + READY_S, self._to_running),
                              (now + READY_S + MAP_AFTER_RUNNING_S, lambda: self._send_init_map(bases))]
        self._pub_server()
        self._pub_mission()
        self.say('처음 위치 잡기 %s: (%.2f~%.2f, %.2f~%.2f) 안을 라이다로 찾음 — 로봇을 움직이지 말 것' % (slot, b['min_x'], b['max_x'], b['min_y'], b['max_y']))

    def _send_init_map(self, bases):
        with self.lock:
            if self.init is not None:
                self.init['map_sent'] = time.monotonic()
        self._pub_field_map(bases, None)

    def _to_running(self):
        with self.lock:
            self.phase = 'running'
        self._pub_server()

    def init_pose(self, x, y, yaw):
        """자세를 직접 알려 준다 (항공 지도가 우리 차를 찍어 준 것과 같은 길). 라이다 정합은 이 자세의 ±10 cm · ±11° 안에서만 찾는다."""
        with self.lock:
            if self.phase != 'running':
                self.phase = 'running'
            self.trail = []
            self.init = {'slot': self.slot, 'region': None, 'how': 'pose', 'started': time.monotonic(), 'map_sent': time.monotonic()}
        self._pub_server()
        self._pub_field_map(self._bases_for_slam(), (float(x), float(y), float(yaw)))
        self.say('자세 직접 지정: (%.2f, %.2f, %.0f°)' % (x, y, math.degrees(yaw)))

    def _fresh_scan(self):
        if lidar_map is None:
            raise ValueError('numpy 가 없어 라이다로 지도를 만들 수 없음')
        with self.lock:
            scan, body = self.scan, self.scan_body
        if scan is None or body is None or time.monotonic() - scan['t'] > 1.0 or len(body) < 30:
            raise ValueError('/scan 이 오지 않음 (라이다가 켜져 있나?)')
        return body

    def build_map(self, x, y, yaw, keep=False):
        """모의 배치용: 사람이 찍은 자세를 벽에 맞춰 다듬고, 지금 라이다에 보이는 상자로 지도를 만든 뒤 그 자세에서 시작한다.

        드론 사진 · perception 이 없을 때 지도를 대신 만드는 것이다. 한 자리에서 보이는 면만 쓴다 (lidar_map.py).
        """
        body = self._fresh_scan()
        with self.lock:
            config = dict(self.config)
            if self.allowed:
                self.stop('지도를 다시 만듦')
        pose, note = lidar_map.fit_walls(body, (float(x), float(y), float(yaw)), config['width'], config['height'])
        known = [tuple(b) for b in config['boxes']] if keep else []
        found = lidar_map.suggest_boxes(body, pose, config['width'], config['height'], config['box_size'], known=known)
        config['boxes'] = [list(b) for b in known + found]
        config = clean_config(dict(config, source='라이다', made=time.strftime('%H:%M:%S')))
        with self.lock:
            self.config = config
            self.new_run()       # SLAM 이 예전 위치 · 출발 칸 · 보정을 모두 지운다
            self.phase, self.trail = 'running', []
            self.init = {'slot': self.slot, 'region': None, 'how': 'lidar', 'started': time.monotonic(), 'map_sent': time.monotonic() + 0.4}
            # 단계(새 run_id)가 먼저 가고 나서 지도가 가야 한다
            self._schedule = [(time.monotonic() + 0.4, lambda: self._pub_field_map(self._bases_for_slam(), pose))]
        self._save_config()
        self._pub_server()
        self._pub_mission()
        self.say('라이다로 지도 만듦: 자세 (%.2f, %.2f, %.0f°) [%s], 상자 %d개%s' % (
            pose[0], pose[1], math.degrees(pose[2]), note, len(config['boxes']), ' (기존 %d개 유지)' % len(known) if keep else ''))
        return {'pose': pose, 'boxes': config['boxes']}

    def add_boxes(self):
        """지금 SLAM 위치에서 라이다에 새로 보이는 상자를 지도에 더한다 (움직이며 지도를 넓힐 때)."""
        body = self._fresh_scan()
        with self.lock:
            pose, config = self.pose, dict(self.config)
        if pose is None or time.monotonic() - pose[3] > 0.6:
            raise ValueError('SLAM 위치가 없음 — 먼저 처음 위치를 잡아야 함')
        known = [tuple(b) for b in config['boxes']]
        found = lidar_map.suggest_boxes(body, pose[:3], config['width'], config['height'], config['box_size'], known=known)
        if found:
            config['boxes'] = [list(b) for b in known + found]
            with self.lock:
                self.config = clean_config(dict(config, made=time.strftime('%H:%M:%S')))
            self._save_config()
            self._pub_field_map(self._bases_for_slam(), None)
        self.say('라이다에 새로 보인 상자 %d개를 지도에 더함' % len(found))
        return {'added': found}

    # ------------------------------------------------------------------ ① 지도: 드론 사진
    def drone_options(self):
        url, code, where = drone_map.server_settings(self.secrets)
        return {'ready': self.mapper.ready, 'error': self.mapper.error, 'code_source': self.mapper.source,
                'samples': sorted(self.mapper.samples()), 'cameras': list(drone_map.CAMERAS),
                'server': url, 'has_code': bool(code), 'secrets_from': where, 'topic_publishers': self.uav_publishers(),
                'uav_age': None if self.uav is None else round(time.monotonic() - self.uav[2], 1)}

    def drone_start(self, raw):
        """사진 한 장을 받아(서버 카메라 · server_bridge 토픽 · 샘플 · 파일) perception 함수로 지도 후보를 만든다. 뒤에서 돈다."""
        if not self.mapper.ready:
            raise ValueError(self.mapper.error)
        source = str(raw.get('source', ''))
        if source == 'file':
            try:
                jpeg = base64.b64decode(str(raw.get('data', '')).split(',')[-1], validate=False)
            except ValueError:
                raise ValueError('파일을 읽지 못함') from None
            if not jpeg.startswith(b'\xff\xd8'):
                raise ValueError('JPEG 사진만 받는다')
            label = '파일 %s' % str(raw.get('name', ''))[:40]
            get = lambda: (jpeg, '')  # noqa: E731
        elif source == 'sample':
            path = self.mapper.samples().get(str(raw.get('sample', '')))
            if path is None:
                raise ValueError('그런 샘플이 없음')
            label = '샘플 %s (연습 경기장)' % path.stem
            get = lambda: (path.read_bytes(), '')  # noqa: E731
        elif source == 'server':
            camera = str(raw.get('camera', 'test1'))
            if camera not in drone_map.CAMERAS:
                raise ValueError('카메라는 %s 중 하나' % ' · '.join(drone_map.CAMERAS))
            url, code, _where = drone_map.server_settings(self.secrets)
            if not code:
                raise ValueError('임무코드가 없음 — nav_web/secrets.json(server_url · mission_code) 이나 환경변수 MAICON_MISSION_CODE')
            label = '서버 카메라 %s' % camera
            get = lambda: drone_map.fetch_drone(url, code, camera)  # noqa: E731
        elif source == 'topic':
            with self.lock:
                frame = self.uav
            if frame is None or time.monotonic() - frame[2] > 5.0:
                raise ValueError('/team/server/uav_image 에 5초 안에 온 사진이 없음 (server_bridge 가 켜져 있나?)')
            label = 'server_bridge 토픽'
            get = lambda: (frame[0], frame[1])  # noqa: E731
        else:
            raise ValueError('사진 출처를 고를 것')
        with self.lock:
            if self.drone['state'] == 'busy':
                raise ValueError('앞 사진을 아직 계산하는 중')
            self.drone.update(state='busy', detail='사진 받는 중', source=label, started=time.monotonic())
        width, height = ARENA['width'], ARENA['height']   # perception 노드와 같은 크기 (실측 612 × 500 — drone_boundary 기본값)
        threading.Thread(target=self._drone_job, args=(get, label, width, height), daemon=True).start()

    def _drone_job(self, get, label, width, height):
        try:
            jpeg, captured = get()
            with self.lock:
                self.drone['detail'] = '지도 만드는 중 (경계 → 상자)'
                self.drone_images = {'photo': jpeg}
            result, images = self.mapper.analyze_apart(jpeg, width, height)
            result.update(source=label, captured=captured, at=time.strftime('%H:%M:%S'))
            with self.lock:
                self.drone_images = images
                self.drone.update(state='done', detail='', result=result, version=self.drone['version'] + 1)
            self.say('드론 사진 → 지도 후보 (%s): 경계 %s, 상자 %d개, %.1f s%s' % (
                label, '맞음' if result['boundary']['ok'] else '검사 실패', len(result['boxes']), result['took_s'],
                '' if result['ok'] else ' — 믿기 어려움: ' + '; '.join(result['notes'][:2])))
        except Exception as error:  # noqa: BLE001  화면에 이유를 보인다
            text = str(error) if isinstance(error, ValueError) else repr(error)
            with self.lock:
                self.drone.update(state='error', detail=text, version=self.drone['version'] + 1)
            self.say('드론 사진 → 지도 실패 (%s): %s' % (label, text))

    def drone_apply(self, add_pillars=True):
        """마지막 후보를 지도로 쓴다. pillars: 고정 기둥 10개를 더한다 (드론 사진은 끝 띠 안 물체를 내지 않는다)."""
        with self.lock:
            result = self.drone['result']
            if result is None or not result.get('boxes'):
                raise ValueError('쓸 지도 후보가 없음')
            width, height = result['field']
            boxes = [(b['x'], b['y']) for b in result['boxes']]
            if add_pillars:
                boxes = with_pillars(boxes, self.config['box_size'], width, height)
            config = dict(self.config, width=width, height=height, boxes=boxes)
            if (round(width, 3), round(height, 3)) != (round(self.config['width'], 3), round(self.config['height'], 3)):
                config['bases'] = default_bases(width, height)   # 경기장 크기가 바뀌면 출발 칸도 그 크기의 도면값으로
        self.set_config(config, source='드론 사진')
        return {'boxes': len(boxes)}

    def drone_discard(self):
        with self.lock:
            self.drone.update(state='idle', detail='', result=None, version=self.drone['version'] + 1)
            self.drone_images = {}

    def on_uav(self, jpeg, captured):
        with self.lock:
            self.uav = (jpeg, captured, time.monotonic())

    def _save_config(self):
        try:
            self.config_path.write_text(json.dumps(self.config, ensure_ascii=False, indent=1), encoding='utf-8')
        except OSError as error:
            self.say('지도 설정을 저장하지 못함: %r' % (error,))

    def send_goal(self, raw):
        mode = MODES[raw.get('mode', 'normal')]
        yaw = raw.get('yaw')
        goal = {'x': float(raw['x']), 'y': float(raw['y']), 'yaw': None if yaw is None else float(yaw), 'mode': mode,
                'pos_tol': float(raw.get('pos_tol', 0.10)), 'yaw_tol': float(raw.get('yaw_tol', 0.10)),
                'timeout_s': float(raw.get('timeout_s', 60.0))}
        with self.lock:
            self.allowed, self.stop_reason = True, ''
            self.goal = dict(goal, sent=time.strftime('%H:%M:%S'), outcome=None, detail='')
            self.feedback = None
        self._pub_control()
        self._send_goal(goal)
        self.say('목표 %s (%.2f, %.2f)%s' % (raw.get('mode', 'normal'), goal['x'], goal['y'],
                                          '' if goal['yaw'] is None else ' 도착 방향 %.0f°' % math.degrees(goal['yaw'])))

    def stop(self, reason='정지 버튼'):
        with self.lock:
            self.allowed, self.stop_reason = False, reason
        self._pub_control()
        self._cancel()
        self.say('정지: ' + reason)

    # ------------------------------------------------------------------ 받은 것
    def on_pose(self, x, y, yaw):
        with self.lock:
            self.pose = (x, y, yaw, time.monotonic())
            if not self.trail or math.hypot(x - self.trail[-1][0], y - self.trail[-1][1]) > 0.03:
                self.trail.append((round(x, 3), round(y, 3)))
                if len(self.trail) > 4000:
                    del self.trail[:1000]

    def on_scan(self, ranges, angle_min, increment, range_min, range_max):
        points = []
        for index in range(0, len(ranges), 2):
            r = ranges[index]
            if r == r and range_min <= r < min(range_max, 12.0):
                angle = angle_min + index * increment
                points.append((round(LIDAR_X - r * math.sin(angle), 2), round(r * math.cos(angle), 2)))
        body = lidar_map.scan_to_body(ranges, angle_min, increment, range_min, range_max) if lidar_map is not None else None
        with self.lock:
            self.scan = {'points': points, 'pose': self.pose[:3] if self.pose else None, 't': time.monotonic()}
            self.scan_body = body

    def on_cmd(self, v, w):
        self.cmd = (v, w, time.monotonic())

    def on_odom(self, x, y):
        self.odom = (x, y, time.monotonic())

    def on_status(self, name, state, detail):
        with self.lock:
            self.status[name] = {'state': STATUS.get(state, str(state)), 'detail': detail, 't': time.monotonic()}

    def on_debug(self, text):
        try:
            data = json.loads(text)
        except ValueError:
            return
        with self.lock:
            self.debug[data.get('node', '?')] = (data, time.monotonic())

    def on_nav_map(self, width, height, box_size, boxes, vehicles):
        with self.lock:
            self.nav_map = {'width': width, 'height': height, 'box_size': box_size, 'boxes': boxes, 'vehicles': vehicles, 't': time.monotonic()}

    def on_feedback(self, phase, remaining, blocked):
        self.feedback = {'phase': phase, 'remaining': remaining, 'blocked': blocked, 't': time.monotonic()}

    def on_result(self, outcome, detail, final=None):
        name = OUTCOMES.get(outcome, str(outcome))
        with self.lock:
            if self.goal is not None:
                self.goal['outcome'], self.goal['detail'] = name, detail
        self.say('목표 끝: %s %s%s' % (name, detail, '' if final is None else ' — 끝난 자리 (%.2f, %.2f)' % final[:2]))

    # ------------------------------------------------------------------ 화면으로
    def state(self, trail_from=0):
        now = time.monotonic()

        def age(t):
            return None if t is None else round(now - t, 2)

        with self.lock:
            out = {
                'sim': self.sim, 'run_id': self.run_id, 'phase': self.phase, 'camp': self.camp, 'slot': self.slot,
                'allowed': self.allowed, 'stop_reason': self.stop_reason, 'config': self.config,
                'pose': None if self.pose is None else {'x': self.pose[0], 'y': self.pose[1], 'yaw': self.pose[2], 'age': age(self.pose[3])},
                'trail_from': min(trail_from, len(self.trail)), 'trail_total': len(self.trail),
                'trail': self.trail[trail_from:] if trail_from <= len(self.trail) else self.trail,
                'scan': None if self.scan is None else {'points': self.scan['points'], 'pose': self.scan['pose'], 'age': age(self.scan['t'])},
                'cmd': {'v': self.cmd[0], 'w': self.cmd[1], 'age': age(self.cmd[2])},
                'odom': None if self.odom is None else {'x': self.odom[0], 'y': self.odom[1], 'age': age(self.odom[2])},
                'status': {name: dict(item, age=age(item['t'])) for name, item in self.status.items()},
                'debug': {name: dict(data, age=age(t)) for name, (data, t) in self.debug.items()},
                'nav_map': None if self.nav_map is None else dict(self.nav_map, age=age(self.nav_map['t'])),
                'goal': self.goal, 'feedback': None if self.feedback is None else dict(self.feedback, age=age(self.feedback['t'])),
                'log': list(self.log), 'links': self.links(), 'warnings': self.warnings(),
                'drone': {'state': self.drone['state'], 'detail': self.drone['detail'], 'version': self.drone['version'],
                          'source': self.drone['source'], 'elapsed': age(self.drone['started']), 'result': self.drone['result'],
                          'images': sorted(self.drone_images)},
                'init': None if self.init is None else {'slot': self.init['slot'], 'region': self.init['region'], 'how': self.init['how'],
                                                        'age': age(self.init['started']), 'map_age': age(self.init['map_sent'])},
            }
            out['drone_options'] = self.drone_options()
            if trail_from > len(self.trail):
                out['trail_from'] = 0
            out.update(self.extra_state())
        return out


# ====================================================================== ROS
class RosBus(Bus):
    def __init__(self, config_path, deadman_s, secrets=()):
        import rclpy
        from geometry_msgs.msg import Point, PoseStamped, Twist
        from nav_msgs.msg import Odometry
        from rclpy.action import ActionClient
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import LaserScan
        from std_msgs.msg import String
        from team_interfaces.action import Navigate
        from team_interfaces.msg import ControlState, FieldMap, FieldRegion, FieldVehicle, MissionState, NodeStatus, ServerState, UavFrame

        self.m = dict(Point=Point, FieldMap=FieldMap, FieldRegion=FieldRegion, FieldVehicle=FieldVehicle, ServerState=ServerState,
                      ControlState=ControlState, MissionState=MissionState, Navigate=Navigate)
        try:
            # SIGINT 를 rclpy 가 먼저 받으면 context 만 닫히고 HTTP 서버는 계속 돈다(그 뒤 /state 마다 RCLError, run.sh 가 강제 종료).
            # 파이썬 기본 처리(KeyboardInterrupt → main 의 finally 에서 정지 명령 · 종료)에 맡긴다
            from rclpy.signals import SignalHandlerOptions
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        except ImportError:
            rclpy.init()
        self.rclpy = rclpy
        self.node = node = rclpy.create_node('web_drive')

        def qos(reliable=True, latched=False, depth=1):
            return QoSProfile(depth=depth, history=HistoryPolicy.KEEP_LAST,
                              reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL if latched else DurabilityPolicy.VOLATILE)

        self.field_pub = node.create_publisher(FieldMap, '/team/perception/field_map', qos())
        self.server_pub = node.create_publisher(ServerState, '/team/server/state', qos())
        self.control_pub = node.create_publisher(ControlState, '/team/control/state', qos())
        self.mission_pub = node.create_publisher(MissionState, '/team/mission/state', qos(latched=True))
        self.client = ActionClient(node, Navigate, '/team/navigation/navigate')
        self._sequence = 0
        self._handle = None
        super().__init__(config_path, deadman_s, secrets)

        node.create_subscription(PoseStamped, '/team/navigation/pose', self._pose, qos(reliable=False, depth=5))
        node.create_subscription(FieldMap, '/team/navigation/map', self._map, qos(latched=True))
        node.create_subscription(LaserScan, '/scan', lambda m: self.on_scan(m.ranges, m.angle_min, m.angle_increment, m.range_min, m.range_max), qos_profile_sensor_data)
        node.create_subscription(Twist, '/cmd_vel', lambda m: self.on_cmd(m.linear.x, m.angular.z), 10)
        node.create_subscription(Odometry, '/odom', lambda m: self.on_odom(m.pose.pose.position.x, m.pose.pose.position.y), qos_profile_sensor_data)
        node.create_subscription(NodeStatus, '/team/node_status', lambda m: self.on_status(m.node_name, m.state, m.detail), qos())
        node.create_subscription(String, '/team/navigation/debug', lambda m: self.on_debug(m.data), 5)
        # server_bridge 가 켜져 있으면 그 사진을 쓸 수 있다 (경기와 같은 출처). 발행 QoS 와 같게 best effort
        node.create_subscription(UavFrame, '/team/server/uav_image', lambda m: self.on_uav(bytes(m.image.data), m.captured_at_utc),
                                 qos(reliable=False, depth=2))
        node.create_timer(0.1, self.tick)
        self.executor = MultiThreadedExecutor(num_threads=4)
        self.executor.add_node(node)
        threading.Thread(target=self.executor.spin, daemon=True).start()

    def _stamp(self):
        return self.node.get_clock().now().to_msg()

    def _pose(self, message):
        self.on_pose(message.pose.position.x, message.pose.position.y, yaw_of(message.pose.orientation))

    def _map(self, message):
        self.on_nav_map(message.width, message.height, message.box_size, [[p.x, p.y] for p in message.boxes],
                        [{'role': int(v.role), 'x': v.pose.position.x, 'y': v.pose.position.y, 'yaw': yaw_of(v.pose.orientation)} for v in message.vehicles])

    def links(self):
        return {'SLAM (지도 구독)': self.field_pub.get_subscription_count(), 'SLAM (단계 구독)': self.server_pub.get_subscription_count(),
                '주행 (허용 구독)': self.control_pub.get_subscription_count(), '주행 (목표 액션)': int(self.client.server_is_ready())}

    def uav_publishers(self):
        return self.node.count_publishers('/team/server/uav_image')

    def warnings(self):
        """이 시험기가 대신 내는 토픽을 다른 노드도 내면, 두 값이 섞인다 (진짜 server_bridge · perception · supervisor 와 같이 켬)."""
        out = []
        for topic, who in (('/team/server/state', 'server_bridge'), ('/team/perception/field_map', 'perception'),
                           ('/team/control/state', 'supervisor'), ('/team/mission/state', 'supervisor')):
            others = self.node.count_publishers(topic) - 1
            if others > 0:
                out.append('%s 를 다른 노드(%s?)도 %d개 냄 — 시험기와 같이 켜면 값이 섞인다' % (topic, who, others))
        return out

    def _pub_server(self):
        m = self.m['ServerState']()
        m.header.stamp = self._stamp()
        m.run_id, m.connected, m.team_assigned = self.run_id, True, bool(self.slot)
        m.slot, m.camp, m.phase, m.ready, m.detail = self.slot, self.camp, self.phase, True, 'web_drive 시험기'
        self.server_pub.publish(m)

    def _pub_control(self):
        m = self.m['ControlState']()
        m.header.stamp = self._stamp()
        self._sequence += 1
        m.run_id, m.sequence, m.motion_allowed, m.lease_ms = self.run_id, self._sequence, bool(self.allowed), 500
        m.reason = '' if self.allowed else 'web_drive: ' + self.stop_reason
        self.control_pub.publish(m)

    def _pub_mission(self):
        m = self.m['MissionState']()
        m.header.stamp = self._stamp()
        m.run_id, m.phase, m.slot, m.camp, m.detail = self.run_id, self.phase, self.slot, self.camp, 'web_drive 시험기'
        self.mission_pub.publish(m)

    def _pub_field_map(self, bases, self_pose):
        config = self.config
        m = self.m['FieldMap']()
        m.header.stamp = self._stamp()
        m.header.frame_id = 'arena'
        m.width, m.height, m.box_size = config['width'], config['height'], config['box_size']
        m.boxes = [self.m['Point'](x=float(x), y=float(y), z=0.0) for x, y in config['boxes']]
        for base in bases:
            name = base['name']
            row = int(name[-1]) if len(name) == 3 and name[1] == '-' and name[-1] in '123' else 0
            m.bases.append(self.m['FieldRegion'](name=name, min_x=base['min_x'], min_y=base['min_y'],
                                                 max_x=base['max_x'], max_y=base['max_y'], row=row))
        if self_pose is not None:
            vehicle = self.m['FieldVehicle'](role=self.m['FieldVehicle'].SELF)
            vehicle.pose.position.x, vehicle.pose.position.y = self_pose[0], self_pose[1]
            vehicle.pose.orientation.z, vehicle.pose.orientation.w = math.sin(self_pose[2] / 2), math.cos(self_pose[2] / 2)
            m.vehicles.append(vehicle)
        self.field_pub.publish(m)

    def _send_goal(self, goal):
        Navigate = self.m['Navigate']
        if not self.client.server_is_ready():
            self.on_result(0, '주행 노드의 목표 액션이 없음 (navigation_drive 가 켜져 있나?)')
            return
        g = Navigate.Goal()
        g.run_id = self.run_id
        g.goal.header.frame_id = 'arena'
        g.goal.header.stamp = self._stamp()
        g.goal.pose.position.x, g.goal.pose.position.y = goal['x'], goal['y']
        yaw = goal['yaw'] or 0.0
        g.goal.pose.orientation.z, g.goal.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        g.mode, g.ignore_yaw = goal['mode'], goal['yaw'] is None
        g.position_tolerance_m, g.yaw_tolerance_rad = goal['pos_tol'], goal['yaw_tol']
        g.execution_timeout_ms = int(goal['timeout_s'] * 1000)

        def accepted(future):
            handle = future.result()
            if not handle.accepted:
                self.on_result(0, '목표가 받아들여지지 않음')
                return
            self._handle = handle
            handle.get_result_async().add_done_callback(finished)

        def finished(future):
            result = future.result().result
            p = result.final_pose.pose
            self.on_result(int(result.outcome), result.detail, (p.position.x, p.position.y, yaw_of(p.orientation)))

        def feedback(message):
            fb = message.feedback
            self.on_feedback(fb.phase, fb.remaining_distance_m if fb.distance_valid else None, bool(fb.blocked))

        self.client.send_goal_async(g, feedback_callback=feedback).add_done_callback(accepted)

    def _cancel(self):
        handle = self._handle
        if handle is not None:
            handle.cancel_goal_async()

    def shutdown(self):
        try:
            with self.lock:
                self.allowed, self.stop_reason = False, '시험기 종료'
            self._pub_control()
            self._cancel()
            time.sleep(0.2)
        finally:
            self.rclpy.try_shutdown()


# ====================================================================== 가상 경기장 (PC)
class SimBus(Bus):
    sim = True

    def __init__(self, config_path, deadman_s, slot='B-2', slip=0.85, speed=1.0, secrets=(), robot='실차'):
        import sim_world as W
        self.W = W
        super().__init__(config_path, deadman_s, secrets)
        config = self.config
        self.world = W.World(config['width'], config['height'], config['box_size'], [tuple(b) for b in config['boxes']])
        self.robot = W.SimRobot(self.world, W.slot_start_pose(slot), odom_offset=1.0, slip=slip, **(W.REAL_ROBOT if robot in ('실차', 'real') else {}))
        self.loop = W.ClosedLoop(self.world, self.robot, drive_params={'debug_hz': 5.0})
        self.loop.listeners.update(pose=self._pose, map=self._map, cmd=lambda m: self.on_cmd(float(m.linear.x), float(m.angular.z)),
                                   status=lambda m: self.on_status(m.node_name, m.state, m.detail),
                                   scan=lambda m: self.on_scan(m.ranges, m.angle_min, m.angle_increment, m.range_min, m.range_max),
                                   debug=lambda m: self.on_debug(m.data))
        self.speed = speed
        self._sync()
        threading.Thread(target=self._run, daemon=True).start()

    def _sync(self):
        self.loop.run_id, self.loop.phase, self.loop.camp, self.loop.slot, self.loop.allowed = self.run_id, self.phase, self.camp, self.slot, self.allowed

    def _run(self):
        step = 0
        next_t = time.monotonic()
        while True:
            self._sync()
            self.loop.step()
            self.on_odom(self.robot.ox, self.robot.oy)
            step += 1
            if step % 2 == 0:
                self.tick()
            next_t += self.loop.DT / self.speed
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()

    def _pose(self, message):
        self.on_pose(message.pose.position.x, message.pose.position.y, yaw_of(message.pose.orientation))

    def _map(self, message):
        self.on_nav_map(message.width, message.height, message.box_size, [[p.x, p.y] for p in message.boxes],
                        [{'role': int(v.role), 'x': v.pose.position.x, 'y': v.pose.position.y, 'yaw': yaw_of(v.pose.orientation)} for v in message.vehicles])

    def links(self):
        return {'가상 경기장': 1}

    def extra_state(self):
        return {'truth': {'x': self.robot.x, 'y': self.robot.y, 'yaw': self.robot.yaw}, 'hidden': [list(b) for b in self.world.hidden],
                'collisions': self.robot.collisions}

    def _pub_server(self):
        with self.loop.lock:
            self._sync()
            self.loop.publish_server()

    def _pub_control(self):
        self.loop.allowed = self.allowed

    def _pub_mission(self):
        from types import SimpleNamespace as NS
        with self.loop.lock:
            self.loop.drive._receive_mission(NS(run_id=self.run_id))

    def _pub_field_map(self, bases, self_pose):
        config = self.config
        with self.loop.lock:
            # 실제 세계의 상자 = 지도의 상자 + 숨은 상자. 경기장 크기를 바꾸면 세계도 따라 바뀐다
            self.world.width, self.world.height, self.world.box_size = config['width'], config['height'], config['box_size']
            self.world.boxes = [tuple(b) for b in config['boxes']]
            self.loop.publish_map(bases=[(b['name'], b['min_x'], b['min_y'], b['max_x'], b['max_y']) for b in bases], self_pose=self_pose)

    def _send_goal(self, goal):
        def feedback(fb):
            self.on_feedback(fb.phase, fb.remaining_distance_m if fb.distance_valid else None, bool(fb.blocked))

        def result(message):
            p = message.final_pose.pose
            self.on_result(int(message.outcome), message.detail, (p.position.x, p.position.y, 0.0))

        self._sync()
        self.loop.send_goal(goal['x'], goal['y'], yaw=goal['yaw'], mode=goal['mode'], pos_tol=goal['pos_tol'],
                            yaw_tol=goal['yaw_tol'], timeout_s=goal['timeout_s'], on_feedback=feedback, on_result=result)

    def _cancel(self):
        self.loop.cancel()

    def toggle_hidden(self, x, y):
        """지도에는 없는 상자를 놓거나 치운다 (장애물 회피를 볼 때)."""
        with self.loop.lock:
            for box in list(self.world.hidden):
                if abs(box[0] - x) < self.world.box_size / 2 and abs(box[1] - y) < self.world.box_size / 2:
                    self.world.hidden.remove(box)
                    return
            self.world.hidden.append((round(x, 2), round(y, 2)))

    def teleport(self, x, y, yaw):
        with self.loop.lock:
            self.robot.x, self.robot.y, self.robot.yaw = x, y, yaw

    def shutdown(self):
        pass


# ====================================================================== HTTP
def make_handler(bus):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, content_type='application/json; charset=utf-8'):
            data = body if isinstance(body, bytes) else body.encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path, _, query = self.path.partition('?')
            if path == '/':
                self._send(200, (HERE / 'page.html').read_bytes(), 'text/html; charset=utf-8')
            elif path.startswith('/drone/') and path.endswith('.jpg'):
                with bus.lock:
                    data = bus.drone_images.get(path[7:-4])
                if data is None:
                    self._send(404, '{"error":"그림 없음"}')
                else:
                    self._send(200, data, 'image/jpeg')
            elif path == '/state':
                bus.client_seen()
                trail_from = 0
                for part in query.split('&'):
                    if part.startswith('trail='):
                        try:
                            trail_from = max(0, int(part[6:]))
                        except ValueError:
                            pass
                self._send(200, json.dumps(bus.state(trail_from), ensure_ascii=False, separators=(',', ':')))
            else:
                self._send(404, '{"error":"없는 주소"}')

        def do_POST(self):
            bus.client_seen()
            try:
                length = int(self.headers.get('Content-Length') or 0)
                if length > 20_000_000:
                    raise ValueError('보낸 것이 너무 큼 (20 MB 넘음)')
                body = json.loads(self.rfile.read(length) or b'{}')
                action = self.path.rsplit('/', 1)[-1]
                if action == 'stop':
                    bus.stop(body.get('reason', '정지 버튼'))
                elif action == 'goal':
                    bus.send_goal(body)
                elif action == 'map':
                    bus.set_config(body, source='직접 편집')
                elif action == 'init_slot':
                    bus.init_region(body['slot'], body.get('region'))
                elif action == 'init_pose':
                    bus.init_pose(body['x'], body['y'], body['yaw'])
                elif action == 'build_map':
                    bus.build_map(body['x'], body['y'], body['yaw'], bool(body.get('keep')))
                elif action == 'add_boxes':
                    bus.add_boxes()
                elif action == 'drone':
                    bus.drone_start(body)
                elif action == 'drone_apply':
                    bus.drone_apply(bool(body.get('pillars', True)))
                elif action == 'drone_discard':
                    bus.drone_discard()
                elif action == 'sim_hidden' and bus.sim:
                    bus.toggle_hidden(float(body['x']), float(body['y']))
                elif action == 'sim_teleport' and bus.sim:
                    bus.teleport(float(body['x']), float(body['y']), float(body['yaw']))
                else:
                    self._send(404, '{"error":"없는 동작"}')
                    return
                self._send(200, '{"ok":true}')
            except (KeyError, ValueError, TypeError) as error:
                self._send(400, json.dumps({'error': str(error)}, ensure_ascii=False))
            except Exception as error:  # noqa: BLE001  화면에 이유를 보여 준다
                self._send(500, json.dumps({'error': repr(error)}, ensure_ascii=False))

    return Handler


def main():
    parser = argparse.ArgumentParser(description='웹 주행 시험기')
    parser.add_argument('--sim', action='store_true', help='ROS 없이 가상 경기장에서 (PC)')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', '5960')))
    parser.add_argument('--config', default=str(HERE / 'map.local.json'), help='지도 설정을 저장할 파일')
    parser.add_argument('--deadman', type=float, default=3.0, help='화면이 이 시간(s) 동안 끊기면 주행 허용을 끈다 (0 = 끄지 않음)')
    parser.add_argument('--sim-slot', default='B-2')
    parser.add_argument('--sim-slip', type=float, default=0.85, help='가상 로봇: 바퀴가 돈 것 대비 차체가 도는 비율')
    parser.add_argument('--sim-robot', choices=('실차', '이상', 'real', 'ideal'), default='실차',
                        help='가상 로봇의 반응: 실차(real) = 지연 0.3 s · 짧은 회전 명령으로는 안 돎 (2026-11-18 팀 테스트와 같게), 이상(ideal) = 지연 0.1 s')
    parser.add_argument('--secrets', action='append', default=None,
                        help='서버 주소 · 임무코드 파일 (server_url · mission_code). 여러 번 줄 수 있다. 기본: $TEAM_SECRETS, 지도 설정 옆 secrets.json, ~/team/secrets.json')
    args = parser.parse_args()
    secrets = args.secrets or [p for p in (os.environ.get('TEAM_SECRETS'), str(Path(args.config).resolve().parent / 'secrets.json'),
                                           os.path.expanduser('~/team/secrets.json')) if p]
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass
    # run.sh 는 bash 의 백그라운드(&)로 띄운다 — 그러면 SIGINT 가 「무시」 로 물려받아져 KeyboardInterrupt 가 나지 않는다. 다시 건다
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    bus = (SimBus(args.config, args.deadman, args.sim_slot, args.sim_slip, secrets=secrets, robot=args.sim_robot) if args.sim
           else RosBus(args.config, args.deadman, secrets=secrets))
    server = ThreadingHTTPServer(('0.0.0.0', args.port), make_handler(bus))
    server.daemon_threads = True
    print('[web_drive] http://<이 컴퓨터의 주소>:%d  (%s)' % (args.port, '가상 경기장' if args.sim else '로봇'), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        bus.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
