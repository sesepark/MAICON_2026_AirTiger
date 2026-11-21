"""드론 사진 → 경기장 지도. perception 의 함수를 고치지 않고 그대로 불러 쓴다 (경기 때 perception 노드가 할 일과 같은 계산).

  drone_boundary.detect_boundary   바닥 흰 선 → 화면 픽셀 ↔ 경기장 좌표
  drone_boxes.detect_boxes         상자 윗면 → 상자 가운데 (높이 보정 포함)
  drone_field_map.field_map        → FieldMap 칸 (bases 는 slot 이름 B-1 … 로)

perception 코드를 찾는 곳 (먼저 찾은 것):
  1. 환경변수 PERCEPTION_SRC (perception 패키지 폴더, 그 안에 perception/drone_boundary.py)
  2. 저장소의 개발/src/perception — perception 브랜치가 main 에 합쳐지면 여기
  3. 이 폴더의 _perception/ — 아직 합쳐지지 않았으면 git 의 origin/perception 에서 꺼내 둔다 (robot.sh deploy 도 같은 일을 한다)
numpy · OpenCV 가 있어야 한다 (로봇 컨테이너: OpenCV 4.5.4).

서버 카메라에서 받기: 서버 주소 · 임무코드를 환경변수(MAICON_SERVER_URL · MAICON_MISSION_CODE) 나 secrets.json
(server_bridge 와 같은 칸: server_url · mission_code) 에서 읽는다. 임무코드는 어디에도 찍지 않는다.
"""
import base64
import importlib
import json
import os
import subprocess
import sys
import time
import types
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULES = ('drone_boundary', 'drone_boxes', 'drone_field_map')
SAMPLES = ('drone_test1.jpg', 'drone_test2.jpg')
BRANCH = 'origin/perception'
CACHE = HERE / '_perception'
PRACTICE_URL = 'http://192.168.110.50:8100'   # 연습 서버 시험 포트 (perception tools/collect_drone_frames.py 기본값)
CAMERAS = ('test1', 'test2', 'main')


def _candidates():
    if os.environ.get('PERCEPTION_SRC'):
        yield Path(os.environ['PERCEPTION_SRC'])
    yield HERE.parents[2] / 'perception'      # web_drive → 시험기록 → navigation → src
    yield CACHE


def _has_code(root):
    return all((root / 'perception' / (name + '.py')).is_file() for name in MODULES)


def _fill_cache():
    """git 의 perception 브랜치에서 필요한 파일만 꺼낸다. 실패하면 False (git 이 없거나 브랜치를 받지 않음)."""
    try:
        top = subprocess.run(['git', '-C', str(HERE), 'rev-parse', '--show-toplevel'], capture_output=True, text=True, timeout=10)
        if top.returncode != 0:
            return False
        prefix = HERE.parents[2].relative_to(Path(top.stdout.strip())).as_posix() + '/perception/'
        files = [('perception/%s.py' % name) for name in MODULES] + [('test/data/' + name) for name in SAMPLES]
        blobs = {}
        for rel in files:
            out = subprocess.run(['git', '-C', str(HERE), 'show', '%s:%s%s' % (BRANCH, prefix, rel)], capture_output=True, timeout=20)
            if out.returncode != 0:
                if rel.startswith('perception/'):
                    return False
                continue
            blobs[rel] = out.stdout
        for rel, data in blobs.items():
            path = CACHE / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (CACHE / 'SOURCE.txt').write_text('git %s 에서 꺼냄 (%s)\n' % (BRANCH, time.strftime('%Y-%m-%d %H:%M')), encoding='utf-8')
        return True
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


class DroneMapper:
    """perception 함수를 불러 두고, 사진 한 장을 지도로 바꾼다."""

    def __init__(self):
        self.error = None
        self.source = None
        self.root = None
        self.m = {}
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
        except ImportError as error:
            self.error = 'OpenCV · numpy 가 없음 (%s)' % error.name
            return
        root = next((r for r in _candidates() if _has_code(r)), None)
        if root is None and _fill_cache():
            root = CACHE
        if root is None:
            self.error = 'perception 의 드론 코드(drone_boundary.py 등)를 찾지 못함 — perception 브랜치를 받거나 PERCEPTION_SRC 를 줄 것'
            return
        name = '_web_drive_perception'
        package = types.ModuleType(name)
        package.__path__ = [str(root / 'perception')]
        sys.modules[name] = package
        try:
            self.m = {module: importlib.import_module(name + '.' + module) for module in MODULES}
        except Exception as error:  # noqa: BLE001  화면에 이유를 보인다
            self.error = 'perception 드론 코드를 불러오지 못함: %r' % (error,)
            return
        self.root = root
        self.source = 'PERCEPTION_SRC' if os.environ.get('PERCEPTION_SRC') else ('git %s 에서 꺼낸 것' % BRANCH if root == CACHE else '저장소 개발/src/perception')

    @property
    def ready(self):
        return self.error is None

    def samples(self):
        if self.root is None:
            return {}
        folder = self.root / 'test' / 'data'
        return {name.rsplit('.', 1)[0]: folder / name for name in SAMPLES if (folder / name).is_file()}

    def analyze_apart(self, jpeg, width, height, timeout=300.0):
        """analyze 를 따로 띄운 파이썬에서 한다 — 시험기 프로세스의 GIL 을 몇십 초 잡고 있으면 주행 허용 임대(10 Hz)가 늦어진다."""
        command = [sys.executable, str(Path(__file__).resolve()), 'analyze', repr(float(width)), repr(float(height))]
        env = dict(os.environ, PERCEPTION_SRC=str(self.root), PYTHONIOENCODING='utf-8')
        try:
            done = subprocess.run(command, input=jpeg, capture_output=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            raise ValueError('지도 계산이 %.0f s 안에 끝나지 않음' % timeout) from None
        if done.returncode != 0:
            tail = done.stderr.decode('utf-8', 'replace').strip().splitlines()[-3:]
            raise ValueError('지도 계산 실패: ' + ' | '.join(tail))
        payload = json.loads(done.stdout.decode('utf-8'))
        images = {name: base64.b64decode(data) for name, data in payload.pop('_images').items()}
        return payload, images

    def analyze(self, jpeg, width, height):
        """JPEG 바이트 → 결과 dict (+ 그림 JPEG 들). 연습 화면 한 장에 PC 로 약 9 초."""
        import cv2
        import numpy as np

        B, X, F = self.m['drone_boundary'], self.m['drone_boxes'], self.m['drone_field_map']
        began = time.monotonic()
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError('JPEG 을 읽지 못함')
        boundary = B.detect_boundary(img, B.BoundaryParams(field_w=float(width), field_h=float(height)))
        out = {'image_size': [int(img.shape[1]), int(img.shape[0])], 'field': [float(width), float(height)],
               'boundary': {'ok': bool(boundary.ok), 'rms_cm': None if boundary.rms_m is None else round(boundary.rms_m * 100, 1),
                            'strip': _round(boundary.strip), 'cm_half': _round(boundary.cm_half), 'circle_r': _round(boundary.circle_r),
                            'points': int(boundary.points), 'notes': [str(n) for n in boundary.notes][:6]}}
        images = {'photo': jpeg}
        if boundary.hom is None:
            out.update(ok=False, boxes=[], rejected=[], camera=None, notes=['경기장 경계를 찾지 못해 상자를 찾지 않음'])
        else:
            view = X.detect_boxes(img, boundary)
            lines, top = B.draw_before_after(img, boundary)
            overlay, _ = X.draw_boxes(lines, boundary, view)
            images['overlay'] = _jpeg(overlay)
            images['top'] = _jpeg(top, 80)
            fm = F.field_map(view, boundary, base_naming='slot')
            out.update(ok=bool(view.ok),
                       boxes=[{'x': round(float(b.x), 3), 'y': round(float(b.y), 3), 'score': round(float(b.score), 2), 'cut': b.cut} for b in view.boxes],
                       rejected=[{'x': round(float(r['x']), 2), 'y': round(float(r['y']), 2), 'reason': str(r.get('reason', ''))} for r in view.rejected][:40],
                       camera={'x': round(float(view.camera[0]), 2), 'y': round(float(view.camera[1]), 2), 'height': round(float(view.camera[2]), 2),
                               'source': view.height_source, 'faces': int(view.faces_used)},
                       notes=[str(n) for n in view.notes][:8],
                       perception_bases=fm['bases'], sectors=fm['sectors'])
        out['took_s'] = round(time.monotonic() - began, 1)
        return out, images


def _round(value, digits=3):
    return None if value is None else round(float(value), digits)


def _jpeg(bgr, quality=85):
    import cv2

    ok, data = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError('그림을 JPEG 으로 만들지 못함')
    return data.tobytes()


# ---------------------------------------------------------------------- 서버 카메라
def server_settings(secrets_paths):
    """(서버 주소, 임무코드 또는 '', 어디서 읽었는지). 임무코드 자체는 돌려주되 찍지 않는다."""
    secrets, where = {}, ''
    for path in secrets_paths:
        try:
            data = json.loads(Path(path).read_text(encoding='utf-8'))
            if isinstance(data, dict):
                secrets, where = data, str(path)
                break
        except (OSError, ValueError):
            continue
    url = os.environ.get('MAICON_SERVER_URL') or secrets.get('server_url') or PRACTICE_URL
    code = os.environ.get('MAICON_MISSION_CODE') or secrets.get('mission_code') or ''
    if os.environ.get('MAICON_MISSION_CODE'):
        where = '환경변수'
    return url.rstrip('/'), str(code).strip(), where


def fetch_drone(url, code, camera, timeout=10.0):
    """GET /api/report/drone → (JPEG, 촬영 시각 글자 또는 ''). 실패하면 ValueError (임무코드가 든 주소는 글에 넣지 않는다)."""
    if not code:
        raise ValueError('임무코드가 없음 — secrets.json(mission_code) 이나 환경변수 MAICON_MISSION_CODE')
    query = {'mission_code': code}
    if camera:
        query['camera'] = camera
    request = urllib.request.Request('%s/api/report/drone?%s' % (url, urllib.parse.urlencode(query)), method='GET')
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
            captured = response.headers.get('X-Captured-At', '')
    except urllib.error.HTTPError as error:
        detail = ''
        try:
            detail = json.loads(error.read().decode('utf-8')).get('detail', '')
        except (ValueError, UnicodeDecodeError, AttributeError, OSError):
            pass
        raise ValueError('서버가 %d 로 답함%s' % (error.code, ' — %s' % str(detail)[:120] if detail else '')) from None
    except (urllib.error.URLError, OSError) as error:
        raise ValueError('서버(%s)에 닿지 못함: %s' % (url, getattr(error, 'reason', error))) from None
    if not data.startswith(b'\xff\xd8'):
        raise ValueError('서버가 JPEG 이 아닌 것을 보냄')
    return data, captured


def _main():
    """python3 drone_map.py analyze <가로 m> <세로 m> < 사진.jpg  →  결과 JSON (그림은 base64). analyze_apart 가 부른다."""
    if len(sys.argv) != 4 or sys.argv[1] != 'analyze':
        print(_main.__doc__, file=sys.stderr)
        return 2
    mapper = DroneMapper()
    if not mapper.ready:
        print(mapper.error, file=sys.stderr)
        return 1
    result, images = mapper.analyze(sys.stdin.buffer.read(), float(sys.argv[2]), float(sys.argv[3]))
    result['_images'] = {name: base64.b64encode(data).decode('ascii') for name, data in images.items()}
    sys.stdout.buffer.write(json.dumps(result, ensure_ascii=False).encode('utf-8'))
    return 0


if __name__ == '__main__':
    sys.exit(_main())
