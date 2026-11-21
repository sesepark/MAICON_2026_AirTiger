"""perception 경계 찾기(drone_boundary.detect_boundary)의 시간과 결과를 설정값만 바꿔 비교한다. perception 코드는 고치지 않고 감싸서 잰다.

    python time_boundary.py                      # test/data 의 drone_test1.jpg · drone_test2.jpg
    python time_boundary.py 사진1.jpg 사진2.jpg   # 다른 사진 (로봇 기록의 perception/drone_used.jpg 등)

numpy · OpenCV 가 있어야 한다. 단계 시간은 안쪽 함수가 겹쳐 세어진다 (rough 안에 circle_points · plumb_k · score 가 들어 있다).
2026-11-18 결과: 개발/src/perception/docs/주행_쪽_수정_제안_2026-11-18.md
"""
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PERCEPTION = REPO / '개발' / 'src' / 'perception'
sys.path.insert(0, str(PERCEPTION))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from perception import drone_boundary as B  # noqa: E402

timers = {}


def timed(name, fn):
    def wrap(*a, **k):
        began = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            timers[name] = timers.get(name, 0.0) + time.perf_counter() - began
    return wrap


for name in ('_rough', '_refine', '_plumb_k', '_circle_points', '_score', '_row_pairs'):
    setattr(B, name, timed(name, getattr(B, name)))

VARIANTS = [
    ('기본', {}),
    ('refine=1', dict(refine=1)),
    ('max_combos=20', dict(max_combos=20)),
    ('max_combos=10 · refine=1', dict(max_combos=10, refine=1)),
    ('full_lens=False', dict(full_lens=False)),
]


def load(path):
    path = Path(path)
    if not path.exists():
        path = PERCEPTION / 'test' / 'data' / path
    return cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)  # imread 는 한글 경로를 못 읽는다


def main():
    for name in sys.argv[1:] or ['drone_test1.jpg', 'drone_test2.jpg']:
        img = load(name)
        for label, change in VARIANTS:
            timers.clear()
            p = replace(B.BoundaryParams(), **change)
            best = None
            for _ in range(2):  # 두 번 재서 빠른 쪽
                began = time.perf_counter()
                b = B.detect_boundary(img, p)
                took = time.perf_counter() - began
                best = took if best is None else min(best, took)
            parts = ' · '.join('%s %.2f' % (k.strip('_'), v / 2) for k, v in sorted(timers.items(), key=lambda kv: -kv[1]))
            print('%s | %-24s | %.2f s | ok=%s 끝띠 %.3f CM %.3f 원 %.3f RMS %.1f cm 후보 %s | %s' % (
                Path(name).name, label, best, b.ok, b.strip or 0, b.cm_half or 0, b.circle_r or 0, (b.rms_m or 0) * 100,
                b.debug.get('candidates'), parts))


if __name__ == '__main__':
    main()
