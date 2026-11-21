"""그림 판독을 그림 크기(박스 높이 px)별로 잰다 — 먼 그림에서 무엇이 틀리는지 보려고 (명령줄 도구).

노드와 같은 길을 그대로 탄다: PictureDetector.detect → (zoom.refine) → pictures.build_pictures.

    python eval_by_size.py --weights pictures.pt --data <YOLO 데이터셋 폴더> [<폴더2> ...]
    python eval_by_size.py --weights pictures.pt --data ds --zoom-max-height 96     확대 재판독을 켜고
    python eval_by_size.py ... --compare                                            끈 것 · 켠 것을 나란히
    python eval_by_size.py ... --shrink 1,0.5,0.33,0.25                             거리 모사: 사진을 줄여 같은 크기 화면 가운데에

데이터셋: data.yaml(names) + images/<split>/*.jpg + labels/<split>/*.txt (UGV 조종 콘솔에서 내려받은 형식).
정답 박스가 그림 전체든 그림 속 물체만이든 맞춘다 — IoU ≥ 0.3 이거나 작은 쪽 상자의 70 % 이상이 겹치면 같은 그림.
그림이 아닌 정답(ugv · target_sensor · antenna 등)은 세지 않는다.

크기 구간마다:
    정답   정답 그림 수
    맞음   종류까지 맞게 답한 비율 (supervisor 가 받는 답 기준: min_confidence 미만이면 unknown)
    unknown  찾았지만 점수가 낮아 unknown 으로 나간 비율
    낮은점수_맞음  그 unknown 중 모델의 1등 종류는 맞았던 비율 (임계값만 고치면 살릴 수 있는 몫)
    다른종류  찾았지만 다른 종류로 답한 비율
    놓침   그림 답이 아예 없는 비율
끝에 정답과 짝이 없는 그림 답(오탐) 수를 적는다 (정답 라벨이 사진의 모든 그림에 있을 때만 뜻이 있다).

거리 모사 (--shrink): 라벨이 붙은 그림은 대부분 가깝다. 사진 전체를 s 배로 줄여(INTER_AREA) 같은 크기의 회색 화면
가운데에 놓으면, 같은 그림이 약 1/s 배 멀리서 찍힌 것과 비슷해진다 (원근 · 초점 · 조명 차이는 모사하지 못한다).
정답 종류를 그대로 알므로 '같은 그림이 멀어지면 어떻게 무너지는가'를 잴 수 있다.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # perception 패키지 (설치 없이 바로 쓰기)

from perception import pictures, zoom  # noqa: E402
from perception.detector import PictureDetector  # noqa: E402


def read_names(folder):
    names, inside = {}, False
    for line in open(os.path.join(folder, 'data.yaml'), encoding='utf-8'):
        if line.strip().startswith('names:'):
            inside = True
            continue
        if inside:
            key, sep, value = line.strip().partition(':')
            if not sep or not key.strip().isdigit():
                break
            names[int(key)] = value.strip().strip('"\'')
    return names


def read_truth(label_path, names, width, height):
    """정답 그림 [(kind, (x1, y1, x2, y2))]. 그림이 아니거나 10가지에 없는 클래스는 뺀다."""
    out = []
    if not os.path.exists(label_path):
        return out
    for line in open(label_path):
        p = line.split()
        if len(p) != 5:
            continue
        name = names.get(int(p[0]), p[0])
        kind = pictures.kind_of(name)
        if kind is None or not pictures.is_picture(name):
            continue
        x, y, w, h = (float(v) for v in p[1:])
        out.append((kind, ((x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height)))
    return out


def overlap(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    iou = inter / (area_a + area_b - inter)
    contained = inter / min(area_a, area_b)
    return iou if iou >= 0.3 else (0.29 * contained if contained >= 0.7 else 0.0)


def match(answers, truth):
    """점수 높은 답부터 탐욕 짝짓기. -> {정답 번호: 답 번호}."""
    pairs, used = {}, set()
    for ai in sorted(range(len(answers)), key=lambda i: -answers[i]['confidence']):
        best, best_g = 0.0, None
        for gi, (_, gbox) in enumerate(truth):
            if gi in used:
                continue
            s = overlap(answers[ai]['box'], gbox)
            if s > best:
                best, best_g = s, gi
        if best_g is not None:
            used.add(best_g)
            pairs[best_g] = ai
    return pairs


def answers_of(detections, width, height, min_confidence):
    """build_pictures 의 답 + 모델이 낸 1등 종류(raw_kind)와 픽셀 박스 (짝짓기용)."""
    found = pictures.build_pictures(detections, width, height, min_confidence=min_confidence)
    for item in found:
        item['box'] = ((item['x_center'] - item['width'] / 2) * width, (item['y_center'] - item['height'] / 2) * height,
                       (item['x_center'] + item['width'] / 2) * width, (item['y_center'] + item['height'] / 2) * height)
        best = max((d for d in detections if pictures.is_picture(d[0])),
                   key=lambda d: overlap(item['box'], d[2:6]) + d[1] * 1e-6, default=None)
        item['raw_kind'] = pictures.kind_of(best[0]) if best else None
    return found


class Tally:
    KEYS = ('정답', '맞음', 'unknown', '낮은점수_맞음', '다른종류', '놓침')

    def __init__(self, edges):
        self.edges = edges
        self.rows = {self.label(i): dict.fromkeys(self.KEYS, 0) for i in range(len(edges) + 1)}
        self.false_alarms = 0
        self.frames = self.exact = 0

    def label(self, i):
        lo = self.edges[i - 1] if i > 0 else 0
        return ('%d~%d px' % (lo, self.edges[i])) if i < len(self.edges) else ('%d px~' % lo)

    def bin_of(self, height):
        for i, edge in enumerate(self.edges):
            if height < edge:
                return self.label(i)
        return self.label(len(self.edges))

    def add(self, truth, answers):
        pairs = match(answers, truth)
        for gi, (kind, box) in enumerate(truth):
            row = self.rows[self.bin_of(box[3] - box[1])]
            row['정답'] += 1
            if gi not in pairs:
                row['놓침'] += 1
                continue
            answer = answers[pairs[gi]]
            if answer['kind'] == kind:
                row['맞음'] += 1
            elif answer['kind'] == pictures.UNKNOWN:
                row['unknown'] += 1
                row['낮은점수_맞음'] += int(answer['raw_kind'] == kind)
            else:
                row['다른종류'] += 1
        self.false_alarms += len(answers) - len(pairs)
        self.frames += 1
        self.exact += int(sorted(k for k, _ in truth) == sorted(a['kind'] for a in answers))

    def lines(self):
        out = ['%-12s %5s %7s %8s %13s %8s %6s' % ('크기', *self.KEYS)]
        total = dict.fromkeys(self.KEYS, 0)
        for name, row in list(self.rows.items()) + [('전체', None)]:
            if row is None:
                row = total
            else:
                for k in self.KEYS:
                    total[k] += row[k]
            n = max(row['정답'], 1)
            out.append('%-12s %5d %7s %8s %13s %8s %6s' % (
                name, row['정답'], *('%5.1f%%' % (100.0 * row[k] / n) for k in self.KEYS[1:])))
        out.append('오탐(정답 없는 그림 답) %d · 프레임 종류별 개수 완전일치 %d/%d' % (self.false_alarms, self.exact, self.frames))
        return out


def shrink_frame(rgb, truth, s, fill=114):
    """사진을 s 배로 줄여 같은 크기 회색 화면 가운데에 놓는다. -> (새 사진, 옮긴 정답)."""
    import cv2
    import numpy as np
    if s >= 1.0:
        return rgb, truth
    height, width = rgb.shape[:2]
    w, h = max(1, int(round(width * s))), max(1, int(round(height * s)))
    canvas = np.full_like(rgb, fill)
    ox, oy = (width - w) // 2, (height - h) // 2
    canvas[oy:oy + h, ox:ox + w] = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    moved = [(k, (b[0] * s + ox, b[1] * s + oy, b[2] * s + ox, b[3] * s + oy)) for k, b in truth]
    return canvas, moved


def image_files(folders):
    files = []
    for folder in folders:
        files += sorted(glob.glob(os.path.join(folder, 'images', '*', '*.jpg')))
        files += sorted(glob.glob(os.path.join(folder, 'images', '*', '*.png')))
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--data', nargs='+', required=True, help='YOLO 데이터셋 폴더 (data.yaml 이 있는 곳)')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--detect-conf', type=float, default=0.15)
    ap.add_argument('--min-confidence', type=float, default=0.5)
    ap.add_argument('--bins', default='48,96', help='크기 구간 경계 (박스 높이 px, 쉼표로)')
    ap.add_argument('--zoom-max-height', type=int, default=0, help='이 높이(px) 이하 그림을 확대해 다시 본다. 0 = 끔')
    ap.add_argument('--zoom-fill', type=float, default=0.25)
    ap.add_argument('--zoom-limit', type=int, default=4)
    ap.add_argument('--shrink', default='1', help='거리 모사 배율들 (쉼표로). 1 = 원본 그대로')
    ap.add_argument('--compare', action='store_true', help='확대 재판독을 끈 것과 켠 것을 나란히 (--zoom-max-height 필요)')
    args = ap.parse_args()

    import numpy as np
    from PIL import Image

    detector = PictureDetector(args.weights, imgsz=args.imgsz, detect_conf=args.detect_conf, device=args.device)
    detector.load()
    edges = [int(v) for v in args.bins.split(',') if v.strip()]
    settings = []
    if args.compare or args.zoom_max_height <= 0:
        settings.append(('확대 끔', 0))
    if args.zoom_max_height > 0:
        settings.append(('확대 켬 (≤%d px, fill %.2f)' % (args.zoom_max_height, args.zoom_fill), args.zoom_max_height))
    shrinks = [float(v) for v in args.shrink.split(',') if v.strip()]
    tallies = {(name, s): Tally(edges) for name, _ in settings for s in shrinks}
    seconds = {name: 0.0 for name, _ in settings}
    runs = 0

    files = image_files(args.data)
    names_by_folder = {}
    for path in files:
        folder = path.split(os.sep + 'images' + os.sep)[0]
        names = names_by_folder.setdefault(folder, read_names(folder))
        rgb = np.asarray(Image.open(path).convert('RGB'))
        height, width = rgb.shape[:2]
        label = path.replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep).rsplit('.', 1)[0] + '.txt'
        full_truth = read_truth(label, names, width, height)
        for s in shrinks:
            frame, truth = shrink_frame(rgb, full_truth, s)
            if s < 1.0 and not truth:
                continue                      # 정답 그림이 없는 사진은 줄여 볼 까닭이 없다
            runs += 1
            started = time.monotonic()
            base = detector.detect(frame)
            base_seconds = time.monotonic() - started
            for name, max_height in settings:
                started = time.monotonic()
                detections = base if max_height <= 0 else zoom.refine(
                    frame, base, detector.detect, pictures.is_picture,
                    max_height=max_height, fill=args.zoom_fill, limit=args.zoom_limit)
                seconds[name] += base_seconds + time.monotonic() - started
                tallies[(name, s)].add(truth, answers_of(detections, width, height, args.min_confidence))

    print('가중치 %s · 사진 %d장 · min_confidence %.2f' % (args.weights, len(files), args.min_confidence))
    for name, _ in settings:
        print('\n[%s]  사진당 %.0f ms (%s)' % (name, 1000.0 * seconds[name] / max(runs, 1), args.device))
        for s in shrinks:
            if len(shrinks) > 1:
                print('  -- 거리 모사 x%.2f (약 %.1f 배 멀리)' % (s, 1.0 / s))
            for line in tallies[(name, s)].lines():
                print('  ' + line)


if __name__ == '__main__':
    main()
