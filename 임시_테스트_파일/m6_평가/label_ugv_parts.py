#!/usr/bin/env python3
"""7판의 실제 경기 사진에 UGV 부품 라벨을 자동으로 붙인다 (부품 모델 다시 학습용).

  ugv        : 지금 부품 모델의 탐지 (점수 0.5 이상)
  antenna    : 색 덮개 찾기(ugvs.find_sleeves) + 모델 탐지(0.4 이상). 덮개 위아래의 검은 막대까지 넣으려고 위로 35 % · 아래로 10 % 넓힌다
  led_bar    : 밝은 막대 찾기(ugvs.find_led_bars) + 모델 탐지 가운데 색이 읽히는 것
  ugv_sensor : 수광 돔 찾기(아래 find_domes: 하얗게 번진 둥근 것 + 초록 테두리) + 모델 탐지(0.3 이상)
  그림 · target_sensor : 그림 모델(pictures_v2)의 탐지 (0.5 이상) — 비워 두면 그림을 배경으로 배운다
UGV 가 한 대도 없는 사진은 쓰지 않는다. 사용: m6v2_labels.py <패키지 폴더> [sheet]
"""
import glob, json, os, sys, random, collections
import cv2
import numpy as np
from ultralytics import YOLO
sys.path.insert(0, sys.argv[1])
from perception import ugvs

BASE = ['military_truck', 'mortar', 'tank', 'missile', 'ammo_box', 'drum', 'comms', 'outpost', 'radar', 'non_object', 'target_sensor', 'ugv']
NAMES = BASE + ['antenna', 'led_bar', 'ugv_sensor']
ROOT = os.path.expanduser('~/maicon-m3-model/raw/runs7')
OUT = os.path.expanduser('~/maicon-m3-model/datasets/m6v2')
SCAN = os.path.expanduser('~/m6work/scan')
HOLD_RUNS = ('20261119_180657', '20261120_003129')     # 학습에 쓰지 않고 남겨 두는 판 (본선 1판 · 연습 경기장 2 의 1판)


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h)


def find_domes(rgb, box):
    """UGV 박스(픽셀) 안의 수광 돔: 하얗게 번진 둥근 토막 + 둘레의 초록빛. -> [(x1, y1, x2, y2)] 픽셀."""
    h, w = rgb.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    x1, x2 = int(max(0, box[0] - 0.10 * bw)), int(min(w, box[2] + 0.10 * bw))
    y1, y2 = int(max(0, box[1] + 0.15 * bh)), int(min(h, box[3] - 0.05 * bh))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return []
    region = rgb[y1:y2, x1:x2].astype(np.int16)
    r, g, b = region[:, :, 0], region[:, :, 1], region[:, :, 2]
    top = region.max(axis=2); low = region.min(axis=2)
    bright = ((top >= 205) & (low >= 110)).astype(np.uint8)
    green = ((g > r + 18) & (g > b + 18) & (g > 90)).astype(np.uint8)
    both = cv2.morphologyEx(np.maximum(bright, green), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(both, connectivity=8)
    out = []
    for i in range(1, n):
        sx, sy, sw, sh, area = [int(v) for v in st[i]]
        d = max(sw, sh)
        if d < max(6, 0.07 * bh) or d > 0.32 * bh or not (0.55 <= sw / float(sh) <= 1.8) or area < 0.55 * sw * sh:
            continue
        gsum = int(green[sy:sy + sh, sx:sx + sw].sum()); bsum = int(bright[sy:sy + sh, sx:sx + sw].sum())
        if bsum < 0.25 * area or gsum < 0.04 * area:      # 하얀 가운데 + 초록 테두리 둘 다 있어야 한다
            continue
        out.append((x1 + sx, y1 + sy, x1 + sx + sw, y1 + sy + sh))
    return out


def main():
    sheet_only = len(sys.argv) > 2
    scan = {(r['run'], r['file']): r for r in json.load(open(SCAN + '/scan.json'))}
    teacher = YOLO(os.path.expanduser('~/m3work/weights/pictures_v2.pt'))
    for sub in ('train', 'hold'):
        os.makedirs('%s/%s/images' % (OUT, sub), exist_ok=True); os.makedirs('%s/%s/labels' % (OUT, sub), exist_ok=True)
    count = collections.Counter(); samples = []
    keys = sorted(scan)
    if sheet_only:
        random.seed(5); keys = random.sample(keys, 260)
    for run, name in keys:
        r = scan[(run, name)]
        f = '%s/%s/supervisor/photos/%s' % (ROOT, run, name)
        robots = [d for d in r['dets'] if d[0] == 'ugv' and d[1] >= 0.35]
        if not robots:
            continue
        bgr = cv2.imread(f); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); h, w = rgb.shape[:2]
        labels = []                                        # (class, x1, y1, x2, y2)
        for u in robots:
            ub = tuple(u[2:6]); labels.append(('ugv',) + ub)
            rb = (ub[0] / w, ub[1] / h, ub[2] / w, ub[3] / h)
            for s in ugvs.find_sleeves(rgb, rb):
                sx1, sy1, sx2, sy2 = s[0] * w, s[1] * h, s[2] * w, s[3] * h
                sh = sy2 - sy1
                labels.append(('antenna', sx1 - 1, max(0, sy1 - 0.35 * sh), sx2 + 1, min(h, sy2 + 0.10 * sh)))
            sens = [(d[2] / w, d[3] / h, d[4] / w, d[5] / h) for d in r['dets'] if d[0] == 'ugv_sensor']
            for l in ugvs.find_led_bars(rgb, rb, avoid=sens):
                crop = ugvs._crop(rgb, l)
                if crop is not None and ugvs.read_led_color(crop):
                    labels.append(('led_bar', l[0] * w, l[1] * h, l[2] * w, l[3] * h))
            for dm in find_domes(rgb, ub):
                labels.append(('ugv_sensor',) + dm)
        for d in r['dets']:                                # 모델이 찾은 부품도 넣는다 (겹치면 하나만)
            need = {'antenna': 0.4, 'led_bar': 0.4, 'ugv_sensor': 0.3}.get(d[0])
            if need is None or d[1] < need:
                continue
            box = tuple(d[2:6])
            if d[0] == 'led_bar':
                crop = ugvs._crop(rgb, (box[0] / w, box[1] / h, box[2] / w, box[3] / h))
                if crop is None or not ugvs.read_led_color(crop):
                    continue                               # 이름판 · 천장 조명을 led_bar 로 잡은 것은 뺀다
            if not any(ugvs._inside(((box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h) * 2, (u[2] / w, u[3] / h, u[4] / w, u[5] / h)) for u in robots):
                continue
            if all(not (l[0] == d[0] and iou(l[1:], box) > 0.3) for l in labels):
                labels.append((d[0],) + box)
        for c, p, b in zip(*[getattr(teacher.predict(bgr, conf=0.5, verbose=False)[0].boxes, k) for k in ('cls', 'conf', 'xyxy')]):
            nm = teacher.names[int(c)]
            if nm in BASE and nm != 'ugv':
                labels.append((nm,) + tuple(float(v) for v in b))
        sub = 'hold' if run in HOLD_RUNS else 'train'
        for l in labels:
            count[sub + ' ' + l[0] if l[0] in ('ugv', 'antenna', 'led_bar', 'ugv_sensor') else sub + ' pictures'] += 1
        count[sub + ' images'] += 1
        samples.append((f, labels))
        if sheet_only:
            continue
        stem = '%s__%s' % (run, name[:-4])
        cv2.imwrite('%s/%s/images/%s.jpg' % (OUT, sub, stem), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open('%s/%s/labels/%s.txt' % (OUT, sub, stem), 'w') as fh:
            for l in labels:
                x1, y1, x2, y2 = [float(v) for v in l[1:]]
                x1, x2 = max(0.0, min(w, x1)), max(0.0, min(w, x2)); y1, y2 = max(0.0, min(h, y1)), max(0.0, min(h, y2))
                if x2 - x1 >= 2 and y2 - y1 >= 2:
                    fh.write('%d %.6f %.6f %.6f %.6f\n' % (NAMES.index(l[0]), (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
    print(dict(sorted(count.items())))
    random.seed(7); random.shuffle(samples)
    colors = {'ugv': (0, 255, 255), 'antenna': (255, 0, 255), 'led_bar': (255, 128, 0), 'ugv_sensor': (0, 255, 0)}
    for page in range(2):
        tiles = []
        for f, labels in samples[page * 12:(page + 1) * 12]:
            img = cv2.imread(f)
            for l in labels:
                cv2.rectangle(img, (int(l[1]), int(l[2])), (int(l[3]), int(l[4])), colors.get(l[0], (160, 160, 160)), 1)
            tiles.append(img)
        while len(tiles) < 12:
            tiles.append(np.zeros((480, 640, 3), np.uint8))
        sheet = np.vstack([np.hstack(tiles[i * 4:(i + 1) * 4]) for i in range(3)])
        cv2.imwrite(SCAN + '/labels_%d.jpg' % page, cv2.resize(sheet, None, fx=0.75, fy=0.75), [cv2.IMWRITE_JPEG_QUALITY, 90])


main()
