#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M6 판독 평가: 경기 사진 폴더에 UGV 부품 모델과 perception 의 판독 코드를 돌려 표를 낸다 (ROS 없이).

쓰는 것은 로봇의 노드와 같다 — 모델 탐지 → perception.ugvs.build_ugvs (진영 · 호출부호 · 목숨) · perception.sensors.build_sensors (수광센서).
정답표가 없으므로 「읽힌 비율」 과 「일관성」(한 판에서 호출부호가 몇 가지로 갈리는가)을 본다. 틀렸는지는 --sheet 로 그림을 만들어 눈으로 본다.

사용 (저장소 루트에서):
  python3 임시_테스트_파일/m6_평가/eval_read_ugvs.py --weights <ugv_parts.pt|.engine> --photos "<로그>/2026*/supervisor/photos" [--sheet out.jpg]

  --photos   사진 폴더(들). 큰따옴표로 감싼 glob. 로봇에서는 /home/ws/team/log/<판>/supervisor/photos
  --package  perception 패키지 폴더 (기본: 이 저장소의 개발/src/perception). 다른 브랜치의 코드와 견줄 때 바꾼다
  --only     판 이름에 이 글자가 들어간 것만 (쉼표로 여러 개)
대회 사진은 저장소에 넣지 않는다. 2026-11-20 의 수치는 README.md 에 있다.
"""
import argparse
import collections
import glob
import os
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))


def size_bin(px):
    return 'h<60' if px < 60 else 'h<120' if px < 120 else 'h<200' if px < 200 else 'h>=200'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--photos', required=True)
    ap.add_argument('--package', default=os.path.join(HERE, '..', '..', '개발', 'src', 'perception'))
    ap.add_argument('--only', default='')
    ap.add_argument('--conf', type=float, default=0.15)
    ap.add_argument('--sheet', default='')
    args = ap.parse_args()
    sys.path.insert(0, os.path.abspath(args.package))
    from perception import sensors, ugvs
    from ultralytics import YOLO

    model = YOLO(args.weights, task='detect')
    files = sorted(f for d in glob.glob(args.photos) for f in glob.glob(os.path.join(d, '*.jpg')))
    only = [s for s in args.only.split(',') if s]
    if only:
        files = [f for f in files if any(s in f for s in only)]
    if not files:
        sys.exit('사진이 없다: %s' % args.photos)

    size = collections.defaultdict(collections.Counter)
    runs = collections.defaultdict(collections.Counter)
    lives = collections.Counter()
    pair = collections.Counter()
    owners = collections.Counter()
    t_model, t_read, shown = [], [], []
    for f in files:
        bgr = cv2.imread(f)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        run = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(f))))
        t0 = time.time()
        r = model.predict(bgr, conf=args.conf, verbose=False)[0]
        dets = [(model.names[int(c)], float(p)) + tuple(float(v) for v in b) for c, p, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]
        t1 = time.time()
        found = ugvs.build_ugvs(dets, w, h, rgb=rgb)
        spots = sensors.build_sensors(dets, w, h, rgb=rgb)           # locate_sensors 가 부품 모델 하나로 답할 때와 같다
        t2 = time.time()
        t_model.append(t1 - t0)
        if found:
            t_read.append(t2 - t1)
        for s in spots:
            owners[s['owner']] += 1
        for u in found:
            b = size_bin(u['height'] * h)
            box = (u['x_center'] - u['width'] / 2, u['y_center'] - u['height'] / 2, u['x_center'] + u['width'] / 2, u['y_center'] + u['height'] / 2)
            mine = [s for s in spots if s['owner'] == sensors.UGV and abs(s['host_x_center'] - u['x_center']) < 1e-6 and abs(s['host_y_center'] - u['y_center']) < 1e-6]
            c = size[b]
            c['ugv'] += 1
            c['camp'] += bool(u['camp'])
            c['callsign'] += bool(u['callsign'])
            c['lives'] += u['lives'] >= 0
            c['sensor'] += bool(mine)
            c['engageable'] += bool(mine) and bool(u['camp'])        # 쏠 수 있는 조건: 수광센서가 보이고 진영을 안다
            runs[run][u['callsign'] or (u['camp'] + '-?' if u['camp'] else 'none')] += 1
            if u['lives'] >= 0:
                lives[u['lives']] += 1
            # 한 UGV 의 두 안테나가 같은 값을 냈는가 (모델이 안테나 둘을 찾은 경우만)
            ants = [(d[2] / w, d[3] / h, d[4] / w, d[5] / h) for d in dets if d[0] == 'antenna' and d[1] >= 0.25]
            reads = [ugvs.read_antenna(ugvs._crop(rgb, a)) for a in ants if ugvs._inside(a, box) and ugvs._crop(rgb, a) is not None]
            reads = [x for x in reads if x[1] > 0]
            if len(reads) == 2:
                pair['same' if reads[0] == reads[1] else 'differ'] += 1
            if args.sheet and u['height'] * h >= 120 and len(shown) < 24:
                shown.append((f, dets, u, mine))
    t_model = t_model[3:] or t_model

    print('사진 %d장, 모델 %s' % (len(files), args.weights))
    print('| UGV 박스 높이 | UGV | camp | callsign | lives | 수광센서 | 쏠 수 있음(센서+진영) |')
    print('|---|---|---|---|---|---|---|')
    for b in ('h>=200', 'h<200', 'h<120', 'h<60'):
        c = size[b]
        n = max(1, c['ugv'])
        print('| %s | %d | %.0f %% | %.0f %% | %.0f %% | %.0f %% | %.0f %% |' % (b, c['ugv'], 100.0 * c['camp'] / n, 100.0 * c['callsign'] / n, 100.0 * c['lives'] / n,
                                                                              100.0 * c['sensor'] / n, 100.0 * c['engageable'] / n))
    print('판별 답 (호출부호가 한 판에서 여러 가지로 갈리면 잘못 읽은 것이 섞인 것이다):')
    for run in sorted(runs):
        print('  %s %s' % (run, dict(sorted(runs[run].items()))))
    print('lives 값: %s   두 안테나 일치: %s   수광센서 owner: %s' % (dict(sorted(lives.items())), dict(pair), dict(owners)))
    print('시간: 모델 %.1f ms, 판독 평균 %.1f ms · 최대 %.1f ms' % (1000 * sum(t_model) / len(t_model), 1000 * sum(t_read) / max(1, len(t_read)),
                                                          1000 * max(t_read) if t_read else 0.0))
    if args.sheet and shown:
        import numpy as np
        tiles = []
        for f, dets, u, mine in shown:
            img = cv2.imread(f)
            hh, ww = img.shape[:2]
            for d in dets:
                color = {'ugv': (0, 255, 255), 'antenna': (255, 0, 255), 'led_bar': (255, 128, 0), 'ugv_sensor': (0, 255, 0)}.get(d[0])
                if color and d[1] >= 0.25:
                    cv2.rectangle(img, (int(d[2]), int(d[3])), (int(d[4]), int(d[5])), color, 1)
            x1, y1 = int((u['x_center'] - u['width'] / 2) * ww), int((u['y_center'] - u['height'] / 2) * hh)
            x2, y2 = int((u['x_center'] + u['width'] / 2) * ww), int((u['y_center'] + u['height'] / 2) * hh)
            pad = 30
            c = img[max(0, y1 - pad):min(hh, y2 + 5), max(0, x1 - pad):min(ww, x2 + pad)]
            k = min(320.0 / c.shape[1], 282.0 / c.shape[0])
            c = cv2.resize(c, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
            tile = np.zeros((300, 320, 3), np.uint8)
            tile[18:18 + c.shape[0], :c.shape[1]] = c
            cv2.putText(tile, '%s %s lives %d sensors %d' % (u['camp'] or '-', u['callsign'] or '-', u['lives'], len(mine)), (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            tiles.append(tile)
        while len(tiles) % 6:
            tiles.append(np.zeros((300, 320, 3), np.uint8))
        cv2.imwrite(args.sheet, np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print('그림: %s' % args.sheet)


if __name__ == '__main__':
    main()
