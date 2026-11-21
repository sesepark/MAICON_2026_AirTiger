"""replay.py 결과 json 여러 개를 나란히 본다.

    python -I report.py 결과1.json 결과2.json …

  끊긴 시간   = 위치가 1 s 넘게 안 나온 구간의 합
  5 cm 넘게 튐 = 이어진 두 위치(0.3 s 안) 사이가 5 cm 넘게 벌어진 횟수 — 정합이 다른 답으로 건너뛴 것
  실차와의 차이 = 같은 시각에 그때 로봇이 실제로 낸 위치와의 거리 (그때 돌던 코드와 같은 코드일 때만 뜻이 있다)
"""
import bisect
import json
import math
import sys

sys.stdout.reconfigure(encoding='utf-8')


def quantile(values, fraction):
    values = sorted(values)
    return values[int(fraction * (len(values) - 1))] if values else float('nan')


for path in sys.argv[1:]:
    d = json.load(open(path, encoding='utf-8'))
    poses, counts = d['poses'], d['scan_counts']
    times = [p[0] / 1e9 for p in poses]
    lost = sum(b - a for a, b in zip(times, times[1:]) if b - a > 1.0)
    steps = [math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(poses, poses[1:]) if 0 < (b[0] - a[0]) / 1e9 < 0.3]
    print(path)
    print('  map_fusion %s · 위치 %d개 · 정합 성공 %s / 실패 %s · 처음 위치 찾기 실패 %s · 끊긴 시간 %.1f s · 5 cm 넘게 튐 %d번 (p99 %.1f cm)'
          % (d['map_fusion'], len(poses), counts.get('정합 성공', 0), counts.get('정합 실패', 0), counts.get('처음 위치 찾기 실패', 0),
             lost, sum(1 for s in steps if s > 0.05), quantile(steps, 0.99) * 100))
    print('  내보낸 지도의 상자 수 %s · 지도 고치기 관측 %s · 한 번 %.2f ms (최대 %.1f)'
          % (d['published_box_counts'], d['skips'], d['fusion_ms']['mean'], d['fusion_ms']['max']))
    stamps = [p[0] for p in d['recorded']]
    gaps = []
    for t, x, y, _ in poses:
        i = bisect.bisect_left(stamps, t)
        if i < len(stamps) and stamps[i] == t:
            gaps.append(math.hypot(x - d['recorded'][i][1], y - d['recorded'][i][2]))
    if gaps:
        print('  실차와의 차이: 중앙값 %.1f cm · p95 %.1f cm · 최대 %.1f cm (%d개)'
              % (quantile(gaps, 0.5) * 100, quantile(gaps, 0.95) * 100, max(gaps) * 100, len(gaps)))
    for event in d['events']:
        removed = [b[:2] for b in event['drone'] if b[3]]
        confirmed = sum(1 for b in event['drone'] if b[2] >= 2 and not b[3])
        print('    %.1f  드론 상자 %d 중 확인 %d · 뺌 %s · 정합에만 쓰는 더한 상자 %s'
              % (event['t'], len(event['drone']), confirmed, removed, [b[:2] for b in event['added']]))
