import json
import statistics
import sys

sys.stdout.reconfigure(encoding='utf-8')
print('| 판 | 도착 | 박힌 판 (닿은 주기 300 넘음) | 지도 상자 · 벽 닿음 (판 · 주기) | 기둥 닿음 (판 · 주기) | 걸린 시간 합 | 상자까지 최소 거리 중앙 |')
print('|---|---|---|---|---|---|---|')
for label, path in zip(sys.argv[1::2], sys.argv[2::2]):
    rows = json.load(open(path, encoding='utf-8'))
    reached = sum(r['reached'] for r in rows)
    n = sum(r['n'] for r in rows)
    stuck = [r for r in rows if r['contacts'] + r['hidden'] > 300]
    cm = [r for r in rows if r['contacts'] > 0]
    ch = [r for r in rows if r['hidden'] > 0]
    print('| %s | %d/%d | %d (%s) | %d · %d | %d · %d | %.0f s | %.3f m |' % (
        label, reached, n, len(stuck), ', '.join('%s s%d' % (r['map'], r['seed']) for r in stuck) or '-',
        len(cm), sum(r['contacts'] for r in rows), len(ch), sum(r['hidden'] for r in rows),
        sum(r['time'] for r in rows), statistics.median(r['box_min'] for r in rows)))
