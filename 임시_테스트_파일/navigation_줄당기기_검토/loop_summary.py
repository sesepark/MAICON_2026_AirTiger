"""loop_compare.py 의 --out 두 개(old · new)를 지도별 표로.

    python loop_summary.py old.json new.json
"""
import json
import statistics
import sys

sys.stdout.reconfigure(encoding='utf-8')
rows = json.load(open(sys.argv[1], encoding='utf-8')) + json.load(open(sys.argv[2], encoding='utf-8'))
print('| 지도 | 격자 | 판 | 도착 | 지도 상자 · 벽에 닿은 판 (주기 합) | 그중 「서서 돌다 박혀 시한까지」 판 | 차체-상자 최소 (중앙 · 가장 작은) | 걸린 시간 합 | 계획 시간 중앙 · 최대 |')
print('|---|---|---|---|---|---|---|---|---|')
for m in ('1차', '2차', '3차', '전체'):
    for w in ('old', 'new'):
        r = [x for x in rows if x['which'] == w and (m == '전체' or x['map'] == m)]
        if not r:
            continue
        touched = [x for x in r if x['contacts'] > 0]
        wedged = [x for x in r if x['contacts'] > 300]
        print('| %s | %s | %d | %d/%d | %d (%d) | %d | %.3f · %.3f m | %.0f s | %.1f · %.1f ms |' % (
            m, '전' if w == 'old' else '후', len(r), sum(x['reached'] for x in r), sum(x['n'] for x in r), len(touched),
            sum(x['contacts'] for x in r), len(wedged), statistics.median(x['box_min'] for x in r), min(x['box_min'] for x in r),
            sum(x['time'] for x in r), statistics.median(x['plan_med'] for x in r), max(x['plan_max'] for x in r)))
# 박힌 판을 뺀 것
print()
for w in ('old', 'new'):
    r = [x for x in rows if x['which'] == w and x['contacts'] <= 300]
    print('%s, 박힌 판 뺌: %d판 · 닿은 판 %d (주기 %d) · 차체-상자 최소 중앙 %.3f m · 0.02 m 안으로 들어간 판 %d' % (
        '전' if w == 'old' else '후', len(r), sum(x['contacts'] > 0 for x in r), sum(x['contacts'] for x in r),
        statistics.median(x['box_min'] for x in r), sum(x['box_min'] < 0.02 for x in r)))
