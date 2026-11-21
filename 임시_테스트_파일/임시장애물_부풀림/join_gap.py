"""compare_real.py gap 결과 파일들(out/)을 한 표로: 칸 = 결과(R 도착 · B 막힘 · T 시한) 걸린 s, ! = 지도에 없는 것에 닿음, 그 뒤 = 그것과의 최소 거리 m."""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
OUT = Path(__file__).resolve().parent / 'out'
FILES = [('지금 0.18', 'gap_0.18.txt'), ('일괄 0.13', 'gap_0.13.txt'), ('일괄 0.08', 'gap_0.08.txt'), ('일괄 0.05', 'gap_0.05.txt'), ('일괄 0', 'gap_0.0.txt'),
         ('2단계 +0', 'proto_gap_0.0.txt'), ('2단계 +0.03', 'proto_gap_0.03.txt'), ('2단계 +0.05', 'proto_gap_0.05.txt'), ('2단계 +0.08', 'proto_gap_0.08.txt'), ('2단계 +0.10', 'proto_gap_0.10.txt')]
rows = {}
names = []
for name, fname in FILES:
    if not (OUT / fname).exists():
        continue
    names.append(name)
    for line in open(OUT / fname, encoding='utf-8'):
        m = re.match(r'(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+\([\d.]+\) \| (\S+)\s+([\d.]+)\s+(\d+)\s+(-?[\d.]+)', line)
        if m:
            k = m.groups()[:4]
            out, t, hit, near = m.groups()[4:]
            ab = {'REACHED': 'R', 'BLOCKED': 'B', 'TIMED_OUT': 'T'}.get(out, out[:1])
            rows.setdefault(k, {})[name] = '%s%4.1f%s%.2f' % (ab, float(t), '!' if int(hit) else ' ', float(near))
print('%-7s %-4s %-4s %-4s | ' % ('종류', '틈', '비킴', '보임') + ' | '.join('%-11s' % n for n in names))
summary = {n: [0, 0, 0] for n in names}
for k in sorted(rows):
    print('%-7s %s %s %s | ' % k + ' | '.join('%-11s' % rows[k].get(n, '-') for n in names))
    for n in names:
        v = rows[k].get(n)
        if v:
            summary[n][0] += v[0] == 'R'
            summary[n][1] += '!' in v
            summary[n][2] += 1
print('도착/전체 · 닿은 판: ' + ' | '.join('%s %d/%d·%d' % (n, s[0], s[2], s[1]) for n, s in summary.items()))
for seen in ('0.23', '0.12'):
    part = {n: [0, 0, 0] for n in names}
    for k, cells in rows.items():
        if k[3] != seen:
            continue
        for n in names:
            v = cells.get(n)
            if v:
                part[n][0] += v[0] == 'R'
                part[n][1] += '!' in v
                part[n][2] += 1
    print('보임 %s 만: ' % seen + ' | '.join('%s %d/%d·%d' % (n, s[0], s[2], s[1]) for n, s in part.items()))
