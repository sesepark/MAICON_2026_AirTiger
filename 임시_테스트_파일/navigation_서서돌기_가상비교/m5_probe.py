"""M5 돌진(B-2 → B-1)을 aaf5a44 의 supervisor 값으로: 벽에서 0.20 m, 방향 무시(yaw_free). 패키지 사본을 바꿔 가며 견준다.

    python m5_probe.py <navigation 패키지 경로>
자리 y: aaf5a44 의 거점 끝(기둥)에서 0.40 m 안쪽 + 가운데, 그리고 1 · 2차에 실제로 보낸 3.817 · 그 반대쪽 4.517. seed 1 ~ 3. 시한 30 s.
잰 것: 결과 · 거점 안에 든 시각 · 끝난 시각 · 서서 돌라는 명령을 냈는데 차체가 닿아 못 돈 주기(pinned) · 닿은 주기
"""
import math
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
pkg = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(pkg / 'test'))
os.chdir(pkg / 'test')
import test_team_test_replay as T  # noqa: E402

W = T.W
NAMES = {1: 'REACHED', 2: 'CANCELED', 3: 'BLOCKED', 4: 'TIMED_OUT'}
x0, y0, x1, y1 = W.base_region('B-1')
SPOTS = [round(y0 + 0.40, 3), round((y0 + y1) / 2, 3), round(y1 - 0.40, 3), 3.817, 4.517]
rows = []
for y in SPOTS:
    for seed in (1, 2, 3):
        loop = T.start(slot='B-2', seed=seed)
        r = loop.robot
        loop.allowed = True
        handle, thread, box = loop.send_goal(0.20, y, yaw=None, mode=2, timeout_s=30.0)
        job = loop.job_of(handle)
        began, entered, pinned = loop.t, None, 0
        while job is not None and not job.done and loop.t < began + 35.0:
            c0 = r.collisions
            loop.step()
            v, w = loop.drive._last_cmd
            if abs(v) < 1e-6 and abs(w) > 1e-6 and r.collisions > c0:
                pinned += 1
            if entered is None and x0 <= r.x <= x1 and y0 <= r.y <= y1:
                entered = loop.t - began
        thread.join(2.0)
        res = box.get('result')
        out = dict(y=y, seed=seed, outcome=NAMES.get(res.outcome, res.outcome) if res else None, t=round(loop.t - began, 1),
                   entered=entered, pinned=pinned, contacts=r.collisions, detail=(res.detail if res else '')[:60])
        rows.append(out)
        print(out, flush=True)
n = len(rows)
print('REACHED %d/%d · 들어감 %d/%d · 평균 끝 %.1f s · 못 돈 주기 합 %d · 닿은 주기 합 %d' % (
    sum(o['outcome'] == 'REACHED' for o in rows), n, sum(o['entered'] is not None for o in rows), n,
    sum(o['t'] for o in rows) / n, sum(o['pinned'] for o in rows), sum(o['contacts'] for o in rows)))
