"""제자리 방향 맞추기: 남은 각도 · 허용값별로 결과를 본다 (navigation 가상 경기장, 실차 흉내 REAL_ROBOT).

돌리기 (저장소 루트에서):
    python 임시_테스트_파일/navigation_방향_맞추기/align_sweep.py [패키지 경로] [--timeline 각도]
패키지 경로를 주면 그 navigation 패키지 사본(예: 8d83ef3 을 풀어 둔 곳)의 test/ 로 돌린다. 없으면 이 저장소의 것.
--timeline 12 → 그 각도(허용 0.10 rad)에서 주기마다 명령 각속도 · 남은 각도(추정 · 실제) · 단계를 찍는다.

출발 칸(R-2)에 선 채로 「지금 방향 + N°」 를 보내고(자리는 그대로) 끝날 때까지 돌린다. 시한 20 s.
"""
import math
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parents[2]
args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith('--') and sys.argv[i - 1] != '--timeline']
PACKAGE = Path(args[0]) if args else ROOT / '개발' / 'src' / 'navigation'
TEST = PACKAGE / 'test'
sys.path.insert(0, str(TEST))
os.chdir(TEST)
import test_team_test_replay as T  # noqa: E402

W = T.W
NAMES = {1: 'REACHED', 2: 'CANCELED', 3: 'BLOCKED', 4: 'TIMED_OUT'}


def wrapd(a):
    return (a + 180.0) % 360.0 - 180.0


def one(left_deg, tol, seed, timeline=False):
    loop = T.start(seed=seed)
    r = loop.robot
    target = math.degrees(r.yaw) + left_deg
    loop.allowed = True
    handle, thread, box = loop.send_goal(r.x, r.y, yaw=math.radians(target), yaw_tol=tol, mode=0, timeout_s=20.0)
    job = loop.job_of(handle)
    began = loop.t
    rows, nonzero, starts, last_w = [], 0, 0, 0.0
    while job is not None and not job.done and loop.t < began + 25.0:
        loop.step()
        w = loop.drive._last_cmd[1]
        if abs(w) > 1e-6:
            nonzero += 1
            if abs(last_w) < 1e-6:
                starts += 1
        last_w = w
        if timeline and loop.t - began <= 4.0:
            est = loop.drive.tracker.estimate()
            rows.append((round(loop.t - began, 2), round(w, 2), round(wrapd(target - math.degrees(est.yaw)), 1) if est else None,
                         round(wrapd(target - math.degrees(r.yaw)), 1), job.phase))
    thread.join(timeout=2.0)
    res = box.get('result')
    return dict(outcome=NAMES.get(res.outcome, res.outcome) if res else None, t=round(loop.t - began, 1),
                true_left=round(wrapd(target - math.degrees(r.yaw)), 1), starts=starts, turn_cycles=nonzero,
                gave_up='끝냄' in (res.detail if res else '')), rows


def main():
    if '--timeline' in sys.argv:
        deg = float(sys.argv[sys.argv.index('--timeline') + 1])
        out, rows = one(deg, 0.10, 1, timeline=True)
        print('t(s) | 명령 w | 남은 각도 추정 | 실제 | 단계')
        for row in rows:
            print(' | '.join(str(v) for v in row))
        print(out)
        return
    cases = [(d, 0.10) for d in (4, 6, 8, 10, 12, 14, 16, 20, 25, 30, 40, 90)] + [(d, 0.5) for d in (30, 33, 36, 39, 42, 46, 50, 60, 90)]
    print('| 남은 각도 | 허용 | seed | 결과 | 걸린 시간 | 끝난 뒤 실제 남은 각도 | 회전 시작 횟수 | 포기 판정 |')
    print('|---|---|---|---|---|---|---|---|')
    summary = {}
    for deg, tol in cases:
        for seed in (1, 2):
            out, _ = one(deg, tol, seed)
            print('| %d° | %.1f° | %d | %s | %.1f s | %.1f° | %d | %s |' % (deg, math.degrees(tol), seed, out['outcome'], out['t'], out['true_left'],
                                                              out['starts'], '예' if out['gave_up'] else ''))
            s = summary.setdefault(round(math.degrees(tol), 1), [0, 0, 0, 0.0])
            s[0] += 1
            s[1] += out['outcome'] == 'REACHED'
            s[2] += out['gave_up']
            s[3] += out['t']
    for tol, (n, ok, gave, tsum) in summary.items():
        print('허용 %.1f°: REACHED %d/%d (그중 포기 판정 %d), 평균 %.1f s' % (tol, ok, n, gave, tsum / n))


if __name__ == '__main__':
    main()
