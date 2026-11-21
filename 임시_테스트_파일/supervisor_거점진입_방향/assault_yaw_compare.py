"""거점 진입(M5) 돌진 목표에 도착 방향을 줄 때와 안 줄 때 비교 (navigation 가상 경기장, 실차 흉내 REAL_ROBOT).

supervisor mission._assault 는 ports.go(gx, y, heading, mode="assault") — yaw_free=False 라 벽을 보는 방향(±yaw_tolerance_rad)까지 맞춘다.
여기서는 같은 목표를 heading 180°(허용 0.10 rad) 로 보낼 때와 방향 무시(yaw_free=True 와 같음)로 보낼 때를 견준다.

돌리기 (저장소 루트에서):
    python 임시_테스트_파일/supervisor_거점진입_방향/assault_yaw_compare.py
navigation 의 시험 도구(개발/src/navigation/test/test_team_test_replay.py 의 start · 지도)를 그대로 쓴다. 코드는 바꾸지 않는다.
"""
import math
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / '개발' / 'src' / 'navigation' / 'test'
sys.path.insert(0, str(TEST))
os.chdir(TEST)
import test_team_test_replay as T  # noqa: E402

W = T.W
REACHED, BLOCKED, TIMED_OUT = 1, 3, 4
NAMES = {1: 'REACHED', 2: 'CANCELED', 3: 'BLOCKED', 4: 'TIMED_OUT', 5: 'CONTROL_LOST', 6: 'LOCALIZATION_LOST', 7: 'INVALID_GOAL'}


def run(slot, goal, yaw, seed, timeout_s=30.0):
    loop = T.start(slot=slot, seed=seed)
    x0, y0, x1, y1 = W.base_region('B-1')
    r = loop.robot
    loop.allowed = True
    handle, thread, box = loop.send_goal(goal[0], goal[1], yaw=yaw, yaw_tol=0.10, mode=2, timeout_s=timeout_s)
    job = loop.job_of(handle)
    began = loop.t
    entered = None
    turning_at_wall = 0.0
    while job is not None and not job.done and loop.t < began + timeout_s + 5.0:
        loop.step()
        if entered is None and x0 <= r.x <= x1 and y0 <= r.y <= y1:
            entered = loop.t - began
        if entered is not None and job.phase in ('aligning', 'rotating', 'settling'):
            turning_at_wall += W.ClosedLoop.DT
    thread.join(timeout=2.0)
    result = box.get('result')
    return dict(outcome=NAMES.get(result.outcome, result.outcome) if result else None, t=round(loop.t - began, 1),
                entered=None if entered is None else round(entered, 1), turning=round(turning_at_wall, 1),
                collisions=r.collisions, yaw=round(math.degrees(r.yaw), 0), pose=(round(r.x, 2), round(r.y, 2)))


def main():
    x0, y0, x1, y1 = W.base_region('B-1')
    spots = [round(y0 + 0.25, 3), round((y0 + y1) / 2, 3), round(y1 - 0.25, 3)]
    print('B-1 거점', tuple(round(v, 3) for v in (x0, y0, x1, y1)), '자리 y =', spots)
    # gx 0.175 = 지금 supervisor (벽 + robot_half_length 0.125 + wall_margin_m 0.05), 0.25 = wall_margin_m 0.125
    cases = [(gx, label, yaw) for gx in (0.175, 0.25) for label, yaw in (('heading 180°', math.pi), ('방향 무시', None))]
    rows = []
    for gx, label, yaw in cases:
        for y in spots:
            for seed in (1, 2, 3):
                out = run('B-2', (gx, y), yaw, seed)
                rows.append((gx, label, y, seed, out))
                print('gx=%.3f %-12s y=%.3f seed=%d %s' % (gx, label, y, seed, out))
    print()
    print('| 돌진 목표 x | 방향 | REACHED | 거점 안까지(평균) | 끝날 때까지(평균) | 거점 안에서 서서 돈 시간(평균) | 닿은 주기(합) |')
    print('|---|---|---|---|---|---|---|')
    for gx, label, _yaw in cases:
        sel = [o for g, lab, _y, _s, o in rows if g == gx and lab == label]
        entered = [o['entered'] for o in sel if o['entered'] is not None]
        print('| %.3f | %s | %d/%d | %.1f s | %.1f s | %.1f s | %d |' % (
            gx, label, sum(o['outcome'] == 'REACHED' for o in sel), len(sel), sum(entered) / max(1, len(entered)),
            sum(o['t'] for o in sel) / len(sel), sum(o['turning'] for o in sel) / len(sel), sum(o['collisions'] for o in sel)))


if __name__ == '__main__':
    main()
