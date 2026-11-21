"""경기 기록에서 지도에 없는 물체 앞에 서 있던 구간마다: 「막은 것은 서 있는 물체」 판정(obstacles.blocker_static)이 언제 켜졌을까.

기록의 지도에 없는 점(navigation/debug.jsonl state 의 unexplained, 2 Hz · 300개까지 · 1 cm 로 반올림)을 실제 ObstacleTracker 의 칸 기억에
그대로 넣는다 (실차는 10 Hz 라 칸이 더 촘촘히 이어진다). 선 뒤 그 판정이 켜지기까지의 시간을 적는다 — 켜지면 obstacle_give_up_static_s(2 s) 뒤에
끝낸다 (전 5 s). 물체가 정말 서 있었는지는 그 무리의 가운데가 구간 동안 움직인 거리로 본다.

쓰기:  python replay_static_blocker.py <기록 폴더>… [-v]
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..')))
from navigation.drive.geometry import Pose2D  # noqa: E402
from navigation.drive.obstacles import ObstacleTracker  # noqa: E402
from navigation.drive.params import DriveParams  # noqa: E402


class Cells:
    """grid.cell_of 만 있는 격자 (칸 기억에는 그것만 쓰인다)."""
    cell = 0.05
    grow = 0.17

    def cell_of(self, x, y):
        return int(math.floor(x / self.cell)), int(math.floor(y / self.cell))


def run(folder, verbose=False):
    path = next(q for q in (os.path.join(folder, 'navigation', 'debug.jsonl'), os.path.join(folder, 'navigation_debug.jsonl')) if os.path.exists(q))
    p = DriveParams()
    grid = Cells()
    tracker = ObstacleTracker(p)
    start = None
    episode = None   # [선 때, 판정이 처음 켜진 때, 처음 무리 가운데, 마지막 무리 가운데, 끝]
    done = []
    for line in open(path, encoding='utf-8'):
        if '"event":"goal"' in line and start is None:
            start = json.loads(line)['t']
        if '"event":"state"' not in line:
            continue
        r = json.loads(line)
        if not r.get('pose') or not r.get('job'):
            continue
        t, pose = r['t'], Pose2D(*r['pose'])
        world = [tuple(q) for q in (r.get('unexplained') or [])]
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        body = [((x - pose.x) * c + (y - pose.y) * s, -(x - pose.x) * s + (y - pose.y) * c) for x, y in world]
        # on_scan 의 칸 기억과 같은 일 (지도 거리는 기록에 없어 모두 「떨어진 점」 으로 넣는다 — 처음 찍힌 때만 쓴다)
        for x, y in world:
            cell = grid.cell_of(x, y)
            seed = tracker._seeds.get(cell)
            if seed is None:
                tracker._seeds[cell] = [p.obstacle_radius, t + p.obstacle_ttl_s, t, t]
            else:
                seed[1], seed[2] = t + p.obstacle_ttl_s, t
        for cell in [k for k, seed in tracker._seeds.items() if seed[1] <= t]:
            del tracker._seeds[cell]
        tracker.unexplained_body, tracker.unexplained_world, tracker.scan_t = body, world, t
        waiting = r['job'].get('phase', '').startswith('waiting_obstacle')
        band = [w for (bx, by), w in zip(body, world) if 0.0 < bx <= p.robot_half_length + 1.3 and abs(by) <= 0.30]
        if waiting and band:
            centre = (sum(x for x, _ in band) / len(band), sum(y for _, y in band) / len(band))
            if episode is None or t - episode[4] > 1.2:
                if episode is not None:
                    done.append(episode)
                episode = [t, None, centre, centre, t]
            episode[3], episode[4] = centre, t
            if episode[1] is None and tracker.blocker_static(t, grid, p.tight_still_s):
                episode[1] = t
        elif episode is not None and t - episode[4] > 1.2:
            done.append(episode)
            episode = None
    if episode is not None:
        done.append(episode)
    long = [e for e in done if e[4] - e[0] >= 2.0]
    on = [e for e in long if e[1] is not None]
    still = [e for e in long if math.hypot(e[3][0] - e[2][0], e[3][1] - e[2][1]) < 0.10]
    waits = sorted(e[1] - e[0] for e in on)
    print('== %s  2 s 넘게 선 구간 %d · 그 물체가 안 움직인 것 %d · 판정이 켜진 것 %d (선 뒤 가운데 %.1f s · 가장 늦게 %.1f s) · 선 시간 합 %.0f s → 판정 + 2 s 에서 끝냈다면 %.0f s' % (
        os.path.basename(os.path.normpath(folder)), len(long), len(still), len(on), waits[len(waits) // 2] if waits else 0, waits[-1] if waits else 0,
        sum(e[4] - e[0] for e in long), sum(min(e[4] - e[0], (e[1] - e[0] + p.obstacle_give_up_static_s) if e[1] is not None else 1e9) for e in long)))
    if verbose:
        for e in long:
            print('   t%+7.1f  선 시간 %5.1f s · 무리 움직임 %.2f m · 판정 %s' % (
                e[0] - (start or 0), e[4] - e[0], math.hypot(e[3][0] - e[2][0], e[3][1] - e[2][1]), '선 뒤 %.1f s' % (e[1] - e[0]) if e[1] is not None else '안 켜짐'))
    return len(long), len(on)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    folders = [a for a in sys.argv[1:] if not a.startswith('-')]
    total = [0, 0]
    for folder in folders:
        n, k = run(folder, '-v' in sys.argv)
        total[0] += n
        total[1] += k
    print('전체: 2 s 넘게 선 구간 %d 가운데 판정이 켜진 것 %d' % tuple(total))


if __name__ == '__main__':
    main()
