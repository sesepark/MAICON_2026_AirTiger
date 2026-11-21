"""재현 시험 「입구로 나오는 UGV」 장면을 여러 변형으로 (상대 속도 · 나오기 시작하는 때 · 보이는 폭 · 잡음 seed).
한 판의 통과 · 실패가 시각에 따라 바뀌어서, 판 수로 견준다.

    python sweep_coming_out.py <이름> [--set 이름=값 ...]      (NAV_ROOT 로 패키지를 바꿀 수 있다)
"""
import math
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
NAV = Path(os.environ["NAV_ROOT"]) if os.environ.get("NAV_ROOT") else Path(__file__).resolve().parents[2] / "개발" / "src" / "navigation"
sys.path.insert(0, str(NAV / "test"))
import test_team_test_replay as T  # noqa: E402

W = T.W
label = sys.argv[1]
params = {}
for a in sys.argv[2:]:
    k, v = a.split("=")
    params[k] = float(v)

rows = []
for speed in (0.2, 0.3, 0.4):
    for trigger in (2.6, 2.75, 2.9):
        for seen, seed in ((0.23, 1), (0.12, 2)):
            world = W.World(boxes=T.BOXES_1118_2, hidden=W.PILLARS, vehicles=[(0.85, 4.45)], vehicle_seen=seen, **W.ARENA)
            robot = W.SimRobot(world, W.slot_start_pose('B-2'), odom_offset=1.0, slip=0.85, seed=seed, **W.REAL_ROBOT)
            loop = W.ClosedLoop(world, robot, drive_params=dict(params))
            loop.start_from_slot('B-2', W.base_region('B-2'))
            loop.run(1.5)
            res = T.drive_with(loop, T.B1_ASSAULT, mode=int(os.environ.get("MODE", "2")), timeout_s=28.0,  # MODE=0 이면 일반 주행으로
                               move=lambda r, y0=trigger: (0.50, 3.80) if r.y > y0 else None, speed=speed)
            rows.append((speed, trigger, seen, getattr(res, 'outcome', None), round(loop.t - loop.drive_started, 1),
                         robot.hidden_contacts, robot.collisions))
hit = [r for r in rows if r[5]]
print("%-22s %s: %d판 중 지도에 없는 것에 닿은 판 %d, 상자 · 벽 포함 닿은 판 %d, 결과 %s" % (
    label, params, len(rows), len(hit), sum(1 for r in rows if r[6]),
    {k: sum(1 for r in rows if r[3] == k) for k in sorted({r[3] for r in rows}, key=str)}), flush=True)
for r in hit:
    print("    속도 %.1f 나옴 y>%.2f 보임 %.2f → 결과 %s %.1f s 닿은 주기 %d (상자·벽 포함 %d)" % r)
