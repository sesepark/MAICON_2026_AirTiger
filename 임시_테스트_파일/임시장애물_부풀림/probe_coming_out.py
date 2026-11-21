"""재현 시험 「입구로 나오는 UGV」(test_does_not_hit_a_ugv_coming_out_to_the_base_entrance)를 obstacle_radius 별로 돌려
닿은 때 우리 속도 · 단계 · 상대가 움직이던 중인지를 찍는다 (누가 들이받았나).

    python probe_coming_out.py [r ...]
"""
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
NAV = Path(__file__).resolve().parents[2] / "개발" / "src" / "navigation"
sys.path.insert(0, str(NAV / "test"))
import test_team_test_replay as T  # noqa: E402

W = T.W

for r in [float(a) for a in sys.argv[1:]] or [0.18, 0.13, 0.08, 0.05]:
    for seen in (0.23, 0.12):
        world = W.World(boxes=T.BOXES_1118_2, hidden=W.PILLARS, vehicles=[(0.85, 4.45)], vehicle_seen=seen, **W.ARENA)
        robot = W.SimRobot(world, W.slot_start_pose('B-2'), odom_offset=1.0, slip=0.85, seed=1, **W.REAL_ROBOT)
        loop = W.ClosedLoop(world, robot, drive_params={'obstacle_radius': r})
        loop.start_from_slot('B-2', W.base_region('B-2'))
        loop.run(1.5)
        loop.allowed = True
        handle, thread, box = loop.send_goal(*T.B1_ASSAULT, yaw=math.pi, mode=2, timeout_s=28.0)
        job = loop.job_of(handle)
        began, target, events, path = loop.t, None, [], []
        while job is not None and not job.done and loop.t < began + 33.0:
            if target is None and robot.y > 2.75:
                target = (0.50, 3.80)
            moving = False
            if target is not None:
                x, y = world.vehicles[0]
                d = math.hypot(target[0] - x, target[1] - y)
                if d > 1e-9:
                    step = min(d, 0.3 * loop.DT)
                    world.vehicles[0] = (x + (target[0] - x) / d * step, y + (target[1] - y) / d * step)
                    moving = True
            h0 = robot.hidden_contacts
            loop.step()
            if robot.hidden_contacts > h0:
                events.append((round(loop.t - began, 2), job.phase, round(robot.v, 2), round(robot.w, 2), moving))
            if int(round((loop.t - began) / loop.DT)) % 20 == 0:
                path.append((round(loop.t - began, 1), round(robot.x, 2), round(robot.y, 2), job.phase))
        thread.join(1.0)
        res = box.get('result')
        print("r %.2f 보임 %.2f → %s %.1f s, 닿은 주기 %d" % (r, seen, getattr(res, 'outcome', None), loop.t - began, robot.hidden_contacts))
        if events:
            print("   처음 닿음 (s, 단계, 우리 v, w, 상대 움직이는 중):", events[0], " 마지막:", events[-1],
                  " 상대가 움직이던 주기 %d / %d" % (sum(e[4] for e in events), len(events)))
            print("   자취:", [p for p in path if p[0] >= events[0][0] - 3.0][:8])
