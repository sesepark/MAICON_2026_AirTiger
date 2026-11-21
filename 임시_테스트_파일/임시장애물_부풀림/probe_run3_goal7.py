"""3차 팀 테스트(2026-11-18 23:39)의 7번 목표 (2.625, 4.625) 를 가상으로: 그 판의 perception 지도(상자 14개) + 기둥 10개 [추정 자리].
실차: 30.8 s 시한에 BLOCKED, 끝 자리 (3.014, 4.364). 목표는 지도 상자에서 0.74 m, 위 가운데 기둥 (3.06, 4.8) 의 왼쪽 면에서 0.24 m.

    python probe_run3_goal7.py [r ...]     (NAV_ROOT 로 패키지를 바꿀 수 있다, --set 이름=값)
"""
import json
import math
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
NAV = Path(os.environ["NAV_ROOT"]) if os.environ.get("NAV_ROOT") else ROOT / "개발" / "src" / "navigation"
sys.path.insert(0, str(NAV / "test"))
sys.path.insert(0, str(NAV / "시험기록" / "web_drive"))
import sim_world as W  # noqa: E402

field = json.load(open(ROOT / "로그_파일" / "1118_팀테스트_경기장1_R2" / "3차_2339" / "perception" / "field_map.json", encoding="utf-8"))
BOXES = [(b["x"], b["y"]) for b in field["boxes"]]
args = [a for a in sys.argv[1:] if not a.startswith("--")]
MAPPED_PILLARS = "--pillars-in-map" in sys.argv  # 기둥 10개를 지도 상자처럼 넣는다 (주행이 기둥을 알 때)
extra = {}
for a in sys.argv[1:]:
    if a.startswith("--set="):
        k, v = a[6:].split("=")
        extra[k] = float(v)
for r in [float(a) for a in args] or [0.18, 0.13, 0.08, 0.05, 0.0]:
    world = W.World(boxes=BOXES + list(W.PILLARS), **W.ARENA) if MAPPED_PILLARS else W.World(boxes=BOXES, hidden=W.PILLARS, **W.ARENA)
    robot = W.SimRobot(world, W.slot_start_pose("R-2"), odom_offset=1.0, slip=0.85, seed=1, **W.REAL_ROBOT)
    loop = W.ClosedLoop(world, robot, drive_params=dict(extra, obstacle_radius=r))
    loop.start_from_slot("R-2", W.base_region("R-2"))
    loop.run(1.5)
    first = loop.drive_to(3.83, 3.62, limit_s=40.0, timeout_s=35.0)
    c0 = robot.collisions
    began = loop.t
    res = loop.drive_to(2.625, 4.625, limit_s=36.0, yaw=math.radians(-90.0), yaw_tol=0.10, timeout_s=30.8)
    print("r %.2f (합 %.2f) %s: 앞 목표 %s → 7번 %s %.1f s, 끝 (%.2f, %.2f), 닿음 %d, 차체 최소 거리 %.3f  %s" % (
        r, r + 0.17, extra, getattr(first, "outcome", None), getattr(res, "outcome", None), loop.t - began, robot.x, robot.y,
        robot.collisions - c0, robot.min_body_clearance, getattr(res, "detail", "")[:40]), flush=True)
