"""드론 화면 → 상자 위치 (명령줄 도구). perception/drone_boundary.py 로 경계를 찾고 perception/drone_boxes.py 로 상자를 찾는다.

    python detect_boxes.py 화면.jpg [화면2.jpg ...]            파일에서
    python detect_boxes.py --fetch test1                        연습 서버에서 한 장 (MAICON_MISSION_CODE 필요)
    python detect_boxes.py --fetch main --base http://<서버>:8000   경기 화면 (camera 없이 부른다)
    python detect_boxes.py --height 2.6 화면.jpg                카메라 높이를 알면 고정 (재지 않는다)

결과 (--out 폴더, 기본 ./boxes_out):
    <이름>_field_map.json  /team/perception/field_map (team_interfaces/FieldMap) 과 같은 칸 — boxes · sectors · faces · bases(--bases) ·
                           header.stamp(서버 사진이면 x-captured-at)
    <이름>.json          상자 가운데 (경기장 m) · 카메라 (x, y, 높이) · 버린 후보와 까닭 · 경계 요약
    <이름>_boxes.png     원본 위에 상자 바닥(초록) · 윗면(노랑) · 버린 후보(빨간 X) · 카메라 바로 아래(하늘색 +)
    <이름>_boxes_top.png 위에서 본 경기장 (1 cm = 1 px) 위에 같은 것
여러 장을 주면 끝에 카메라 높이 중앙값을 알려 준다 (--height 로 고정할 때 쓴다).
끝 코드: 모두 ok 면 0, 하나라도 아니면 1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # perception 패키지 (설치 없이 바로 쓰기)
sys.path.insert(0, HERE)

from detect_boundary import _fetch  # noqa: E402
from perception.drone_boundary import detect_boundary, read_image, write_image  # noqa: E402
from perception.drone_boxes import BoxParams, detect_boxes, draw_boxes  # noqa: E402


def _stamp(text):
    """서버 머리글 x-captured-at (ISO 8601) → 초. 없거나 못 읽으면 None."""
    if not text:
        return None
    from datetime import datetime

    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def main(argv=None):
    import cv2
    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("images", nargs="*")
    ap.add_argument("--fetch", metavar="CAMERA", help="test1 · test2 · main")
    ap.add_argument("--base", default="http://192.168.110.50:8100")
    ap.add_argument("--height", type=float, help="카메라 높이 (m). 주지 않으면 상자 옆면으로 잰다")
    ap.add_argument("--bases", choices=("none", "slot", "report"), default="none",
                    help="field_map 의 거점 이름: none 비움 (기본) · slot B-1 ~ R-3 · report BLUE_COMMAND_POST 등 (--base-kinds 필요)")
    ap.add_argument("--base-kinds", help='report 일 때 1 · 2 · 3번 줄의 거점 종류 JSON, 예 {"BLUE": ["COMMAND_POST", "MAINTENANCE", "SUPPLY_DEPOT"], "RED": [...]}')
    ap.add_argument("--out", default="boxes_out")
    ap.add_argument("--no-images", action="store_true", help="그림은 저장하지 않는다")
    args = ap.parse_args(argv)

    jobs = []
    for path in args.images:
        img = read_image(path)
        if img is None:
            raise SystemExit("그림을 읽지 못함: %s" % path)
        jobs.append((os.path.splitext(os.path.basename(path))[0], img, {}))
    os.makedirs(args.out, exist_ok=True)
    if args.fetch:
        data, headers = _fetch(args.base, args.fetch)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        stamp = "".join(ch for ch in headers.get("x-captured-at", time.strftime("%Y%m%dT%H%M%S")) if ch.isdigit())[:17]
        jobs.append(("%s_%s" % (args.fetch, stamp), img, headers))
        write_image(os.path.join(args.out, jobs[-1][0] + ".jpg"), img)
    if not jobs:
        ap.error("그림 파일이나 --fetch 를 주세요")

    params = BoxParams(camera_height=args.height)
    all_ok, heights = True, []
    for name, img, headers in jobs:
        t0 = time.time()
        b = detect_boundary(img)
        view = detect_boxes(img, b, params)
        out = view.to_dict()
        out.update({"source": name, "seconds": round(time.time() - t0, 2), "captured_at": headers.get("x-captured-at"),
                    "camera_name": headers.get("x-camera"),
                    "boundary": {"ok": b.ok, "notes": list(b.notes), "corners_px": None if b.hom is None else np.round(b.corners_px(), 1).tolist()}})
        with open(os.path.join(args.out, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        if b.hom is not None:
            fm = view.field_map(b, _stamp(headers.get("x-captured-at")), None if args.bases == "none" else args.bases,
                                json.loads(args.base_kinds) if args.base_kinds else None)
            with open(os.path.join(args.out, name + "_field_map.json"), "w", encoding="utf-8") as f:
                json.dump(fm, f, ensure_ascii=False, indent=1)
        if b.hom is not None and not args.no_images:
            raw, top = draw_boxes(img, b, view)
            write_image(os.path.join(args.out, name + "_boxes.png"), raw)
            write_image(os.path.join(args.out, name + "_boxes_top.png"), top)
        print("%s: %s (%.1fs) 상자 %d개 · 카메라 (%.2f, %.2f) 높이 %.2f m [%s, 옆면 %d개]" % (
            name, "OK" if view.ok else "확인 필요", out["seconds"], len(view.boxes), view.camera[0], view.camera[1], view.camera[2],
            view.height_source, view.faces_used))
        for i, bx in enumerate(view.boxes):
            print("    %2d  (%.3f, %.3f)  점수 %.2f%s" % (i, bx.x, bx.y, bx.score, "  윗면 %s 변이 화면 밖" % bx.cut if bx.cut else ""))
        for note in view.notes:
            print("    " + note)
        if view.height_source == "measured":
            heights.append(view.camera[2])
        all_ok &= view.ok
    if len(heights) > 1:
        print("카메라 높이 중앙값 %.2f m (%d장) — 고정하려면 --height %.2f" % (float(np.median(heights)), len(heights), float(np.median(heights))))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
