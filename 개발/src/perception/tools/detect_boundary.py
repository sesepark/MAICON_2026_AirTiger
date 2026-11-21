"""드론 화면 → 경기장 경계 (명령줄 도구). perception/drone_boundary.py 를 부른다.

    python detect_boundary.py 화면.jpg [화면2.jpg ...]           파일에서
    python detect_boundary.py --fetch test1                       연습 서버에서 한 장 (MAICON_MISSION_CODE 필요)
    python detect_boundary.py --fetch main --base http://<서버>:8000   경기 화면 (camera 없이 부른다)

결과 (--out 폴더, 기본 ./boundary_out):
    <이름>.json          ok · 모서리 픽셀 · 렌즈 왜곡 k · 사영 변환 · 잰 치수(끝 띠 · CM 반폭 · 원 반지름) · 검사 결과
    <이름>_before.png    원본 위에 경계(통과 초록 / 실패 빨강) · 구역선 · 찾은 선 점(분홍)
    <이름>_after.png     위에서 본 경기장 (1 cm = 1 px)
끝 코드: 모두 ok 면 0, 하나라도 실패면 1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # perception 패키지 (설치 없이 바로 쓰기)

from perception.drone_boundary import detect_boundary, draw_before_after, read_image, write_image  # noqa: E402


def _fetch(base, camera):
    import urllib.parse
    import urllib.request

    code = os.environ.get("MAICON_MISSION_CODE", "").strip()
    if not code:
        raise SystemExit("환경변수 MAICON_MISSION_CODE 에 임무코드를 넣으세요")
    query = {"mission_code": code}
    if camera != "main":
        query["camera"] = camera
    url = "%s/api/report/drone?%s" % (base.rstrip("/"), urllib.parse.urlencode(query))
    with urllib.request.urlopen(url, timeout=10) as res:
        return res.read(), {k.lower(): v for k, v in res.headers.items()}


def main(argv=None):
    import cv2
    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("images", nargs="*")
    ap.add_argument("--fetch", metavar="CAMERA", help="test1 · test2 · main")
    ap.add_argument("--base", default="http://192.168.110.50:8100")
    ap.add_argument("--out", default="boundary_out")
    ap.add_argument("--no-images", action="store_true", help="그림은 저장하지 않는다")
    args = ap.parse_args(argv)

    jobs = []
    for path in args.images:
        jobs.append((os.path.splitext(os.path.basename(path))[0], read_image(path), {}))
        if jobs[-1][1] is None:
            raise SystemExit("그림을 읽지 못함: %s" % path)
    if args.fetch:
        data, headers = _fetch(args.base, args.fetch)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        stamp = "".join(ch for ch in headers.get("x-captured-at", time.strftime("%Y%m%dT%H%M%S")) if ch.isdigit())[:17]
        jobs.append(("%s_%s" % (args.fetch, stamp), img, headers))
        os.makedirs(args.out, exist_ok=True)
        write_image(os.path.join(args.out, jobs[-1][0] + ".jpg"), img)
    if not jobs:
        ap.error("그림 파일이나 --fetch 를 주세요")

    os.makedirs(args.out, exist_ok=True)
    all_ok = True
    for name, img, headers in jobs:
        t0 = time.time()
        b = detect_boundary(img)
        out = b.to_dict()
        out.update({"source": name, "seconds": round(time.time() - t0, 2), "score": b.debug.get("score"), "margin": b.debug.get("margin"),
                    "captured_at": headers.get("x-captured-at"), "camera": headers.get("x-camera")})
        with open(os.path.join(args.out, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        if b.hom is not None and not args.no_images:
            before, after = draw_before_after(img, b)
            write_image(os.path.join(args.out, name + "_before.png"), before)
            write_image(os.path.join(args.out, name + "_after.png"), after)
        corners = out.get("corners_px")
        print("%s: %s (%.1fs) 모서리 %s · 끝 띠 %s m · CM 반폭 %s m · 원 %s m" % (
            name, "OK" if b.ok else "실패", out["seconds"], corners,
            *("%.2f" % v if v is not None else "-" for v in (b.strip, b.cm_half, b.circle_r))))
        for note in b.notes:
            print("    " + note)
        all_ok &= b.ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
