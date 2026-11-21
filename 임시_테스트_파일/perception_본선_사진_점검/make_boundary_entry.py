#!/usr/bin/env python3
"""드론 사진 한 장 → 그 카메라의 경계를 맞추고, 눈으로 볼 겹친 그림을 쓰고, 코드에 붙여 넣을 값을 찍는다.

    python make_boundary_entry.py 사진.jpg --camera main
    python make_boundary_entry.py 사진.jpg --camera test2 --out /tmp/겹친그림.jpg

하는 일
  1) 경계를 처음부터 찾는다 (지난 경계 · 코드에 실린 경계를 쓰지 않는다 — 이 사진만으로 맞는지 보려는 것).
     --use-built-in 을 주면 코드에 실린 그 카메라의 경계에서 출발한다 (이미 넣은 값이 이 사진과 맞는지 볼 때).
  2) 겹친 그림을 쓴다: 왼쪽 = 원본 위에 찾은 선(주황) · 경기장 테두리(초록 = 검사 통과, 빨강 = 실패) · 맞추는 데 쓴 점(분홍),
     오른쪽 = 위에서 본 지도. 선이 바닥의 흰 선 위에 정확히 놓였는지, 가운데 원이 원 위에 놓였는지 눈으로 본다.
  3) 검사를 통과했으면 `개발/src/perception/perception/camera_boundaries.py` 의 BUILT_IN 에 붙여 넣을 한 줄을 찍는다.
     통과하지 못했으면 까닭을 찍고 끝낸다 (값을 찍지 않는다 — 틀린 경계를 코드에 넣지 않게).

사진은 저장소에 넣지 않는다. numpy · OpenCV 만 쓴다 (ROS 없이 돈다).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "개발", "src", "perception"))

KEYS = ("ok", "k", "hom", "image_size", "lens", "strip", "cm_half", "circle_r", "rms_m", "points", "field_w", "field_h")


def main():
    import numpy as np

    from perception import camera_boundaries
    from perception.drone_boundary import detect_boundary, draw_before_after, read_image, write_image

    ap = argparse.ArgumentParser(description="드론 사진 한 장으로 카메라의 경계 값을 만든다")
    ap.add_argument("image")
    ap.add_argument("--camera", default="main", help="카메라 이름: main (본 경기) · test1 · test2")
    ap.add_argument("--out", default=None, help="겹친 그림 파일 (기본: 사진 옆에 <이름>_경계.jpg)")
    ap.add_argument("--use-built-in", action="store_true", help="코드에 실린 그 카메라의 경계에서 출발한다")
    args = ap.parse_args()

    img = read_image(args.image)
    if img is None:
        print("사진을 읽지 못함:", args.image)
        return 2
    prior = camera_boundaries.built_in(args.camera) if args.use_built_in else None
    began = time.perf_counter()
    b = detect_boundary(img, prior=prior or None)
    took = time.perf_counter() - began
    print("사진 %s (%d x %d) · 찾은 길 %s · %.2f초" % (os.path.basename(args.image), img.shape[1], img.shape[0], b.debug.get("via"), took))
    for note in b.notes:
        print("   ", note)
    if b.hom is not None:
        raw, top = draw_before_after(img, b)
        h = max(raw.shape[0], top.shape[0])
        pad = [np.pad(a, ((0, h - a.shape[0]), (0, 0), (0, 0))) for a in (raw, top)]
        out = args.out or os.path.splitext(args.image)[0] + "_경계.jpg"
        write_image(out, np.hstack(pad))
        print("겹친 그림:", out)
    if not b.ok:
        print("검사를 통과하지 못했다 — 값을 찍지 않는다. 다른 사진(상자 · 로봇 · 사람이 선을 덜 가린 것)으로 다시 해 본다.")
        return 1
    d = b.to_dict()
    entry = {k: d[k] for k in KEYS}
    print("끝 띠 %.3f m · CM 반폭 %.3f m · 원 반지름 %.3f m · 잔차 RMS %.1f cm · 점 %d개" % (b.strip, b.cm_half, b.circle_r, b.rms_m * 100, b.points))
    print("겹친 그림을 눈으로 확인한 뒤, 아래 한 줄을 camera_boundaries.py 의 BUILT_IN 에 넣는다 (같은 이름이 있으면 바꾼다):")
    print('    "%s": %s,' % (args.camera, json.dumps(entry, ensure_ascii=False)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
