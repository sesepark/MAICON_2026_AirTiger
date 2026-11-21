"""드론 사진(/team/server/uav_image) → 지도(/team/perception/field_map). perception 노드에 붙는 연결 부분.

계산은 drone_boundary · drone_boxes · drone_field_map 이 한다 (perception 담당이 만든 것 — 여기서는 고치지 않는다).
이 파일은 받아서 넘기고 내보내는 연결만 한다 (2026-11-18 supervisor 쪽에서 붙임 — perception 담당의 요청).

  - 드론 사진이 올 때마다 가장 새 것 한 장만 들고 있다. 경기가 시작되면(/team/mission/state 의 phase 가 running) 그때의 가장 새 사진으로 계산한다.
    본 경기에서는 드론 사진이 시작 뒤에만 오므로 어차피 그렇게 되고, 연습용 카메라는 시작 전에도 사진을 주지만 같은 때에 계산하게 맞췄다
    (시작 전 사진에는 사람이 서 있을 수 있고, 본 경기와 같은 순서로 시험해야 걸리는 시간을 미리 알 수 있다)
  - 계산은 따로 띄운 프로세스가 한다 (경계 찾기에 몇 초 ~ 수십 초가 걸린다 → 그동안에도 이 노드의 사진 판독이 느려지지 않게).
    그 프로세스는 노드가 켜질 때 미리 띄워 두고(numpy · OpenCV 를 읽고 기다린다), map_timeout_s 안에 답이 없으면 끊고 새 사진으로 다시 한다.
    프로세스를 띄우지 못하거나 답이 깨져 오면 그 뒤로는 이 노드 안의 스레드에서 계산한다
  - 경계 · 상자 검사가 모두 통과한 첫 결과를 지도로 내보낸다. 그 뒤로는 계산하지 않고 같은 지도를 1초마다 다시 내보낸다
    (늦게 켜진 프로그램도 받게. 상자는 경기 중에 움직이지 않는다)
  - 검사를 통과하지 못한 결과는 내보내지 않는다 (틀린 지도로 움직이는 것보다 서 있는 편이 낫다). map_retry_s 뒤의 새 사진으로 다시 해 본다
  - 좌표는 계산 모듈이 내는 그대로다: 벽에서 벽까지 약 612 x 500 cm, x 는 BLUE 벽 → RED 벽, y 는 아래 → 위, frame_id "arena"
  - 거점 이름은 자리 이름(B-1 ~ R-3). navigation 은 이 이름으로 출발 칸을 찾고, supervisor 는 자리 이름과 보고값을 모두 읽는다
  - 그 여섯 개 뒤에, 벽 그림으로 줄마다 어느 거점인지 알아낸 진영은 보고값 이름(RED_MAINTENANCE 등) 세 개를 더 붙인다 (2026-11-19, base_panels.py).
    supervisor 는 뒤에 온 보고값 이름으로 덮어써서 그 종류의 거점이 실제로 있는 줄을 안다. 알아내지 못한 진영은 붙이지 않는다

지난 경계 다시 쓰기 (2026-11-19): 드론 카메라는 경기장 천장에 붙어 있어 경기마다 같다. 검사를 통과한 경계(렌즈 · 사영)를
  map_boundary_cache 파일(기본 ~/team/cache/drone_boundary.json)에 남겨 두고, 다음 계산(다음 시도 · 다음 경기)에서는 그것으로 선 조합을
  바로 골라 맞춘다 — 경계 찾기가 PC 에서 약 2 ~ 9초 → 0.2 ~ 0.5초. 이 화면과 맞지 않으면(카메라가 바뀜 · 움직임) 처음부터 찾는다.
  상자는 경기마다 다시 놓일 수 있으므로 상자 찾기는 늘 이 화면으로 한다.
  저장하는 것은 검사를 통과하고 치수(끝 띠 · CM 반폭)도 맞는 경계뿐이다 (boundary_trusted) — 구역선을 잘못 고른 경계가 좋은 경계를 밀어내지 않게.

코드에 실린 경계 (2026-11-19): 저장 파일이 없는 곳(처음 가는 경기장 · 새 로봇)에서도 「처음부터 찾기」 에만 기대지 않게,
  카메라마다 미리 맞춘 경계를 camera_boundaries.BUILT_IN 에 넣어 둔다 (지금은 연습 카메라 test1 · test2, 본 경기 카메라 main 은 사진을 받은 뒤).
  사진의 카메라(frame_id "uav" · "uav:test:test1" …)로 골라 저장 파일의 것 다음 차례로 대 본다. 다루는 법은 지난 경계와 같다 —
  이 화면의 흰 선으로 다시 맞추고 검사를 통과해야 쓴다. map.jsonl 의 boundary_prior 가 cache(저장 파일) · built_in(코드) 중 어느 것이었는지 알려 준다.

상자 찾기 (2026-11-19): 기본은 학습한 YOLO OBB 모델 (drone_box_model, map_box_detector="model"). 드론 화면을 학습 그림과 같은
  map 그림(drone_map_view — 경기장으로 잘라 상자 윗면 높이 평면 1 cm/px)으로 편 뒤 찾으므로 높이 보정이 들어간 상자 바닥 자리가 바로 나온다.
  상자들이 바닥 격자에 놓였으면 규칙과 같은 방법(drone_boxes._grid_divisors)으로 축마다 격자에 맞춘다. 거르기 · 순서 · 지도 형식은 규칙과 같다.
  모델을 못 쓰면(파일 · ultralytics 없음 · 읽기 실패) 그 시도는 예전 규칙(drone_boxes)으로 한다 — notes 에 남는다.
  카메라 높이: map_camera_height(> 0) > 경계 저장 파일에 카메라마다 쌓은 측정값의 중앙값(HEIGHT_MIN 개 이상) > 이 화면에서
  drone_boxes 가 옆면 폭으로 잰 값 (잴 때마다 저장 파일에 쌓는다 — 경기를 거듭할수록 흔들림이 준다). 학습 그림도 같은 방법(여러 장 중앙값)으로 폈다.

기록: 환경변수 TEAM_LOG_DIR 이 있으면 그 아래 perception/ 에 시도마다 한 줄(map.jsonl), 내보낸 지도(field_map.json), 그때의 드론 사진을 남긴다.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

from perception import arena_height, base_panels, camera_boundaries, drone_field_map
from perception.drone_boundary import detect_boundary
from perception.drone_boxes import BoxParams, detect_boxes


CACHE_KEEP = 4  # 경계 저장 파일에 남기는 수 (카메라 test1 · test2 · main …) — 새 것이 앞
HEIGHT_KEEP = 15  # 카메라마다 쌓는 높이 측정값 수
HEIGHT_MIN = 3  # 이만큼 쌓이면 중앙값을 쓰고 더 재지 않는다 (한 장의 측정은 ±15 % 흔들린다)
DETECTORS = ("mixed", "model", "rules")


def default_cache_path():
    return os.path.join(os.path.expanduser("~"), "team", "cache", "drone_boundary.json")


def load_boundaries(path):
    """경계 저장 파일 → [Boundary.to_dict()] (새 것이 앞). 없거나 깨졌으면 []."""
    if not path:
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    items = data.get("boundaries") if isinstance(data, dict) else None
    return [d for d in items or [] if isinstance(d, dict) and d.get("ok") and d.get("hom") is not None]


def _same_camera(a, b):
    """두 경계(to_dict)가 같은 카메라로 보이는가 (경기장 모서리 픽셀이 40 px 안)."""
    ca, cb = np.asarray(a.get("corners_px") or [], float), np.asarray(b.get("corners_px") or [], float)
    return bool(ca.shape == cb.shape and ca.size and np.isfinite(ca).all() and np.isfinite(cb).all() and np.abs(ca - cb).max() < 40)


def cached_heights(boundary, old):
    """이 경계와 같은 카메라로 보이는 저장 항목에 쌓인 카메라 높이 측정값들."""
    if boundary is None or boundary.hom is None:
        return []
    d = boundary.to_dict()
    for o in old:
        if _same_camera(o, d):
            return [float(h) for h in o.get("camera_heights") or [] if isinstance(h, (int, float))]
    return []


def save_boundary(path, boundary, old, heights=None):
    """검사를 통과한 경계를 맨 앞에 넣고 저장한다. 같은 카메라(같은 선 조합 자리)로 보이는 옛 것은 뺀다. 실패해도 조용히 넘어간다.
    heights: 그 카메라의 높이 측정값들 (없으면 옛 항목의 것을 이어 쓴다)."""
    if not path or not boundary.ok:
        return
    new = boundary.to_dict()
    new["saved_at"] = round(time.time(), 3)
    same = _same_camera
    if heights is None:
        heights = cached_heights(boundary, old)
    if heights:
        new["camera_heights"] = [round(float(h), 4) for h in heights][-HEIGHT_KEEP:]

    keep = [new] + [d for d in old if not same(d, new)]
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"boundaries": keep[:CACHE_KEEP]}, fh, ensure_ascii=False, default=float)
        os.replace(tmp, path)
    except OSError:
        pass


def boundary_trusted(boundary, params=None):
    """이 경계를 저장해 두고 다음에 다시 써도 되는가: 검사를 통과했고, 끝 띠 · CM 반폭이 상자 찾기가 받는 범위 안일 때만.

    2026-11-19 까지는 boundary.ok 만 보고 저장해서, 구역선을 잘못 고른 경계(CM 반폭 1.05 m — 상자 찾기가 「범위 밖」 으로 버린 것)가
    같은 카메라의 좋은 경계를 밀어내고 저장됐다. 그 뒤로는 그 카메라의 사진마다 처음부터 찾았다.
    """
    p = params or BoxParams()
    return bool(boundary is not None and boundary.ok and boundary.hom is not None and boundary.strip is not None
                and p.strip_range[0] <= boundary.strip <= p.strip_range[1] and p.cm_range[0] <= boundary.cm_half <= p.cm_range[1])


def compute(jpeg, stamp_s=None, base_naming="slot", camera_height=None, boundary_cache=None, arena=None, camera=None, border_wall_gap=None,
            box_detector="rules", box_model=None, box_conf=0.4, predict=None):
    """JPEG 바이트 → (지도 dict 또는 None, 설명 dict). 계산 모듈을 부르기만 한다. ROS 를 쓰지 않는다 (시험하기 쉽게).

    boundary_cache: 경계 저장 파일 경로 (None 이나 빈 문자열이면 쓰지 않는다). 있으면 지난 경계에서 출발하고, 통과한 경계를 저장한다.
    arena: 지도에 적을 벽에서 벽까지의 크기 (가로, 세로) m. None 이면 6.12 × 5.00 그대로 (drone_field_map.field_map 의 설명 참고).
    camera: 사진을 찍은 카메라 (frame_id "uav" · "uav:test:test1" 이나 이름 "main" · "test1"). 주면 코드에 실린 그 카메라의 경계
      (camera_boundaries.BUILT_IN)도 지난 경계처럼 대 본다 — 저장 파일의 것 다음 차례로. None 이면 쓰지 않는다.
    border_wall_gap: 화면 끝에 걸린 상자(옆면의 바닥선으로 찾은 것)를 위아래 벽에서 얼마나 떨어진 것부터 받을지 (m, 상자 가운데 기준).
      None 이면 BoxParams 의 기본값 0.45 (벽에 붙은 상자는 내지 않는다). 연습 경기장처럼 벽에 붙여 놓은 상자도 받으려면 0.2.
    box_detector: "model" (학습한 모델, drone_box_model) · "rules" (drone_boxes). 모델을 못 쓰면 규칙으로.
    box_model: 모델 파일 (None 이면 drone_box_model.DEFAULT_MODEL). predict: 시험용 — 주면 모델 파일 대신 이 함수를 쓴다.
    """
    import cv2

    began = time.monotonic()
    info = {"ok": False, "boxes": 0, "notes": []}
    image = cv2.imdecode(np.frombuffer(bytes(jpeg), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        info["notes"].append("사진을 풀지 못함")
        return None, info
    info["image"] = [int(image.shape[1]), int(image.shape[0])]
    priors = load_boundaries(boundary_cache)
    shipped = camera_boundaries.built_in(camera)
    if shipped:
        boundary = detect_boundary(image, prior=(priors + shipped) or None)
    else:
        # 코드에 실린 경계가 없는 카메라(본선 카메라): 처음부터 찾는 것을 먼저 하고, 그것이 안 됐을 때만 저장 파일의 지난 경계를 대 본다.
        # 2026-11-19 17:06 모의 경기 2차(본선 경기장): 1차에서 저장한 경계로 다시 맞춘 것(prior_refit)이 검사는 다 통과했는데 상자를 11개만 냈다
        # (실제 16개 — 같은 사진을 처음부터 찾으면 16개). 출발 칸 앞의 진짜 상자가 지도에 없어 로봇이 5분 내내 그 앞에 갇혔다 (막힘 21번).
        # 같은 사진을 제 저장 파일로 다시 넣어도(prior) 17개가 15개로 준다 — 이 카메라에서는 지난 경계에서 출발한 맞춤이 화면 끝에 걸린 상자를 놓친다
        boundary = detect_boundary(image, prior=None)
        if not boundary_trusted(boundary) and priors:
            again = detect_boundary(image, prior=priors)
            if again.ok and again.debug.get("via") != "search":
                boundary = again
    info["boundary_ok"] = bool(boundary.ok)
    info["boundary_s"] = round(time.monotonic() - began, 2)
    info["boundary_via"] = boundary.debug.get("via")
    index = boundary.debug.get("prior_index")
    info["boundary_prior"] = None if index is None else ("cache" if index < len(priors) else "built_in")  # 맞은 지난 경계가 어디 것인가
    # 어느 세로선 5개를 골랐고 렌즈를 어떻게 맞췄는가 (2026-11-20). 본선 카메라에서 끝 띠 선 대신 거점 테두리 선을 고른 지도가 검사를 통과해 나갔는데,
    # 기록에는 그것을 알아볼 줄이 없었다 (notes 는 앞의 여섯 줄만 남기고 렌즈 줄은 그 뒤다)
    info["boundary_lines"] = [None if u is None else int(u) for u in boundary.debug.get("combo") or ()] or None
    info["boundary_lens"] = next((str(n) for n in boundary.notes if str(n).startswith("렌즈")), None)
    info["boundary_tried"] = len(boundary.debug.get("tried") or ())
    if boundary_trusted(boundary):  # 검사를 통과하지 못했거나 치수가 이상한 경계는 저장하지 않는다 (좋은 경계를 밀어내지 않게)
        save_boundary(boundary_cache, boundary, priors)
    info["notes"] += [str(n) for n in list(boundary.notes)[:6]]
    if boundary.hom is None:
        return None, info
    params = BoxParams(camera_height=camera_height)
    if border_wall_gap is not None:
        params.border_wall_gap = float(border_wall_gap)
    view = None
    info["box_detector"] = "rules"
    if box_detector == "model":
        try:
            view = _model_boxes(image, boundary, priors, camera_height, boundary_cache, box_model, box_conf, predict, params, info)
            info["box_detector"] = "model"
        except Exception as error:  # noqa: BLE001 — 모델을 못 쓰면 이 시도는 규칙으로 (지도는 어떻게든 낸다)
            info["notes"].append("상자 모델을 못 씀 → 규칙으로: %s: %s" % (type(error).__name__, error))
            view = None
    if view is None:
        view = detect_boxes(image, boundary, params)
    info.update(boxes=len(view.boxes), boxes_ok=bool(view.ok), seconds=round(time.monotonic() - began, 2),
                strip=None if boundary.strip is None else round(float(boundary.strip), 3),
                cm_half=None if boundary.cm_half is None else round(float(boundary.cm_half), 3))
    info["notes"] += [str(n) for n in list(view.notes)[:6]]
    info["ok"] = bool(boundary.ok and view.ok and len(view.boxes) > 0)
    info.update(scale=round(float(view.scale), 4), divisor=[round(float(v), 4) for v in view.divisor], divisor_source=list(view.divisor_source),
                joined=sum(1 for b in view.boxes if b.joined), border=sum(1 for b in view.boxes if b.border))
    data = view.field_map(boundary, stamp_s, base_naming, arena=tuple(arena) if arena else None)
    # 이 지도를 만든 경계 · 카메라 높이 — 경기 중 드론 사진에서 UGV 를 찾는 쪽(ugv_watch)이 같은 것으로 화면을 편다. 기록에는 넣지 않는다 (attempt 가 뺀다)
    info["view"] = {"boundary": boundary.to_dict(), "camera_height": float(view.camera[2]) if view.camera is not None else None}
    # 거점 뒤 벽 그림으로 줄마다 어느 거점(작전지휘소 · 정비소 · 군수보급소)인지 읽는다 (2026-11-19 공지: 배치가 바뀔 수 있다 — base_panels.py).
    # 알아낸 진영만 bases 뒤에 보고값 이름(예 RED_MAINTENANCE)으로 덧붙인다. 앞의 자리 이름 여섯 개는 그대로다. 못 읽어도 지도는 그대로 나간다
    info["base_panels"] = base_panels.annotate(data, image, boundary, view)
    return data, info


def map_score(data, link_m=0.56, edge_m=1.15):
    """지도 한 장의 점수 (클수록 좋다): (짝을 이룬 상자 수, −남는 상자 수, 상자 수).

    객체 상자는 둘씩 붙어 서고(가운데 사이 0.40 m) 쌍끼리는 1 m 넘게 떨어져 있다 → 가운데가 link_m 안인 상자끼리 한 무리로 묶는다.
      - 짝을 이룬 상자: 둘 넘는 무리마다 둘씩. 붙은 쌍을 상자 하나로 뭉쳐 본 지도는 그 쌍이 통째로 빠진다 (모의 6차의 첫 사진: 12개 가운데 8개).
      - 남는 상자: 한 무리의 셋째부터, 그리고 좌우 벽에서 edge_m 안에 홀로 선 상자. 거점 사이의 노란 줄무늬 블록을 틀린 자리에 상자로 그린 것이 이렇게 나온다
        (모의 1차 (5.10, 3.75) — 진짜 쌍에서 0.48 m · 3차 (5.05, 3.41) · 5차 (5.06, 1.36)). 지금까지 본 경기장의 객체 상자는 좌우 벽에서 1.4 m 넘게 떨어져 있다.
      - 가운데에 홀로 선 상자가 있는 경기장도 있다 (연습 경기장의 둘) → 버리지 않고, 앞의 둘이 같을 때 상자 수가 많은 쪽을 고른다.
    """
    import math

    boxes = [(float(b["x"]), float(b["y"])) for b in (data or {}).get("boxes") or []]
    width = float((data or {}).get("width") or 6.12)
    group = list(range(len(boxes)))  # 무리 번호 (가까운 상자끼리 같은 번호가 되게 합친다)
    for i in range(len(boxes)):
        for j in range(i):
            if math.hypot(boxes[i][0] - boxes[j][0], boxes[i][1] - boxes[j][1]) <= link_m and group[i] != group[j]:
                old, new = group[i], group[j]
                group = [new if g == old else g for g in group]
    sizes = {}
    for g in group:
        sizes[g] = sizes.get(g, 0) + 1
    paired = sum(2 for n in sizes.values() if n >= 2)
    spare = sum(n - 2 for n in sizes.values() if n > 2)
    spare += sum(1 for (x, _y), g in zip(boxes, group) if sizes[g] == 1 and (x < edge_m or x > width - edge_m))
    return paired, -spare, len(boxes)


def pick_best(results):
    """여러 사진의 계산 결과 [(지도 dict 또는 None, 설명 dict)] 가운데 쓸 것의 번호 (없으면 None).
    지도가 나온 것(ok) 가운데 map_score 가 가장 큰 것, 같으면 먼저 찍힌 것 (앞 번호)."""
    best = None
    for k, (data, info) in enumerate(results):
        if data is None or not (info or {}).get("ok"):
            continue
        try:
            score = map_score(data)
        except Exception:  # noqa: BLE001 — 점수를 못 내도 지도는 골라야 한다 → 상자 수만으로
            score = (0, 0, len(data.get("boxes") or []))
        if best is None or score > best[0]:
            best = (score, k)
    return None if best is None else best[1]


def asymmetry(data):
    """상자 수가 BLUE · RED 로 맞보지 않는 줄의 목록 [(줄, BLUE 쪽 개수, RED 쪽 개수)]. 비어 있으면 맞본다.

    경기장은 좌우로 맞보는 꼴이라 같은 줄의 B-S{n} 과 R-S{n} 에는 상자가 같은 수만큼 있다 (지금까지 본 경기장 셋 모두).
    한쪽에서 상자를 놓쳤거나 헛것이 붙은 지도는 이 수가 어긋난다. 가운데 구역(CM)은 짝이 없어 보지 않는다.
    지도에 구역이 적혀 있지 않으면 판단하지 않는다 ([]).
    """
    sectors = {str(s.get("name")): s for s in (data or {}).get("sectors") or []}
    counts = dict.fromkeys(sectors, 0)
    for box in (data or {}).get("boxes") or []:
        for name, s in sectors.items():
            if s["min_x"] <= box["x"] < s["max_x"] and s["min_y"] <= box["y"] < s["max_y"]:
                counts[name] += 1
                break
    rows = sorted({name[3:] for name in sectors if name[:3] in ("B-S", "R-S")})
    return [(row, counts.get("B-S" + row, 0), counts.get("R-S" + row, 0)) for row in rows
            if counts.get("B-S" + row, 0) != counts.get("R-S" + row, 0)]


def pick_symmetric(results):
    """pick_best 와 같되, 상자 수가 BLUE · RED 로 맞보지 않는 지도(asymmetry)를 먼저 버린다 → (고른 번호 또는 None, 버린 번호들, 되돌렸나).

    설명 dict 마다 "symmetric" · "asymmetry" 를 적는다 (지도가 나온 것만). 하나도 남지 않으면 버리지 않은 것으로 치고
    전체에서 고른다 (되돌림 = True) — 맞보지 않는 경기장이거나 네 장 모두 한쪽을 놓친 때에도 지도는 나가야 한다.
    왜 (2026-11-19 본선 경기장 모의 경기 여섯 판의 드론 사진을 PC 에서 모델로 계산): 상자를 놓친 지도는 모두 한쪽(R-S1)만 놓쳐
      이 검사에 걸린다 (점수 한도 0.40 일 때 1차의 첫 장: 1번 줄 BLUE 2 · RED 1). 규칙이 내던 「줄무늬 블록을 상자로 그린 17개짜리」 도
      한쪽에만 붙으므로 걸린다. 양쪽에서 똑같이 놓친 지도와 가운데 구역의 잘못은 걸리지 않는다.
    """
    kept, dropped = [], []
    for k, (data, info) in enumerate(results):
        if data is None or not (info or {}).get("ok"):
            kept.append((None, info))
            continue
        try:
            bad = asymmetry(data)
        except Exception:  # noqa: BLE001 — 검사를 못 해도 지도는 골라야 한다 → 버리지 않는다
            bad = []
        info["symmetric"], info["asymmetry"] = not bad, [list(row) for row in bad]
        if bad:
            dropped.append(k)
        kept.append((None, info) if bad else (data, info))
    pick = pick_best(kept)
    if pick is not None or not dropped:
        return pick, dropped, False
    return pick_best(results), dropped, True


def fill_base_kinds(results, pick, mirror=True):
    """고른 사진이 읽지 못한 진영의 거점 배치(1 · 2 · 3번 줄이 작전지휘소 · 정비소 · 군수보급소 가운데 무엇인가)를 채워 지도에 덧붙인다.

    results = [(지도 dict 또는 None, 설명 dict)], pick = 고른 사진의 번호. 돌려주는 값: {진영: {"order": [...], "from": "frame" · "mirror"}} (채운 것만).
      1. 함께 계산한 다른 사진이 그 진영을 읽었으면 그것을 쓴다 ("frame"). 사진마다 답이 다르면 쓰지 않는다.
      2. 그래도 모르고 맞은편 진영은 알면, 같은 줄에 같은 종류가 있다고 본다 ("mirror", mirror=False 면 하지 않는다).
    왜 (2026-11-19 본선 경기장 모의 경기): 본선 카메라는 오른쪽 벽이 화면 밖에 걸려 RED 진영의 벽 그림을 사진의 70% 에서 읽지 못한다
      (BLUE 는 거의 늘 읽는다. 읽은 답은 사진 209장에서 한 번도 틀리지 않았다). 못 읽으면 supervisor 는 기본 배치(작전지휘소 · 정비소 · 군수보급소)로
      치는데, 오늘은 그것이 실제 배치와 우연히 같았다. 배치는 바뀔 수 있다 (주최측 공지).
      지금까지 본 경기장 셋(연습 둘 · 본선)은 모두 두 진영의 같은 줄에 같은 종류가 있었다 (경기장이 좌우로 맞보는 꼴이다).
      그렇지 않은 경기장이면 맞은편 것을 따라 적은 답이 틀리지만, 그때는 기본 배치로 쳐도 맞을 까닭이 없다 —
      어느 쪽이든 supervisor 는 목표가 아니면 다음 거점으로 들어가 본다 (mission._assault_order).
    """
    data, info = results[pick]
    panels = (info or {}).get("base_panels")
    if data is None or not isinstance(panels, dict):
        return {}
    kinds = dict(panels.get("kinds") or {})
    filled = {}
    for camp in base_panels.CAMPS:
        if kinds.get(camp):
            continue
        seen = set()
        for k, (_other, other_info) in enumerate(results):
            order = ((((other_info or {}).get("base_panels") or {}).get("kinds")) or {}).get(camp) if k != pick else None
            if order:
                seen.add(tuple(order))
        if len(seen) == 1:
            kinds[camp] = list(seen.pop())
            filled[camp] = {"order": kinds[camp], "from": "frame"}
    if mirror:
        for camp, other in (("BLUE", "RED"), ("RED", "BLUE")):
            if not kinds.get(camp) and kinds.get(other) and filled.get(other, {}).get("from") != "mirror":
                kinds[camp] = list(kinds[other])
                filled[camp] = {"order": kinds[camp], "from": "mirror"}
    if filled:
        base_panels.add_to_map(data, {camp: entry["order"] for camp, entry in filled.items()})
        panels["read"] = panels.get("kinds")  # 이 사진에서 읽은 것 그대로
        panels["kinds"] = kinds
        panels["filled"] = filled
    return filled


def _model_boxes(image, boundary, priors, camera_height, boundary_cache, box_model, box_conf, predict, params, info):
    """모델로 상자 찾기 (카메라 높이 고르기 포함). 실패하면 예외 — compute 가 규칙으로 돌아간다."""
    from perception import drone_box_model

    if predict is None:
        predict = drone_box_model.get_model(box_model or drone_box_model.DEFAULT_MODEL).predict
    heights = cached_heights(boundary, priors)
    if camera_height:
        hc, source = float(camera_height), "given"
    elif len(heights) >= HEIGHT_MIN:
        hc, source = float(np.median(heights)), "cache"
    else:  # 아직 모름 → 이 화면에서 규칙(drone_boxes)이 옆면 폭으로 잰다. 잰 값은 쌓아 둔다
        t0 = time.monotonic()
        rv = detect_boxes(image, boundary, BoxParams())
        hc, source = float(rv.camera[2]), rv.height_source
        info["height_measure_s"] = round(time.monotonic() - t0, 2)
        if source == "measured" and boundary_trusted(boundary):
            save_boundary(boundary_cache, boundary, priors, heights + [hc])
    info.update(camera_height=round(hc, 3), height_source=source, heights_cached=len(heights))
    t0 = time.monotonic()
    view, _mv = drone_box_model.detect_boxes_model(image, boundary, predict, hc, source, drone_box_model.ModelBoxParams(conf=box_conf), params)
    info["model_s"] = round(time.monotonic() - t0, 2)
    return view


# ---- 따로 띄우는 계산 프로세스. 주고받는 형식: 표준 입력 = 설정(JSON 한 줄) + 줄바꿈 + JPEG, 표준 출력 = {"data": 지도, "info": 설명} (JSON)
class ChildError(RuntimeError):
    """계산 프로세스가 답을 주지 못했다 (죽었거나, 답이 JSON 이 아니다)."""


def start_child(python=None, box_model=None, box_gpu=False):
    """계산 프로세스를 띄운다. 띄우면 곧바로 numpy · OpenCV 를 (box_model 을 주면 상자 모델도) 읽고 표준 입력을 기다린다.

    box_gpu=False (기본): 계산 프로세스는 상자 모델을 CPU 로 돌린다 (CUDA 를 가린다). 로봇(Jetson 8 GB)에서 새 프로세스로 잰 값 2026-11-19:
      GPU 면 프로세스마다 953 MB (쓸 수 있는 메모리 −700 MB) · 추론 37 ms, CPU 면 503 MB (−300 MB) · 추론 0.49초.
      사진 세 장을 프로세스 셋으로 함께 계산하므로(map_pick_frames) GPU 로는 2 GB 넘게 든다 — perception 노드의 모델 둘과 같이 쓰기에는 위험하다.
      CPU 로 돌리면 TensorRT(.engine)는 쓸 수 없으므로 .pt 를 읽는다 (DEFAULT_MODEL 자리에 .engine 만 있으면 git 에 넣은 .pt 로 간다).
    """
    env = dict(os.environ)
    if box_model:
        env["PERCEPTION_MAP_BOX_MODEL"] = box_model
        if not box_gpu:
            env["CUDA_VISIBLE_DEVICES"] = ""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # perception 꾸러미가 든 폴더
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    # 계산 프로세스 하나는 수학 라이브러리(BLAS)의 스레드를 하나만 쓴다. 로봇(Jetson, CPU 6개)에서 잰 값 2026-11-19:
    #   그대로 두면 프로세스마다 스레드를 6개씩 잡아, 사진 세 장을 프로세스 셋으로 함께 계산할 때 70초가 걸린다 (하나만 돌리면 4.2초).
    #   하나로 묶으면 하나만 돌릴 때 4.1초(같다), 셋을 함께 돌려도 4.7초.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[name] = "1"
    return subprocess.Popen([python or sys.executable, "-m", "perception.map_maker"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)


def ask_child(child, jpeg, options, timeout_s):
    """기다리던 계산 프로세스에 사진을 넘기고 답을 받는다 → (지도 dict 또는 None, 설명 dict). 시한을 넘기면 끊고 TimeoutError."""
    payload = json.dumps(options).encode("utf-8") + b"\n" + bytes(jpeg)
    try:
        out, _ = child.communicate(payload, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
        for pipe in (child.stdin, child.stdout):
            try:
                pipe.close()
            except OSError:
                pass
        raise TimeoutError("계산이 %.0f초 안에 끝나지 않아 끊음" % timeout_s) from None
    try:
        answer = json.loads(out.decode("utf-8"))
        return answer["data"], answer["info"]
    except (ValueError, KeyError, TypeError):
        raise ChildError("계산 프로세스의 답을 읽지 못함 (종료 코드 %s, %d바이트)" % (child.returncode, len(out))) from None


def child_main():
    out = sys.stdout.buffer
    sys.stdout = sys.stderr  # 계산 모듈이 무엇을 찍어도 답(JSON)에 섞이지 않게
    import cv2  # noqa: F401 — 사진을 기다리는 동안 미리 읽어 둔다

    preload = os.environ.get("PERCEPTION_MAP_BOX_MODEL")
    if preload:  # 상자 모델도 미리 읽어 둔다 (torch · TensorRT 를 읽는 데 몇 초 걸린다). 실패하면 compute 가 규칙으로 돌아간다
        try:
            from perception import drone_box_model

            drone_box_model.get_model(preload)
        except Exception:  # noqa: BLE001
            pass
    raw = sys.stdin.buffer.read()
    if not raw:
        return 0  # 일을 받지 못하고 끝남 (노드가 먼저 꺼졌다)
    head, _, jpeg = raw.partition(b"\n")
    try:
        options = json.loads(head.decode("utf-8"))
        data, info = compute(jpeg, options.get("stamp_s"), options.get("base_naming"), options.get("camera_height"), options.get("boundary_cache"),
                             options.get("arena"), options.get("camera"), options.get("border_wall_gap"),
                             options.get("box_detector", "rules"), options.get("box_model"), float(options.get("box_conf", 0.4)))
        text = json.dumps({"data": data, "info": info}, ensure_ascii=False, default=float)
    except Exception as error:  # noqa: BLE001 — 무슨 일이 있어도 부모가 읽을 수 있는 답을 준다
        text = json.dumps({"data": None, "info": {"ok": False, "boxes": 0, "notes": ["계산 중 오류 %s: %s" % (type(error).__name__, error)]}}, ensure_ascii=False)
    out.write(text.encode("utf-8"))
    out.flush()
    return 0


class MapMaker:
    """perception 노드 안에서 도는 지도 만들기. node 는 rclpy 노드, group 은 그 노드의 콜백 묶음."""

    def __init__(self, node, group):
        # ROS 쪽 import 는 여기서 한다 — 위의 compute() 는 ROS 없이도 불러 쓸 수 있어야 한다 (시험 · 계산 프로세스)
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from team_interfaces.msg import FieldMap, UavFrame

        self.node = node
        declare = node.declare_parameter
        self.enabled = bool(declare("field_map", True).value)
        self.base_naming = str(declare("map_base_naming", "slot").value) or None
        self.retry_s = float(declare("map_retry_s", 3.0).value)
        self.timeout_s = float(declare("map_timeout_s", 120.0).value)
        self.use_child = bool(declare("map_separate_process", True).value)
        self.started = not bool(declare("map_wait_for_start", True).value)  # 경기 시작(phase running)을 본 뒤에만 계산한다
        height = float(declare("map_camera_height", 0.0).value)  # 0 이면 사진에서 잰다
        self.camera_height = height if height > 0 else None
        # 본선 카메라의 높이 (상자 윗면이 바닥보다 크게 찍히는 배율 s = 높이 / (높이 − 0.4)). 0 이면 본선 카메라도 사진에서 잰다.
        # 2026-11-19 모의 경기 여섯 판(본선 경기장): 사진 한 장에서 잰 s 가 1.198 ~ 1.258 로 흔들렸다 (연습 경기장은 바닥 격자로 고쳐 1.17 ~ 1.23,
        # 본선 카메라는 화면이 좁아 그 격자 맞추기가 거의 안 된다). s 가 크게 재진 판(6차 1.258)은 상자가 가운데 쪽으로 8 cm 쯤 끌려가고
        # 붙은 상자 쌍을 하나로 보아 16개 가운데 12개만 냈다 — 그 판은 정찰이 CM 구역에서 끝났다. 잘 된 판(5차)의 격자 값이 1.180 이다.
        # 2.55 m → s 1.186. 그 여섯 판의 사진 233장: 상자 15개 넘게 낸 사진 145 → 159장 (높이 2.43 으로 견준 값), 자리의 흩어짐 0.056 → 0.042 m
        main_height = float(declare("map_camera_height_main", 2.55).value)
        self.camera_height_main = main_height if main_height > 0 else None
        # 처음 지도는 사진 한 장으로 정하지 않는다: 경기 시작 뒤의 사진을 pick_frames 장까지 모아(첫 사진 뒤 pick_wait_s 까지만 기다린다)
        # 한꺼번에 계산하고, 상자가 가장 많이 나온 것을 쓴다. 같은 장면인데도 사진마다 붙은 상자 쌍을 하나로 보거나 놓치는 일이 있다
        # (6차의 처음 여섯 장: 상자 14 · 16 · 14 · 15 · 16 · 16개 — 로봇은 첫 장으로 지도를 만들었다). 지도는 2 ~ 4초 늦게 나온다.
        # 로봇이 상자로 잘못 잡혀 하나 더 많은 사진이 뽑힐 수 있다 — 그것은 navigation 이 라이다로 뺀다 (1차에서 46초에 뺐다)
        # 네 장 (전: 세 장) — 모두 모델로 찾고, 상자 수가 BLUE · RED 로 맞보지 않는 지도는 버린 뒤 고른다 (pick_symmetric, map_pick_symmetric).
        # 본선의 드론 사진은 1.0 ~ 1.6초 간격이라 넷째 장은 첫 장 3.0 ~ 3.6초 뒤에 온다 → pick_wait_s 2.6 → 3.8 (지도가 1초쯤 더 늦게 나온다)
        self.pick_frames = max(1, int(declare("map_pick_frames", 4).value))
        self.pick_symmetric = bool(declare("map_pick_symmetric", True).value)
        # 고른 사진이 한 진영의 거점 벽 그림을 못 읽었을 때: 함께 계산한 다른 사진의 답을 쓰고, 그것도 없으면 맞은편 진영과 같은 줄로 본다 (fill_base_kinds)
        self.base_mirror = bool(declare("map_base_mirror", True).value)
        self.pick_wait_s = float(declare("map_pick_wait_s", 3.8).value)
        gap = float(declare("map_border_wall_gap", 0.0).value)  # 화면 끝에 걸린 상자를 벽에서 얼마나 떨어진 것부터 받을지 (m). 0 이면 기본값 0.45
        self.border_wall_gap = gap if gap > 0 else None
        # 지난 경계의 저장 파일: 기본으로 쓰지 않는다 (빈 문자열). 2026-11-19 모의 경기 2차에서 저장 파일의 경계로 만든 지도가 상자 5개를 놓쳤다
        # (compute 의 설명). 처음부터 찾는 데 로봇에서 2초쯤 걸릴 뿐이고(모의 경기 1차 1.94초), 경기마다 한 번만 한다. 쓰려면 경로를 준다
        cache = str(declare("map_boundary_cache", "").value)
        self.boundary_cache = os.path.expanduser(cache) if cache else None
        from perception.drone_box_model import DEFAULT_MODEL

        # 상자 찾기: mixed (기본) = 함께 계산하는 사진들 가운데 첫 장은 학습한 모델(drone_box_model)로, 나머지는 규칙(drone_boxes)으로 찾고
        # map_score 가 가장 큰 지도를 고른다 (같으면 앞 번호 = 모델의 것). model = 모두 모델로 · rules = 모두 규칙으로.
        # 왜 섞는가 (2026-11-19 본선 경기장 모의 경기 다섯 판의 드론 사진 693장으로 견줌 — 인식 담당의 map-yolo 가지):
        #   모델: 시작 사진 한 장에서 상자 16개를 모두 찾은 것 15장 가운데 14장 (규칙 9장) · 경기 내내의 사진 138장 가운데 132장 (규칙 32장),
        #         줄무늬 블록을 상자로 그린 것 0장 (규칙은 시작 사진 15장 가운데 3장). 라이다로 본 위쪽 쌍의 자리와도 더 잘 맞는다.
        #   다만 모델은 윗면이 화면 밖으로 다 나간 상자는 찾지 못한다 (규칙은 옆면의 바닥선으로 찾는다). 본선 카메라는 위쪽 오른편 쌍의
        #   윗면이 5 ~ 10 cm 만 보인다 (점수 0.40 ~ 0.66) — 카메라가 조금만 달라져도 모델만으로는 그 쌍을 잃을 수 있다.
        #   → 서로의 약한 곳을 막는다. 계산 프로세스도 모델을 읽은 것은 하나면 된다 (로봇에서 프로세스마다 500 MB).
        # 2026-11-19 밤: 기본을 model 로 (전: mixed). 모의 경기 여섯 판의 처음 네 장씩 24장을 이 코드로 PC 에서 계산하니(점수 한도 0.30)
        #   24장 모두 16개 · 헛것 0 이었다 (한도 0.40 이면 한 장이 위쪽 오른편 상자를 놓치고, 그 지도는 pick_symmetric 이 버린다).
        #   규칙의 지도는 같은 여섯 판의 처음 열 장씩 60장에서 헛것 39개를 냈다.
        #   위의 걱정(윗면이 화면 밖으로 나간 쌍을 모델이 잃는 것)은 그대로다 — 네 장 모두 그 쌍을 잃으면 되돌림으로 그 지도가 나간다.
        #   계산 프로세스 넷이 모두 모델을 읽는다 (CPU 로 프로세스마다 500 MB — 로봇에서 넷을 함께 띄워 재 보지는 않았다). 무거우면 mixed 로 되돌린다.
        self.box_detector = str(declare("map_box_detector", "model").value)
        if self.box_detector not in DETECTORS:
            node.get_logger().error("map_box_detector=%r 는 mixed · model · rules 중 하나 → mixed" % self.box_detector)
            self.box_detector = "mixed"
        self.box_model = os.path.expanduser(str(declare("map_box_model", DEFAULT_MODEL).value))  # .engine 이 없으면 같은 이름의 .pt
        # 모델이 낸 상자의 점수 한도. 0.40 이면 모의 1차의 첫 사진에서 위쪽 오른편 상자(0.397)를 버려 15개, 0.30 이면 시작 사진 15장 모두 16개 · 헛것 0
        self.box_conf = float(declare("map_box_conf", 0.30).value)
        self.box_gpu = bool(declare("map_box_gpu", False).value)  # 계산 프로세스가 상자 모델을 GPU 로 돌릴지 (start_child 의 설명 — 기본은 CPU)
        # 지도에 적을 벽에서 벽까지의 크기 (m). 상자는 정중앙 기준 자리를 지키고 벽 · 구역 · 거점만 이 크기로 낸다 (drone_field_map.field_map 의 arena).
        # 세로는 경기장마다 다르다 (2026-11-19 팀장 전달: 연습 경기장 6.12 × 4.8, 본선 경기장 6.12 × 5 — 줄자로 정확히 재지는 못함):
        #   연습 경기장 4.80 — 오전 2경기장 실차 두 판에서 라이다가 아래 벽을 y ≈ 0.13, 위쪽 가운데 기둥 면을 y ≈ 4.5 에서 봤다 (5.00 지도보다 위아래 10 cm 씩 안쪽)
        #   본선 경기장 — 사전 자료는 「약 612 × 500」 이었으나 2026-11-19 모의 경기에서 라이다로 재니 4.83 이었다 (아래 height_main 의 설명)
        # 마감 뒤에는 설정을 바꿀 수 없으므로 사진이 어느 카메라의 것인지로 스스로 고른다: server_bridge 가 팀 테스트(시험 포트)의 사진에는
        # frame_id "uav:test:<카메라>" 를, 본 경기의 사진에는 "uav" 를 적는다 (arena_for). map_arena_height 를 0 보다 크게 주면 그 값으로 고정한다.
        # 가로 6.12 는 드론 사진에서 잰 값이다 (벽이 바닥 격자보다 6 ~ 8 cm 바깥, 끝 띠 선이 벽에서 0.88 m)
        self.arena_width = float(declare("map_arena_width", 6.12).value)
        self.arena_height = float(declare("map_arena_height", 0.0).value)  # 0 = 카메라로 고른다
        self.height_practice = float(declare("map_height_practice", 4.80).value)
        # 2026-11-19 17:05 본선 경기장 값 5.00 → 4.80. 16:56 모의 경기(본선 경기장 · 본 경기 방식)의 라이다 원본 349장으로 직접 쟀다:
        # 안쪽 세로 가운데값 4.830 m (10 ~ 90% 가 4.811 ~ 4.847), 가로 6.081 m — 연습 경기장(4.813 · 6.040)과 사실상 같다.
        # 「본선은 약 612 × 500」 은 사실이 아니었다. 그 판은 5.00 지도라 위아래 벽이 8 ~ 9 cm 씩 바깥에 있었고, 로봇이 진짜 벽 · 상자를
        # 「지도에 없는 장애물」 로 보아 막힘 12번 · 한자리에 105초를 서 있었다 (서서 찍은 사진 0장)
        self.height_main = float(declare("map_height_main", 4.80).value)
        # 라이다로 직접 재기 (arena_height.py): 경기 시작 전 출발 칸에 서 있는 동안의 스캔에서 위아래 벽 사이를 잰다. 4.80 이나 5.00 에 분명히 맞으면
        # 카메라로 고른 값보다 그것을 쓴다 (본선 경기장의 5.00 은 전해 들은 값이라 로봇이 스스로 확인하게 한다). 애매하면 카메라로 고른 값 그대로.
        self.height_from_lidar = bool(declare("map_height_from_lidar", True).value)
        # 라이다 표의 후보에는 5.00 을 남겨 둔다 (어느 경기장이 정말 5.00 이고 라이다가 분명히 그렇게 재면 그 값을 쓴다)
        self.vote = arena_height.HeightVote(candidates=tuple(sorted({self.height_practice, self.height_main, 5.00})))
        self.scan = None  # 가장 새 /scan (LaserScan) — work() 가 1초에 한 번쯤 꺼내 잰다
        self.scan_used = None
        self.arena = self.arena_for(None)  # 마지막으로 고른 크기 (사진을 한 장도 못 받았으면 본 경기 것 — 임시 지도에 쓴다)
        # 경기가 시작되고 이만큼 지나도 지도를 못 만들었으면(사진이 안 오거나, 본선 카메라 화면에서 경계 · 상자를 못 찾거나) 벽만 있는 임시 지도를 낸다.
        # 지도가 없으면 로봇은 경기 내내 출발 칸에 서 있다 — 임시 지도가 있으면 적어도 거점 진입(M5)은 한다. 0 이면 끔.
        # 본선 카메라(camera=main)는 경기 밖에서는 받을 수 없어 지도 계산을 그 화면으로 시험해 보지 못했다 (2026-11-19)
        self.fallback_s = float(declare("map_fallback_s", 30.0).value)
        self.give_up_s = float(declare("map_give_up_s", 150.0).value)  # 임시 지도를 낸 뒤에도 이때까지는 진짜 지도를 계속 시도한다 (시작부터 초)
        self.started_at = time.monotonic() if self.started else None
        self.bare = False  # 지금 내보내는 것이 벽만 있는 임시 지도인가
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.frame = None  # 가장 새 드론 사진 (UavFrame)
        self.view_info = None  # 내보낸 지도를 만든 경계 · 카메라 높이 · 지도 크기 {"boundary", "camera_height", "arena"} (ugv_watch 가 읽는다)
        self.frames = 0
        self.recent = []  # 경기 시작 뒤에 받은 서로 다른 드론 사진들 (가장 새 pick_frames 장)
        self.first_frame_at = None  # 경기 시작 뒤 첫 사진을 받은 때
        self.spare = []  # 여러 장을 한꺼번에 계산할 때 쓸 나머지 계산 프로세스들
        self.tries = 0
        self.message = None  # 내보낸 지도 (FieldMap)
        self.child = None  # 사진을 기다리는 계산 프로세스
        self.detail = "지도: 드론 사진을 기다림" if self.enabled else "지도: 끔"
        shared = os.environ.get("TEAM_LOG_DIR")
        self.log_dir = os.path.join(shared, "perception") if shared and os.environ.get("TEAM_LOG_OFF") != "1" else None
        if not self.enabled:
            return
        # QoS 는 노드_관리/interfaces.yaml 에 적힌 설계 그대로 (지도: reliable · volatile · 1, 드론 사진: best_effort · volatile · 5)
        latest = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        frames = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        self.publisher = node.create_publisher(FieldMap, "/team/perception/field_map", latest)
        node.create_subscription(UavFrame, "/team/server/uav_image", self.on_frame, frames, callback_group=group)
        if self.height_from_lidar:
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import LaserScan

            node.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data, callback_group=group)
        node.create_timer(1.0, self.republish, callback_group=group)
        self.child = self.spawn()
        self.spare = [c for c in (self.spawn(first=False) for _ in range(self.pick_frames - 1)) if c is not None]
        threading.Thread(target=self.work, name="map-maker", daemon=True).start()

    # ---- ROS 콜백 (빨리 끝낸다)
    def arena_for(self, frame):
        """이 드론 사진에 맞는 지도 크기 [가로, 세로]. 팀 테스트(연습 경기장)의 사진이면 세로 4.80, 본 경기의 사진이면 5.00."""
        if self.arena_width <= 0.0:
            return None  # 끔: 경계를 맞출 때 쓴 크기(6.12 × 5.00) 그대로
        if self.arena_height > 0.0:
            return [self.arena_width, self.arena_height]
        measured = self.vote.decide() if self.height_from_lidar else None
        if measured is not None:
            return [self.arena_width, measured]  # 라이다로 잰 값이 분명하다
        where = str(getattr(getattr(getattr(frame, "image", None), "header", None), "frame_id", "") or "")
        return [self.arena_width, self.height_practice if where.startswith("uav:test") else self.height_main]

    def on_scan(self, message):
        self.scan = message  # 재는 것은 work() 에서 한다 (여기서는 받아 두기만)

    def measure_walls(self):
        """받아 둔 스캔 한 장으로 위아래 벽 사이를 잰다 (지도를 내기 전, 로봇이 출발 칸에 서 있는 동안에만 부른다)."""
        scan = self.scan
        if not self.height_from_lidar or scan is None or scan is self.scan_used:
            return
        self.scan_used = scan
        try:
            self.vote.add(arena_height.wall_spans(scan.ranges, scan.angle_min, scan.angle_increment))
        except Exception as error:  # 재다가 난 오류로 지도 계산이 멈추면 안 된다
            self.node.get_logger().warning("라이다로 벽 사이를 재다가 오류: %s" % error)

    def on_frame(self, message):
        with self.lock:
            self.frame = message
            self.frames += 1
            self.arena = self.arena_for(message)
            if self.started:  # 경기 시작 뒤의 사진만 모은다 (같은 사진이 다시 오면 넣지 않는다)
                self.collect(message)
        if self.message is None and not self.started:
            self.detail = "지도: 경기 시작을 기다림 (드론 사진 %d장 받음)" % self.frames
        self.wake.set()

    def collect(self, message):
        """지도에 쓸 사진으로 모은다 (lock 을 쥐고 부른다). 바로 앞의 것과 같은 사진이면 넣지 않는다. 가장 새 pick_frames 장만 남긴다."""
        stamp = message.header.stamp
        key = (stamp.sec, stamp.nanosec, str(getattr(message, "captured_at_utc", "")))
        if self.recent and self.recent[-1][0] == key:
            return
        self.recent = (self.recent + [(key, message)])[-self.pick_frames:]
        if self.first_frame_at is None:
            self.first_frame_at = time.monotonic()

    def on_mission(self, message):
        """/team/mission/state 가 올 때마다 노드가 불러 준다."""
        if message.phase == "running" and not self.started:
            with self.lock:
                self.recent, self.first_frame_at = [], None
                if self.frame is not None:  # 시작 전에 받아 둔 가장 새 사진 (팀 테스트) — 첫 장으로 쓰고, 뒤따라 오는 사진을 더 모은다
                    self.collect(self.frame)
            self.started = True
            self.started_at = time.monotonic()
            self.wake.set()

    def republish(self):
        if self.message is not None:
            self.publisher.publish(self.message)

    # ---- 계산 (따로 도는 스레드가 계산 프로세스에 맡기고 기다린다)
    def detector_for(self, k):
        """함께 계산하는 사진들 가운데 k 번째(0부터)의 상자 찾기: mixed 면 첫 장만 모델."""
        if self.box_detector == "mixed":
            return "model" if k == 0 else "rules"
        return self.box_detector

    def spawn(self, first=True):
        """계산 프로세스를 띄운다. first = 첫 장을 맡을 프로세스인가 (mixed 면 그것만 상자 모델을 미리 읽는다)."""
        if not self.use_child:
            return None
        model = self.detector_for(0 if first else 1) == "model"
        try:
            return start_child(box_model=self.box_model if model else None, box_gpu=self.box_gpu)
        except OSError as error:
            self.use_child = False
            self.node.get_logger().error("지도 계산 프로세스를 띄우지 못함 → 노드 안에서 계산: %s" % error)
            return None

    def work(self):
        last_try = -1e9
        while self.message is None or self.bare:
            self.wake.wait(1.0)
            self.wake.clear()
            with self.lock:
                frame = self.frame
            if self.message is None:
                self.measure_walls()  # 지도를 내기 전(로봇이 서 있는 동안)에만 잰다
            waited = time.monotonic() - self.started_at if self.started_at is not None else 0.0
            if self.message is None and self.started and self.fallback_s > 0.0 and waited >= self.fallback_s:
                self.publish_bare(waited)
            if self.bare and waited >= self.give_up_s:
                self.detail += " · 진짜 지도는 포기"
                return
            retry = max(self.retry_s, 10.0) if self.bare else self.retry_s  # 임시 지도를 낸 뒤에는 드물게 (계산이 다른 프로그램의 CPU 를 먹는다)
            if frame is None or not self.started or time.monotonic() - last_try < retry:
                continue
            with self.lock:
                batch = [m for _key, m in self.recent[-self.pick_frames:]] or [frame]
                first_at = self.first_frame_at
            if (self.message is None and len(batch) < self.pick_frames and first_at is not None
                    and time.monotonic() - first_at < self.pick_wait_s):
                continue  # 사진을 더 모은다 (다음 사진이나 1초 뒤에 다시 본다)
            last_try = time.monotonic()
            self.tries += 1
            self.detail = "지도: 계산 중 (%d번째 시도, 사진 %d장)" % (self.tries, len(batch))
            try:
                self.attempt(batch)
            except Exception as error:  # noqa: BLE001 — 계산이 죽어도 perception 의 다른 일(사진 판독)은 계속한다
                self.detail = "지도: 계산 중 오류 %s (%d번째 시도)" % (type(error).__name__, self.tries)
                self.node.get_logger().error("지도 계산 오류: %s: %s" % (type(error).__name__, error))
                self.write_log(e="error", error="%s: %s" % (type(error).__name__, error))
            if (self.message is None or self.bare) and self.child is None:
                self.child = self.spawn()  # 다음 시도에 쓸 프로세스를 미리 띄워 둔다
        for child in [self.child] + self.spare:  # 지도를 냈다 → 기다리던 계산 프로세스들을 끝낸다
            if child is not None:
                try:
                    child.kill()
                except OSError:
                    pass
        self.child, self.spare = None, []

    def publish_bare(self, waited):
        """벽만 있는 임시 지도를 낸다. 시각은 0 으로 둔다 — 나중에 진짜 지도(사진을 찍은 시각)가 오면 navigation 이 더 새것으로 받게."""
        size = {} if self.arena is None else {"width": self.arena[0], "height": self.arena[1], "strip": 0.88 * self.arena[0] / 6.12,
                                              "cm_half": 0.60 * self.arena[0] / 6.12}  # 구역선은 가로 비율만큼 (field_map 의 arena 와 같은 셈)
        data = drone_field_map.bare_field_map(None, self.base_naming, **size)
        self.message = drone_field_map.field_map_msg(data)
        self.bare = True
        self.publisher.publish(self.message)
        self.detail = "지도: 임시 (벽만, 상자 0개) — 시작 뒤 %.0f초 동안 지도를 못 만듦 (시도 %d번, 사진 %d장)" % (waited, self.tries, self.frames)
        self.node.get_logger().warning(self.detail)
        self.write_log(e="bare", waited_s=round(waited, 1), tries=self.tries, frames=self.frames)

    def camera_height_for(self, frame):
        """이 사진에 쓸 카메라 높이 (None = 사진에서 잰다). 본선 카메라의 사진이면 map_camera_height_main."""
        if self.camera_height is not None:
            return self.camera_height
        where = str(getattr(getattr(getattr(frame, "image", None), "header", None), "frame_id", "") or "")
        return None if where.startswith("uav:test") else self.camera_height_main

    def compute_one(self, frame, child, detector="rules"):
        """사진 한 장을 계산한다 → (지도 dict 또는 None, 설명 dict, 어떻게 "process" · "thread", 걸린 초, 지도 크기). detector = model · rules."""
        stamp = frame.header.stamp.sec + frame.header.stamp.nanosec * 1e-9
        arena = self.arena_for(frame)
        options = {"stamp_s": stamp or None, "base_naming": self.base_naming, "camera_height": self.camera_height_for(frame),
                   "boundary_cache": self.boundary_cache, "arena": arena,
                   "camera": str(frame.image.header.frame_id),  # 코드에 실린 그 카메라의 경계를 고르는 데 쓴다 (camera_boundaries)
                   "border_wall_gap": self.border_wall_gap,
                   "box_detector": detector, "box_model": self.box_model, "box_conf": self.box_conf}
        began = time.monotonic()
        data, info, how = None, None, "process"
        if child is not None:
            try:
                data, info = ask_child(child, frame.image.data, options, self.timeout_s)
            except TimeoutError as error:
                info = {"ok": False, "boxes": 0, "notes": [str(error)]}
            except (ChildError, OSError) as error:
                self.use_child = False
                self.node.get_logger().error("지도 계산 프로세스가 답하지 못함 → 노드 안에서 계산: %s" % error)
                self.write_log(e="child_failed", error=str(error))
        if info is None:
            how = "thread"
            data, info = compute(frame.image.data, **options)
        return data, info, how, time.monotonic() - began, arena

    def attempt(self, frames):
        """사진들(한 장이면 그 한 장)을 한꺼번에 계산하고, 지도가 나온 것 가운데 상자 수가 BLUE · RED 로 맞보지 않는 것을 버린 뒤
        (pick_symmetric) map_score 가 가장 큰 것을 내보낸다 (같으면 먼저 찍힌 것)."""
        frames = list(frames) if isinstance(frames, (list, tuple)) else [frames]
        began = time.monotonic()
        children, self.child = [self.child], None
        while len(children) < len(frames):
            children.append(self.spare.pop() if self.spare else self.spawn(first=False))
        # 계산 프로세스를 얻지 못한 사진은 뺀다 (노드 안에서 여러 장을 함께 계산하면 사진 판독이 밀린다). 하나도 없으면 첫 장만 노드 안에서 규칙으로
        # (노드 안에서 상자 모델까지 읽으면 사진 판독이 그동안 멈춘다).
        usable = [(f, c, self.detector_for(k)) for k, (f, c) in enumerate(zip(frames, children)) if c is not None] or [(frames[0], None, "rules")]
        frames, children, detectors = [f for f, _c, _d in usable], [c for _f, c, _d in usable], [d for _f, _c, d in usable]
        results = [None] * len(frames)

        def run(k):
            try:
                results[k] = self.compute_one(frames[k], children[k], detectors[k])
            except Exception as error:  # noqa: BLE001 — 한 장의 계산이 죽어도 다른 장은 쓴다
                results[k] = (None, {"ok": False, "boxes": 0, "notes": ["계산 중 오류 %s: %s" % (type(error).__name__, error)]}, "error", time.monotonic() - began, None)

        workers = [threading.Thread(target=run, args=(k,), name="map-frame-%d" % k, daemon=True) for k in range(1, len(frames))]
        for worker in workers:
            worker.start()
        run(0)
        for worker in workers:
            worker.join(self.timeout_s + 5.0)
        results = [r if r is not None else (None, {"ok": False, "boxes": 0, "notes": ["시한 안에 끝나지 않음"]}, "timeout", time.monotonic() - began, None) for r in results]
        views = [info.pop("view", None) for _data, info, _how, _took, _arena in results]
        pairs = [(data, info) for data, info, _how, _took, _arena in results]
        dropped, restored = [], False
        if getattr(self, "pick_symmetric", True):
            pick, dropped, restored = pick_symmetric(pairs)
        else:
            pick = pick_best(pairs)
        if pick is not None:
            try:
                fill_base_kinds([(data, info) for data, info, _how, _took, _arena in results], pick, self.base_mirror)
            except Exception as error:  # noqa: BLE001 — 거점 배치를 못 채워도 지도는 나가야 한다
                self.node.get_logger().warning("거점 배치 채우기 오류: %s: %s" % (type(error).__name__, error))
        took = time.monotonic() - began
        middle = self.vote.median()
        for k, (data, info, how, each, arena) in enumerate(results):
            info.update(tries=self.tries, frames=self.frames, captured=frames[k].captured_at_utc, how=how, took_s=round(each, 2),
                        arena=arena, camera=str(frames[k].image.header.frame_id), batch=len(frames), picked=(k == pick),
                        dropped_asymmetric=(k in dropped), symmetry_restored=restored,
                        lidar_height={"decided": self.vote.decide(), "median": None if middle is None else round(middle, 3),
                                      "values": [round(v, 3) for v in self.vote.values], "scans": self.vote.scans})
            self.write_log(e="try", **info)
        if pick is None:
            info = results[-1][1]
            why = "; ".join([str(n) for n in info.get("notes", []) if not str(n).startswith("OK")][-2:]) or "검사를 통과하지 못함"
            self.detail = "지도: 아직 없음 (%d번째 시도 %.1f초, 사진 %d장, 상자 %d개 — %s)" % (self.tries, took, len(frames), info.get("boxes", 0), why[:120])
            self.node.get_logger().warning(self.detail)
            return
        data = results[pick][0]
        message = drone_field_map.field_map_msg(data)
        self.message = message
        self.bare = False
        self.publisher.publish(message)
        if views[pick]:  # 드론 사진의 UGV 찾기(ugv_watch)가 쓴다
            self.view_info = dict(views[pick], arena=results[pick][4])
        counts = "·".join(str(len(d["boxes"])) if d is not None and i.get("ok") else "x" for d, i, _h, _t, _a in results)
        self.detail = "지도: 상자 %d개, 면 %d개 (%.1f초 걸림, %d번째 시도, 사진 %d장의 상자 수 %s 가운데 %d번째, 상자 찾기 %s, 맞보지 않아 버린 것 %d장%s)" % (
            len(data["boxes"]), len(data["faces"]), took, self.tries, len(frames), counts, pick + 1, results[pick][1].get("box_detector", "?"),
            len(dropped), " — 남은 것이 없어 전체에서 고름" if restored else "")
        self.node.get_logger().info(self.detail)
        self.save(frames[pick], data)

    # ---- 기록
    def write_log(self, **record):
        if self.log_dir is None:
            return
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(os.path.join(self.log_dir, "map.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(record, wall=round(time.time(), 3)), ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass

    def save(self, frame, data):
        if self.log_dir is None:
            return
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(os.path.join(self.log_dir, "field_map.json"), "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=1)
            if os.environ.get("TEAM_LOG_LIGHT") != "1":
                with open(os.path.join(self.log_dir, "drone_used.jpg"), "wb") as fh:
                    fh.write(bytes(frame.image.data))
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(child_main())
