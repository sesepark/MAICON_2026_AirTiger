"""시야 끝 자르기 변형 — 카메라 시야 끝(map 그림의 검정 테두리)이 상자를 가로지르는 화면을 만든다.

까닭 (2026-11-19 본선 항공뷰 0162.jpg): 화면 위쪽 끝에 윗면이 잘린 상자 3개를 v1 모델이 못 찾았다. 학습 라벨에는 윗면이
절반 넘게 잘린 상자가 하나도 없었다 (상자 라벨 609 개 중 일부 잘림 47 개, 모두 50 % 넘게 보임).

map 그림은 경기장 좌표에 고정(1 cm/px)이라 그림을 잘라 옮기면(crop) 좌표가 어긋난다 → 그림 크기 · 좌표는 그대로 두고
시야 끝(검정)을 안쪽으로 밀어 상자 하나를 가로지르게 한다. 실제 map 그림의 시야 끝처럼 조금 기울고 휜 곡선이다.

라벨: 남은 비율이 keep_min 이상이면 **상자 전체** 라벨을 그대로 둔다 (잘려도 상자 가운데 = 지도 자리를 내야 하니까).
keep_min 보다 적게 보이면 헷갈리지 않게 그 상자 자리를 마저 검게 칠하고 라벨을 뺀다.

  cut(img, labels, cfg, rng) → (그림, 라벨 줄들, 설명 dict 또는 None)
    labels: YOLO OBB 줄 ("cls x1 y1 ... x4 y4", 0 ~ 1 정규화)
"""
from __future__ import annotations

import numpy as np

DEFAULTS = {
    "p": 0.5,              # 변형 1장마다 시야 끝 자르기를 걸 확률
    "sides": {"top": 0.35, "bottom": 0.35, "left": 0.15, "right": 0.15},  # 어느 쪽 끝을 밀까 (본선 화면은 위 · 아래가 잘린다)
    "visible": [0.2, 0.85],  # 겨냥한 상자에 남길 비율
    "keep_min": 0.2,         # 이보다 적게 보이는 라벨은 빼고 그 자리를 검게 칠한다
    "tilt": 0.08,            # 시야 끝 기울기 (픽셀/픽셀, ±)
    "bend": 60.0,            # 그림 폭 끝에서의 휨 (px, ±)
    "second": 0.2,           # 다른 쪽 끝도 하나 더 자를 확률
    "max_depth": 0.3,        # 시야 끝을 그림 가장자리에서 이 비율(그 축 길이)보다 깊이 밀지 않는다 — 실제 시야 끝은 경기장 끝 쪽만 가린다
    "target_class": "0",     # 겨냥할 라벨 종류 (0 = box)
}


def _parse(lines, w, h):
    out = []
    for line in lines:
        v = line.split()
        if len(v) != 9:
            continue
        c = [float(x) for x in v[1:]]
        out.append((v[0], np.array([(c[i] * w, c[i + 1] * h) for i in range(0, 8, 2)], float)))
    return out


def _poly_mask(shape, pts):
    import cv2

    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return m > 0


def _cut_mask(shape, side, anchor, edge, tilt, bend):
    """side 쪽에서 곡선 너머를 True 로. 곡선은 겨냥한 상자의 anchor(가로 · 세로 위치)에서 edge 를 지난다."""
    h, w = shape
    if side in ("top", "bottom"):
        x = np.arange(w, dtype=float)
        curve = edge + tilt * (x - anchor) + bend * (((x - w / 2.0) / (w / 2.0)) ** 2 - ((anchor - w / 2.0) / (w / 2.0)) ** 2)
        yy = np.arange(h, dtype=float)[:, None]
        return yy < curve[None, :] if side == "top" else yy > curve[None, :]
    y = np.arange(h, dtype=float)
    curve = edge + tilt * (y - anchor) + bend * (((y - h / 2.0) / (h / 2.0)) ** 2 - ((anchor - h / 2.0) / (h / 2.0)) ** 2)
    xx = np.arange(w, dtype=float)[None, :]
    return xx < curve[:, None] if side == "left" else xx > curve[:, None]


def _one_cut(shape, labs, seen, cfg, rng, exclude_side=None):
    h, w = shape
    sides = {k: float(v) for k, v in cfg["sides"].items() if k != exclude_side and float(v) > 0}
    if not sides:
        return None, None
    names = list(sides)
    side = names[int(rng.choice(len(names), p=np.array([sides[n] for n in names]) / sum(sides.values())))]
    # 아직 잘 보이는 상자 중 그 쪽 끝에 가까운 것일수록 자주 고른다 (실제로도 끝 쪽 상자가 잘린다)
    cand = []
    for i, (cls, pts) in enumerate(labs):
        if cls != str(cfg["target_class"]):
            continue
        area = seen[i][1]
        if area == 0 or seen[i][0] / area < 0.95:
            continue
        cx, cy = pts.mean(axis=0)
        dist = {"top": cy / h, "bottom": 1 - cy / h, "left": cx / w, "right": 1 - cx / w}[side]
        if dist > float(cfg["max_depth"]):
            continue
        cand.append((i, np.exp(-3.0 * dist)))
    if not cand:
        return None, None
    wts = np.array([c[1] for c in cand])
    i = cand[int(rng.choice(len(cand), p=wts / wts.sum()))][0]
    pts = labs[i][1]
    t = float(rng.uniform(*cfg["visible"]))
    if side in ("top", "bottom"):
        lo, hi, anchor = pts[:, 1].min(), pts[:, 1].max(), pts[:, 0].mean()
    else:
        lo, hi, anchor = pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].mean()
    edge = hi - t * (hi - lo) if side in ("top", "left") else lo + t * (hi - lo)
    tilt = float(rng.uniform(-cfg["tilt"], cfg["tilt"]))
    bend = float(rng.uniform(-cfg["bend"], cfg["bend"]))
    if side in ("left", "right"):
        bend *= h / float(w)  # 세로 곡선은 짧은 축이라 덜 휜다
    mask = _cut_mask(shape, side, anchor, edge, tilt, bend)
    return mask, {"side": side, "target": i, "visible_target": round(t, 3), "tilt": round(tilt, 3), "bend": round(bend, 1)}


def cut(img, lines, cfg=None, rng=None, force=False):
    import cv2

    c = dict(DEFAULTS)
    c.update(cfg or {})
    rng = rng if rng is not None else np.random.default_rng()
    if not force and rng.random() >= float(c["p"]):
        return img, list(lines), None
    h, w = img.shape[:2]
    labs = _parse(lines, w, h)
    lit = ~(img <= 3).all(axis=2)  # 원래 찍힌 곳 (시야 안)
    polys = [_poly_mask((h, w), pts) for _, pts in labs]
    seen = [(int((p & lit).sum()), int(p.sum())) for p in polys]
    mask, info = _one_cut((h, w), labs, seen, c, rng)
    if mask is None:
        return img, list(lines), None
    infos = [info]
    if rng.random() < float(c["second"]):
        m2, i2 = _one_cut((h, w), labs, seen, c, rng, exclude_side=info["side"])
        if m2 is not None:
            mask |= m2
            infos.append(i2)
    out = img.copy()
    out[mask] = 0
    keep, dropped, blacked = [], 0, 0
    vis_lit = lit & ~mask
    valid = [l for l in lines if len(l.split()) == 9]
    for k, (p, line) in enumerate(zip(polys, valid)):
        area = int(p.sum())
        frac = float((p & vis_lit).sum()) / area if area else 0.0
        before = seen[k][0] / area if area else 0.0
        if frac >= float(c["keep_min"]) or (frac >= before - 1e-6 and frac > 0):  # 자르기로 줄지 않은 것(원래 일부만 보이던 것)은 원래 규칙대로 둔다
            keep.append(line.strip())
            continue
        dropped += 1
        if frac > 0:  # 조금만 남은 것은 마저 검게 — 보이는데 라벨이 없는 상자를 만들지 않는다
            grow = cv2.dilate(p.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            out[grow] = 0
            blacked += 1
    for inf in infos:
        inf.pop("target", None)
    return out, keep, {"cuts": infos, "dropped": dropped, "blacked": blacked}
