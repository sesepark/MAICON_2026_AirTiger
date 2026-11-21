"""조명 · 색 · 화질 변형 (기하는 건드리지 않는다 → 라벨을 그대로 쓴다).

드론(상공 카메라) 화면은 경기장 조명 · 시간 · 포그머신 · 카메라 자동 노출에 따라 밝기와 색이 크게 달라진다.
augment.py 는 학습 그림마다 이 변형을 무작위로 섞어 여러 장을 만들고, evaluate.py 는 정해진 값으로 걸어
「조명이 바뀌어도 찾는가」 를 잰다 (STRESS).

설정 (config/augment.yaml 의 ops) — 연산 이름: {p: 걸 확률, 값...}. 값이 [a, b] 면 그 사이에서 고르고, 수 하나면 그 값.
  exposure        밝기 배율 (곱)              range: [0.5, 1.6]
  brightness      밝기 더하기 (0~255)         range: [-50, 50]
  contrast        대비 배율 (평균 기준)       range: [0.6, 1.4]
  gamma           감마                        range: [0.6, 1.6]
  white_balance   채널(B, G, R)별 배율        range: [0.85, 1.15]  — 색온도 · 조명 색
  hsv             색상 · 채도 · 명도 흔들기    hue: 8 (도, OpenCV 0~180 단위의 ±) · sat: 0.35 · val: 0.25 (배율 ±)
  gradient_light  한쪽이 어두운 선형 조명     strength: [0.2, 0.6]  (어두운 쪽이 1 − strength 배)
  spotlight       둥근 밝은(어두운) 자리      strength: [-0.4, 0.6] · radius: [0.15, 0.5] (그림 대각선 비)
  shadow          흐릿한 다각형 그림자        strength: [0.2, 0.55]
  glare           밝은 반사 얼룩              strength: [0.3, 0.9] · count: [1, 3] · radius: [0.02, 0.08]
  fog             포그머신 연막 (회백색 섞기)  strength: [0.1, 0.45] · gray: [180, 235]
  blur            가우스 흐림                 ksize: [3, 7] (홀수로 맞춘다)
  noise           가우스 잡음                 sigma: [2, 10]
  jpeg            JPEG 다시 압축              quality: [35, 90]
적용 순서는 ORDER (조명 → 노출 · 색 → 연막 → 흐림 → 잡음 → 압축).

numpy · OpenCV 만 쓴다.
"""
from __future__ import annotations

import math

import numpy as np

ORDER = ("gradient_light", "spotlight", "shadow", "glare", "exposure", "brightness", "contrast", "gamma", "white_balance", "hsv",
         "fog", "blur", "noise", "jpeg")


def pick(rng, v):
    """[a, b] → a ~ b 사이 균일, 수 → 그대로."""
    if isinstance(v, (list, tuple)):
        if len(v) != 2:
            raise ValueError("범위는 [최소, 최대]: %r" % (v,))
        return float(rng.uniform(float(v[0]), float(v[1])))
    return float(v)


def _pick_int(rng, v):
    if isinstance(v, (list, tuple)):
        return int(rng.integers(int(v[0]), int(v[1]) + 1))
    return int(v)


def _coords(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return xx / max(w - 1, 1), yy / max(h - 1, 1)


# ------------------------------------------------------------------ 연산 (img: float32 HxWx3, 0~255 근처. BGR)
def op_exposure(img, rng, c):
    return img * pick(rng, c.get("range", [0.5, 1.6]))


def op_brightness(img, rng, c):
    return img + pick(rng, c.get("range", [-50, 50]))


def op_contrast(img, rng, c):
    m = float(img.mean())
    return (img - m) * pick(rng, c.get("range", [0.6, 1.4])) + m


def op_gamma(img, rng, c):
    g = max(pick(rng, c.get("range", [0.6, 1.6])), 1e-3)
    return 255.0 * np.power(np.clip(img, 0, 255) / 255.0, g)


def op_white_balance(img, rng, c):
    if "gains" in c:  # 정해진 값 (평가용)
        gains = np.array([float(v) for v in c["gains"]], np.float32)
    else:
        r = c.get("range", [0.85, 1.15])
        gains = np.array([pick(rng, r) for _ in range(3)], np.float32)
    return img * gains[None, None, :]


def op_hsv(img, rng, c):
    import cv2

    u8 = np.clip(img, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    def sym(key, default):  # 수 하나 v 면 ±v 에서 고른다, [a, b] 면 그 사이
        v = c.get(key, default)
        return pick(rng, v) if isinstance(v, (list, tuple)) else pick(rng, [-float(v), float(v)])

    hue, sat, val = sym("hue", 8), sym("sat", 0.35), sym("val", 0.25)
    hsv[..., 0] = (hsv[..., 0] + hue) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + sat), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1.0 + val), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)


def op_gradient_light(img, rng, c):
    h, w = img.shape[:2]
    x, y = _coords(h, w)
    ang = math.radians(pick(rng, c.get("angle", [0, 360])))
    t = (x - 0.5) * math.cos(ang) + (y - 0.5) * math.sin(ang)
    t = (t - t.min()) / max(float(t.max() - t.min()), 1e-6)  # 0 (밝은 쪽) ~ 1 (어두운 쪽)
    s = pick(rng, c.get("strength", [0.2, 0.6]))
    return img * (1.0 - s * t)[..., None]


def op_spotlight(img, rng, c):
    h, w = img.shape[:2]
    x, y = _coords(h, w)
    cx, cy = pick(rng, c.get("cx", [0.1, 0.9])), pick(rng, c.get("cy", [0.1, 0.9]))
    diag = math.hypot(w, h)
    r = pick(rng, c.get("radius", [0.15, 0.5])) * diag
    d2 = ((x - cx) * w) ** 2 + ((y - cy) * h) ** 2
    s = pick(rng, c.get("strength", [-0.4, 0.6]))
    return img * (1.0 + s * np.exp(-d2 / (2.0 * r * r)))[..., None]


def op_shadow(img, rng, c):
    import cv2

    h, w = img.shape[:2]
    n = _pick_int(rng, c.get("vertices", [3, 6]))
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    rad = pick(rng, c.get("size", [0.15, 0.45])) * max(w, h)
    angs = np.sort(rng.uniform(0, 2 * math.pi, n))
    pts = np.stack([cx + rad * rng.uniform(0.5, 1.0, n) * np.cos(angs), cy + rad * rng.uniform(0.5, 1.0, n) * np.sin(angs)], 1)
    mask = np.zeros((h, w), np.float32)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1.0)
    k = max(3, int(round(pick(rng, c.get("softness", [0.02, 0.08])) * max(w, h))) | 1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    s = pick(rng, c.get("strength", [0.2, 0.55]))
    return img * (1.0 - s * mask)[..., None]


def op_glare(img, rng, c):
    import cv2

    h, w = img.shape[:2]
    out = img.copy()
    for _ in range(_pick_int(rng, c.get("count", [1, 3]))):
        mask = np.zeros((h, w), np.float32)
        r = max(2.0, pick(rng, c.get("radius", [0.02, 0.08])) * max(w, h))
        center = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        axes = (int(r), int(r * rng.uniform(0.4, 1.0)))
        cv2.ellipse(mask, center, axes, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        k = max(3, int(r) | 1)
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        out = out + pick(rng, c.get("strength", [0.3, 0.9])) * 255.0 * mask[..., None]
    return out


def op_fog(img, rng, c):
    h, w = img.shape[:2]
    x, y = _coords(h, w)
    g = pick(rng, c.get("gray", [180, 235]))
    s = pick(rng, c.get("strength", [0.1, 0.45]))
    if c.get("uniform", False):
        a = np.full((h, w), s, np.float32)
    else:  # 연막은 한쪽(측면)에서 퍼진다 → 진하기가 한 방향으로 옅어진다
        ang = math.radians(pick(rng, c.get("angle", [0, 360])))
        t = (x - 0.5) * math.cos(ang) + (y - 0.5) * math.sin(ang)
        t = (t - t.min()) / max(float(t.max() - t.min()), 1e-6)
        a = s * (0.35 + 0.65 * (1.0 - t))
    a = a[..., None]
    return img * (1.0 - a) + g * a


def op_blur(img, rng, c):
    import cv2

    k = _pick_int(rng, c.get("ksize", [3, 7])) | 1
    return cv2.GaussianBlur(img, (k, k), 0)


def op_noise(img, rng, c):
    return img + rng.normal(0.0, pick(rng, c.get("sigma", [2, 10])), img.shape).astype(np.float32)


def op_jpeg(img, rng, c):
    import cv2

    q = _pick_int(rng, c.get("quality", [35, 90]))
    ok, buf = cv2.imencode(".jpg", np.clip(img, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32) if ok else img


OPS = {name: globals()["op_" + name] for name in ORDER}


def check_ops(ops):
    bad = [k for k in ops if k not in OPS]
    if bad:
        raise ValueError("모르는 변형: %s (있는 것: %s)" % (", ".join(bad), ", ".join(ORDER)))


def apply(bgr, ops, rng, force=False):
    """bgr: uint8 HxWx3. ops: {이름: 설정}. force=True 면 p 를 무시하고 다 건다 (평가용).
    돌려주는 값: (uint8 그림, 건 연산 이름 목록)."""
    check_ops(ops)
    # map 그림에서 카메라 시야 밖은 (0, 0, 0) 으로 채워져 있다 → 변형 뒤에도 검정으로 둔다 (연막 · 밝기가 그 자리를 칠하지 않게)
    outside = (bgr <= 3).all(axis=2)  # JPEG 로 저장돼 0 근처로 흔들린다
    img = bgr.astype(np.float32)
    used = []
    for name in ORDER:
        c = ops.get(name)
        if c is None:
            continue
        if not force and rng.random() >= float(c.get("p", 1.0)):
            continue
        img = OPS[name](img, rng, c)
        used.append(name)
    out = np.clip(img, 0, 255).astype(np.uint8)
    out[outside] = 0
    return out, used


# 평가용 정해진 조건 (evaluate.py --stress). 한 조건 = 연산 몇 개를 정해진 값으로 (seed 고정)
STRESS = {
    "어둡게 ×0.45": {"exposure": {"range": 0.45}},
    "밝게 ×1.6": {"exposure": {"range": 1.6}},
    "한쪽 어둡게": {"gradient_light": {"strength": 0.6, "angle": 0}},
    "그림자": {"shadow": {"strength": 0.5, "size": 0.4, "vertices": 5, "softness": 0.04}},
    "반사 얼룩": {"glare": {"strength": 0.8, "count": 3, "radius": 0.06}},
    "연막": {"fog": {"strength": 0.45, "gray": 215, "angle": 90}},
    "따뜻한 조명": {"white_balance": {"gains": [0.82, 1.0, 1.18]}},
    "차가운 조명": {"white_balance": {"gains": [1.18, 1.0, 0.85]}},
    "대비 낮음": {"contrast": {"range": 0.55}, "brightness": {"range": 25}},
    "흐림 · 잡음 · 압축": {"blur": {"ksize": 5}, "noise": {"sigma": 8}, "jpeg": {"quality": 40}},
}
