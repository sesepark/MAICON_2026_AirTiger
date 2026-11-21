"""탐지 결과를 ReadUgvs 응답의 UGV 목록으로 바꾸는 순수 계산 (ROS · 모델 없이 시험 가능).

규칙은 개발/src/supervisor/docs/요청사항.md 2절 8번과 UgvState.msg 를 따른다.
- 사진에 보이는 UGV 마다 하나 (아군도 넣는다).
- 못 읽은 값은 지어내지 않는다: 글자는 빈 문자열, 수는 -1.

읽는 방법 (2026-11-19)
- 위치: 모델의 ugv 박스.
- 진영(camp): 안테나에 끼운 색 덮개의 색. 빨강 = RED, 파랑 = BLUE.
- 호출부호(callsign): 색 덮개의 검은 띠 수. 덮개가 사진에서 40 픽셀 이상일 때만 센다.
  덮개를 따라 내려가며 색이 약해지는 골을 센다 (sleeve_profile · count_bands — 2026-11-20 에 실제 경기 사진으로 다시 만듦).
- 목숨(lives): 뒷면 LED 막대의 색 (파랑 4 · 초록 3 · 노랑 2 · 주황 1 · 빨강 0).
- 탄(ammo): 읽지 않는다 (-1). LED 가 사진에서 번져 켜진 개수를 셀 수 없다.
안테나와 LED 막대의 위치는 모델의 antenna · led_bar 박스로 찾는다. 모델에 그 클래스가 없거나 못 찾으면 그 값은 "못 읽음"이다.
"""
import numpy as np

UNREAD_TEXT = ''
UNREAD_NUMBER = -1

# 모델(학습 데이터)의 클래스 이름
_UGV_NAMES = ('ugv', 'robot', 'enemy ugv', 'friendly ugv', '로봇')
_ANTENNA_NAMES = ('antenna', '안테나')
_LED_NAMES = ('led bar', 'led')
_SENSOR_NAMES = ('ugv sensor', 'target sensor', 'sensor')

# 색 덮개가 이 길이(픽셀)보다 작게 찍히면 검은 띠가 1 픽셀 안팎이라 셀 수 없다.
# 실제 사진(2026-11-20): 40 이상이면 한 UGV 의 두 안테나가 같은 수를 낸 것이 110 / 118, 그보다 작으면 거의 못 읽는다
MIN_SLEEVE_PX_FOR_BANDS = 40
# LED 색 -> 남은 목숨 (UgvState.msg)
LIVES_BY_COLOR = {'blue': 4, 'green': 3, 'yellow': 2, 'orange': 1, 'red': 0}


def _norm(name):
    return str(name).strip().lower().replace('-', ' ').replace('_', ' ')


def is_ugv(class_name):
    return _norm(class_name) in _UGV_NAMES


def _iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _crop(rgb, box):
    """box = (x1, y1, x2, y2) 비율 -> 잘라낸 영상 (int16, 음수 계산용)."""
    h, w = rgb.shape[:2]
    x1, y1 = max(0, int(round(box[0] * w))), max(0, int(round(box[1] * h)))
    x2, y2 = min(w, int(round(box[2] * w))), min(h, int(round(box[3] * h)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return rgb[y1:y2, x1:x2].astype(np.int16)


def sleeve_masks(crop):
    """안테나 덮개의 빨강 · 파랑 화소. 하늘 그림처럼 옅은 색은 채도 조건으로 뺀다."""
    r, g, b = crop[:, :, 0], crop[:, :, 1], crop[:, :, 2]
    top = np.maximum(np.maximum(r, g), b)
    sat = (top - np.minimum(np.minimum(r, g), b)) / np.maximum(top, 1)
    red = (r > 90) & (r > 1.6 * g) & (r > 1.6 * b) & (sat > 0.55)
    blue = (b > 70) & (b > 1.4 * r) & (b > 1.15 * g) & (sat > 0.55)
    return red, blue


def read_antenna(crop):
    """안테나 한 개 -> (진영, 검은 띠 수). 못 읽으면 ('', -1)."""
    red, blue = sleeve_masks(crop)
    n_red, n_blue = int(red.sum()), int(blue.sum())
    need = max(6, int(0.02 * crop.shape[0] * crop.shape[1]))
    if max(n_red, n_blue) < need or min(n_red, n_blue) * 2 > max(n_red, n_blue):
        return UNREAD_TEXT, UNREAD_NUMBER          # 색이 너무 적거나 두 색이 비슷하게 섞임
    camp, mask = ('RED', red) if n_red > n_blue else ('BLUE', blue)
    profile = sleeve_profile(crop, camp, mask)
    if profile is None or len(profile) < MIN_SLEEVE_PX_FOR_BANDS:
        return camp, UNREAD_NUMBER
    bands = count_bands(profile)
    return camp, (bands if 1 <= bands <= 3 else UNREAD_NUMBER)


def sleeve_profile(crop, camp, mask):
    """덮개를 위에서 아래로 따라가며 줄마다 잰 색의 세기 (덮개 길이만큼의 배열). 덮개를 못 찾으면 None.

    검은 띠가 있는 줄은 색이 약하다. 안테나가 기울어 있어도 되도록 덮개의 가운데 선을 곧은 선으로 맞추고 그 둘레만 잰다.
    """
    h, w = mask.shape
    has = mask.sum(axis=1) >= 1
    # 덮개가 있는 세로 구간 = 색이 있는 줄들의 가장 긴 묶음. 띠 때문에 생기는 짧은 끊김은 잇고,
    # 멀리 떨어진 같은 색(안테나 위의 경광등 · 뒤의 그림)은 끊는다
    gap_ok = max(3, int(0.10 * h))
    best, start, last = None, None, None
    for y in range(h):
        if not has[y]:
            continue
        if start is None:
            start = y
        elif y - last - 1 > gap_ok:
            if best is None or last - start > best[1] - best[0]:
                best = (start, last)
            start = y
        last = y
    if start is not None and (best is None or last - start > best[1] - best[0]):
        best = (start, last)
    if best is None:
        return None
    top, bottom = best
    ys = np.arange(top, bottom + 1)
    counts = mask[top:bottom + 1].sum(axis=1)
    ok = counts > 0
    if int(ok.sum()) < 3:
        return None
    cx = (mask[top:bottom + 1] * np.arange(w)).sum(axis=1)[ok] / counts[ok]
    k, b0 = np.polyfit(ys[ok].astype(np.float32), cx, 1)
    resid = np.abs(cx - (k * ys[ok] + b0))
    keep = resid <= max(1.5, 2.0 * float(np.median(resid)))
    if int(keep.sum()) >= 3:
        k, b0 = np.polyfit(ys[ok][keep].astype(np.float32), cx[keep], 1)
    centers = k * ys + b0
    half = max(0.5, 0.30 * float(np.median(counts[ok])))
    r, g, b = [crop[:, :, i].astype(np.float32) for i in range(3)]
    strength = (r - np.maximum(g, b)) if camp == 'RED' else (b - np.maximum(r, 0.9 * g))
    profile = np.zeros(len(ys), np.float32)
    for i, y in enumerate(ys):
        x1 = int(max(0, np.floor(centers[i] - half)))
        x2 = int(min(w, np.ceil(centers[i] + half) + 1))
        if x2 > x1:
            profile[i] = float(strength[y, x1:x2].mean())
    return profile


def count_bands(profile, depth=0.45, top_frac=0.10, bottom_frac=0.68):
    """줄마다의 색 세기에서 골(검은 띠)의 수를 센다.

    골 = 둘레의 색 세기보다 depth 넘게 약해진 줄들의 토막. 본선 경기장 6판 + 연습 경기장 1판의 실제 사진으로 맞춘 값이다 (2026-11-20):
    - 띠는 덮개의 위쪽에 모여 있다 → 맨 위(top_frac 위)와 아래쪽(bottom_frac 아래: 고정 고리 · 전선이 가리는 곳)의 골은 세지 않는다.
    - 띠끼리는 붙어 있다 → 서로 가까운 골들의 가장 큰 묶음만 센다.
    """
    n = len(profile)
    p = np.maximum(np.asarray(profile, np.float32), 0.0)
    win = max(3, int(round(0.18 * n)))
    base = np.array([p[max(0, i - win):i + win + 1].max() for i in range(n)], np.float32)
    base = np.maximum(base, 0.6 * float(np.percentile(p, 80)))
    low = p / np.maximum(base, 1.0) < (1.0 - depth)
    lo, hi = max(1, int(round(top_frac * n))), int(round(bottom_frac * n))
    found, i = [], 0
    while i < n:
        if not low[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and low[j + 1]:
            j += 1
        if lo <= (i + j) / 2.0 <= hi and (j - i + 1) <= max(3, 0.16 * n):
            found.append((i, j))
        i = j + 1
    if len(found) > 1:
        groups, cur = [], [found[0]]
        for band in found[1:]:
            if band[0] - cur[-1][1] <= max(4, 0.20 * n):
                cur.append(band)
            else:
                groups.append(cur)
                cur = [band]
        groups.append(cur)
        found = max(groups, key=len)
    return len(found)


def find_sleeves(rgb, box, most=2):
    """UGV 박스 둘레에서 색 덮개처럼 생긴 것(진한 빨강 · 파랑의 길고 가는 세로 토막)을 찾는다 -> [(x1, y1, x2, y2) 비율].

    모델이 안테나를 못 찾았을 때만 쓴다. 본선 경기장 사진에서 가까운 UGV 의 안테나를 모델이 자주 놓쳤다 (특히 파랑 — 학습 사진 14장이 모두 빨강).
    경광등(둥글다) · LED 막대(가로로 길다) · 전선(너무 가늘다) · 뒤에 선 사람의 옷(너무 넓다)은 모양으로 거른다.
    """
    try:
        import cv2
    except ImportError:                                   # 영상 라이브러리가 없으면 모델의 안테나만 쓴다
        return []
    h, w = rgb.shape[:2]
    bw, bh = (box[2] - box[0]) * w, (box[3] - box[1]) * h
    x1, x2 = int(max(0, box[0] * w - 0.15 * bw)), int(min(w, box[2] * w + 0.15 * bw))
    y1, y2 = int(max(0, box[1] * h - 0.35 * bh)), int(min(h, box[1] * h + 0.75 * bh))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return []
    region = rgb[y1:y2, x1:x2].astype(np.int16)
    found = []
    close = np.ones((max(3, int(0.05 * bh)) | 1, 1), np.uint8)        # 검은 띠로 끊긴 덮개를 세로로 잇는다
    for mask in sleeve_masks(region):
        joined = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, close)
        count, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
        for i in range(1, count):
            sx, sy, sw, sh, area = [int(v) for v in stats[i]]
            # 덮개의 길이는 UGV 박스 높이의 1/4 쯤이다. 기운 안테나는 박스가 넓어지므로 폭은 넉넉히 본다
            if sh < max(12, 0.17 * bh) or sh < 2.2 * sw:              # 짧거나 가로로 퍼진 것
                continue
            if sw < max(3, 0.02 * bw) or sw > 0.25 * bw:              # 전선처럼 가늘거나 옷처럼 넓은 것
                continue
            if area < 0.35 * sw * sh:                                 # 휘었거나 듬성듬성한 것
                continue
            found.append((area, ((x1 + sx - 2) / float(w), (y1 + sy - 2) / float(h),
                                 (x1 + sx + sw + 2) / float(w), (y1 + sy + sh + 2) / float(h))))
    found.sort(reverse=True)
    return [b for _, b in found[:most]]


def find_led_bars(rgb, box, avoid=(), most=2):
    """UGV 박스 아래쪽에서 LED 막대처럼 생긴 것(하얗게 번진 가로로 긴 밝은 토막 + 둘레의 색 번짐)을 찾는다 -> [(x1, y1, x2, y2) 비율].

    모델이 LED 막대를 못 찾았을 때만 쓴다. 실제 사진(2026-11-20)에서 LED 막대는 차체 뒷면의 밝은 막대로 찍힌다.
    - 앞면의 흰 이름판: 밝지만 둘레에 색 번짐이 없다 → 번짐 조건으로 거른다.
    - 양옆의 수광 돔(흰 공, 초록빛이 돈다): 둥글다 → 가로로 긴 것만 받고, 모델이 찾은 수광부(avoid)와 겹치면 버린다.
    """
    try:
        import cv2
    except ImportError:
        return []
    h, w = rgb.shape[:2]
    bw, bh = (box[2] - box[0]) * w, (box[3] - box[1]) * h
    # 막대는 차체 아래쪽 가운데에 있다. 박스의 위 · 가장자리에는 뒤의 그림 · 노란 장애물 · 바닥 표시가 들어온다
    x1, x2 = int(max(0, box[0] * w + 0.08 * bw)), int(min(w, box[2] * w - 0.08 * bw))
    y1, y2 = int(max(0, box[1] * h)), int(min(h, box[3] * h - 0.03 * bh))
    low_from = 0.55 * bh                                  # 토막의 가운데가 박스의 아래 45 % 안에 있어야 한다
    # (위쪽까지 함께 보는 까닭: 수광 돔이 아래쪽에 걸치면 잘린 반쪽이 가로로 긴 토막처럼 보인다. 통째로 보면 둥글어서 걸러진다)
    if x2 - x1 < 12 or y2 - y1 < 8:
        return []
    region = rgb[y1:y2, x1:x2]
    top = region.max(axis=2).astype(np.int16)
    low = region.min(axis=2).astype(np.int16)
    # 막대의 가운데는 어떤 색이든 하얗게 번진다 (세 채널이 다 밝다). 노란 장애물 · 바닥 표시처럼 색이 진한 것은 여기서 빠진다
    bright = ((top >= 215) & (low >= 110)).astype(np.uint8)
    colored = ((top > 110) & ((top - low) > 0.30 * top)).astype(np.uint8)       # read_led_color 가 쓰는 화소와 같다
    count, _, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    found = []
    for i in range(1, count):
        sx, sy, sw, sh, area = [int(v) for v in stats[i]]
        if sw < max(8, 0.12 * bw) or sw > 0.8 * bw or sh < 3 or sw < 1.5 * sh:   # 작거나 · 차체만큼 넓거나 · 둥근 것
            continue
        if area < 0.5 * sw * sh or sy + sh / 2.0 < low_from:
            continue
        pad = max(2, int(0.5 * sh))
        ax1, ay1 = max(0, sx - pad), max(0, sy - pad)
        ax2, ay2 = min(region.shape[1], sx + sw + pad), min(region.shape[0], sy + sh + pad)
        ring = int(colored[ay1:ay2, ax1:ax2].sum()) - int(colored[sy:sy + sh, sx:sx + sw].sum())
        ring_area = (ax2 - ax1) * (ay2 - ay1) - sw * sh
        if ring_area <= 0 or ring < 0.15 * ring_area:                            # 둘레에 색 번짐이 없다 (이름판 · 바닥의 흰 선)
            continue
        # 막대는 검은 차체에 박혀 있다: 번짐 바깥(색이 없는 화소)은 어두워야 한다. 뒤의 그림 · 밝은 바닥 위의 것을 거른다
        wide = max(3, sh)
        bx1, by1 = max(0, sx - pad - wide), max(0, sy - pad - wide)
        bx2, by2 = min(region.shape[1], sx + sw + pad + wide), min(region.shape[0], sy + sh + pad + wide)
        around = top[by1:by2, bx1:bx2][colored[by1:by2, bx1:bx2] == 0]
        around = around[around < 215]
        if around.size < 6 or float(np.median(around)) > 90:
            continue
        cand = ((x1 + ax1) / float(w), (y1 + ay1) / float(h), (x1 + ax2) / float(w), (y1 + ay2) / float(h))
        if any(_iou(cand, a) > 0.1 for a in avoid):
            continue
        found.append((area, cand))
    found.sort(reverse=True)
    return [b for _, b in found[:most]]


def read_led_color(crop):
    """LED 막대 -> 'blue' | 'green' | 'yellow' | 'orange' | 'red' | ''.

    가운데는 하얗게 번지므로, 밝으면서 색이 남아 있는 화소(번짐의 가장자리)만 본다.
    """
    r, g, b = [crop[:, :, i].astype(np.float32) for i in range(3)]
    top = np.maximum(np.maximum(r, g), b)
    low = np.minimum(np.minimum(r, g), b)
    span = top - low
    use = (top > 110) & (span / np.maximum(top, 1) > 0.30)
    if int(use.sum()) < max(4, int(0.03 * use.size)):
        return ''
    r, g, b, top, span = r[use], g[use], b[use], top[use], np.maximum(span[use], 1)
    hue = np.where(top == r, ((g - b) / span) % 6, np.where(top == g, (b - r) / span + 2, (r - g) / span + 4)) * 60.0
    votes = {'red': int(((hue < 15) | (hue >= 330)).sum()), 'orange': int(((hue >= 15) & (hue < 40)).sum()),
             'yellow': int(((hue >= 40) & (hue < 70)).sum()), 'green': int(((hue >= 70) & (hue < 170)).sum()),
             'blue': int(((hue >= 170) & (hue < 270)).sum())}
    best = max(votes, key=votes.get)
    return best if votes[best] >= 0.5 * len(hue) else ''     # 색이 섞여 있으면 못 읽음


def read_state(rgb, box, antennas=(), leds=(), sensors=()):
    """UGV 한 대의 진영 · 호출부호 · 목숨 · 탄.

    rgb: HxWx3 uint8 (없으면 전부 못 읽음), box · antennas · leds · sensors: (x1, y1, x2, y2) 사진 크기에 대한 비율.
    sensors = 모델이 찾은 수광부. LED 막대를 색으로 찾을 때 수광 돔을 막대로 잘못 잡지 않게 하는 데만 쓴다.
    """
    state = {'camp': UNREAD_TEXT, 'callsign': UNREAD_TEXT, 'lives': UNREAD_NUMBER, 'ammo': UNREAD_NUMBER}
    if rgb is None:
        return state
    def read_all(boxes):
        got = []
        for a in boxes:
            crop = _crop(rgb, a)
            if crop is not None:
                camp, bands = read_antenna(crop)
                if camp:
                    got.append((camp, bands))
        return got

    # 색 덮개를 색 · 모양으로 직접 찾은 것을 먼저 쓴다: 덮개에 꼭 맞는 박스라 띠를 세기 좋다. 모델의 안테나 박스는 막대 전체를 감싸
    # 기운 안테나에서는 뒤의 것까지 들어온다 (2026-11-20 부품 모델 v2 로 견줌: 모델 박스 먼저면 callsign 이 덜 읽혔다). 못 찾았을 때만 모델 박스를 쓴다
    reads = read_all(find_sleeves(rgb, box))
    if not reads:
        reads = read_all(antennas)
    camps = {c for c, _ in reads}
    if len(camps) == 1:                               # 안테나들의 색이 서로 다르면 못 읽음
        state['camp'] = camps.pop()
        counts = {n for _, n in reads if n != UNREAD_NUMBER}
        if len(counts) == 1:                          # 띠 수가 안테나마다 다르면 못 읽음
            state['callsign'] = '%s-%d' % (state['camp'][0], counts.pop())
    def led_lives(boxes):
        for led in boxes:
            crop = _crop(rgb, led)
            color = read_led_color(crop) if crop is not None else ''
            if color:
                return LIVES_BY_COLOR[color]
        return UNREAD_NUMBER

    state['lives'] = led_lives(leds)
    if state['lives'] == UNREAD_NUMBER:               # 모델이 LED 막대를 못 찾았거나 색을 못 읽음 → 밝은 막대를 직접 찾는다
        state['lives'] = led_lives(find_led_bars(rgb, box, avoid=sensors))
    return state


def _inside(part, box, margin=0.15):
    """부품 박스의 가운데가 UGV 박스(조금 넓힌 것) 안에 있는가."""
    cx, cy = (part[0] + part[2]) / 2.0, (part[1] + part[3]) / 2.0
    mx, my = (box[2] - box[0]) * margin, (box[3] - box[1]) * margin
    return box[0] - mx <= cx <= box[2] + mx and box[1] - my <= cy <= box[3] + my


def build_ugvs(detections, image_width, image_height, rgb=None, min_confidence=0.35, same_ugv_iou=0.6,
               min_part_confidence=0.25):
    """탐지 목록 -> UGV 목록(dict), 왼쪽부터.

    detections: (class_name, confidence, x1, y1, x2, y2) 의 목록. 좌표는 픽셀.
    min_confidence: 이보다 낮은 탐지는 UGV 로 치지 않는다 (없는 UGV 를 만들어 내지 않기 위해).
    same_ugv_iou: 두 박스가 이보다 많이 겹치면 같은 UGV 로 보고 점수가 높은 하나만 남긴다.
    """
    w, h = float(image_width), float(image_height)

    def ratio(x1, y1, x2, y2):
        return (max(0.0, min(w, x1)) / w, max(0.0, min(h, y1)) / h, max(0.0, min(w, x2)) / w, max(0.0, min(h, y2)) / h)

    boxes, antennas, leds, sensors = [], [], [], []
    for name, conf, x1, y1, x2, y2 in detections:
        box = ratio(x1, y1, x2, y2)
        if box[2] - box[0] <= 0.0 or box[3] - box[1] <= 0.0:
            continue
        key = _norm(name)
        if key in _UGV_NAMES and conf >= min_confidence:
            boxes.append((float(conf), box))
        elif key in _ANTENNA_NAMES and conf >= min_part_confidence:
            antennas.append(box)
        elif key in _LED_NAMES and conf >= min_part_confidence:
            leds.append(box)
        elif key in _SENSOR_NAMES:                    # 낮은 점수도 받는다 (LED 막대를 색으로 찾을 때 피할 자리)
            sensors.append(box)

    kept = []
    for conf, box in sorted(boxes, key=lambda b: b[0], reverse=True):
        if all(_iou(box, other) <= same_ugv_iou for _, other in kept):
            kept.append((conf, box))

    found = []
    for conf, box in kept:
        item = read_state(rgb, box, [a for a in antennas if _inside(a, box)], [l for l in leds if _inside(l, box)], sensors)
        item.update({
            'x_center': (box[0] + box[2]) / 2.0,
            'y_center': (box[1] + box[3]) / 2.0,
            'width': box[2] - box[0],
            'height': box[3] - box[1],
            'confidence': max(0.0, min(1.0, conf)),
        })
        found.append(item)
    found.sort(key=lambda u: u['x_center'])
    return found
