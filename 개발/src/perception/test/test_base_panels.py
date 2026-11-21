"""거점 벽 그림 읽기 (base_panels): 줄마다 어느 거점(작전지휘소 · 정비소 · 군수보급소)인지. ROS 없이 돈다.

1) 그려 만든 화면 — 기준 그림 석 장을 벽 자리에 아는 순서로 그려 넣고 그 순서가 돌아오는지 본다. 대회 사진이 필요 없다.
   그리는 식은 이 파일에 따로 적었다 (base_panels 의 식을 가져다 쓰지 않는다 → 좌우 · 위아래를 거꾸로 읽으면 여기서 걸린다).
   로봇이 여섯 거점 모두에 서 있는 화면(출발할 때) · 벽이 모델과 다른 자리에 있는 화면 · 그림이 없는 화면 · 벽이 안 찍힌 화면도 그려서 본다.
2) 저장소의 연습 화면 두 장 (test/data/drone_test1 · 2.jpg) — map_maker.compute 를 끝까지 돌린다.
   연습 경기장 두 곳은 두 진영 모두 위부터 정비소 · 작전지휘소 · 군수보급소다 (2026-11-18 · 19 드론 화면에서 눈으로 확인).
   그 한 장을 180도 돌려 「다른 배치」(위부터 군수보급소 · 작전지휘소 · 정비소)도 읽어 본다.
3) 이 PC 에만 있는 연습 화면들 — 없으면 건너뛴다. 환경변수 BASE_PANEL_PICTURES (경로 무늬, os.pathsep 으로 구분)로 더 줄 수 있다.
   연습 경기장의 벽 그림을 누가 바꿔 달면 이 시험은 틀린다 (그때는 PRACTICE 를 고친다).
4) 경기장의 좌우가 화면에서 잘린 경우 (본선 카메라, 2026-11-19 주최측 안내) — 저장소의 연습 화면 두 장의 좌우를 메모리에서 칠하거나 잘라 내 만든다.
   벽이 안 보이는 진영은 모름, 보이는 진영은 그대로 읽고, 틀린 배치를 내지 않아야 한다.
"""
import glob
import json
import itertools
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from perception import base_panels, drone_field_map, map_maker  # noqa: E402
from perception.base_panels import KINDS  # noqa: E402
from perception.drone_boundary import Boundary  # noqa: E402

DATA = os.path.join(HERE, "data")
COMMAND, MAINT, SUPPLY = KINDS
PRACTICE = [MAINT, COMMAND, SUPPLY]  # 연습 경기장 1 · 2: 1번 줄 정비소, 2번 줄 작전지휘소, 3번 줄 군수보급소 (두 진영 같다)
SLOTS = ["B-1", "B-2", "B-3", "R-1", "R-2", "R-3"]
LOCAL = [
    "/tmp/claude-1005/-home-shnoh-military-ai-contest/f287eec4-6d05-405c-9dec-68be366f2941/scratchpad/venue/now_test1.jpg",
    "/tmp/claude-1005/-home-shnoh-military-ai-contest/f287eec4-6d05-405c-9dec-68be366f2941/scratchpad/venue/now_test2.jpg",
    os.path.expanduser("~/.config/maicon-match/sessions/2026111*/perception/drone_used.jpg"),
]

W, H = 6.12, 5.00  # 경기장 (m)
SIZE = (1280, 720)  # 그려 만든 화면 (가로, 세로)
PPM = 110.0  # 그 화면의 바닥 축척 (px / m) — 연습 카메라와 비슷하다


def flat_boundary(ppm=PPM, size=SIZE):
    """렌즈 왜곡이 없는 카메라의 경계: 경기장 가운데가 화면 가운데, x 는 오른쪽, y 는 위."""
    cx, cy = size[0] / 2.0, size[1] / 2.0
    hom = np.array([[1.0 / ppm, 0.0, W / 2.0 - cx / ppm], [0.0, -1.0 / ppm, H / 2.0 + cy / ppm], [0.0, 0.0, 1.0]])
    return Boundary(True, k=0.0, hom=hom, image_size=size, strip=0.88, cm_half=0.60)


def to_px(x, y, ppm=PPM, size=SIZE):
    return size[0] / 2.0 + (x - W / 2.0) * ppm, size[1] / 2.0 - (y - H / 2.0) * ppm


def scene(orders, s=1.18, inset=0.0, hide=(), seed=0, robots=0.0):
    """벽 그림을 그려 넣은 화면 (BGR).

    orders: {"BLUE": [1번 줄 종류, 2번 줄, 3번 줄], "RED": [...]}. s: 벽 위쪽 끝이 화면 가운데에서 밀려 보이는 배율 (카메라 높이 Hc 면 Hc / (Hc − 0.4)).
    inset: 실제 벽이 경계 모델의 벽 자리보다 안쪽에 있는 거리 (m). hide: 그리지 않을 (진영, 줄).
    robots: 0 보다 크면 거점마다 로봇 크기(가로 0.25 m)의 어두운 덩어리를 그린다 — 거점 바닥에서 벽 그림 높이의 이 비율까지 가린다 (자리는 거점마다 다르게).
    """
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(np.clip(rng.normal(95, 14, (SIZE[1], SIZE[0])), 0, 255).astype(np.uint8), (0, 0), 1.5)  # 무늬 없는 바닥 + 잡음
    for camp, order in orders.items():
        for row, kind in zip((1, 2, 3), order):
            if (camp, row) in hide:
                continue
            mid = H * (3.5 - row) / 3.0  # 1번 줄이 위
            wall = inset if camp == "BLUE" else W - inset
            # 경기장 안에서 그 벽을 보고 선 사람의 왼쪽 · 오른쪽: BLUE 벽을 보면 왼쪽이 y 가 작은 쪽, RED 벽을 보면 왼쪽이 y 가 큰 쪽
            left, right = (mid - 0.6, mid + 0.6) if camp == "BLUE" else (mid + 0.6, mid - 0.6)
            floor = [(wall, left), (wall, right)]
            top = [(W / 2.0 + (x - W / 2.0) * s, H / 2.0 + (y - H / 2.0) * s) for x, y in floor]
            quad = np.float32([to_px(*top[0]), to_px(*top[1]), to_px(*floor[1]), to_px(*floor[0])])  # 그림의 왼쪽 위 · 오른쪽 위 · 오른쪽 아래 · 왼쪽 아래
            picture = base_panels.reference(kind)
            h, w = picture.shape
            warp = cv2.getPerspectiveTransform(np.float32([(0, 0), (w, 0), (w, h), (0, h)]), quad)
            drawn = cv2.warpPerspective(picture, warp, SIZE, flags=cv2.INTER_LINEAR)
            where = cv2.warpPerspective(np.full_like(picture, 255), warp, SIZE, flags=cv2.INTER_LINEAR) > 127
            img[where] = drawn[where]
            if robots > 0:
                at = float(rng.uniform(0.2, 0.8))  # 그림 가로의 어디쯤에 서 있는가
                edge = lambda t, up: tuple(np.add(floor[0], np.multiply(np.subtract(floor[1], floor[0]), t)) + np.multiply(np.subtract(top[0], floor[0]), up))  # noqa: E731
                body = [to_px(*edge(at - 0.105, robots)), to_px(*edge(at + 0.105, robots)), to_px(*edge(at + 0.105, -0.7)), to_px(*edge(at - 0.105, -0.7))]
                cv2.fillConvexPoly(img, np.int32(body), 28)
                cv2.circle(img, tuple(int(v) for v in to_px(*edge(at, robots - 0.3))), 4, 235, -1)  # 밝은 점 (수광 돔)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


class DrawnSceneTests(unittest.TestCase):
    """그려 만든 화면 — 대회 사진 없이 도는 시험."""

    def test_every_order_comes_back(self):
        boundary = flat_boundary()
        for order in itertools.permutations(KINDS):
            orders = {"BLUE": list(order), "RED": list(order[::-1])}
            kinds, detail = base_panels.identify(scene(orders), boundary, scale=1.18)
            self.assertEqual(kinds, orders, detail)
            for camp in ("BLUE", "RED"):
                self.assertTrue(detail[camp]["accepted"])
                self.assertGreater(detail[camp]["gap"], 0.5, detail[camp])  # 그림이 다 보이면 차이가 크다

    def test_wall_need_not_be_where_the_model_says(self):
        """경계 모델의 벽 자리는 몇 cm ~ 20 cm 틀린다 (연습 화면에서 잰 값). 벽이 15 cm 안쪽에 있고 벽이 모델보다 낮게 보여도 읽는다."""
        orders = {"BLUE": [SUPPLY, COMMAND, MAINT], "RED": [COMMAND, SUPPLY, MAINT]}
        boundary = flat_boundary()
        for inset, seen_scale in ((0.15, 1.12), (0.08, 1.14), (0.0, 1.22)):
            kinds, detail = base_panels.identify(scene(orders, s=seen_scale, inset=inset), boundary, scale=1.18)
            self.assertEqual(kinds, orders, (inset, seen_scale, detail))

    def test_camera_height_may_be_missing(self):
        orders = {"BLUE": [MAINT, SUPPLY, COMMAND], "RED": [SUPPLY, MAINT, COMMAND]}
        kinds, _detail = base_panels.identify(scene(orders), flat_boundary())  # 배율을 주지 않으면 기본 카메라 높이 2.65 m
        self.assertEqual(kinds, orders)

    def test_no_pictures_on_the_wall_gives_none(self):
        orders = {"BLUE": list(KINDS), "RED": list(KINDS)}
        everything = [(camp, row) for camp in orders for row in (1, 2, 3)]
        kinds, detail = base_panels.identify(scene(orders, hide=everything), flat_boundary(), scale=1.18)
        self.assertEqual(kinds, {"BLUE": None, "RED": None})
        for camp in ("BLUE", "RED"):
            self.assertFalse(detail[camp]["accepted"])
            self.assertLess(detail[camp]["gap"], base_panels.PanelParams().min_gap)
            self.assertTrue(detail[camp]["note"])

    def test_two_hidden_pictures_give_none_but_one_hidden_is_still_read(self):
        orders = {"BLUE": [SUPPLY, MAINT, COMMAND], "RED": [MAINT, COMMAND, SUPPLY]}
        boundary = flat_boundary()
        kinds, _detail = base_panels.identify(scene(orders, hide=[("RED", 1), ("RED", 3)]), boundary, scale=1.18)
        self.assertEqual(kinds, {"BLUE": orders["BLUE"], "RED": None})  # 두 장이 안 보이면 모름 — 다른 진영은 그대로 읽는다
        kinds, _detail = base_panels.identify(scene(orders, hide=[("RED", 2)]), boundary, scale=1.18)
        self.assertEqual(kinds, orders)  # 한 장이 안 보여도 나머지 두 장으로 정해진다

    def test_robots_standing_in_every_base_do_not_hide_the_answer(self):
        """출발할 때는 여섯 거점 모두에 로봇이 있다. 로봇은 카메라에서 볼 때 벽 그림의 아래쪽을 가린다 (연습 화면에서는 그림 높이의 10 % 남짓).
        그림 높이의 35 % 까지 가려도 읽고, 틀린 답은 어떤 경우에도 내지 않는다 (모르면 None)."""
        orders = {"BLUE": [COMMAND, SUPPLY, MAINT], "RED": [MAINT, SUPPLY, COMMAND]}
        boundary = flat_boundary()
        for seed in range(4):
            kinds, detail = base_panels.identify(scene(orders, robots=0.35, seed=seed), boundary, scale=1.18)
            self.assertEqual(kinds, orders, (seed, detail))
        for seed in range(4):  # 그림을 거의 다 가리면 모름이 될 수 있다 — 그래도 틀린 배치를 내지는 않는다
            kinds, detail = base_panels.identify(scene(orders, robots=0.9, seed=seed), boundary, scale=1.18)
            for camp in ("BLUE", "RED"):
                self.assertIn(kinds[camp], (None, orders[camp]), (seed, detail[camp]))

    def test_walls_outside_the_picture_give_none(self):
        """카메라가 벽을 찍지 않는 화면 (경기장이 화면보다 넓게 찍힘): 짐작하지 않고 모름."""
        orders = {"BLUE": list(KINDS), "RED": list(KINDS)}
        kinds, detail = base_panels.identify(scene(orders), flat_boundary(ppm=260.0), scale=1.18)
        self.assertEqual(kinds, {"BLUE": None, "RED": None})
        self.assertIn("화면 밖", detail["BLUE"]["note"])

    def test_wall_seen_almost_from_above_gives_none(self):
        """카메라가 아주 높아 벽 그림이 화면에서 몇 px 두께로만 보이면 모름."""
        orders = {"BLUE": list(KINDS), "RED": list(KINDS)}
        kinds, detail = base_panels.identify(scene(orders, s=1.03), flat_boundary(), scale=1.03)
        self.assertEqual(kinds, {"BLUE": None, "RED": None})
        self.assertIn("너무 작음", detail["RED"]["note"])

    def test_missing_boundary_or_picture_gives_none(self):
        orders = {"BLUE": list(KINDS), "RED": list(KINDS)}
        for bgr, boundary in ((scene(orders), None), (scene(orders), Boundary(False)), (None, flat_boundary())):
            kinds, detail = base_panels.identify(bgr, boundary)
            self.assertEqual(kinds, {"BLUE": None, "RED": None})
            self.assertTrue(detail["BLUE"]["note"])

    def test_one_call_is_quick(self):
        orders = {"BLUE": list(KINDS), "RED": list(KINDS)}
        img, boundary = scene(orders), flat_boundary()
        base_panels.identify(img, boundary, scale=1.18)  # 첫 번은 기준 그림을 푼다
        took = min(base_panels.identify(img, boundary, scale=1.18)[1]["seconds"] for _ in range(3))
        self.assertLess(took, 0.3)  # 이 PC 에서 0.05초쯤. 로봇은 4 ~ 5배 느리다 (경기마다 한 번)


class MapRegionTests(unittest.TestCase):
    """알아낸 것을 지도 dict 에 넣는 법: 자리 이름 여섯 개는 그대로 두고 그 뒤에 보고값 이름을 덧붙인다."""

    def setUp(self):
        self.data = drone_field_map.bare_field_map(None, "slot")
        self.before = [dict(b) for b in self.data["bases"]]

    def test_recognised_camp_is_appended_after_the_slot_names(self):
        base_panels.add_to_map(self.data, {"BLUE": None, "RED": [MAINT, COMMAND, SUPPLY]})
        bases = self.data["bases"]
        self.assertEqual(bases[:6], self.before)  # navigation 이 출발 칸을 찾는 여섯 개는 순서도 값도 그대로
        self.assertEqual([b["name"] for b in bases[6:]], ["RED_MAINTENANCE", "RED_COMMAND_POST", "RED_SUPPLY_DEPOT"])
        self.assertEqual([b["row"] for b in bases[6:]], [1, 2, 3])
        by_slot = {b["name"]: b for b in self.before}
        for extra, slot in zip(bases[6:], ("R-1", "R-2", "R-3")):  # 네모는 그 줄 자리 이름의 것과 같다
            self.assertEqual([extra[k] for k in ("min_x", "min_y", "max_x", "max_y")], [by_slot[slot][k] for k in ("min_x", "min_y", "max_x", "max_y")])
        base_panels.add_to_map(self.data, {"BLUE": None, "RED": [MAINT, COMMAND, SUPPLY]})
        self.assertEqual(len(self.data["bases"]), 9)  # 두 번 불러도 겹쳐 넣지 않는다

    def test_nothing_is_appended_when_nothing_is_known(self):
        base_panels.add_to_map(self.data, {"BLUE": None, "RED": None})
        base_panels.add_to_map(self.data, {"BLUE": [MAINT, MAINT, SUPPLY]})  # 세 종류가 하나씩이 아니면 넣지 않는다
        self.assertEqual(self.data["bases"], self.before)

    def test_annotate_never_raises(self):
        note = base_panels.annotate(self.data, np.zeros((10, 10, 7)), flat_boundary())  # 사진이 아닌 것 — 안에서 오류가 나도 밖으로 내지 않는다
        self.assertEqual(note["kinds"], {"BLUE": None, "RED": None})
        self.assertIn("오류", note["note"])
        note = base_panels.annotate(self.data, "사진이 아님", flat_boundary())
        self.assertEqual(note["kinds"], {"BLUE": None, "RED": None})
        note = base_panels.annotate(self.data, scene({"BLUE": list(KINDS), "RED": list(KINDS)}), Boundary(False))
        self.assertEqual(note["kinds"], {"BLUE": None, "RED": None})
        self.assertEqual(self.data["bases"], self.before)

    def test_annotate_fills_the_map_and_the_log(self):
        orders = {"BLUE": [SUPPLY, COMMAND, MAINT], "RED": [COMMAND, MAINT, SUPPLY]}
        note = base_panels.annotate(self.data, scene(orders), flat_boundary())
        self.assertEqual(note["kinds"], orders)
        self.assertEqual([b["name"] for b in self.data["bases"][6:]],
                         ["BLUE_SUPPLY_DEPOT", "BLUE_COMMAND_POST", "BLUE_MAINTENANCE", "RED_COMMAND_POST", "RED_MAINTENANCE", "RED_SUPPLY_DEPOT"])
        import json

        json.dumps(note)  # 기록(map.jsonl) · 계산 프로세스의 답(JSON)에 그대로 들어갈 수 있어야 한다


def check_practice_picture(case, path, cache=None):
    """연습 경기장 화면 한 장을 map_maker.compute 로 끝까지 돌려, 두 진영 모두 PRACTICE 순서로 읽고 지도에 덧붙였는지 본다."""
    with open(path, "rb") as fh:
        data, info = map_maker.compute(fh.read(), boundary_cache=cache)
    panels = info.get("base_panels")
    case.assertIsNotNone(data, (path, info))
    case.assertEqual(panels["kinds"], {"BLUE": PRACTICE, "RED": PRACTICE}, (path, panels))
    case.assertEqual([b["name"] for b in data["bases"][:6]], SLOTS, path)
    want = ["%s_%s" % (camp, kind) for camp in ("BLUE", "RED") for kind in PRACTICE]
    case.assertEqual([b["name"] for b in data["bases"][6:]], want, path)
    case.assertEqual([b["row"] for b in data["bases"][6:]], [1, 2, 3, 1, 2, 3], path)
    return panels


class RepoPictureTests(unittest.TestCase):
    """저장소에 든 연습 화면 두 장 (카메라 test1 · test2, 2026-11-18)."""

    def test_both_practice_cameras_read_maintenance_command_supply(self):
        for name in ("drone_test1.jpg", "drone_test2.jpg"):
            panels = check_practice_picture(self, os.path.join(DATA, name))
            for camp in ("BLUE", "RED"):
                self.assertGreater(panels[camp]["gap"], 0.4, (name, panels[camp]))


class TurnedPictureTests(unittest.TestCase):
    """실제 화면으로 「다른 배치」 를 읽어 본다. 가진 연습 화면은 모두 같은 배치(정비소 · 작전지휘소 · 군수보급소)뿐이다.
    화면을 180도 돌리면 BLUE 벽 자리에 RED 벽의 그림이 (경기장 안에서 보아 똑바로 선 채) 오고 줄 순서가 거꾸로다 →
    두 진영 모두 위부터 군수보급소 · 작전지휘소 · 정비소인 경기장의 화면이 된다. 경계는 다시 찾지 않고 같은 식으로 돌린다."""

    def test_turned_picture_reads_the_reversed_order(self):
        from perception.drone_boundary import Lens, detect_boundary, read_image
        from perception.drone_boxes import BoxParams, detect_boxes

        img = read_image(os.path.join(DATA, "drone_test2.jpg"))
        found = detect_boundary(img)
        self.assertTrue(found.ok, found.notes)
        view = detect_boxes(img, found, BoxParams())
        upright, _detail = base_panels.identify(img, found, view.scale, view.camera[:2])
        self.assertEqual(upright, {"BLUE": PRACTICE, "RED": PRACTICE})
        # 돌린 화면의 경계: 픽셀 (u, v) → (가로 − 1 − u, 세로 − 1 − v), 경기장 (x, y) → (W − x, H − y), 렌즈 가운데도 같이 돈다
        wide, high = found.image_size
        lens = found._lens()
        turn_px = np.array([[-1.0, 0.0, wide - 1.0], [0.0, -1.0, high - 1.0], [0.0, 0.0, 1.0]])
        turn_field = np.array([[-1.0, 0.0, found.field_w], [0.0, -1.0, found.field_h], [0.0, 0.0, 1.0]])
        turned = Boundary(True, k=found.k, hom=turn_field @ found.hom @ turn_px, image_size=found.image_size,
                          lens=Lens(lens.k1, lens.k2, lens.aspect, wide - 1.0 - lens.cx, high - 1.0 - lens.cy, lens.n),
                          strip=found.strip, cm_half=found.cm_half, field_w=found.field_w, field_h=found.field_h)
        kinds, detail = base_panels.identify(np.ascontiguousarray(img[::-1, ::-1]), turned, view.scale, view.camera[:2])
        self.assertEqual(kinds, {"BLUE": PRACTICE[::-1], "RED": PRACTICE[::-1]}, detail)
        for camp in ("BLUE", "RED"):
            self.assertGreater(detail[camp]["gap"], 0.4, detail[camp])


def painted(img, camp, cut, colour):
    """그 진영의 벽이 있는 쪽 화면 끝에서 cut px 를 한 가지 색으로 칠한 화면 (저장소의 두 화면은 BLUE 벽이 왼쪽, RED 벽이 오른쪽이다)."""
    out = img.copy()
    if camp == "BLUE":
        out[:, :cut] = colour
    else:
        out[:, img.shape[1] - cut:] = colour
    return out


def cropped(img, found, left, right):
    """화면의 왼쪽 left px · 오른쪽 right px 를 잘라 낸 화면과, 그 화면에 맞게 옮긴 경계 (렌즈 가운데와 사영을 같이 옮긴다)."""
    from perception.drone_boundary import Lens

    wide = img.shape[1] - left - right
    lens = found._lens()
    move = np.array([[1.0, 0.0, float(left)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # 잘린 화면의 픽셀 → 원래 화면의 픽셀
    moved = Boundary(True, k=found.k, hom=found.hom @ move, image_size=(wide, img.shape[0]),
                     lens=Lens(lens.k1, lens.k2, lens.aspect, lens.cx - left, lens.cy, lens.n),
                     strip=found.strip, cm_half=found.cm_half, field_w=found.field_w, field_h=found.field_h)
    return np.ascontiguousarray(img[:, left:img.shape[1] - right]), moved


class CutPictureTests(unittest.TestCase):
    """경기장의 좌우가 화면에서 잘린 경우 — 저장소의 연습 화면 두 장으로 메모리에서 만든다 (본선 카메라는 좌우가 잘릴 수 있다, 2026-11-19 주최측 안내).

    두 가지로 만든다.
      (가) 잘린 쪽을 한 가지 색으로 칠하고 경계는 그대로 둔다 — 벽은 찍히지 않았는데 벽 띠는 화면 안에 놓인 경우
      (나) 화면을 실제로 잘라 내고 경계를 그만큼 옮긴다 — 벽 띠가 화면 밖으로 나간 경우
    벽이 안 보이는 진영은 모름(None), 보이는 진영은 그대로 읽고, 어느 경우에도 틀린 배치나 예외를 내지 않아야 한다.
    이 두 화면에서 BLUE 벽 띠는 왼쪽 끝에서 190 ~ 330 px, RED 벽 띠는 오른쪽 끝에서 165 ~ 330 px 자리에 있다 (2026-11-19 잼).
    """

    HIDE = 360  # 화면 끝에서 이만큼(px) 가리면 그쪽 벽 띠가 모두 가려진다
    GREY, BLACK, WHITE = 128, 0, 255

    @classmethod
    def setUpClass(cls):
        from perception.drone_boundary import detect_boundary, read_image
        from perception.drone_boxes import BoxParams, detect_boxes

        cls.cases = []
        for name in ("drone_test1.jpg", "drone_test2.jpg"):
            img = read_image(os.path.join(DATA, name))
            found = detect_boundary(img)
            cls.cases.append((name, img, found, detect_boxes(img, found, BoxParams())))

    def read(self, picture, boundary, view):
        note = base_panels.annotate(None, picture, boundary, view)  # 예외가 나면 여기서 시험이 틀린다
        self.assertNotIn("오류", note.get("note", ""), note)
        return note["kinds"], note

    def test_these_pictures_have_blue_on_the_left(self):
        for name, _img, found, _view in self.cases:
            self.assertTrue(found.ok, (name, found.notes))
            left, right = found.field_to_pixel([(0.0, found.field_h / 2.0), (found.field_w, found.field_h / 2.0)])
            self.assertLess(left[0], right[0], name)

    def test_uncut_pictures_are_read_as_before(self):
        """자르지 않은 화면의 답과 차이(gap)는 「무늬 없는 자리」 검사를 켜도 꺼도 같다 (연습 화면의 벽 띠에는 그런 자리가 없다)."""
        off = base_panels.PanelParams(blank_window=0)
        for name, img, found, view in self.cases:
            kinds, detail = base_panels.identify(img, found, view.scale, view.camera[:2])
            before, detail_before = base_panels.identify(img, found, view.scale, view.camera[:2], off)
            self.assertEqual(kinds, {"BLUE": PRACTICE, "RED": PRACTICE}, (name, detail))
            self.assertEqual(kinds, before, name)
            for camp in ("BLUE", "RED"):
                self.assertEqual(detail[camp]["gap"], detail_before[camp]["gap"], (name, camp))
                self.assertEqual(detail[camp]["scores"], detail_before[camp]["scores"], (name, camp))

    def test_painted_out_wall_gives_none_and_the_other_camp_is_still_read(self):
        for name, img, found, view in self.cases:
            for colour in (self.GREY, self.BLACK, self.WHITE):
                kinds, note = self.read(painted(img, "BLUE", self.HIDE, colour), found, view)
                self.assertEqual(kinds, {"BLUE": None, "RED": PRACTICE}, (name, colour, note))
                self.assertIn("무늬가 없음", note["BLUE"]["note"])
                kinds, note = self.read(painted(img, "RED", self.HIDE, colour), found, view)
                self.assertEqual(kinds, {"BLUE": PRACTICE, "RED": None}, (name, colour, note))
                both = painted(painted(img, "BLUE", self.HIDE, colour), "RED", self.HIDE, colour)
                kinds, note = self.read(both, found, view)
                self.assertEqual(kinds, {"BLUE": None, "RED": None}, (name, colour, note))

    def test_cut_off_wall_gives_none_and_the_other_camp_is_still_read(self):
        for name, img, found, view in self.cases:
            kinds, note = self.read(*cropped(img, found, self.HIDE, 0), view)
            self.assertEqual(kinds, {"BLUE": None, "RED": PRACTICE}, (name, note))
            self.assertIn("화면 밖", note["BLUE"]["note"])
            kinds, note = self.read(*cropped(img, found, 0, self.HIDE), view)
            self.assertEqual(kinds, {"BLUE": PRACTICE, "RED": None}, (name, note))
            kinds, note = self.read(*cropped(img, found, self.HIDE, self.HIDE), view)
            self.assertEqual(kinds, {"BLUE": None, "RED": None}, (name, note))

    def test_cut_through_the_wall_never_gives_a_wrong_layout(self):
        """자르는 자리를 벽 띠를 지나가며 조금씩 옮긴다. 그 진영의 답은 맞는 배치이거나 모름이어야 하고, 반대쪽 진영은 늘 그대로 읽어야 한다.
        이 검사(blank_mask)가 없을 때는 회색 · 흰색으로 칠한 432번 중 9번이 틀린 배치를 냈다 (칠한 자리와 사진이 만나는 선이 윤곽으로 잡혀서)."""
        for name, img, found, view in self.cases:
            for camp, other in (("BLUE", "RED"), ("RED", "BLUE")):
                for colour in (self.GREY, self.WHITE):
                    answers = []
                    for cut in range(200, 335, 5):
                        kinds, note = self.read(painted(img, camp, cut, colour), found, view)
                        self.assertIn(kinds[camp], (None, PRACTICE), (name, camp, colour, cut, note[camp]))
                        self.assertEqual(kinds[other], PRACTICE, (name, camp, colour, cut))
                        answers.append(kinds[camp])
                    self.assertEqual(answers[0], PRACTICE, (name, camp, colour))  # 벽 띠에 닿기 전에는 읽는다
                    self.assertIsNone(answers[-1], (name, camp, colour))  # 벽 띠가 다 가려지면 모름
                for cut in range(150, 420, 30):
                    left, right = (cut, 0) if camp == "BLUE" else (0, cut)
                    kinds, note = self.read(*cropped(img, found, left, right), view)
                    self.assertIn(kinds[camp], (None, PRACTICE), (name, camp, cut, note[camp]))
                    self.assertEqual(kinds[other], PRACTICE, (name, camp, cut))

    def test_picture_of_one_colour_gives_none(self):
        name, img, found, view = self.cases[0]
        for colour in (self.GREY, self.BLACK, self.WHITE):
            kinds, note = self.read(np.full_like(img, colour), found, view)
            self.assertEqual(kinds, {"BLUE": None, "RED": None}, (name, colour, note))


def main_camera_picture(path):
    """경기 기록 폴더의 드론 사진이면, 그 판의 지도 기록(map.jsonl)에 적힌 카메라가 본선 카메라("uav")인가."""
    record = os.path.join(os.path.dirname(path), "map.jsonl")
    try:
        with open(record, encoding="utf-8") as fh:
            lines = [line for line in fh if line.strip()]
        return bool(lines) and json.loads(lines[-1]).get("camera") == "uav"
    except (OSError, ValueError):
        return False


class LocalPictureTests(unittest.TestCase):
    """이 PC 에만 있는 연습 화면 (오늘 받은 화면 · 경기 기록의 드론 사진). 없으면 건너뛴다."""

    def test_local_practice_pictures(self):
        patterns = LOCAL + [p for p in os.environ.get("BASE_PANEL_PICTURES", "").split(os.pathsep) if p]
        paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
        if not paths:
            self.skipTest("이 PC 에 연습 화면이 없음")
        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "drone_boundary.json")  # 같은 카메라의 다음 화면은 지난 경계에서 출발한다 (빠르게)
            for path in paths:
                if main_camera_picture(path):
                    continue  # 본선 카메라의 사진 (모의 경기 기록): 거점 벽이 화면 밖이고 배치도 다르다 — 연습 경기장의 답으로 견줄 수 없다
                with self.subTest(path=path):
                    check_practice_picture(self, path, cache)


if __name__ == "__main__":
    unittest.main()
