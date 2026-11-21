"""드론 사진 한 장 → 지도 dict (map_maker.compute). 계산 모듈을 이어 붙인 것이 메시지 형식에 맞는지 본다. ROS 없이 돈다."""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from perception import map_maker  # noqa: E402

DATA = os.path.join(HERE, "data")


class ComputeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(DATA, "drone_test2.jpg"), "rb") as fh:
            cls.data, cls.info = map_maker.compute(fh.read(), stamp_s=1234.5)

    def test_good_frame_gives_a_trusted_map(self):
        self.assertTrue(self.info["ok"], self.info)
        self.assertGreaterEqual(self.info["boxes"], 12)
        self.assertEqual(self.data["header"], {"frame_id": "arena", "stamp": 1234.5})
        self.assertAlmostEqual(self.data["width"], 6.12)
        self.assertAlmostEqual(self.data["height"], 5.0)

    def test_bases_carry_call_sign_names_for_navigation(self):
        # 자리 이름 여섯 개가 먼저, 하나씩 (navigation 은 name 이 slot 인 칸이 꼭 하나여야 출발 칸으로 쓴다)
        self.assertEqual([b["name"] for b in self.data["bases"][:6]], ["B-1", "B-2", "B-3", "R-1", "R-2", "R-3"])
        self.assertEqual({b["name"]: b["row"] for b in self.data["bases"]}["R-3"], 3)

    def test_base_kinds_read_from_the_wall_pictures_come_after_them(self):
        """2026-11-19 공지(거점 종류의 배치는 바뀔 수 있다): 벽 그림으로 읽은 종류를 보고값 이름으로 뒤에 덧붙인다 (base_panels.py).
        이 화면(연습 경기장 2)은 두 진영 모두 위부터 정비소 · 작전지휘소 · 군수보급소다."""
        order = ["MAINTENANCE", "COMMAND_POST", "SUPPLY_DEPOT"]
        self.assertEqual(self.info["base_panels"]["kinds"], {"BLUE": order, "RED": order}, self.info["base_panels"])
        extra = self.data["bases"][6:]
        self.assertEqual([b["name"] for b in extra], ["%s_%s" % (camp, kind) for camp in ("BLUE", "RED") for kind in order])
        self.assertEqual([b["row"] for b in extra], [1, 2, 3, 1, 2, 3])
        by_name = {b["name"]: b for b in self.data["bases"]}
        for slot, value in (("B-1", "BLUE_MAINTENANCE"), ("R-2", "RED_COMMAND_POST"), ("R-3", "RED_SUPPLY_DEPOT")):
            self.assertEqual([by_name[value][k] for k in ("min_x", "min_y", "max_x", "max_y")], [by_name[slot][k] for k in ("min_x", "min_y", "max_x", "max_y")])
        json.dumps(self.info)  # 기록(map.jsonl) · 계산 프로세스의 답에 그대로 들어간다

    def test_sectors_and_faces_are_filled(self):
        self.assertEqual(sorted(s["name"] for s in self.data["sectors"]), ["B-S1", "B-S2", "B-S3", "CM", "R-S1", "R-S2", "R-S3"])
        cm = next(s for s in self.data["sectors"] if s["name"] == "CM")
        self.assertAlmostEqual(cm["max_x"] - cm["min_x"], 1.2, delta=0.06)  # 가운데 구역 폭 120 cm (포털의 값과 같아야 한다)
        self.assertGreater(len(self.data["faces"]), 3 * len(self.data["boxes"]) - 12)
        for box in self.data["boxes"]:
            self.assertTrue(0.0 < box["x"] < 6.12 and 0.0 < box["y"] < 5.0)

    def test_boundary_cache_is_written_and_reused(self):
        """저장 파일의 지난 경계는 코드에 실린 경계가 있는 카메라(연습 경기장)에서만 먼저 쓴다."""
        import tempfile

        with open(os.path.join(DATA, "drone_test2.jpg"), "rb") as fh:
            jpeg = fh.read()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "drone_boundary.json")
            data1, info1 = map_maker.compute(jpeg, 1.0, "slot", None, path, camera="uav:test:test2")
            self.assertEqual((info1["boundary_via"], info1["boundary_prior"]), ("prior", "built_in"))
            self.assertEqual(len(map_maker.load_boundaries(path)), 1)
            data2, info2 = map_maker.compute(jpeg, 1.0, "slot", None, path, camera="uav:test:test2")
            self.assertEqual((info2["boundary_via"], info2["boundary_prior"]), ("prior", "cache"))
            self.assertTrue(info2["ok"], info2)
            self.assertEqual(len(data1["boxes"]), len(data2["boxes"]))
            self.assertEqual(len(map_maker.load_boundaries(path)), 1)  # 같은 카메라는 하나만 남긴다
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{broken")
            self.assertEqual(map_maker.load_boundaries(path), [])
            _data3, info3 = map_maker.compute(jpeg, 1.0, "slot", None, path, camera="uav:test:test2")  # 깨진 파일은 무시한다
            self.assertEqual((info3["boundary_via"], info3["boundary_prior"]), ("prior", "built_in"))

    def test_unknown_camera_searches_first_even_with_a_saved_boundary(self):
        """코드에 실린 경계가 없는 카메라(본선 카메라)는 저장 파일이 있어도 처음부터 찾는다.

        2026-11-19 모의 경기 2차: 1차에서 저장한 경계로 다시 맞춘 지도가 상자 16개 가운데 11개만 냈고, 로봇은 지도에 없는 상자 앞에 5분을 갇혔다.
        """
        import tempfile

        with open(os.path.join(DATA, "drone_test2.jpg"), "rb") as fh:
            jpeg = fh.read()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "drone_boundary.json")
            data1, info1 = map_maker.compute(jpeg, 1.0, "slot", None, path)
            self.assertEqual(info1["boundary_via"], "search")
            self.assertEqual(len(map_maker.load_boundaries(path)), 1)  # 저장은 한다
            data2, info2 = map_maker.compute(jpeg, 1.0, "slot", None, path)
            self.assertEqual((info2["boundary_via"], info2["boundary_prior"]), ("search", None))  # 그래도 처음부터 찾는다
            self.assertEqual(data1, data2)


    def test_arena_size_only_moves_the_walls(self):
        """arena 를 주면 지도의 벽 크기만 바뀌고 상자는 경기장 정중앙 기준 자리를 지킨다 (기본은 전과 같은 6.12 x 5.00)."""
        with open(os.path.join(DATA, "drone_test2.jpg"), "rb") as fh:
            data, info = map_maker.compute(fh.read(), 1234.5, "slot", None, None, [6.0, 4.8])
        self.assertTrue(info["ok"], info)
        self.assertEqual((data["width"], data["height"]), (6.0, 4.8))
        self.assertEqual(len(data["boxes"]), len(self.data["boxes"]))
        for new, old in zip(data["boxes"], self.data["boxes"]):
            self.assertAlmostEqual(new["x"] - 3.0, old["x"] - 3.06, places=3)
            self.assertAlmostEqual(new["y"] - 2.4, old["y"] - 2.5, places=3)

    def test_info_tells_how_positions_were_made(self):
        self.assertEqual(len(self.info["divisor"]), 2)
        self.assertTrue(set(self.info["divisor_source"]) <= {"grid", "s"}, self.info)
        self.assertEqual(self.info["joined"], 0)  # 이 사진에는 사이 모서리가 안 보이는 줄이 없다

    def test_wrong_boundary_is_never_saved(self):
        """검사를 통과했다고 나와도 치수가 이상한 경계(구역선을 잘못 고른 것)는 저장하지 않는다 — 좋은 경계를 밀어내지 않게."""
        import tempfile

        import numpy as np

        from perception.drone_boundary import Boundary

        good = Boundary(True, hom=np.eye(3), image_size=(1280, 720), strip=0.82, cm_half=0.62, circle_r=0.63)
        seam = Boundary(True, hom=np.eye(3), image_size=(1280, 720), strip=0.81, cm_half=1.05, circle_r=0.63)  # 2026-11-19 에 저장돼 버린 것
        failed = Boundary(False, hom=np.eye(3), image_size=(1280, 720), strip=0.82, cm_half=0.62, circle_r=0.63)
        self.assertTrue(map_maker.boundary_trusted(good))
        self.assertFalse(map_maker.boundary_trusted(seam))
        self.assertFalse(map_maker.boundary_trusted(failed))
        self.assertFalse(map_maker.boundary_trusted(Boundary(False)))
        with open(os.path.join(DATA, "drone_test2.jpg"), "rb") as fh:
            jpeg = fh.read()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "drone_boundary.json")
            map_maker.compute(jpeg, 1.0, "slot", None, path)
            kept = map_maker.load_boundaries(path)
            self.assertEqual(len(kept), 1)
            real = map_maker.detect_boundary
            try:  # 같은 사진에서 틀린 경계가 나왔다고 치고 — 저장 파일은 그대로여야 한다
                def wrong(image, prior=None):
                    b = real(image, prior=prior)
                    b.cm_half = 1.05
                    return b
                map_maker.detect_boundary = wrong
                _data, info = map_maker.compute(jpeg, 1.0, "slot", None, path)
            finally:
                map_maker.detect_boundary = real
            self.assertFalse(info["ok"], info)
            self.assertEqual(map_maker.load_boundaries(path), kept)

    def test_built_in_boundary_of_the_camera_is_tried(self):
        """camera 를 주면 코드에 실린 그 카메라의 경계에서 출발한다 (저장 파일이 없어도 「처음부터 찾기」 를 하지 않는다)."""
        from perception import camera_boundaries

        self.assertEqual([camera_boundaries.camera_key(n) for n in ("uav", "uav:test:test1", "uav:test:test2", "test2", "main", "", None, "uav:test:other")],
                         ["main", "test1", "test2", "test2", "main", None, None, None])
        self.assertEqual(len(camera_boundaries.built_in("uav:test:test2")), 1)
        self.assertEqual(camera_boundaries.built_in(None), [])
        self.assertEqual(self.info["boundary_via"], "search")  # camera 를 주지 않으면 쓰지 않는다
        self.assertIsNone(self.info["boundary_prior"])
        with open(os.path.join(DATA, "drone_test2.jpg"), "rb") as fh:
            jpeg = fh.read()
        data, info = map_maker.compute(jpeg, 1234.5, "slot", None, None, None, "uav:test:test2")
        self.assertTrue(info["ok"], info)
        self.assertEqual((info["boundary_via"], info["boundary_prior"]), ("prior", "built_in"))
        self.assertEqual(len(data["boxes"]), len(self.data["boxes"]))
        for new, old in zip(data["boxes"], self.data["boxes"]):  # 같은 선 조합이면 같은 방법으로 맞추므로 답도 같다
            self.assertAlmostEqual(new["x"], old["x"], places=3)
            self.assertAlmostEqual(new["y"], old["y"], places=3)
        _data, other = map_maker.compute(jpeg, 1234.5, "slot", None, None, None, "uav:test:test1")  # 다른 카메라의 것은 맞지 않아 처음부터 찾는다
        self.assertTrue(other["ok"], other)
        self.assertEqual(other["boundary_via"], "search")

    def test_unreadable_bytes_give_no_map(self):
        data, info = map_maker.compute(b"not a picture")
        self.assertIsNone(data)
        self.assertFalse(info["ok"])


class BareMapTests(unittest.TestCase):
    """지도를 못 만들 때 내는 임시 지도 (벽 · 거점 · 구역만, 상자 0개)."""

    def test_bare_map_has_walls_bases_and_sectors_but_no_boxes(self):
        from perception import drone_field_map

        data = drone_field_map.bare_field_map(None, "slot")
        self.assertEqual((data["width"], data["height"], data["boxes"], data["faces"]), (6.12, 5.0, [], []))
        self.assertEqual([b["name"] for b in data["bases"]], ["B-1", "B-2", "B-3", "R-1", "R-2", "R-3"])
        self.assertEqual({s["name"] for s in data["sectors"]}, {"B-S1", "B-S2", "B-S3", "CM", "R-S1", "R-S2", "R-S3"})
        self.assertIsNone(data["header"]["stamp"])  # 시각 없음 → 나중에 오는 진짜 지도가 더 새것이 된다
        json.dumps(data)


class ArenaSizeTests(unittest.TestCase):
    """지도의 경기장 높이는 사진이 어느 카메라의 것인지로 고른다 (연습 경기장 4.80 · 본선 경기장 5.00)."""

    @staticmethod
    def maker(width=6.12, height=0.0):
        m = map_maker.MapMaker.__new__(map_maker.MapMaker)  # ROS 없이 고르는 부분만 본다
        m.arena_width, m.arena_height, m.height_practice, m.height_main = width, height, 4.80, 5.00
        m.height_from_lidar = True
        from perception import arena_height

        m.vote = arena_height.HeightVote(candidates=(4.80, 5.00))
        return m

    @staticmethod
    def frame(frame_id):
        from types import SimpleNamespace as N

        return N(image=N(header=N(frame_id=frame_id)))

    def test_practice_camera_gives_480_and_main_camera_gives_500(self):
        m = self.maker()
        self.assertEqual(m.arena_for(self.frame("uav:test:test2")), [6.12, 4.80])
        self.assertEqual(m.arena_for(self.frame("uav:test:")), [6.12, 4.80])  # 시험 포트인데 카메라 이름이 비어 있어도 연습 경기장이다
        self.assertEqual(m.arena_for(self.frame("uav")), [6.12, 5.00])
        self.assertEqual(m.arena_for(None), [6.12, 5.00])  # 사진을 못 받았으면 본 경기 것
        self.assertEqual(self.maker(height=4.9).arena_for(self.frame("uav")), [6.12, 4.9])  # 값을 주면 고정
        self.assertIsNone(self.maker(width=0.0).arena_for(self.frame("uav:test:test1")))  # 끔

    def test_a_clear_lidar_measurement_overrides_the_camera(self):
        m = self.maker()
        for value in (4.99, 5.01, 5.00, 4.98):  # 라이다가 5.00 을 쟀다 (팀 테스트 카메라인데도)
            m.vote.add([value])
        self.assertEqual(m.arena_for(self.frame("uav:test:test2")), [6.12, 5.00])
        m = self.maker()
        for value in (4.79, 4.81, 4.80):  # 본 경기 카메라인데 라이다가 4.80 을 쟀다
            m.vote.add([value])
        self.assertEqual(m.arena_for(self.frame("uav")), [6.12, 4.80])
        m = self.maker()
        for value in (4.90, 4.89, 4.91, 4.90):  # 두 값의 중간 → 정하지 않는다 (카메라로 고른 값)
            m.vote.add([value])
        self.assertEqual(m.arena_for(self.frame("uav")), [6.12, 5.00])
        m = self.maker()
        m.vote.add([4.80])  # 한 번만 쟀다 → 모자라다
        self.assertEqual(m.arena_for(self.frame("uav")), [6.12, 5.00])
        m = self.maker()
        m.height_from_lidar = False
        for value in (4.80, 4.80, 4.80):
            m.vote.add([value])
        self.assertEqual(m.arena_for(self.frame("uav")), [6.12, 5.00])  # 끄면 카메라로만

    def test_boxes_keep_their_distance_from_the_centre(self):
        with open(os.path.join(DATA, "drone_test2.jpg"), "rb") as fh:
            jpeg = fh.read()
        tall, _ = map_maker.compute(jpeg, stamp_s=1.0, arena=(6.12, 5.00))
        short, _ = map_maker.compute(jpeg, stamp_s=1.0, arena=(6.12, 4.80))
        self.assertEqual((tall["height"], short["height"]), (5.0, 4.8))
        for a, b in zip(tall["boxes"], short["boxes"]):
            self.assertAlmostEqual(a["x"], b["x"], places=6)
            self.assertAlmostEqual(a["y"] - 2.5, b["y"] - 2.4, places=6)
        rows = {r["name"]: (r["min_y"], r["max_y"]) for r in short["sectors"]}
        self.assertEqual(rows["B-S2"], (1.6, 3.2))  # 줄 경계는 높이를 셋으로 나눈 자리


class PickFramesTests(unittest.TestCase):
    """처음 지도는 사진 한 장으로 정하지 않는다: 시작 뒤의 사진 몇 장을 한꺼번에 계산해 상자가 가장 많이 나온 것을 쓴다.

    2026-11-19 모의 경기 6차(본선 경기장): 첫 사진 한 장으로 만든 지도에 상자가 16개 가운데 12개뿐이었다 (같은 판의 다음 사진은 16개).
    """

    @staticmethod
    def frame(n, frame_id="uav"):
        from types import SimpleNamespace as N

        return N(header=N(stamp=N(sec=100 + n, nanosec=0)), captured_at_utc="t%d" % n, image=N(header=N(frame_id=frame_id), data=b"jpeg%d" % n))

    @staticmethod
    def maker(pick_frames=3):
        import threading
        from types import SimpleNamespace as N

        from perception import arena_height

        m = map_maker.MapMaker.__new__(map_maker.MapMaker)
        m.lock, m.wake = threading.Lock(), threading.Event()
        m.pick_frames, m.pick_wait_s, m.timeout_s = pick_frames, 2.6, 5.0
        m.frame, m.frames, m.recent, m.first_frame_at = None, 0, [], None
        m.started, m.started_at, m.message, m.bare, m.tries = False, None, None, False, 1
        m.child, m.spare, m.use_child = None, [], False
        m.camera_height, m.camera_height_main, m.base_mirror = None, 2.55, True
        m.box_detector, m.box_model, m.box_conf, m.box_gpu = "mixed", "/nowhere/drone_boxes.engine", 0.30, False
        m.arena_width, m.arena_height, m.height_practice, m.height_main, m.height_from_lidar = 6.12, 0.0, 4.80, 4.80, False
        m.vote = arena_height.HeightVote(candidates=(4.80, 5.00))
        m.arena = None
        m.logged, m.saved, m.sent = [], [], []
        m.write_log = lambda **record: m.logged.append(record)
        m.save = lambda frame, data: m.saved.append((frame.captured_at_utc, len(data["boxes"])))
        m.publisher = N(publish=m.sent.append)
        m.node = N(get_logger=lambda: N(info=lambda *_: None, warning=lambda *_: None, error=lambda *_: None))
        m.spawn = lambda first=True: None
        return m

    @staticmethod
    def boxes(*pairs, singles=()):
        """붙은 쌍(왼쪽 상자의 가운데로 준다)과 홀로 선 상자로 지도 dict 를 만든다."""
        out = []
        for x, y in pairs:
            out += [{"x": x, "y": y}, {"x": x + 0.40, "y": y}]
        return {"width": 6.12, "height": 4.8, "boxes": out + [{"x": x, "y": y} for x, y in singles]}

    SIX = ((1.45, 3.9), (1.45, 2.4), (1.45, 0.8), (4.25, 3.95), (4.25, 2.36), (4.25, 0.8))  # 좌우의 쌍 여섯 (가운데 쌍 둘은 뺐다)

    def test_pick_best_takes_the_map_with_the_most_paired_boxes(self):
        ok, bad = {"ok": True}, {"ok": False}
        full = self.boxes(*self.SIX)
        merged = self.boxes(*self.SIX[:4], singles=[(4.45, 2.4), (4.45, 0.95)])  # 붙은 쌍 둘을 상자 하나씩으로 뭉쳐 본 지도 (모의 6차의 첫 사진)
        self.assertEqual(map_maker.map_score(full)[0], 12)
        self.assertEqual(map_maker.map_score(merged)[0], 8)
        self.assertEqual(map_maker.pick_best([(merged, ok), (full, ok), (merged, ok)]), 1)
        self.assertEqual(map_maker.pick_best([(full, ok), (full, ok)]), 0)  # 같으면 먼저 찍힌 것
        self.assertEqual(map_maker.pick_best([(None, bad), (merged, ok)]), 1)  # 지도가 안 나온 사진은 뺀다
        self.assertEqual(map_maker.pick_best([(full, bad), (merged, ok)]), 1)  # 검사를 통과하지 못한 것도 뺀다
        self.assertIsNone(map_maker.pick_best([(None, bad), (None, {})]))
        self.assertIsNone(map_maker.pick_best([]))

    def test_pick_best_avoids_a_lone_box_near_the_side_wall_but_keeps_lone_boxes_in_the_middle(self):
        ok = {"ok": True}
        full = self.boxes(*self.SIX)
        ghost = self.boxes(*self.SIX, singles=[(5.05, 3.25)])  # 거점 사이의 줄무늬 블록을 상자로 본 사진 (모의 3차의 둘째 사진: 17개)
        self.assertEqual(map_maker.map_score(ghost), (12, -1, 13))
        self.assertEqual(map_maker.pick_best([(ghost, ok), (full, ok)]), 1)  # 상자 수는 적어도 헛것이 없는 쪽
        self.assertEqual(map_maker.pick_best([(full, ok), (ghost, ok)]), 0)
        third = self.boxes(*self.SIX, singles=[(5.10, 3.75)])  # 진짜 쌍에서 0.48 m — 한 무리의 셋째 (모의 1차의 첫 사진)
        self.assertEqual(map_maker.map_score(third), (12, -1, 13))
        self.assertEqual(map_maker.pick_best([(third, ok), (full, ok)]), 1)
        wide = self.boxes(*self.SIX[:5], singles=[(4.29, 0.8), (4.81, 0.8)])  # 쌍의 사이가 0.52 m 로 그려진 지도도 짝으로 센다 (모의 2차)
        self.assertEqual(map_maker.map_score(wide), (12, 0, 12))
        centre = self.boxes(*self.SIX, singles=[(3.06, 3.4), (3.06, 1.5)])  # 연습 경기장: 가운데에 홀로 선 상자 둘
        one = self.boxes(*self.SIX, singles=[(3.06, 3.4)])
        self.assertEqual(map_maker.pick_best([(one, ok), (centre, ok)]), 1)  # 가운데의 홀로 선 상자는 많이 찾은 쪽

    def test_main_camera_gets_the_fixed_height_and_practice_cameras_are_measured(self):
        m = self.maker()
        self.assertEqual(m.camera_height_for(self.frame(1, "uav")), 2.55)
        self.assertEqual(m.camera_height_for(self.frame(1, "")), 2.55)  # 이름이 없으면 본 경기 것으로 본다 (arena_for 와 같은 규칙)
        self.assertIsNone(m.camera_height_for(self.frame(1, "uav:test:test1")))  # 연습 경기장은 사진에서 잰다 (전과 같다)
        m.camera_height_main = None
        self.assertIsNone(m.camera_height_for(self.frame(1, "uav")))  # 0 으로 끄면 본선 카메라도 사진에서 잰다
        m.camera_height = 2.3
        self.assertEqual(m.camera_height_for(self.frame(1, "uav:test:test1")), 2.3)  # map_camera_height 를 주면 어느 카메라든 그 값

    def test_frames_are_collected_only_after_the_start_and_without_repeats(self):
        from types import SimpleNamespace as N

        m = self.maker()
        m.on_frame(self.frame(1))
        self.assertEqual(m.recent, [])  # 시작 전의 사진은 모으지 않는다
        m.on_mission(N(phase="running"))
        self.assertEqual([f.captured_at_utc for _k, f in m.recent], ["t1"])  # 시작 전에 받아 둔 가장 새 사진이 첫 장
        self.assertIsNotNone(m.first_frame_at)
        m.on_frame(self.frame(1))  # 같은 사진이 또 옴
        m.on_frame(self.frame(2))
        m.on_frame(self.frame(3))
        m.on_frame(self.frame(4))
        self.assertEqual([f.captured_at_utc for _k, f in m.recent], ["t2", "t3", "t4"])  # 가장 새 세 장
        fresh = self.maker()
        fresh.on_mission(N(phase="running"))  # 본 경기: 시작 전에는 사진이 오지 않는다
        self.assertEqual((fresh.recent, fresh.first_frame_at), ([], None))
        fresh.on_frame(self.frame(5))
        self.assertEqual(len(fresh.recent), 1)

    def test_attempt_publishes_the_frame_with_the_most_boxes(self):
        m = self.maker()
        m.child, m.spare = "A", ["B", "C"]  # 계산 프로세스 셋 (가짜 — compute_one 을 바꿔 끼운다)
        counts = {b"jpeg1": 14, b"jpeg2": 16, b"jpeg3": 15}
        asked = []

        def compute_one(frame, child, detector="rules"):
            asked.append(frame.captured_at_utc)
            n = counts[frame.image.data]
            return {"boxes": [{"x": 1.0 + 0.4 * (i % 2), "y": 0.5 + 1.0 * (i // 2)} for i in range(n)], "faces": []}, {"ok": True, "boxes": n, "notes": []}, "process", 0.1, [6.12, 4.8]

        m.compute_one = compute_one
        original = map_maker.drone_field_map.field_map_msg
        map_maker.drone_field_map.field_map_msg = lambda data: ("msg", len(data["boxes"]))
        try:
            m.attempt([self.frame(1), self.frame(2), self.frame(3)])
        finally:
            map_maker.drone_field_map.field_map_msg = original
        self.assertEqual(sorted(asked), ["t1", "t2", "t3"])
        self.assertEqual(m.message, ("msg", 16))
        self.assertEqual(m.sent, [("msg", 16)])
        self.assertEqual(m.saved, [("t2", 16)])  # 기록에 남는 사진도 고른 것
        tries = [r for r in m.logged if r.get("e") == "try"]
        self.assertEqual([(r["captured"], r["boxes"], r["picked"], r["batch"]) for r in tries], [("t1", 14, False, 3), ("t2", 16, True, 3), ("t3", 15, False, 3)])
        self.assertIn("14·16·15", m.detail)

    SECTORS = [{"name": n, "min_x": x0, "max_x": x1, "min_y": y0, "max_y": y1}
               for side, x0, x1 in (("B", 0.89, 2.44), ("R", 3.68, 5.23)) for n, y0, y1 in (("%s-S1" % side, 3.2, 4.8), ("%s-S2" % side, 1.6, 3.2), ("%s-S3" % side, 0.0, 1.6))
               ] + [{"name": "CM", "min_x": 2.44, "max_x": 3.68, "min_y": 0.0, "max_y": 4.8}]

    def sided(self, *pairs, singles=()):
        return dict(self.boxes(*pairs, singles=singles), sectors=self.SECTORS)

    def test_asymmetry_lists_the_rows_whose_blue_and_red_counts_differ(self):
        full = self.sided(*self.SIX, singles=((3.08, 3.4), (3.08, 3.0)))
        self.assertEqual(map_maker.asymmetry(full), [])
        lost = self.sided(*self.SIX[:3], *self.SIX[4:], singles=((4.25, 3.95),))  # R-S1 의 오른쪽 상자를 놓침
        self.assertEqual(map_maker.asymmetry(lost), [("1", 2, 1)])
        ghost = self.sided(*self.SIX, singles=((5.05, 1.2),))  # R-S3 에 헛것 하나
        self.assertEqual(map_maker.asymmetry(ghost), [("3", 2, 3)])
        self.assertEqual(map_maker.asymmetry(self.sided(*self.SIX, singles=((3.08, 3.4),))), [])  # 가운데 구역은 보지 않는다
        self.assertEqual(map_maker.asymmetry(self.boxes(*self.SIX[:5])), [])  # 구역이 없는 지도는 판단하지 않는다

    def test_pick_symmetric_drops_lopsided_maps_and_falls_back_when_none_is_left(self):
        ok, bad = {"ok": True}, {"ok": False}
        full = self.sided(*self.SIX)
        lost = self.sided(*self.SIX[:3], *self.SIX[4:], singles=((4.25, 3.95),))
        ghost = self.sided(*self.SIX, singles=((5.05, 1.2), (5.05, 2.4)))
        first, second = dict(ok), dict(ok)
        self.assertEqual(map_maker.pick_symmetric([(lost, first), (full, second), (ghost, dict(ok)), (full, dict(ok))]), (1, [0, 2], False))
        self.assertEqual((first["symmetric"], first["asymmetry"], second["symmetric"]), (False, [["1", 2, 1]], True))
        self.assertEqual(map_maker.pick_symmetric([(ghost, dict(ok)), (lost, dict(ok))]), (0, [0, 1], True))  # 남은 것이 없으면 전체에서 (map_score)
        self.assertEqual(map_maker.pick_symmetric([(None, dict(bad)), (full, dict(bad))]), (None, [], False))
        self.assertEqual(map_maker.pick_symmetric([]), (None, [], False))

    def test_attempt_drops_the_lopsided_map_even_when_it_has_more_boxes(self):
        m = self.maker(pick_frames=4)
        maps = {"t1": self.sided(*self.SIX, singles=((5.05, 1.2), (5.05, 2.4))), "t2": self.sided(*self.SIX), "t3": self.sided(*self.SIX), "t4": self.sided(*self.SIX)}
        m.child, m.spare = "A", ["D", "C", "B"]
        m.compute_one = lambda frame, child, detector="rules": (
            dict(maps[frame.captured_at_utc], faces=[]), {"ok": True, "boxes": len(maps[frame.captured_at_utc]["boxes"]), "notes": []}, "process", 0.1, None)
        original = map_maker.drone_field_map.field_map_msg
        map_maker.drone_field_map.field_map_msg = lambda data: ("msg", len(data["boxes"]))
        try:
            m.attempt([self.frame(k) for k in (1, 2, 3, 4)])
            self.assertEqual(m.saved, [("t2", 12)])
            tries = [r for r in m.logged if r.get("e") == "try"]
            self.assertEqual([(r["dropped_asymmetric"], r["picked"]) for r in tries], [(True, False), (False, True), (False, False), (False, False)])
            off = self.maker(pick_frames=4)
            off.pick_symmetric, off.child, off.spare, off.compute_one = False, "A", ["D", "C", "B"], m.compute_one
            off.attempt([self.frame(k) for k in (1, 2, 3, 4)])
            self.assertEqual(off.saved[0][0], "t2")  # 끄면 map_score 만으로 (헛것 둘은 남는 상자로 깎인다)
        finally:
            map_maker.drone_field_map.field_map_msg = original

    def test_mixed_mode_uses_the_model_for_the_first_frame_and_the_rules_for_the_others(self):
        m = self.maker()
        m.child, m.spare = "A", ["C", "B"]
        seen = {}

        def compute_one(frame, child, detector="rules"):
            seen[frame.captured_at_utc] = (child, detector)
            return {"boxes": [{}] * 16, "faces": []}, {"ok": True, "boxes": 16, "notes": [], "box_detector": detector}, "process", 0.1, None

        m.compute_one = compute_one
        original = map_maker.drone_field_map.field_map_msg
        map_maker.drone_field_map.field_map_msg = lambda data: ("msg", len(data["boxes"]))
        try:
            m.attempt([self.frame(1), self.frame(2), self.frame(3)])
            self.assertEqual(seen, {"t1": ("A", "model"), "t2": ("B", "rules"), "t3": ("C", "rules")})
            self.assertEqual(m.saved, [("t1", 16)])  # 점수가 같으면 앞 번호 = 모델의 지도
            self.assertIn("model", m.detail)
            for mode, want in (("model", ["model", "model", "model"]), ("rules", ["rules", "rules", "rules"])):
                other = self.maker()
                other.box_detector, other.child, other.spare = mode, "A", ["C", "B"]
                seen.clear()
                other.compute_one = compute_one
                other.attempt([self.frame(1), self.frame(2), self.frame(3)])
                self.assertEqual([seen[k][1] for k in ("t1", "t2", "t3")], want)
            lone = self.maker()  # 계산 프로세스가 하나도 없으면 첫 장만 노드 안에서, 규칙으로 (노드 안에서 모델을 읽지 않는다)
            seen.clear()
            lone.compute_one = compute_one
            lone.attempt([self.frame(1), self.frame(2)])
            self.assertEqual(seen, {"t1": (None, "rules")})
        finally:
            map_maker.drone_field_map.field_map_msg = original

    def test_spawn_gives_the_model_only_to_the_first_child_in_mixed_mode(self):
        from unittest import mock

        m = self.maker()
        del m.spawn  # 진짜 spawn 을 쓴다
        m.use_child = True
        with mock.patch.object(map_maker, "start_child", return_value="child") as start:
            m.spawn()
            m.spawn(first=False)
            self.assertEqual([c.kwargs["box_model"] for c in start.call_args_list], ["/nowhere/drone_boxes.engine", None])
            m.box_detector = "model"
            start.reset_mock()
            m.spawn(first=False)
            self.assertEqual(start.call_args.kwargs["box_model"], "/nowhere/drone_boxes.engine")
            m.box_detector = "rules"
            start.reset_mock()
            m.spawn()
            self.assertIsNone(start.call_args.kwargs["box_model"])

    def test_without_child_processes_only_the_first_frame_is_computed(self):
        m = self.maker()  # 이 시험용 maker 는 계산 프로세스를 띄우지 못한다 (spawn 이 None)
        asked = []

        def compute_one(frame, child, detector="rules"):
            asked.append((frame.captured_at_utc, child))
            return {"boxes": [{}] * 15, "faces": []}, {"ok": True, "boxes": 15, "notes": []}, "thread", 0.1, None

        m.compute_one = compute_one
        original = map_maker.drone_field_map.field_map_msg
        map_maker.drone_field_map.field_map_msg = lambda data: ("msg", len(data["boxes"]))
        try:
            m.attempt([self.frame(1), self.frame(2), self.frame(3)])
            self.assertEqual(asked, [("t1", None)])  # 노드 안에서 세 장을 함께 계산하지 않는다
            self.assertEqual(m.message, ("msg", 15))
            some = self.maker()
            some.child, some.spare = "A", ["B"]  # 프로세스가 둘뿐이면 두 장만
            asked.clear()
            some.compute_one = compute_one
            some.attempt([self.frame(1), self.frame(2), self.frame(3)])
            self.assertEqual(sorted(asked), [("t1", "A"), ("t2", "B")])
        finally:
            map_maker.drone_field_map.field_map_msg = original

    def test_attempt_survives_a_frame_whose_computation_dies_and_reports_when_none_works(self):
        m = self.maker()
        m.child, m.spare = "A", ["B", "C"]

        def compute_one(frame, child, detector="rules"):
            if frame.image.data == b"jpeg1":
                raise RuntimeError("boom")
            if frame.image.data == b"jpeg2":
                return None, {"ok": False, "boxes": 3, "notes": ["상자가 너무 적다"]}, "process", 0.1, None
            return {"boxes": [{}] * 15, "faces": []}, {"ok": True, "boxes": 15, "notes": []}, "process", 0.1, None

        m.compute_one = compute_one
        original = map_maker.drone_field_map.field_map_msg
        map_maker.drone_field_map.field_map_msg = lambda data: ("msg", len(data["boxes"]))
        try:
            m.attempt([self.frame(1), self.frame(2), self.frame(3)])
            self.assertEqual(m.message, ("msg", 15))  # 한 장이 죽고 한 장이 실패해도 남은 한 장으로 낸다
            none = self.maker()
            none.child, none.spare = "A", ["B"]
            none.compute_one = lambda frame, child, detector="rules": (None, {"ok": False, "boxes": 0, "notes": ["경계를 못 찾음"]}, "process", 0.1, None)
            none.attempt([self.frame(1), self.frame(2)])
            self.assertIsNone(none.message)
            self.assertEqual(none.sent, [])
            self.assertIn("아직 없음", none.detail)
            single = self.maker()
            single.compute_one = compute_one
            single.attempt(self.frame(3))  # 한 장만 줘도 된다 (전과 같은 쓰임)
            self.assertEqual(single.message, ("msg", 15))
        finally:
            map_maker.drone_field_map.field_map_msg = original


class FillBaseKindsTests(unittest.TestCase):
    """고른 사진이 한 진영의 거점 벽 그림을 못 읽었을 때 (본선 카메라는 RED 벽이 화면 밖에 걸린다)."""

    CP_FIRST = ["COMMAND_POST", "MAINTENANCE", "SUPPLY_DEPOT"]
    SD_FIRST = ["SUPPLY_DEPOT", "COMMAND_POST", "MAINTENANCE"]

    @staticmethod
    def result(blue, red):
        from perception import base_panels

        data = {"width": 6.12, "height": 4.8, "bases": [{"name": n} for n in ("B-1", "B-2", "B-3", "R-1", "R-2", "R-3")], "boxes": []}
        kinds = {"BLUE": blue, "RED": red}
        base_panels.add_to_map(data, kinds)  # compute 가 하는 것처럼, 읽은 진영은 이미 지도에 있다
        return data, {"ok": True, "base_panels": {"kinds": kinds}}

    @staticmethod
    def rows(data, camp):
        return {b["name"]: b["row"] for b in data["bases"] if b["name"].startswith(camp + "_")}

    def test_takes_the_other_frames_reading_first(self):
        results = [self.result(self.SD_FIRST, None), self.result(self.SD_FIRST, self.SD_FIRST), self.result(None, None)]
        filled = map_maker.fill_base_kinds(results, 0)
        self.assertEqual(filled, {"RED": {"order": self.SD_FIRST, "from": "frame"}})
        self.assertEqual(self.rows(results[0][0], "RED"), {"RED_SUPPLY_DEPOT": 1, "RED_COMMAND_POST": 2, "RED_MAINTENANCE": 3})
        self.assertEqual(results[0][1]["base_panels"]["read"], {"BLUE": self.SD_FIRST, "RED": None})  # 이 사진에서 읽은 것은 따로 남는다

    def test_follows_the_other_camp_when_no_frame_read_it(self):
        results = [self.result(self.SD_FIRST, None), self.result(self.SD_FIRST, None)]
        filled = map_maker.fill_base_kinds(results, 1)
        self.assertEqual(filled["RED"]["from"], "mirror")
        self.assertEqual(self.rows(results[1][0], "RED"), {"RED_SUPPLY_DEPOT": 1, "RED_COMMAND_POST": 2, "RED_MAINTENANCE": 3})
        self.assertEqual(self.rows(results[0][0], "RED"), {})  # 고르지 않은 사진의 지도는 건드리지 않는다
        off = [self.result(self.SD_FIRST, None)]
        self.assertEqual(map_maker.fill_base_kinds(off, 0, mirror=False), {})
        self.assertEqual(self.rows(off[0][0], "RED"), {})

    def test_frames_that_disagree_are_not_used(self):
        results = [self.result(self.CP_FIRST, None), self.result(None, self.SD_FIRST), self.result(None, self.CP_FIRST)]
        filled = map_maker.fill_base_kinds(results, 0)
        self.assertEqual(filled["RED"], {"order": self.CP_FIRST, "from": "mirror"})  # 사진마다 답이 달라 맞은편 진영을 따른다

    def test_nothing_to_do_when_both_are_read_or_none_is(self):
        both = [self.result(self.CP_FIRST, self.SD_FIRST)]
        self.assertEqual(map_maker.fill_base_kinds(both, 0), {})
        self.assertNotIn("filled", both[0][1]["base_panels"])
        none = [self.result(None, None), self.result(None, None)]
        self.assertEqual(map_maker.fill_base_kinds(none, 0), {})
        self.assertEqual(map_maker.fill_base_kinds([(None, {"ok": False})], 0), {})


class ChildTests(unittest.TestCase):
    """계산을 따로 띄운 프로세스에 맡기는 길 (노드가 쓰는 길)."""

    def test_child_with_the_box_model_hides_the_gpu_unless_asked(self):
        """계산 프로세스는 상자 모델을 CPU 로 돌린다 (로봇에서 GPU 면 프로세스마다 +700 MB)."""
        import subprocess
        from unittest import mock

        with mock.patch.object(subprocess, "Popen") as popen:
            map_maker.start_child(box_model="/x/drone_boxes.pt")
            env = popen.call_args.kwargs["env"]
            self.assertEqual((env["PERCEPTION_MAP_BOX_MODEL"], env["CUDA_VISIBLE_DEVICES"]), ("/x/drone_boxes.pt", ""))
            with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}):
                map_maker.start_child(box_model="/x/drone_boxes.pt", box_gpu=True)
            self.assertEqual(popen.call_args.kwargs["env"].get("CUDA_VISIBLE_DEVICES"), "0")
            map_maker.start_child()
            self.assertNotIn("PERCEPTION_MAP_BOX_MODEL", popen.call_args.kwargs["env"])

    def test_child_uses_one_math_thread(self):
        """프로세스 셋을 함께 돌려도 서로 CPU 를 밀치지 않게 (로봇에서 70초 → 4.7초, 2026-11-19)."""
        import subprocess
        from unittest import mock

        with mock.patch.object(subprocess, "Popen") as popen:
            map_maker.start_child()
        env = popen.call_args.kwargs["env"]
        self.assertEqual([env[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")], ["1", "1", "1"])

    def test_child_answers_like_compute(self):
        with open(os.path.join(DATA, "drone_test2.jpg"), "rb") as fh:
            jpeg = fh.read()
        child = map_maker.start_child()
        data, info = map_maker.ask_child(child, jpeg, {"stamp_s": 1234.5, "base_naming": "slot", "camera_height": None}, 120.0)
        self.assertTrue(info["ok"], info)
        self.assertEqual(data["header"], {"frame_id": "arena", "stamp": 1234.5})
        self.assertEqual([b["name"] for b in data["bases"][:6]], ["B-1", "B-2", "B-3", "R-1", "R-2", "R-3"])
        self.assertEqual(info["base_panels"]["kinds"]["RED"], ["MAINTENANCE", "COMMAND_POST", "SUPPLY_DEPOT"])  # 벽 그림으로 읽은 것도 프로세스를 거쳐 온다
        self.assertEqual(len(data["bases"]), 12)
        direct, _info = map_maker.compute(jpeg, 1234.5, "slot", None)
        self.assertEqual(len(data["boxes"]), len(direct["boxes"]))
        self.assertEqual(len(data["faces"]), len(direct["faces"]))
        self.assertEqual(json.loads(json.dumps(direct)), data)  # 프로세스를 거쳐도 같은 지도

    def test_child_reports_a_bad_picture_instead_of_dying(self):
        child = map_maker.start_child()
        data, info = map_maker.ask_child(child, b"not a picture", {"stamp_s": None, "base_naming": "slot", "camera_height": None}, 60.0)
        self.assertIsNone(data)
        self.assertFalse(info["ok"])

    def test_slow_child_is_cut_off(self):
        with open(os.path.join(DATA, "drone_test2.jpg"), "rb") as fh:
            jpeg = fh.read()
        child = map_maker.start_child()
        with self.assertRaises(TimeoutError):
            map_maker.ask_child(child, jpeg, {"stamp_s": None, "base_naming": "slot", "camera_height": None}, 0.05)
        self.assertIsNotNone(child.poll())  # 끊긴 프로세스가 남아 있지 않다

    def test_broken_child_is_noticed(self):
        child = map_maker.start_child(python="false")  # 아무것도 하지 않고 끝나는 프로그램
        with self.assertRaises(map_maker.ChildError):
            map_maker.ask_child(child, b"x", {}, 10.0)


if __name__ == "__main__":
    unittest.main()
