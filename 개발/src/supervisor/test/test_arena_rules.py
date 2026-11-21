"""경기장 수치와 구역 판정이 참가팀 포털의 경기장 그림(서버의 채점 규칙)과 같은가. 2026-11-18 포털에서 읽은 값."""
import unittest
from types import SimpleNamespace

from supervisor import arena as A


def portal_sector(x_cm, y_cm):
    """포털 스크립트의 sectorOfPoint 를 그대로 옮긴 것 (좌표는 cm, y 는 위에서 아래로)."""
    if x_cm < 80 or x_cm >= 520:
        return None
    if 240 <= x_cm < 360:
        return "CM"
    row = 3 if y_cm >= 320 else 2 if y_cm >= 160 else 1
    return "%s-S%d" % ("B" if x_cm < 240 else "R", row)


PORTAL = dict(width=6.0, height=4.8, end_strip=0.8, cm_width=1.2, base_w=0.8, base_d=0.4)  # 포털의 경기장 그림(서버의 모형)과 같은 수치


class ArenaRuleTests(unittest.TestCase):
    def test_default_plan_is_the_measured_arena(self):
        """기본값은 실제 경기장에서 잰 값 (벽에서 벽까지 약 612 x 500 cm, 끝 띠 선 0.88 m, CM 폭 1.2 m, 줄은 높이의 1/3)."""
        p = A.ArenaPlan()
        self.assertEqual((p.width, p.height), (6.12, 5.0))
        self.assertEqual(tuple(round(v, 3) for v in p.sector_rect("CM")), (2.46, 0.0, 3.66, 5.0))
        self.assertEqual(tuple(round(v, 3) for v in p.sector_rect("B-S1")), (0.88, 3.333, 2.46, 5.0))  # 1번 줄이 위
        self.assertEqual(tuple(round(v, 3) for v in p.sector_rect("R-S3")), (3.66, 0.0, 5.24, 1.667))
        self.assertEqual(tuple(round(v, 3) for v in p.base_rect("BLUE", 2)), (0.0, 1.9, 0.4, 3.1))

    def test_portal_numbers_give_the_portal_layout(self):
        p = A.ArenaPlan(**PORTAL)
        self.assertEqual(tuple(round(v, 6) for v in p.sector_rect("CM")), (2.4, 0.0, 3.6, 4.8))
        self.assertEqual(tuple(round(v, 6) for v in p.sector_rect("B-S1")), (0.8, 3.2, 2.4, 4.8))
        self.assertEqual(tuple(round(v, 6) for v in p.base_rect("BLUE", 1)), (0.0, 3.6, 0.4, 4.4))  # 가운데 높이 위에서 80 cm
        self.assertEqual(tuple(round(v, 6) for v in p.base_rect("RED", 3)), (5.6, 0.4, 6.0, 1.2))

    def test_points_follow_the_server_rule_including_boundaries(self):
        fmap = A.field_map([], A.ArenaPlan(**PORTAL))
        for x_cm in range(0, 601, 20):
            for y_cm in range(0, 481, 20):
                for dx, dy in ((0, 0), (7, 0), (0, 7), (13, 13)):
                    xc, yc = x_cm + dx, y_cm + dy
                    if not (0 < xc < 600 and 0 < yc < 480):
                        continue
                    ours = fmap.sector_at(xc / 100.0, (480 - yc) / 100.0)
                    self.assertEqual(ours, portal_sector(xc, yc), (xc, yc))

    def test_each_face_gets_the_sector_of_its_plate_centre(self):
        """경계에 걸친 상자: 면마다 구역이 다를 수 있다 (판의 한가운데 = 면에서 5 cm 안쪽). 잰 위치가 몇 cm 틀려도 같아야 한다."""
        for gx, gy in ((11, 3), (17, 7), (3, 15), (25, 10), (12, 8), (14, 11), (10, 6), (18, 14)):  # 상자의 왼쪽 위 칸
            cx_cm, cy_cm = (gx + 1) * 20, (gy + 1) * 20  # 상자 가운데 (cm, y 는 위에서 아래로)
            for noise in ((0.0, 0.0), (0.04, -0.03), (-0.06, 0.05)):
                fmap = A.field_map([(cx_cm / 100.0 + noise[0], (480 - cy_cm) / 100.0 + noise[1])], A.ArenaPlan(**PORTAL))
                self.assertEqual(len(fmap.faces), 4)
                for face in fmap.faces:
                    px = cx_cm + face.nx * 15  # 판의 한가운데: 가운데에서 15 cm (면까지 20 cm − 5 cm)
                    py = cy_cm - face.ny * 15  # 우리 y 는 위로 자란다
                    self.assertEqual(face.sector, portal_sector(px, py), (gx, gy, face.side, noise))

    def test_same_rule_holds_in_wall_based_coordinates(self):
        """지도가 벽 기준 좌표(612 x 500, 바닥 격자가 6 · 10 cm 안쪽)로 와도, 그 지도의 경계선에 맞춰 같은 구역이 나온다."""
        plan = A.ArenaPlan()
        for gx, gy in ((11, 3), (12, 8), (3, 15), (25, 10)):
            x = (gx + 1) * 0.2 + 0.06  # 바닥 격자가 벽에서 6 cm 들어가 있고, 끝 띠 선 0.88 = 0.06 + 0.82 가까이
            y = plan.height / 2.0 + (2.4 - (gy + 1) * 0.2) * (plan.height / 3.0) / 1.6  # 줄 경계가 높이의 1/3 에 오도록 늘린 자리
            clean = A.field_map([(x, y)], plan)
            noisy = A.field_map([(x + 0.04, y - 0.05)], plan)
            self.assertEqual([f.sector for f in clean.faces], [f.sector for f in noisy.faces], (gx, gy))


class TouchingBoxTests(unittest.TestCase):
    def test_measured_pair_hides_the_faces_between(self):
        """드론 사진에서 잰 붙은 쌍(가운데 사이 0.37 ~ 0.43 m, 옆으로 몇 cm 어긋남): 두 상자 사이의 면 둘은 없어야 한다."""
        for gap, skew in ((0.40, 0.0), (0.37, 0.03), (0.43, -0.04), (0.345, 0.05)):
            fmap = A.field_map([(1.48, 4.12), (1.48 + gap, 4.12 + skew)], A.ArenaPlan())
            sides = sorted((f.box, f.side) for f in fmap.faces)
            self.assertEqual(sides, [(0, "N"), (0, "S"), (0, "W"), (1, "E"), (1, "N"), (1, "S")], (gap, skew))

    def test_boxes_one_grid_step_apart_keep_their_faces(self):
        """격자 한 칸(20 cm) 떨어진 상자는 붙은 것이 아니다 — 면을 빼지 않는다."""
        fmap = A.field_map([(1.48, 4.12), (2.08, 4.12)], A.ArenaPlan())
        self.assertEqual(len(fmap.faces), 8)
        fmap = A.field_map([(1.48, 4.12), (1.88, 3.92)], A.ArenaPlan())  # 옆으로 반 칸 어긋나 붙은 상자: 면이 반쯤 보인다
        self.assertEqual(len(fmap.faces), 8)


class MapMessageTests(unittest.TestCase):
    """지도 메시지의 거점 이름: navigation 은 자리 이름(B-1 …)으로 찾고 supervisor 는 보고값으로 찾는다 → 둘 다 받는다."""

    @staticmethod
    def message(bases, sectors=()):
        region = lambda name, r, row=0: SimpleNamespace(name=name, min_x=r[0], min_y=r[1], max_x=r[2], max_y=r[3], row=row)
        return SimpleNamespace(width=6.0, height=4.8, box_size=0.4, beam_depth=0.2, boxes=[SimpleNamespace(x=3.0, y=2.4)],
                               bases=[region(*b) for b in bases], sectors=[region(*s) for s in sectors])

    def test_slot_names_become_report_keys(self):
        self.assertEqual(A.base_key("B-1"), "BLUE_COMMAND_POST")
        self.assertEqual(A.base_key("r-3"), "RED_SUPPLY_DEPOT")
        self.assertEqual(A.base_key(" red_maintenance "), "RED_MAINTENANCE")
        self.assertIsNone(A.base_key("X-9"))
        msg = self.message([("B-1", (0.0, 3.55, 0.4, 4.45)), ("R-2", (5.6, 2.0, 6.0, 2.8), 2)])
        fmap = A.field_map_from_msg(msg)
        self.assertEqual(fmap.bases["BLUE_COMMAND_POST"], (0.0, 3.55, 0.4, 4.45))  # 메시지의 네모를 쓴다
        self.assertEqual(fmap.base_rows["BLUE_COMMAND_POST"], 1)  # 줄 번호가 비어 있어도 높이로 안다
        self.assertEqual(fmap.bases["RED_MAINTENANCE"], (5.6, 2.0, 6.0, 2.8))
        self.assertEqual(fmap.base_in_row("RED", 2), "RED_MAINTENANCE")

    def test_missing_or_unknown_regions_fall_back_to_the_drawing(self):
        msg = self.message([("거점 가", (0.0, 0.0, 1.0, 1.0)), ("B-2", (0.0, 0.0, 0.0, 0.0))], sectors=[("7", (2.0, 0.0, 4.0, 4.8)), ("CM", (2.5, 0.0, 3.5, 4.8))])
        fmap = A.field_map_from_msg(msg)
        plan = A.ArenaPlan()
        self.assertEqual(len(fmap.bases), 6)  # 여섯 거점이 늘 다 있다 (게시된 목표거점을 못 찾아 멈추지 않게)
        self.assertEqual(fmap.bases["BLUE_MAINTENANCE"], plan.base_rect("BLUE", 2))
        self.assertEqual(fmap.sectors["CM"], (2.5, 0.0, 3.5, 4.8))
        self.assertEqual(set(fmap.sectors), set(A.SECTORS))
        for camp in A.CAMPS:
            self.assertEqual(sorted(fmap.base_rows[v] for v in fmap.bases_of(camp)), [1, 2, 3])

    # ---- 2026-11-19 공지: 거점 종류의 배치는 바뀔 수 있다. perception 이 벽 그림으로 읽은 종류를 자리 이름 여섯 개 뒤에 보고값 이름으로 덧붙여 보낸다
    @staticmethod
    def slot_rects():
        plan = A.ArenaPlan()
        return {(camp, row): plan.base_rect(camp, row) for camp in A.CAMPS for row in (1, 2, 3)}

    def slots_then(self, kinds):
        """지도 메시지의 bases: 자리 이름 여섯 개(B-1 … R-3) + kinds = {진영: [1번 줄 종류, 2번 줄, 3번 줄]} 의 보고값 이름."""
        rects = self.slot_rects()
        bases = [("%s-%d" % (camp[0], row), rects[(camp, row)], row) for camp in A.CAMPS for row in (1, 2, 3)]
        bases += [("%s_%s" % (camp, kind), rects[(camp, row)], row) for camp, order in kinds.items() for row, kind in zip((1, 2, 3), order)]
        return self.message(bases)

    def test_kind_names_after_the_slot_names_say_where_each_kind_really_is(self):
        """열두 개: 자리 이름은 기본 배치(위부터 작전지휘소 · 정비소 · 군수보급소)로 먼저 읽히고, 뒤에 온 보고값 이름이 그 종류의 줄 · 네모를 덮어쓴다."""
        rects = self.slot_rects()
        kinds = {"BLUE": ["MAINTENANCE", "COMMAND_POST", "SUPPLY_DEPOT"], "RED": ["SUPPLY_DEPOT", "MAINTENANCE", "COMMAND_POST"]}
        fmap = A.field_map_from_msg(self.slots_then(kinds))
        self.assertEqual(len(fmap.bases), 6)
        for camp, order in kinds.items():
            for row, kind in zip((1, 2, 3), order):
                value = "%s_%s" % (camp, kind)
                self.assertEqual(fmap.base_rows[value], row, value)
                self.assertEqual(fmap.bases[value], rects[(camp, row)], value)
                self.assertEqual(fmap.base_in_row(camp, row), value)  # 줄마다 거점이 하나씩 그대로다
            self.assertEqual(sorted(fmap.base_rows[v] for v in fmap.bases_of(camp)), [1, 2, 3])
        self.assertEqual(fmap.bases["RED_MAINTENANCE"], rects[("RED", 2)])  # 기본 배치와 같은 줄에 있는 것도 그대로
        self.assertEqual(fmap.bases["BLUE_COMMAND_POST"], rects[("BLUE", 2)])  # 기본 배치로는 1번 줄이었다
        self.assertEqual(fmap.kinds_read, frozenset(A.CAMPS))  # 두 진영 모두 종류를 보고값 이름으로 받았다

    def test_kinds_read_is_set_only_for_camps_whose_kinds_came_by_report_name(self):
        """자리 이름(B-1 …)만 온 진영은 기본 배치로 읽은 것 — 「읽었다」 로 치지 않는다 (대기할 거점을 고를 때 쓴다)."""
        self.assertEqual(A.field_map_from_msg(self.slots_then({})).kinds_read, frozenset())
        only_red = A.field_map_from_msg(self.slots_then({"RED": ["SUPPLY_DEPOT", "MAINTENANCE", "COMMAND_POST"]}))
        self.assertEqual(only_red.kinds_read, frozenset({"RED"}))
        rects = self.slot_rects()
        partly = self.message([("B-%d" % row, rects[("BLUE", row)], row) for row in (1, 2, 3)] + [("BLUE_COMMAND_POST", rects[("BLUE", 3)], 3)])
        self.assertEqual(A.field_map_from_msg(partly).kinds_read, frozenset())  # 세 종류가 다 와야 한다
        self.assertEqual(A.field_map([(3.0, 2.4)]).kinds_read, frozenset())  # 메시지 없이 만든 지도

    def test_only_the_recognised_camp_changes(self):
        """아홉 개: 한 진영만 알아냈으면 그 진영만 바뀌고, 다른 진영은 기본 배치로 남는다 (supervisor 가 차례로 들어가 본다)."""
        rects = self.slot_rects()
        fmap = A.field_map_from_msg(self.slots_then({"RED": ["COMMAND_POST", "SUPPLY_DEPOT", "MAINTENANCE"]}))
        self.assertEqual([fmap.base_in_row("RED", row) for row in (1, 2, 3)], ["RED_COMMAND_POST", "RED_SUPPLY_DEPOT", "RED_MAINTENANCE"])
        self.assertEqual(fmap.bases["RED_MAINTENANCE"], rects[("RED", 3)])
        self.assertEqual([fmap.base_in_row("BLUE", row) for row in (1, 2, 3)], ["BLUE_COMMAND_POST", "BLUE_MAINTENANCE", "BLUE_SUPPLY_DEPOT"])
        self.assertEqual(len(fmap.bases), 6)

    def test_slot_names_alone_keep_the_default_layout(self):
        fmap = A.field_map_from_msg(self.slots_then({}))  # 알아내지 못한 경기: 여섯 개만 온다
        for camp in A.CAMPS:
            self.assertEqual([fmap.base_in_row(camp, row) for row in (1, 2, 3)], ["%s_%s" % (camp, kind) for kind in A.BASE_KINDS])


if __name__ == "__main__":
    unittest.main()
