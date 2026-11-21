"""서버에 보내는 JSON 의 모양 (경기서버연동API 2026-11-04 기준). ROS 없이 돈다."""
import unittest

from server_bridge import constants as C
from server_bridge import reports as R


class ReportBodyTests(unittest.TestCase):
    def test_m1_drops_empty_items_and_orders_sectors(self):
        self.assertEqual(R.m1_body("RED_RADAR", ["cm", "R-S3", "R-S1"], "RED_SUPPLY_DEPOT"),
                         {"target": "RED_RADAR", "search_sectors": ["CM", "R-S1", "R-S3"], "base": "RED_SUPPLY_DEPOT"})
        self.assertEqual(R.m1_body("", [], "RED_MAINTENANCE"), {"base": "RED_MAINTENANCE"})
        self.assertEqual(R.m1_body("", ["7", "X-1"], ""), {})  # 모르는 구역 이름(숫자 등)은 보내지 않는다 — 422 가 된다
        self.assertEqual(R.m1_body("레이더시설", ["CM"], "RED_HQ"), {"search_sectors": ["CM"]})  # 명칭 · 모르는 값은 보내지 않는다 (422)
        self.assertEqual(R.m1_body("red_radar", [], " blue_maintenance "), {"target": "RED_RADAR", "base": "BLUE_MAINTENANCE"})

    def test_m3_keeps_every_given_sector_and_only_known_kinds(self):
        body = R.m3_body([("R-S1", {"tank": 2, "drum": 0, "radar": 1}), ("CM", {}), ("9", {"tank": 1})])
        self.assertEqual(body, {"sectors": [{"sector": "R-S1", "objects": [{"type": "tank", "count": 2}]}, {"sector": "CM", "objects": []}]})
        for sector in body["sectors"]:
            for obj in sector["objects"]:
                self.assertIn(obj["type"], C.OBJECT_TYPES)

    def test_m6_ranking_is_three_distinct_opponents(self):
        self.assertEqual(R.m6_body(["R-2", "r-2", "B-1", "R-3"], "BLUE"), {"ranking": ["R-2", "R-3", "R-1"]})
        self.assertEqual(R.m6_body([], "RED"), {"ranking": ["B-3", "B-2", "B-1"]})          # 아무것도 모르면 3 · 2 · 1 (2026-11-20 팀장 결정)
        self.assertEqual(R.m6_body(["R-2"], "BLUE"), {"ranking": ["R-2", "R-3", "R-1"]})    # 빠진 것은 번호가 큰 것부터
        self.assertEqual(R.m6_body(["R-3", "R-1", "R-2"], None), {"ranking": ["R-3", "R-1", "R-2"]})  # 진영을 모르면 받은 그대로

    def test_slot_and_objective_are_validated(self):
        self.assertEqual(R.clean_slot(" b-2 "), "B-2")
        self.assertIsNone(R.clean_slot("B-4"))
        self.assertIsNone(R.clean_slot(None))
        self.assertEqual(R.clean_objective({"objective": "RED_MAINTENANCE", "objective_label": "정비소"}, "BLUE"), "RED_MAINTENANCE")
        self.assertEqual(R.clean_objective({"objective": "??", "objective_label": "군수보급소"}, "RED"), "BLUE_SUPPLY_DEPOT")  # 이름으로 대신 찾는다 (상대 진영의 거점)
        self.assertIsNone(R.clean_objective({"objective": "??", "objective_label": "??"}, "BLUE"))


if __name__ == "__main__":
    unittest.main()
