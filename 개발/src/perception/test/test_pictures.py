import unittest

from perception import pictures


def det(name, conf, x1, y1, x2, y2):
    return (name, conf, x1, y1, x2, y2)


class BuildPicturesTest(unittest.TestCase):
    def test_left_to_right_with_ratios(self):
        found = pictures.build_pictures(
            [det('tank', 0.9, 400, 120, 560, 360), det('drum', 0.8, 80, 120, 240, 360)], 640, 480)
        self.assertEqual([p['kind'] for p in found], ['drum', 'tank'])
        self.assertAlmostEqual(found[0]['x_center'], 0.25)
        self.assertAlmostEqual(found[0]['y_center'], 0.5)
        self.assertAlmostEqual(found[0]['width'], 0.25)
        self.assertAlmostEqual(found[0]['height'], 0.5)
        self.assertFalse(found[0]['cut_off'])

    def test_low_score_is_kept_as_unknown(self):
        found = pictures.build_pictures(
            [det('tank', 0.9, 80, 120, 240, 360), det('mortar', 0.2, 400, 120, 560, 360)], 640, 480)
        self.assertEqual([p['kind'] for p in found], ['tank', 'unknown'])
        self.assertAlmostEqual(found[1]['confidence'], 0.2)

    def test_class_outside_the_ten_kinds_is_unknown(self):
        found = pictures.build_pictures([det('person', 0.9, 80, 120, 240, 360)], 640, 480)
        self.assertEqual(found[0]['kind'], 'unknown')

    def test_aliases(self):
        self.assertEqual(pictures.kind_of('Military-Truck'), 'military_truck')
        self.assertEqual(pictures.kind_of('탄약상자'), 'ammo_box')
        self.assertEqual(pictures.kind_of('레이더시설'), 'radar')
        self.assertEqual(pictures.kind_of('non_object'), 'non_object')
        self.assertIsNone(pictures.kind_of('person'))

    def test_one_answer_per_picture(self):
        found = pictures.build_pictures(
            [det('tank', 0.9, 100, 100, 300, 300), det('military_truck', 0.6, 105, 102, 298, 305)], 640, 480)
        self.assertEqual([p['kind'] for p in found], ['tank'])

    def test_cut_off_at_edge(self):
        found = pictures.build_pictures(
            [det('radar', 0.9, 0, 120, 150, 360), det('comms', 0.9, 300, 120, 450, 360),
             det('outpost', 0.9, 520, 120, 660, 360)], 640, 480)
        self.assertEqual([p['cut_off'] for p in found], [True, False, True])
        self.assertLessEqual(found[2]['x_center'] + found[2]['width'] / 2, 1.0)

    def test_sensor_is_not_a_picture(self):
        found = pictures.build_pictures(
            [det('radar', 0.9, 100, 100, 300, 300), det('target_sensor', 0.95, 190, 105, 210, 125),
             det('antenna', 0.9, 400, 100, 420, 300), det('led_bar', 0.9, 430, 300, 480, 320)], 640, 480)
        self.assertEqual([p['kind'] for p in found], ['radar'])

    def test_sensors_are_told_apart_from_pictures(self):
        """수광부는 그림 목록에는 없고 수광부 목록에만 있다 (자리 · 크기는 사진에 대한 비율, 왼쪽부터)."""
        detections = [det('radar', 0.9, 100, 100, 300, 300), det('target_sensor', 0.95, 190, 130, 210, 150),
                      det('Target-Sensor', 0.4, 500, 200, 512, 212), det('target_sensor', 0.1, 20, 20, 30, 30),
                      det('target_sensor', 0.6, 191, 131, 211, 151), det('antenna', 0.9, 400, 100, 420, 300)]
        self.assertEqual([p['kind'] for p in pictures.build_pictures(detections, 640, 480)], ['radar'])
        sensors = pictures.build_sensors(detections, 640, 480)
        self.assertEqual(len(sensors), 2)                      # 점수 0.1 은 빠지고, 겹친 둘은 하나로
        self.assertAlmostEqual(sensors[0]['x_center'], 200 / 640)
        self.assertAlmostEqual(sensors[0]['y_center'], 140 / 480)
        self.assertAlmostEqual(sensors[0]['width'], 20 / 640)
        self.assertAlmostEqual(sensors[0]['confidence'], 0.95)
        self.assertAlmostEqual(sensors[1]['x_center'], 506 / 640)
        self.assertEqual(pictures.build_sensors([det('tank', 0.9, 10, 10, 90, 90)], 640, 480), [])

    def test_sensors_inside_a_ugv_box_are_left_out(self):
        """UGV 상자 안의 수광부는 UGV 에 달린 것이다 → 지정표적 조준용 목록에서 뺀다 (표적 앞에 선 UGV 의 돔을 겨누지 않게)."""
        detections = [det('radar', 0.9, 200, 60, 440, 300), det('target_sensor', 0.8, 310, 100, 330, 120),
                      det('ugv', 0.9, 260, 280, 420, 420), det('target_sensor', 0.9, 330, 330, 350, 350)]
        ugv_boxes = [d[2:6] for d in detections if d[0] == 'ugv']
        sensors = pictures.build_sensors(detections, 640, 480, exclude_boxes=ugv_boxes)
        self.assertEqual(len(sensors), 1)
        self.assertAlmostEqual(sensors[0]['y_center'], 110 / 480)
        self.assertEqual(len(pictures.build_sensors(detections, 640, 480)), 2)   # 상자를 안 주면 전과 같다

    def test_sensors_text_goes_at_the_end_of_detail(self):
        sensors = pictures.build_sensors([det('target_sensor', 0.954, 190, 130, 210, 150)], 640, 480)
        self.assertEqual(pictures.sensors_text(sensors), 'sensors=[[0.3125,0.2917,0.0312,0.0417,0.95]]')
        self.assertEqual(pictures.sensors_text([]), 'sensors=[]')

    def test_no_pictures(self):
        self.assertEqual(pictures.build_pictures([], 640, 480), [])
        self.assertEqual(pictures.build_pictures([det('tank', 0.9, 10, 10, 10, 50)], 640, 480), [])


if __name__ == '__main__':
    unittest.main()
