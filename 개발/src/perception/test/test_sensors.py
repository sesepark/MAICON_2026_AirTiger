"""LocateSensors 응답 계산(perception/sensors.py) 시험. ROS · 모델 없이 돈다."""
import unittest

from perception import sensors

W, H = 640, 480


def det(name, conf, cx, cy, w, h):
    return (name, conf, cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


class BuildSensorsTests(unittest.TestCase):
    def test_sensor_by_a_designated_target_picture_is_a_target_sensor(self):
        found = sensors.build_sensors([det('radar', 0.9, 320, 260, 160, 160), det('target_sensor', 0.7, 320, 165, 20, 20)], W, H)
        self.assertEqual(len(found), 1)
        s = found[0]
        self.assertEqual((s['owner'], s['host_kind'], s['has_host']), ('target', 'radar', True))
        self.assertAlmostEqual(s['x_center'], 0.5)
        self.assertAlmostEqual(s['host_width'], 0.25)

    def test_sensor_inside_a_ugv_box_is_a_ugv_sensor_whatever_the_model_called_it(self):
        found = sensors.build_sensors([det('ugv', 0.8, 200, 300, 150, 120), det('target_sensor', 0.6, 210, 310, 16, 16)], W, H)
        self.assertEqual((found[0]['owner'], found[0]['host_kind']), ('ugv', 'ugv'))

    def test_ugv_in_front_of_a_target_keeps_the_two_sensors_apart(self):
        found = sensors.build_sensors([det('comms', 0.9, 320, 200, 200, 200), det('target_sensor', 0.7, 320, 120, 18, 18),
                                       det('ugv', 0.8, 330, 330, 160, 120), det('target_sensor', 0.6, 335, 335, 16, 16)], W, H)
        owners = sorted((s['owner'], s['host_kind']) for s in found)
        self.assertEqual(owners, [('target', 'comms'), ('ugv', 'ugv')])

    def test_lonely_sensor_follows_the_model_class(self):
        found = sensors.build_sensors([det('target_sensor', 0.7, 100, 100, 20, 20)], W, H)
        self.assertEqual((found[0]['owner'], found[0]['has_host'], found[0]['host_kind']), ('target', False, ''))
        found = sensors.build_sensors([], W, H, part_detections=[det('ugv_sensor', 0.7, 100, 100, 20, 20)])
        self.assertEqual(found[0]['owner'], 'unknown')

    def test_parts_model_adds_ugv_sensors_and_duplicates_are_merged(self):
        pictures = [det('ugv', 0.8, 200, 300, 150, 120), det('target_sensor', 0.5, 210, 310, 16, 16)]
        parts = [det('ugv', 0.9, 200, 300, 150, 120), det('ugv_sensor', 0.8, 211, 311, 16, 16)]
        found = sensors.build_sensors(pictures, W, H, part_detections=parts)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['owner'], 'ugv')
        self.assertAlmostEqual(found[0]['confidence'], 0.8)

    def test_low_confidence_is_dropped_and_order_is_left_to_right(self):
        found = sensors.build_sensors([det('target_sensor', 0.1, 50, 50, 20, 20), det('target_sensor', 0.6, 500, 50, 20, 20),
                                       det('target_sensor', 0.6, 300, 50, 20, 20)], W, H)
        self.assertEqual([round(s['x_center'] * W) for s in found], [300, 500])

    def test_no_sensor_is_an_empty_list(self):
        self.assertEqual(sensors.build_sensors([det('tank', 0.9, 320, 240, 100, 100)], W, H), [])


if __name__ == '__main__':
    unittest.main()
