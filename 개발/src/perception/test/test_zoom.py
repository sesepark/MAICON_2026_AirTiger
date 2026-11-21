import unittest

import numpy as np

from perception import pictures, zoom


class CropWindowTest(unittest.TestCase):
    def test_box_is_centered_and_fills_the_given_ratio(self):
        x0, y0, x1, y1 = zoom.crop_window((300, 200, 340, 240), 640, 480, fill=0.4)
        self.assertEqual((x1 - x0, y1 - y0), (100, 100))
        self.assertEqual((x0, y0), (270, 170))

    def test_window_is_pushed_inside_the_image(self):
        self.assertEqual(zoom.crop_window((0, 0, 40, 40), 640, 480, fill=0.4), (0, 0, 100, 100))
        self.assertEqual(zoom.crop_window((620, 460, 640, 480), 640, 480, fill=0.2), (540, 380, 640, 480))

    def test_window_never_exceeds_the_image(self):
        x0, y0, x1, y1 = zoom.crop_window((100, 100, 400, 400), 640, 480, fill=0.4)
        self.assertEqual((y1 - y0, y0), (480, 0))
        self.assertTrue(0 <= x0 and x1 <= 640)


class FakeModel:
    """잘라 낸 사진에서 낼 답을 정해 두는 가짜 모델. 받은 사진 크기를 적어 둔다."""

    def __init__(self, answers):
        self.answers = answers      # 잘라 낸 사진 좌표의 탐지 목록
        self.shapes = []

    def __call__(self, rgb):
        self.shapes.append(rgb.shape)
        return list(self.answers)


IMAGE = np.zeros((480, 640, 3), np.uint8)


class RefineTest(unittest.TestCase):
    def test_small_picture_takes_the_zoomed_answer_and_keeps_its_box(self):
        # 그림 (300, 200)~(340, 240) → 창 (270, 170)~(370, 270), 창 안에서는 (30, 30)~(70, 70)
        model = FakeModel([('tank', 0.9, 31, 29, 70, 71)])
        out = zoom.refine(IMAGE, [('military_truck', 0.3, 300, 200, 340, 240)], model, pictures.is_picture, fill=0.4)
        self.assertEqual(out, [('tank', 0.9, 300, 200, 340, 240)])
        self.assertEqual(model.shapes, [(100, 100, 3)])

    def test_lower_zoomed_score_keeps_the_first_answer(self):
        model = FakeModel([('drum', 0.4, 30, 30, 70, 70)])
        first = [('tank', 0.8, 300, 200, 340, 240)]
        self.assertEqual(zoom.refine(IMAGE, first, model, pictures.is_picture, fill=0.4), first)

    def test_no_picture_at_the_same_place_keeps_the_first_answer(self):
        model = FakeModel([('tank', 0.9, 0, 0, 20, 20), ('ugv', 0.9, 30, 30, 70, 70)])
        first = [('mortar', 0.3, 300, 200, 340, 240)]
        self.assertEqual(zoom.refine(IMAGE, first, model, pictures.is_picture, fill=0.4), first)

    def test_large_pictures_and_non_pictures_are_not_zoomed(self):
        model = FakeModel([('tank', 0.9, 0, 0, 100, 100)])
        first = [('drum', 0.3, 0, 0, 200, 200), ('ugv', 0.3, 300, 200, 340, 240),
                 ('target_sensor', 0.3, 400, 200, 410, 210)]
        self.assertEqual(zoom.refine(IMAGE, first, model, pictures.is_picture, max_height=96), first)
        self.assertEqual(model.shapes, [])

    def test_limit_zooms_the_smallest_first(self):
        model = FakeModel([])
        first = [('tank', 0.3, 0, 0, 90, 90), ('tank', 0.3, 200, 0, 220, 20), ('tank', 0.3, 400, 0, 450, 50)]
        zoom.refine(IMAGE, first, model, pictures.is_picture, limit=2, fill=0.4)
        self.assertEqual(model.shapes, [(64, 64, 3), (125, 125, 3)])

    def test_off_returns_the_same_list(self):
        model = FakeModel([('tank', 0.9, 30, 30, 70, 70)])
        first = [('drum', 0.3, 300, 200, 340, 240)]
        self.assertEqual(zoom.refine(IMAGE, first, model, pictures.is_picture, max_height=0), first)
        self.assertEqual(model.shapes, [])


if __name__ == '__main__':
    unittest.main()
