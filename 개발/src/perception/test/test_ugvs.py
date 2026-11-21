import unittest

from perception import pictures, ugvs


def det(name, conf, x1, y1, x2, y2):
    return (name, conf, x1, y1, x2, y2)


class BuildUgvsTest(unittest.TestCase):
    def test_only_ugvs_left_to_right(self):
        found = ugvs.build_ugvs(
            [det('ugv', 0.9, 400, 200, 560, 360), det('tank', 0.95, 0, 0, 200, 200),
             det('target_sensor', 0.9, 90, 5, 110, 25), det('UGV', 0.8, 80, 240, 240, 360)], 640, 480)
        self.assertEqual(len(found), 2)
        self.assertAlmostEqual(found[0]['x_center'], 0.25)
        self.assertAlmostEqual(found[0]['y_center'], 0.625)
        self.assertAlmostEqual(found[0]['width'], 0.25)
        self.assertAlmostEqual(found[0]['height'], 0.25)
        self.assertAlmostEqual(found[1]['confidence'], 0.9)

    def test_unread_values_are_not_made_up(self):
        found = ugvs.build_ugvs([det('ugv', 0.9, 100, 100, 300, 300)], 640, 480)
        self.assertEqual((found[0]['camp'], found[0]['callsign']), ('', ''))
        self.assertEqual((found[0]['lives'], found[0]['ammo']), (-1, -1))

    def test_low_score_is_not_a_ugv(self):
        self.assertEqual(ugvs.build_ugvs([det('ugv', 0.2, 100, 100, 300, 300)], 640, 480), [])

    def test_one_answer_per_ugv(self):
        found = ugvs.build_ugvs(
            [det('ugv', 0.9, 100, 100, 300, 300), det('ugv', 0.6, 105, 102, 298, 305)], 640, 480)
        self.assertEqual(len(found), 1)

    def test_no_ugvs(self):
        self.assertEqual(ugvs.build_ugvs([], 640, 480), [])
        self.assertEqual(ugvs.build_ugvs([det('tank', 0.9, 0, 0, 100, 100)], 640, 480), [])

    # ---- 색으로 읽기 (합성 영상) ----
    @staticmethod
    def scene(sleeve, bands, led=None, sleeve_h=60):
        """640x480 회색 배경에 UGV 한 대: 안테나 덮개(색 · 검은 띠 수)와 LED 막대(색)를 그린다."""
        import numpy as np
        img = np.full((480, 640, 3), 120, np.uint8)
        img[200:400, 250:400] = 30                                 # 차체
        img[20:60, 0:640] = (130, 170, 220)                        # 뒤쪽의 하늘 그림 (옅은 파랑 — 덮개로 세면 안 된다)
        x, y = 262, 110
        img[y:y + sleeve_h, x:x + 8] = sleeve
        # 실제 덮개처럼: 띠는 덮개 위쪽에 모여 있고 가늘다 (덮개 길이의 5 % 쯤, 10 % 간격)
        thick = max(2, sleeve_h // 20)
        for k in range(bands):
            y0 = y + int(sleeve_h * (0.25 + 0.10 * k))
            img[y0:y0 + thick, x:x + 8] = 15
        dets = [det('ugv', 0.9, 250, 90, 400, 400), det('antenna', 0.6, x - 5, y - 30, x + 13, y + sleeve_h + 10)]
        if led is not None:
            img[360:372, 290:360] = 250                            # 하얗게 번진 가운데
            img[356:360, 286:364] = led; img[372:376, 286:364] = led
            dets.append(det('led_bar', 0.6, 284, 354, 366, 378))
        return img, dets

    def test_red_three_bands_blue_led(self):
        img, dets = self.scene((215, 35, 45), 3, led=(40, 90, 235))
        u = ugvs.build_ugvs(dets, 640, 480, rgb=img)[0]
        self.assertEqual((u['camp'], u['callsign'], u['lives'], u['ammo']), ('RED', 'R-3', 4, -1))

    def test_blue_one_band_green_led(self):
        img, dets = self.scene((30, 60, 220), 1, led=(60, 230, 90))
        u = ugvs.build_ugvs(dets, 640, 480, rgb=img)[0]
        self.assertEqual((u['camp'], u['callsign'], u['lives']), ('BLUE', 'B-1', 3))

    def test_led_colors_to_lives(self):
        for color, lives in (((240, 60, 50), 0), ((245, 150, 40), 1), ((235, 220, 50), 2)):
            img, dets = self.scene((215, 35, 45), 2, led=color)
            self.assertEqual(ugvs.build_ugvs(dets, 640, 480, rgb=img)[0]['lives'], lives, color)

    def test_small_sleeve_gives_camp_but_no_callsign(self):
        img, dets = self.scene((215, 35, 45), 2, sleeve_h=24)
        u = ugvs.build_ugvs(dets, 640, 480, rgb=img)[0]
        self.assertEqual((u['camp'], u['callsign']), ('RED', ''))

    def test_sleeve_is_found_by_color_when_the_model_misses_the_antenna(self):
        img, dets = self.scene((215, 35, 45), 2, led=(40, 90, 235))
        u = ugvs.build_ugvs([d for d in dets if d[0] == 'ugv'], 640, 480, rgb=img)[0]
        self.assertEqual((u['camp'], u['callsign']), ('RED', 'R-2'))
        self.assertEqual((u['lives'], u['ammo']), (4, -1))           # LED 막대도 밝은 막대 + 색 번짐으로 찾는다

    def test_led_bar_is_found_by_color_for_every_lives_color(self):
        for color, lives in (((40, 90, 235), 4), ((60, 230, 90), 3), ((235, 220, 50), 2), ((245, 150, 40), 1), ((240, 60, 50), 0)):
            img, dets = self.scene((215, 35, 45), 2, led=color)
            u = ugvs.build_ugvs([d for d in dets if d[0] == 'ugv'], 640, 480, rgb=img)[0]
            self.assertEqual(u['lives'], lives, color)

    def test_things_that_are_not_the_led_bar(self):
        import numpy as np
        img = np.full((480, 640, 3), 120, np.uint8)
        img[200:400, 250:400] = 30
        img[350:372, 290:360] = 245                                  # 앞면의 흰 이름판: 밝지만 둘레에 색 번짐이 없다
        self.assertEqual(ugvs.build_ugvs([det('ugv', 0.9, 250, 90, 400, 400)], 640, 480, rgb=img)[0]['lives'], -1)
        img[350:372, 290:360] = 30
        yy, xx = np.ogrid[:480, :640]                                # 수광 돔: 흰 공에 초록빛 테두리 (둥글다), 박스 아래쪽에 걸침
        img[(yy - 270) ** 2 + (xx - 300) ** 2 <= 26 ** 2] = (90, 230, 110)
        img[(yy - 270) ** 2 + (xx - 300) ** 2 <= 22 ** 2] = 250
        self.assertEqual(ugvs.build_ugvs([det('ugv', 0.9, 250, 90, 400, 400)], 640, 480, rgb=img)[0]['lives'], -1)
        img[200:400, 250:400] = 30
        img[340:356, 280:350] = (250, 205, 40)                       # 뒤의 노란 장애물 조각: 가운데가 하얗게 번지지 않았다
        self.assertEqual(ugvs.build_ugvs([det('ugv', 0.9, 250, 90, 400, 400)], 640, 480, rgb=img)[0]['lives'], -1)

    def test_wide_or_round_color_patches_are_not_sleeves(self):
        import numpy as np
        img = np.full((480, 640, 3), 120, np.uint8)
        img[200:400, 250:400] = 30
        img[120:180, 270:330] = (30, 60, 220)                        # 뒤에 선 사람의 파란 옷 (넓다)
        img[100:112, 300:312] = (230, 30, 30)                        # 경광등 (둥글다)
        img[360:368, 280:370] = (230, 30, 30)                        # 빨간 LED 막대 (가로로 길다)
        u = ugvs.build_ugvs([det('ugv', 0.9, 250, 90, 400, 400)], 640, 480, rgb=img)[0]
        self.assertEqual((u['camp'], u['callsign']), ('', ''))

    def test_tilted_sleeve_is_still_counted(self):
        import numpy as np
        img = np.full((480, 640, 3), 120, np.uint8)
        img[200:400, 250:400] = 30
        for i in range(70):                                          # 오른쪽으로 기운 파란 덮개, 띠 3개
            x = 270 + int(0.12 * i)
            dark = any(int(70 * f) <= i < int(70 * f) + 3 for f in (0.25, 0.35, 0.45))
            img[110 + i, x:x + 8] = 15 if dark else (30, 60, 220)
        u = ugvs.build_ugvs([det('ugv', 0.9, 250, 90, 400, 400)], 640, 480, rgb=img)[0]
        self.assertEqual((u['camp'], u['callsign']), ('BLUE', 'B-3'))

    def test_white_led_is_not_read(self):
        img, dets = self.scene((215, 35, 45), 2, led=(250, 250, 250))
        self.assertEqual(ugvs.build_ugvs(dets, 640, 480, rgb=img)[0]['lives'], -1)

    def test_parts_of_another_ugv_are_not_used(self):
        img, dets = self.scene((215, 35, 45), 2)
        far = [det('ugv', 0.9, 500, 300, 600, 420)]
        both = ugvs.build_ugvs(dets + far, 640, 480, rgb=img)
        self.assertEqual([u['camp'] for u in both], ['RED', ''])

    def test_ugv_is_never_a_picture(self):
        found = pictures.build_pictures(
            [det('ugv', 0.9, 400, 200, 560, 360), det('drum', 0.9, 80, 120, 240, 360)], 640, 480)
        self.assertEqual([p['kind'] for p in found], ['drum'])


if __name__ == '__main__':
    unittest.main()
