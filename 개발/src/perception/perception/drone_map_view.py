"""드론 화면 → 지도 좌표 그림 (map 그림): 렌즈 왜곡을 펴고 경기장 테두리로 잘라, 상자 윗면 높이 평면으로 1 cm/px 로 편다.

상자 찾기 모델(drone_box_model)이 학습한 그림과 경기 중에 넣는 그림이 **같은 코드로** 만들어지게 여기 한 곳에 둔다
(학습 데이터 전처리 tools/undistort_drone_frames.py --mode map 도 이것을 쓴다).

  view = MapView(boundary, camera_height)      # boundary: drone_boundary.Boundary (ok 인 것)
  img = view.render(bgr)                       # 612 × 500 (경기장 6.12 × 5.00 m 일 때)
  xy = view.pixel_to_field([(u, v), ...])      # 그림 픽셀 → 경기장 (x, y) m = 그 높이 평면의 점 = 상자 바닥 자리

평면 높이 plane_h (기본 0.40 m = 상자 높이)
  카메라(경기장 정중앙 위, 높이 Hc — drone_boxes 와 같은 모델, 현장 확인 2026-11-18)에서 보면 높이 h 인 점은 바닥 지도에서
  카메라 바로 아래 점 N 에서 s = Hc / (Hc − h) 배 먼 곳에 찍힌다. map 그림은 높이 h 평면의 점 P 를 그릴 때 바닥 점 N + (P − N)·s 의
  픽셀을 가져온다 → 상자 윗면이 자기 바닥 자리 바로 위에 40 × 40 px 로 찍힌다. 바닥 그림(선 · 원)은 가운데 쪽으로 1/s 만큼 줄어 보인다.
  그림 좌표: x = u / ppm − margin, y = H + margin − v / ppm (위 = 1번 줄). 시야 밖은 (0, 0, 0).

numpy · OpenCV 만 쓴다. ROS 에 묶이지 않는다.
"""
from __future__ import annotations

import numpy as np

PLANE_H = 0.40  # 상자 높이 (공식 40 cm)
PPM = 100.0  # px / m


class MapView:
    def __init__(self, boundary, camera_height, plane_h=PLANE_H, ppm=PPM, margin=0.0, camera_xy=None):
        if boundary is None or boundary.hom is None:
            raise ValueError("경계(hom)가 없습니다")
        if camera_height is None or camera_height <= plane_h + 0.1:
            raise ValueError("카메라 높이가 이상합니다: %r" % (camera_height,))
        self.boundary = boundary
        w, h, m = float(boundary.field_w), float(boundary.field_h), float(margin)
        self.field_w, self.field_h, self.margin, self.ppm, self.plane_h = w, h, m, float(ppm), float(plane_h)
        cx, cy = camera_xy if camera_xy is not None else (w / 2.0, h / 2.0)
        self.camera = (float(cx), float(cy), float(camera_height))
        self.s = float(camera_height) / (float(camera_height) - float(plane_h))
        self.size = (int(round((w + 2 * m) * ppm)), int(round((h + 2 * m) * ppm)))
        # 그림 픽셀 (u, v, 1) → 경기장 (x, y, 1) — 높이 plane_h 평면
        self.out_to_field = np.array([[1.0 / ppm, 0.0, -m], [0.0, -1.0 / ppm, h + m], [0.0, 0.0, 1.0]])
        self._maps = None

    def floor_of(self, xy):
        """높이 plane_h 평면의 점들 → 같은 픽셀에 찍히는 바닥 점들 (N + (P − N)·s)."""
        p = np.asarray(xy, float).reshape(-1, 2)
        n = np.array(self.camera[:2])
        return n + (p - n) * self.s

    def plane_of_floor(self, xy):
        """floor_of 의 반대: 바닥 지도의 점 → 높이 plane_h 평면의 점 (N + (T − N)/s)."""
        p = np.asarray(xy, float).reshape(-1, 2)
        n = np.array(self.camera[:2])
        return n + (p - n) / self.s

    def pixel_to_field(self, uv):
        p = np.asarray(uv, float).reshape(-1, 2)
        m = self.out_to_field
        return np.stack([p[:, 0] * m[0, 0] + m[0, 2], p[:, 1] * m[1, 1] + m[1, 2]], 1)

    def field_to_pixel(self, xy):
        p = np.asarray(xy, float).reshape(-1, 2)
        return np.stack([(p[:, 0] + self.margin) * self.ppm, (self.field_h + self.margin - p[:, 1]) * self.ppm], 1)

    def maps(self):
        """cv2.remap 표 (처음 부를 때 한 번 만든다)."""
        if self._maps is None:
            ow, oh = self.size
            gu, gv = np.meshgrid(np.arange(ow, dtype=np.float64), np.arange(oh, dtype=np.float64))
            plane = self.pixel_to_field(np.stack([gu.ravel(), gv.ravel()], 1))
            src = self.boundary.field_to_pixel(self.floor_of(plane)).reshape(oh, ow, 2)  # 렌즈 왜곡 포함
            src[~np.isfinite(src)] = -1.0  # 화면 밖 → 테두리 색
            self._maps = (src[..., 0].astype(np.float32), src[..., 1].astype(np.float32))
        return self._maps

    def render(self, bgr):
        import cv2

        mu, mv = self.maps()
        return cv2.remap(bgr, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    def to_dict(self):
        return {"size": list(self.size), "out_to_field": self.out_to_field.tolist(), "plane_h": self.plane_h, "ppm": self.ppm,
                "margin": self.margin, "camera_xyh": list(self.camera), "s": self.s}
