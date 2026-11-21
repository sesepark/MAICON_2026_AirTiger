"""그림 탐지 모델 감싸개. ultralytics(YOLO) 가중치(.engine 또는 .pt)를 읽어 사진 한 장을 추론한다.

torch · ultralytics 는 load() 안에서만 불러온다 — 로봇에 없으면 노드는 뜨되 상태가 ERROR 가 된다.
"""
import os


class PictureDetector:
    def __init__(self, model_path, imgsz=640, detect_conf=0.15, device='0'):
        self.model_path = model_path
        self.imgsz = imgsz
        self.detect_conf = detect_conf    # 낮게 잡아 그림을 놓치지 않는다. 낮은 점수는 unknown 으로 나간다
        self.device = device
        self.model = None
        self.names = {}

    def load(self):
        """모델을 읽고 한 번 돌려 둔다(첫 요청이 느려지지 않게). 실패하면 예외."""
        path = self.model_path
        if not os.path.isfile(path):          # .engine 이 없으면 같은 이름의 .pt
            alt = os.path.splitext(path)[0] + '.pt'
            if not os.path.isfile(alt):
                raise FileNotFoundError('model file not found: %s' % path)
            path = self.model_path = alt
        import numpy as np
        from ultralytics import YOLO
        model = YOLO(path, task='detect')
        names = model.names
        self.names = dict(names) if isinstance(names, dict) else dict(enumerate(names))
        model.predict(np.zeros((480, 640, 3), np.uint8), imgsz=self.imgsz, device=self.device, verbose=False)
        self.model = model

    def detect(self, rgb):
        """rgb: HxWx3 uint8 (RGB). -> [(class_name, confidence, x1, y1, x2, y2)] 픽셀 좌표."""
        bgr = rgb[:, :, ::-1]                 # ultralytics 는 numpy 입력을 BGR 로 본다
        result = self.model.predict(bgr, imgsz=self.imgsz, conf=self.detect_conf, device=self.device,
                                    agnostic_nms=True, verbose=False)[0]
        if result.boxes is None:
            return []
        return [(self.names.get(int(c), str(int(c))), float(p), b[0], b[1], b[2], b[3])
                for c, p, b in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist())]
