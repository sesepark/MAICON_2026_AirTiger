#!/usr/bin/env python3
"""부품 모델 v2 학습: pictures_m6_v1 에서 이어서 10 epoch (기존 m6 학습 목록 + 실제 경기 사진 자동 라벨 3번 반복)."""
import os
import shutil

from ultralytics import YOLO

if __name__ == '__main__':
    os.chdir(os.path.expanduser('~/maicon-m3-model'))
    m = YOLO('weights/pictures_m6_v1.pt')
    m.train(data='datasets/m6v2/data.yaml', epochs=10, imgsz=640, batch=16, lr0=0.002, close_mosaic=3, degrees=5.0, translate=0.15, scale=0.6,
            hsv_h=0.02, hsv_s=0.7, hsv_v=0.5, mixup=0.1, perspective=0.0005, erasing=0.3,
            project='/home/admin/maicon-m3-model/runs/pictures', name='m6_v2', exist_ok=True, device=0, seed=0)
    shutil.copy('runs/pictures/m6_v2/weights/best.pt', 'weights/pictures_m6_v2.pt')
    print('TRAIN_DONE')
