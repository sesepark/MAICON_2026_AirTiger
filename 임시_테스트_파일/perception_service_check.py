#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""로봇에서 perception 노드의 두 서비스를 저장된 사진으로 불러 보는 점검 도구.

perception_node 가 켜져 있어야 한다. 사진 폴더(img/*.jpg)와 YOLO 라벨 폴더(lbl/*.txt)를 주면
사진마다 classify_image · read_ugvs 를 부르고, 라벨과 견준 결과를 한 줄씩 적는다.

    python3 perception_service_check.py /경로/rt        # rt/img, rt/lbl
"""
import glob
import os
import sys
import time

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from team_interfaces.msg import NodeStatus
from team_interfaces.srv import ClassifyImage, ReadUgvs

NAMES = ['military_truck', 'mortar', 'tank', 'missile', 'ammo_box', 'drum', 'comms', 'outpost', 'radar', 'non_object',
         'target_sensor', 'ugv', 'antenna', 'led_bar', 'ugv_sensor']
PICTURES = set(NAMES[:10])


def to_image(bgr, node):
    message = Image()
    message.header.stamp = node.get_clock().now().to_msg()
    message.height, message.width = bgr.shape[:2]
    message.encoding = 'rgb8'
    message.step = message.width * 3
    message.data = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).tobytes()
    return message


def call(node, client, request, timeout=20.0):
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    return future.result()


def main():
    root = sys.argv[1]
    rclpy.init()
    node = Node('perception_service_check')
    status = {}
    node.create_subscription(NodeStatus, '/team/node_status',
                             lambda m: status.__setitem__(m.node_name, (m.state, m.detail)), 10)
    classify = node.create_client(ClassifyImage, '/team/perception/classify_image')
    read_ugvs = node.create_client(ReadUgvs, '/team/perception/read_ugvs')
    if not classify.wait_for_service(timeout_sec=20.0) or not read_ugvs.wait_for_service(timeout_sec=5.0):
        print('SERVICES_NOT_FOUND')
        return 1
    deadline = time.monotonic() + 120.0          # 엔진 두 개를 읽을 때까지 기다린다
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.5)
        if status.get('perception', (NodeStatus.INITIALIZING, ''))[0] != NodeStatus.INITIALIZING:
            break
    print('STATUS', status.get('perception'))

    total = hit = extra = frames_exact = n_frames = 0
    times_c, times_u = [], []
    for path in sorted(glob.glob(os.path.join(root, 'img', '*.jpg'))):
        bgr = cv2.imread(path)
        expected, n_ugv = [], 0
        label = os.path.join(root, 'lbl', os.path.basename(path)[:-4] + '.txt')
        for line in open(label):
            p = line.split()
            if len(p) < 5:
                continue
            name = NAMES[int(p[0])]
            if name in PICTURES:
                expected.append((float(p[1]), name))
            elif name == 'ugv':
                n_ugv += 1
        expected = [n for _, n in sorted(expected)]

        request = ClassifyImage.Request()
        request.run_id = 'check'
        request.image = to_image(bgr, node)
        t = time.monotonic()
        c = call(node, classify, request)
        times_c.append(time.monotonic() - t)
        request = ReadUgvs.Request()
        request.run_id = 'check'
        request.image = to_image(bgr, node)
        t = time.monotonic()
        u = call(node, read_ugvs, request)
        times_u.append(time.monotonic() - t)
        if c is None or u is None:
            print('%-34s NO_ANSWER classify=%s read_ugvs=%s' % (os.path.basename(path), c is not None, u is not None))
            continue
        got = [p.kind for p in c.pictures]
        rest = list(got)
        for kind in expected:
            total += 1
            if kind in rest:
                rest.remove(kind)
                hit += 1
        extra += len(rest)
        n_frames += 1
        frames_exact += int(got == expected)
        print('%-34s pictures want %s got %s %s | ugv want %d got %s (%s)' % (
            os.path.basename(path), expected, ['%s:%.2f' % (p.kind, p.confidence) for p in c.pictures],
            'OK' if got == expected else 'DIFF', n_ugv,
            [(s.camp, s.callsign, s.lives, s.ammo, round(s.confidence, 2)) for s in u.ugvs], u.detail if u.ok else 'FAILED ' + u.detail))
    if n_frames:
        print('SUMMARY frames %d | pictures found %d/%d, not in labels %d, frames exact %d | call ms classify %.0f (first %.0f) read_ugvs %.0f (first %.0f)' % (
            n_frames, hit, total, extra, frames_exact,
            1000 * sum(times_c[1:]) / max(1, len(times_c) - 1), 1000 * times_c[0],
            1000 * sum(times_u[1:]) / max(1, len(times_u) - 1), 1000 * times_u[0]))
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
