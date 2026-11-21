"""라이다 한 번으로 시험용 지도를 만든다 (드론 사진 · perception 이 없는 모의 배치에서 SLAM · 주행을 시험할 때).

경기에서는 지도를 perception 이 드론 사진으로 만든다. 여기서는 사람이 로봇 자세를 대충 찍어 주면
  1. fit_walls      : 벽·상자가 경기장 축에 나란하다는 것과 외벽 위치로 자세를 다듬고
  2. suggest_boxes  : 벽으로 설명되지 않는 라이다 점에 맞는 상자(정사각형)를 찾아 준다.
한 자리에서 보이는 면만으로 정하므로 가려진 상자는 못 찾고, 면이 조금만 보이는 상자는 자리가 어긋날 수 있다.
로봇이 움직여 새 면이 보이면 다시 불러 더한다.
"""
import math

import numpy as np

LIDAR_X = 0.0398   # base_link → base_lidar_link


def scan_to_body(ranges, angle_min, increment, range_min, range_max, limit=12.0):
    """LaserScan → 차체 좌표 점 (P, 2). 노드와 같은 장착 변환."""
    r = np.asarray(ranges, dtype=float)
    angles = angle_min + np.arange(len(r)) * increment
    keep = np.isfinite(r) & (r >= range_min) & (r < min(range_max, limit))
    r, angles = r[keep], angles[keep]
    return np.stack((LIDAR_X - r * np.sin(angles), r * np.cos(angles)), axis=1)


def to_world(points, pose):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack((x + c * points[:, 0] - s * points[:, 1], y + s * points[:, 0] + c * points[:, 1]), axis=1)


def _alignment(points, yaw):
    """점들이 축에 나란한 선 위에 모여 있을수록 큰 값."""
    c, s = math.cos(yaw), math.sin(yaw)
    total = 0.0
    for values in (c * points[:, 0] - s * points[:, 1], s * points[:, 0] + c * points[:, 1]):
        counts = np.bincount(((values - values.min()) / 0.03).astype(int))
        total += float((counts.astype(float) ** 2).sum())
    return total


def fit_walls(points, guess, width, height, yaw_window=math.radians(30), shift=0.6):
    """사람이 찍은 자세 guess 를 다듬는다. (x, y, yaw), 그리고 무엇으로 맞췄는지 설명."""
    gx, gy, gyaw = guess
    best = max((gyaw + math.radians(k * 0.5) for k in range(-int(math.degrees(yaw_window) * 2), int(math.degrees(yaw_window) * 2) + 1)),
               key=lambda yaw: _alignment(points, yaw))
    c, s = math.cos(best), math.sin(best)
    rotated = (c * points[:, 0] - s * points[:, 1], s * points[:, 0] + c * points[:, 1])
    out, notes = [], []
    for values, start, size, name in ((rotated[0], gx, width, 'x'), (rotated[1], gy, height, 'y')):
        offsets = start + np.arange(-shift, shift + 1e-9, 0.01)
        world = values[None, :] + offsets[:, None]
        on_wall = (np.abs(world) < 0.04) | (np.abs(world - size) < 0.04)
        outside = (world < -0.08) | (world > size + 0.08)
        score = on_wall.sum(axis=1) - 3 * outside.sum(axis=1)
        top = score.max()
        candidates = offsets[score >= top - 0.5]
        choice = float(candidates[np.argmin(np.abs(candidates - start))])
        hits = int(on_wall[np.argmin(np.abs(offsets - choice))].sum())
        notes.append('%s: 벽에 맞은 점 %d개%s' % (name, hits, '' if hits >= 6 else ' (벽이 잘 안 보여 찍은 자리를 믿음)'))
        out.append(choice)
    return (out[0], out[1], math.atan2(math.sin(best), math.cos(best))), ' · '.join(notes)


def _structure_distance(world, boxes, half, width, height):
    distance = np.minimum(np.minimum(np.abs(world[:, 0]), np.abs(width - world[:, 0])),
                          np.minimum(np.abs(world[:, 1]), np.abs(height - world[:, 1])))
    for bx, by in boxes:
        ax, ay = np.abs(world[:, 0] - bx) - half, np.abs(world[:, 1] - by) - half
        outside = np.hypot(np.maximum(ax, 0.0), np.maximum(ay, 0.0))
        inside = np.minimum(-ax, -ay)
        distance = np.minimum(distance, np.where((ax <= 0) & (ay <= 0), inside, outside))
    return distance


def _rays_through(origin, world, centre, half):
    """origin → 각 점으로 가는 광선 중, 끝점에 닿기 전에 상자(조금 줄인 것) 안을 지나는 것의 수."""
    direction = world - origin[None, :]
    length = np.hypot(direction[:, 0], direction[:, 1])
    near = np.zeros(len(world))
    far = np.maximum(0.0, 1.0 - 0.08 / np.maximum(length, 1e-6))   # 끝 8 cm 는 빼고 본다
    shrunk = half - 0.05
    with np.errstate(divide='ignore', invalid='ignore'):
        for axis in (0, 1):
            a = (centre[axis] - shrunk - origin[axis]) / direction[:, axis]
            b = (centre[axis] + shrunk - origin[axis]) / direction[:, axis]
            parallel = np.abs(direction[:, axis]) < 1e-9
            inside = abs(origin[axis] - centre[axis]) < shrunk
            low = np.where(parallel, -np.inf if inside else np.inf, np.minimum(a, b))
            high = np.where(parallel, np.inf if inside else -np.inf, np.maximum(a, b))
            near, far = np.maximum(near, low), np.minimum(far, high)
    return int((near < far).sum())


def suggest_boxes(points, pose, width, height, box_size, known=(), step=0.05, minimum=5, limit=40):
    """벽·알고 있는 상자로 설명되지 않는 점에 맞는 상자 중심 [(x, y)] 을 찾는다.

    상자마다: 라이다 쪽을 향한 면 위에 점이 minimum 개 넘게 있고, 어떤 광선도 그 상자 속을 지나가지 않아야 한다.
    """
    half = box_size / 2
    world = to_world(points, pose)
    origin = np.array([pose[0] + LIDAR_X * math.cos(pose[2]), pose[1] + LIDAR_X * math.sin(pose[2])])
    inside_arena = (world[:, 0] > -0.05) & (world[:, 0] < width + 0.05) & (world[:, 1] > -0.05) & (world[:, 1] < height + 0.05)
    remaining = inside_arena & (_structure_distance(world, list(known), half, width, height) > 0.07)
    xs = np.arange(0.0, width + 1e-9, step)
    ys = np.arange(0.0, height + 1e-9, step)
    centres = np.array([(x, y) for x in xs for y in ys])
    # 로봇이 서 있는 자리와 겹치는 상자는 없다
    centres = centres[np.maximum(np.abs(centres[:, 0] - pose[0]), np.abs(centres[:, 1] - pose[1])) > half + 0.12]
    dx = world[None, :, 0] - centres[:, None, 0]
    dy = world[None, :, 1] - centres[:, None, 1]
    on_x_face = (np.abs(np.abs(dx) - half) < 0.035) & (np.abs(dy) < half + 0.035)
    on_y_face = (np.abs(np.abs(dy) - half) < 0.035) & (np.abs(dx) < half + 0.035)
    # 그 면이 라이다 쪽을 향해야 보인다
    lidar_dx = origin[0] - centres[:, 0]
    lidar_dy = origin[1] - centres[:, 1]
    support = (on_x_face & (np.sign(dx) == np.sign(lidar_dx)[:, None]) & (np.abs(lidar_dx)[:, None] > half)) \
        | (on_y_face & (np.sign(dy) == np.sign(lidar_dy)[:, None]) & (np.abs(lidar_dy)[:, None] > half))
    found = []
    for _ in range(limit):
        counts = (support & remaining[None, :]).sum(axis=1)
        order = np.argsort(-counts)[:60]
        chosen = None
        for index in order:
            if counts[index] < minimum:
                break
            if _rays_through(origin, world, centres[index], half) <= 1:
                chosen = index
                break
        if chosen is None:
            break
        cx, cy = centres[chosen]
        found.append((round(float(cx), 3), round(float(cy), 3)))
        remaining &= ~(support[chosen] | ((np.abs(dx[chosen]) < half + 0.05) & (np.abs(dy[chosen]) < half + 0.05)))
        # 방금 고른 상자와 겹치는 후보는 뺀다
        overlap = (np.abs(centres[:, 0] - cx) < box_size - 0.05) & (np.abs(centres[:, 1] - cy) < box_size - 0.05)
        support[overlap] = False
    return found
