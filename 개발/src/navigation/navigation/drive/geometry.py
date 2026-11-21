"""2D 자세 계산. 좌표는 경기장 좌표(arena): x 는 BLUE 벽(0) → RED 벽, y 는 아래(0) → 위, yaw 는 +x 기준 반시계."""
from __future__ import annotations

import math
from typing import NamedTuple


class Pose2D(NamedTuple):
    x: float
    y: float
    yaw: float


def wrap(angle: float) -> float:
    """각도를 (-π, π] 로."""
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quaternion_from_yaw(yaw: float):
    """(x, y, z, w)."""
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def compose(a: Pose2D, b: Pose2D) -> Pose2D:
    """a ⊕ b : a 좌표계에서 b 만큼 더 간 자세."""
    c, s = math.cos(a.yaw), math.sin(a.yaw)
    return Pose2D(a.x + c * b.x - s * b.y, a.y + s * b.x + c * b.y, wrap(a.yaw + b.yaw))


def inverse(a: Pose2D) -> Pose2D:
    c, s = math.cos(a.yaw), math.sin(a.yaw)
    return Pose2D(-c * a.x - s * a.y, s * a.x - c * a.y, wrap(-a.yaw))


def relative(a: Pose2D, b: Pose2D) -> Pose2D:
    """a 에서 본 b (a⁻¹ ⊕ b)."""
    return compose(inverse(a), b)


def to_local(pose: Pose2D, x: float, y: float):
    """세계 점 (x, y) 를 pose 의 차체 좌표(앞 +x, 왼쪽 +y)로."""
    dx, dy = x - pose.x, y - pose.y
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return c * dx + s * dy, -s * dx + c * dy


def advance(pose: Pose2D, v: float, w: float, dt: float) -> Pose2D:
    """명령 (v, w) 로 dt 만큼 간 자세 (반응 지연을 내다볼 때)."""
    if dt <= 0.0:
        return pose
    mid = pose.yaw + w * dt / 2.0
    return Pose2D(pose.x + v * dt * math.cos(mid), pose.y + v * dt * math.sin(mid), wrap(pose.yaw + w * dt))


def path_length(start, points) -> float:
    total, prev = 0.0, start
    for p in points or ():
        total += math.hypot(p[0] - prev[0], p[1] - prev[1])
        prev = p
    return total
