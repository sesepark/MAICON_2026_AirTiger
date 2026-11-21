"""Replay recorded aerial, odometry and IMU callbacks without commanding hardware.

Use an isolated ROS_DOMAIN_ID. Field observations in straight04 are synthetic.
Scans are intentionally excluded because its mock map has no measured structures.
"""
import argparse
import json
import math
import sqlite3
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray
from team_interfaces.msg import FieldMap
from rclpy.serialization import deserialize_message

from navigation.slam_node import SlamNode


class CapturedPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    args = parser.parse_args()
    inputs = {
        "/odom": (Odometry, "_receive_odometry"),
        "/imu/data": (Imu, "_receive_imu"),
        "/odom/odom_raw": (Float32MultiArray, "_on_wheels"),
        "/slam_trial/field": (FieldMap, "_receive_perception_map"),
        "/team/perception/field_map": (FieldMap, "_receive_perception_map"),
    }
    records = []
    for path in sorted(args.bag.glob("*.db3")):
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            records.extend(db.execute(
                "SELECT timestamp,name,data FROM messages JOIN topics "
                "ON topics.id=messages.topic_id ORDER BY timestamp"
            ))
    if not records:
        parser.error("bag has no recorded SQLite messages")
    rclpy.init()
    node = SlamNode()
    node.map_publisher = CapturedPublisher()
    node.pose_publisher = CapturedPublisher()
    # 방향 소스: 바퀴 누적 거리(/odom/odom_raw)가 기록돼 있으면 노드 기본(wheel), 없으면(예: straight04) IMU.
    if hasattr(node, "yaw_source") and not any(row[1] == "/odom/odom_raw" for row in records):
        node.yaw_source = "imu"
    counts = {}
    try:
        for timestamp, topic, data in sorted(records, key=lambda row: row[0]):
            if topic not in inputs:
                continue
            kind, callback = inputs[topic]
            if not hasattr(node, callback):
                continue
            message = deserialize_message(data, kind)
            if callback == "_on_wheels":
                # odom_raw 에는 header 가 없다. 노드는 받은 시각을 쓰므로 기록된 수신 시각을 넘긴다.
                if len(message.data) < 2:
                    continue
                node._on_wheels(timestamp, float(message.data[0]), float(message.data[1]))
            else:
                getattr(node, callback)(message)
            counts[topic] = counts.get(topic, 0) + 1
        poses = node.pose_publisher.messages
        report = {
            "heading_source": getattr(node, "yaw_source", "imu"),
            "input_counts": counts,
            "pose_count": len(poses),
            "note": "Direct timestamped callbacks; synthetic map and no scan matching.",
        }
        if poses:
            first, last = poses[0], poses[-1]
            report.update(
                first_stamp_ns=first.header.stamp.sec * 10**9 + first.header.stamp.nanosec,
                last_stamp_ns=last.header.stamp.sec * 10**9 + last.header.stamp.nanosec,
                first_xy=[first.pose.position.x, first.pose.position.y],
                last_xy=[last.pose.position.x, last.pose.position.y],
                published_displacement_m=math.hypot(
                    last.pose.position.x - first.pose.position.x,
                    last.pose.position.y - first.pose.position.y,
                ),
                odom_axis_rotation_rad=getattr(node, "_delta", None),
            )
        print(json.dumps(report, indent=2))
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
