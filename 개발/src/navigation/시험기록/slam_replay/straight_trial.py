"""One supervised, 10 cm straight-drive trial on a cleared practice course."""

import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan
from team_interfaces.action import Navigate
from team_interfaces.msg import ControlState, FieldMap, FieldVehicle, NodeStatus


def main():
    rclpy.init()
    node = rclpy.create_node('straight_trial_operator')
    run = 'straight-trial'
    latest = {}
    for message_type, topic, key, qos in (
        (Odometry, '/odom', 'odom', qos_profile_sensor_data),
        (LaserScan, '/scan', 'scan', qos_profile_sensor_data),
        (Imu, '/imu/data', 'imu', qos_profile_sensor_data),
        (PoseStamped, '/slam_trial/pose', 'pose', 5),
        (Twist, '/slam_trial/cmd_vel', 'cmd', 5),
        (NodeStatus, '/slam_trial/status', 'status', 5),
    ):
        node.create_subscription(
            message_type, topic,
            lambda msg, key=key: latest.update({key: (time.monotonic(), msg)}), qos,
        )
    aerial = node.create_publisher(FieldMap, '/slam_trial/field', 1)
    control = node.create_publisher(ControlState, '/slam_trial/control', 5)
    commands = node.create_publisher(Twist, '/cmd_vel', 1)
    client = ActionClient(node, Navigate, '/team/navigation/navigate')
    start_time = None

    def wait_until(predicate, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.02)
            if predicate():
                return True
        return False

    def permit(allowed):
        message = ControlState()
        message.header.stamp = node.get_clock().now().to_msg()
        message.run_id = run
        message.motion_allowed = allowed
        message.lease_ms = 150
        control.publish(message)

    try:
        if not wait_until(
            lambda: all(k in latest for k in ('odom', 'scan', 'imu'))
            and aerial.get_subscription_count() > 0 and client.server_is_ready(), 8,
        ):
            raise RuntimeError(f'Inputs={list(latest)}, map subscribers={aerial.get_subscription_count()}, action={client.server_is_ready()}')
        start_pose = latest['odom'][1].pose.pose
        start = start_pose.position
        q = latest['imu'][1].orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
        observed = FieldMap(width=6.0, height=6.0, box_size=0.4)
        observed.header.frame_id = 'arena'
        observed.header.stamp = latest['odom'][1].header.stamp
        own = FieldVehicle(role=FieldVehicle.SELF)
        own.pose.position.x = own.pose.position.y = 3.0
        own.pose.orientation.w = 1.0
        observed.vehicles = [own]
        aerial.publish(observed)
        if not wait_until(lambda: 'pose' in latest, 0.5):
            raise RuntimeError('SLAM did not publish a pose from the mock map')
        if not wait_until(lambda: 'status' in latest
                          and latest['status'][1].state == NodeStatus.READY, 1.2):
            raise RuntimeError('Driving did not confirm map and pose readiness')
        observed.header.stamp = latest['odom'][1].header.stamp
        aerial.publish(observed)
        goal = Navigate.Goal()
        goal.run_id = run
        goal.goal.header.frame_id = 'arena'
        goal.goal.pose.position.x = 3.10
        goal.goal.pose.position.y = 3.0
        goal.goal.pose.orientation.w = 1.0
        goal.mode = Navigate.Goal.MODE_STRAIGHT
        goal.ignore_yaw = True
        goal.position_tolerance_m = 0.035
        goal.yaw_tolerance_rad = 0.2
        goal.execution_timeout_ms = 3200
        permit(True)
        request = client.send_goal_async(
            goal, feedback_callback=lambda item: print('feedback', item.feedback.phase,
                item.feedback.remaining_distance_m, flush=True),
        )
        if not wait_until(request.done, 0.35) or not request.result().accepted:
            raise RuntimeError('Navigate goal was not accepted')
        result = request.result().get_result_async()
        start_time = time.monotonic()
        last_forward = start_time - 0.06
        while time.monotonic() - start_time < 3.0 and not result.done():
            rclpy.spin_once(node, timeout_sec=0.02)
            permit(True)
            now = time.monotonic()
            if any(now - latest[key][0] > 0.3 for key in ('scan', 'odom', 'pose', 'imu')):
                print('stopping: stale sensor or pose', flush=True)
                commands.publish(Twist())
                break
            position = latest['odom'][1].pose.pose.position
            distance = math.hypot(position.x - start.x, position.y - start.y)
            q = latest['imu'][1].orientation
            heading = math.atan2(2 * (q.w * q.z + q.x * q.y),
                                 1 - 2 * (q.y**2 + q.z**2))
            turn = math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))
            if distance > 0.18 or abs(turn) > math.radians(10):
                print(f'stopping: distance={distance:.3f} m, IMU turn={math.degrees(turn):.1f} deg',
                      flush=True)
                break
            if now - last_forward >= 0.055:
                last_forward = now
                forwarded = Twist()
                # Continue the lease while settling, with motion capped at 1.6 s.
                if ('cmd' in latest and now - latest['cmd'][0] < 0.12
                        and now - start_time < 1.6):
                    cmd = latest['cmd'][1]
                    forwarded.linear.x = min(0.10, max(0.0, cmd.linear.x))
                    forwarded.angular.z = min(0.25, max(-0.25, cmd.angular.z))
                commands.publish(forwarded)
        permit(False)
        commands.publish(Twist())
        wait_until(result.done, 1.5)
        if result.done():
            reply = result.result().result
            print(f'outcome={reply.outcome} detail={reply.detail}', flush=True)
        end = latest['odom'][1].pose.pose.position
        print(f'odom_delta=({end.x-start.x:.4f}, {end.y-start.y:.4f})', flush=True)
        if not result.done() or result.result().result.outcome != Navigate.Result.REACHED:
            raise RuntimeError('Trial did not reach its goal; inspect the recorded outcome')
    finally:
        permit(False)
        for _ in range(4):
            commands.publish(Twist())
            rclpy.spin_once(node, timeout_sec=0.03)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
