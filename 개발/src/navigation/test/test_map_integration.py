import math
import time
import unittest
from uuid import uuid4

try:
    import rclpy
except ModuleNotFoundError:
    rclpy = None


@unittest.skipIf(rclpy is None, 'requires ROS 2')
class MapIntegrationTest(unittest.TestCase):
    def test_field_map_round_trip(self):
        from geometry_msgs.msg import Point
        from navigation.driving_node import DrivingNode
        from navigation.slam_node import SlamNode
        from team_interfaces.msg import FieldFace, FieldMap, FieldRegion, FieldVehicle

        suffix = uuid4().hex
        input_topic = f'/test_navigation/input_{suffix}'
        output_topic = f'/test_navigation/output_{suffix}'
        rclpy.init(args=[
            '--ros-args',
            '-r', f'/team/perception/field_map:={input_topic}',
            '-r', f'/team/navigation/map:={output_topic}',
        ])
        self.addCleanup(rclpy.try_shutdown)
        perception = rclpy.create_node(f'test_perception_{suffix}')
        self.addCleanup(perception.destroy_node)
        input_publisher = perception.create_publisher(FieldMap, input_topic, 1)
        slam = SlamNode()
        self.addCleanup(slam.destroy_node)
        drive = DrivingNode()
        self.addCleanup(drive.destroy_node)

        self.assertIsNone(slam.latest_perception_map)
        self.assertIsNone(drive.latest_map)
        for revision in (1, 2):
            with self.subTest(revision=revision):
                field = FieldMap(width=8.0, height=4.0, box_size=0.4)
                field.header.frame_id = 'arena'
                field.header.stamp.sec = revision
                field.boxes = [Point(x=2.0, y=1.0)]
                field.bases = [FieldRegion(name='base', max_x=1.0, max_y=1.0, row=1)]
                field.sectors = [FieldRegion(name='sector', max_x=4.0, max_y=4.0)]
                field.faces = [
                    FieldFace(box=0, side='E', cx=2.2, cy=1.0, nx=1.0,
                              sector='sector', image_ref=f'camera/image_{revision}')
                ]
                field.vehicles = [FieldVehicle(role=FieldVehicle.SELF)]
                field.vehicles[0].pose.orientation.w = 1.0

                deadline = time.monotonic() + 5
                while drive.latest_map != field and time.monotonic() < deadline:
                    input_publisher.publish(field)
                    rclpy.spin_once(slam, timeout_sec=0.05)
                    rclpy.spin_once(drive, timeout_sec=0.05)
                self.assertEqual(drive.latest_map, field)
                self.assertIsNotNone(drive.grid)
                self.assertEqual(drive.map_update.spec.boxes, ((2.0, 1.0),))


    def test_pose_expires_then_slot_and_scan_restore_tracking(self):
        from geometry_msgs.msg import Point, PoseStamped
        from nav_msgs.msg import Odometry
        from navigation.slam_node import SlamNode
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import LaserScan
        from team_interfaces.msg import FieldMap, FieldRegion, FieldVehicle, ServerState

        prefix = f'/test_navigation/gap_{uuid4().hex}'
        remaps = [
            ('/team/perception/field_map', '/field'),
            ('/team/navigation/map', '/map'),
            ('/team/navigation/pose', '/pose'),
            ('/odom', '/odom'),
            ('/scan', '/scan'),
            ('/team/server/state', '/server'),
        ]
        args = ['--ros-args']
        for source, target in remaps:
            args.extend(('-r', f'{source}:={prefix}{target}'))
        rclpy.init(args=args)
        self.addCleanup(rclpy.try_shutdown)
        sensor = rclpy.create_node(f'gap_sensor_{uuid4().hex}')
        self.addCleanup(sensor.destroy_node)
        slam = SlamNode()
        self.addCleanup(slam.destroy_node)
        received = []
        sensor.create_subscription(PoseStamped, prefix + '/pose', received.append, 5)
        aerial = sensor.create_publisher(FieldMap, prefix + '/field', 1)
        odom = sensor.create_publisher(
            Odometry, prefix + '/odom', qos_profile_sensor_data
        )

        server = sensor.create_publisher(ServerState, prefix + '/server', 1)
        laser = sensor.create_publisher(LaserScan, prefix + '/scan', qos_profile_sensor_data)

        def wait_until(predicate, send=None):
            deadline = time.monotonic() + 5
            while not predicate() and time.monotonic() < deadline:
                if send:
                    send()
                rclpy.spin_once(slam, timeout_sec=0.03)
                rclpy.spin_once(sensor, timeout_sec=0.03)
            self.assertTrue(predicate())

        def send_odom(sec, x):
            msg = Odometry()
            msg.header.frame_id = 'odom'
            msg.header.stamp.sec = sec
            msg.child_frame_id = 'base_footprint'
            msg.pose.pose.position.x = x
            msg.pose.pose.orientation.w = 1.0
            odom.publish(msg)

        wait_until(lambda: aerial.get_subscription_count() and odom.get_subscription_count())
        field = FieldMap(width=8.0, height=4.0, box_size=0.4)
        field.header.frame_id = 'arena'
        vehicle = FieldVehicle(role=FieldVehicle.SELF)
        vehicle.pose.orientation.w = 1.0
        field.vehicles = [vehicle]
        # Four paired observations establish the odom-to-arena axis alignment.
        for stamp, x in ((7, 3.4), (8, 3.6), (9, 3.8), (10, 4.0)):
            wait_until(lambda: slam._odometry and slam._odometry[-1][0] == stamp * 10**9,
                       lambda: send_odom(stamp, x))
            field.header.stamp.sec = stamp
            field.vehicles[0].pose.position.x = x - 2.0
            wait_until(lambda: slam.latest_perception_map is not None
                       and slam.latest_perception_map.header.stamp.sec == stamp,
                       lambda: aerial.publish(field))
        wait_until(lambda: any(msg.header.stamp.sec == 11 for msg in received),
                   lambda: send_odom(11, 4.5))
        self.assertEqual(received[-1].header.frame_id, 'arena')
        self.assertAlmostEqual(received[-1].pose.position.x, 2.5)

        wait_until(lambda: slam._odometry[-1][0] == 13_000_000_000,
                   lambda: send_odom(13, 5.0))
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            rclpy.spin_once(sensor, timeout_sec=0.05)
        self.assertTrue(all(msg.header.stamp.sec <= 11 for msg in received))

        field.header.stamp.sec = 14
        field.box_size = 0.6
        field.height = 5.0
        slot_x, slot_y = 3.0, 2.0
        field.boxes = [Point(x=x + slot_x, y=y + slot_y) for x, y in (
            (2.0, 0.2), (0.5, 2.0), (-1.2, 1.3), (-1.4, -0.3), (0.6, -1.6)
        )]
        field.bases = [FieldRegion(name='B-2', min_x=2.8, max_x=3.2,
                                   min_y=1.8, max_y=2.2, row=2)]
        vehicle = FieldVehicle(role=FieldVehicle.UNKNOWN)
        vehicle.pose.position.x, vehicle.pose.position.y = slot_x, slot_y
        field.vehicles = [vehicle]
        assignment = ServerState(team_assigned=True, camp='BLUE', slot='B-2', phase='ready')
        assignment.header.stamp = field.header.stamp
        wait_until(lambda: slam._assignment is not None, lambda: server.publish(assignment))
        wait_until(lambda: slam.latest_perception_map.header.stamp.sec == 14,
                   lambda: aerial.publish(field))
        wait_until(lambda: slam._odometry[-1][0] == 14_000_000_000,
                   lambda: send_odom(14, 5.0))
        wait_until(lambda: slam._odometry[-1][0] == 15_000_000_000,
                   lambda: send_odom(15, 5.0))

        def send_scan(sec):
            scan = LaserScan()
            scan.header.frame_id = 'base_lidar_link'
            scan.header.stamp.sec = sec
            scan.angle_increment = math.tau / 500
            scan.range_min, scan.range_max = 0.05, 20.0
            ranges = []
            for index in range(500):
                angle = index * scan.angle_increment
                dx, dy = -math.sin(angle), math.cos(angle)
                nearest = math.inf
                for box in field.boxes:
                    if abs(dx) < 1e-8 or abs(dy) < 1e-8:
                        continue
                    x_near, x_far = sorted((box.x - slot_x + edge - 0.0398) / dx
                                           for edge in (-0.3, 0.3))
                    y_near, y_far = sorted((box.y - slot_y + edge) / dy for edge in (-0.3, 0.3))
                    intersect = max(x_near, y_near)
                    if scan.range_min < intersect <= min(x_far, y_far):
                        nearest = min(nearest, intersect)
                ranges.append(nearest)
            scan.ranges = ranges
            laser.publish(scan)

        # Initialize with a scan close to the aerial capture, then send new scans
        # while the map stays fixed.
        wait_until(lambda: any(msg.header.stamp.sec == 15 for msg in received),
                   lambda: send_scan(14))
        wait_until(lambda: slam._scan_anchor[0] == 15_000_000_000, lambda: send_scan(15))
        wait_until(lambda: any(msg.header.stamp.sec == 17 for msg in received),
                   lambda: send_odom(17, 5.0))
        self.assertAlmostEqual(received[-1].pose.position.x, slot_x)
        self.assertAlmostEqual(received[-1].pose.position.y, slot_y)
        self.assertAlmostEqual(received[-1].pose.orientation.w, 1.0)
