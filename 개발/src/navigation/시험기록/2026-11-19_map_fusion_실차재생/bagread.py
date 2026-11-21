"""ROS 2 sqlite bag(.db3) 의 메시지를 ROS 없이 읽는다 (CDR 을 직접 푼다). replay.py 가 쓰는 타입만.

    for 받은_시각_ns, 토픽, 메시지 in bagread.read('bag_0.db3', {'/scan', '/odom'}): ...
"""
import sqlite3
import struct
from types import SimpleNamespace as NS


class R:
    def __init__(self, data):
        self.d = data
        self.o = 4  # encapsulation header

    def al(self, n):
        rel = self.o - 4
        self.o += (-rel) % n

    def u8(self):
        v = self.d[self.o]; self.o += 1; return v

    def b(self):
        return bool(self.u8())

    def u32(self):
        self.al(4); v = struct.unpack_from('<I', self.d, self.o)[0]; self.o += 4; return v

    def i32(self):
        self.al(4); v = struct.unpack_from('<i', self.d, self.o)[0]; self.o += 4; return v

    def f32(self):
        self.al(4); v = struct.unpack_from('<f', self.d, self.o)[0]; self.o += 4; return v

    def f64(self):
        self.al(8); v = struct.unpack_from('<d', self.d, self.o)[0]; self.o += 8; return v

    def s(self):
        n = self.u32(); v = self.d[self.o:self.o + n - 1].decode('utf-8', 'replace'); self.o += n; return v

    def f32s(self):
        n = self.u32(); self.al(4)
        v = struct.unpack_from('<%df' % n, self.d, self.o); self.o += 4 * n; return list(v)

    def f64n(self, n):
        self.al(8); v = struct.unpack_from('<%dd' % n, self.d, self.o); self.o += 8 * n; return list(v)


def header(r):
    sec = r.i32(); nsec = r.u32()
    return NS(stamp=NS(sec=sec, nanosec=nsec), frame_id=r.s())


def point(r):
    return NS(x=r.f64(), y=r.f64(), z=r.f64())


def pose(r):
    p = point(r)
    q = NS(x=r.f64(), y=r.f64(), z=r.f64(), w=r.f64())
    return NS(position=p, orientation=q)


def scan(r):
    h = header(r)
    vals = [r.f32() for _ in range(7)]
    m = NS(header=h, angle_min=vals[0], angle_max=vals[1], angle_increment=vals[2], time_increment=vals[3],
           scan_time=vals[4], range_min=vals[5], range_max=vals[6])
    m.ranges = r.f32s(); m.intensities = r.f32s()
    return m


def odom(r):
    h = header(r); child = r.s(); p = pose(r); r.f64n(36)
    return NS(header=h, child_frame_id=child, pose=NS(pose=p))


def f32array(r):
    n = r.u32()
    for _ in range(n):
        r.s(); r.u32(); r.u32()
    r.u32()
    return NS(data=r.f32s())


def imu(r):
    h = header(r)
    q = NS(x=r.f64(), y=r.f64(), z=r.f64(), w=r.f64())
    return NS(header=h, orientation=q)


def region(r):
    return NS(name=r.s(), min_x=r.f64(), min_y=r.f64(), max_x=r.f64(), max_y=r.f64(), row=r.u8())


def face(r):
    return NS(box=r.u32(), side=r.s(), cx=r.f64(), cy=r.f64(), nx=r.f64(), ny=r.f64(), sector=r.s(), image_ref=r.s())


class Vehicle(NS):
    """team_interfaces/FieldVehicle 흉내: 상수와 기본 pose (SLAM 노드가 FieldVehicle(role=SELF) 로 새로 만들기도 한다)."""
    UNKNOWN, SELF, OTHER = 0, 1, 2

    def __init__(self, role=0, pose=None):
        super().__init__(role=role, pose=pose or NS(position=NS(x=0.0, y=0.0, z=0.0), orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0)))


def vehicle(r):
    return Vehicle(role=r.u8(), pose=pose(r))


def fieldmap(r):
    m = NS(header=header(r), width=r.f64(), height=r.f64(), box_size=r.f64())
    m.boxes = [point(r) for _ in range(r.u32())]
    m.bases = [region(r) for _ in range(r.u32())]
    m.sectors = [region(r) for _ in range(r.u32())]
    m.beam_depth = r.f64()
    m.faces = [face(r) for _ in range(r.u32())]
    m.vehicles = [vehicle(r) for _ in range(r.u32())]
    return m


def server(r):
    return NS(header=header(r), run_id=r.s(), connected=r.b(), team_assigned=r.b(), slot=r.s(), camp=r.s(),
              phase=r.s(), ready=r.b(), detail=r.s())


def posestamped(r):
    return NS(header=header(r), pose=pose(r))


def string(r):
    return NS(data=r.s())


def twist(r):
    return NS(linear=point(r), angular=point(r))


DECODERS = {
    'sensor_msgs/msg/LaserScan': scan, 'nav_msgs/msg/Odometry': odom, 'std_msgs/msg/Float32MultiArray': f32array,
    'sensor_msgs/msg/Imu': imu, 'team_interfaces/msg/FieldMap': fieldmap, 'team_interfaces/msg/ServerState': server,
    'geometry_msgs/msg/PoseStamped': posestamped, 'std_msgs/msg/String': string, 'geometry_msgs/msg/Twist': twist,
}


def read(db, topics=None):
    """(수신 ns, 토픽, 메시지) 를 시간 순으로."""
    con = sqlite3.connect(db)
    types = {i: (n, t) for i, n, t in con.execute('select id, name, type from topics')}
    for tid, ts, data in con.execute('select topic_id, timestamp, data from messages order by timestamp'):
        name, typ = types[tid]
        if topics is not None and name not in topics:
            continue
        dec = DECODERS.get(typ)
        if dec is None:
            continue
        yield ts, name, dec(R(bytes(data)))
