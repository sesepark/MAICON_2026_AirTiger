"""Check field experiment input stamps and lidar geometry in a SQLite rosbag2."""

import argparse
import json
import math
import sqlite3
from bisect import bisect_left
from collections import Counter
from pathlib import Path
from statistics import median

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


INPUT_TOPICS = ("/scan", "/odom", "/team/perception/field_map", "/team/server/state")


def inspect(directory):
    streams = {}
    for path in sorted(directory.glob("*.db3")):
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            for name, type_name in connection.execute("SELECT name, type FROM topics"):
                if name not in INPUT_TOPICS:
                    continue
                message_type = get_message(type_name)
                samples = streams.setdefault(name, [])
                for data, received_ns in connection.execute(
                    "SELECT m.data, m.timestamp FROM messages m JOIN topics t "
                    "ON t.id = m.topic_id WHERE t.name = ? ORDER BY m.timestamp", (name,)
                ):
                    msg = deserialize_message(data, message_type)
                    header = msg.header
                    stamped_ns = header.stamp.sec * 1_000_000_000 + header.stamp.nanosec
                    sample = {"received": received_ns, "stamped": stamped_ns,
                              "frame": header.frame_id}
                    if name == "/scan":
                        sample["beams"] = len(msg.ranges)
                        sample["valid"] = sum(
                            math.isfinite(value) and msg.range_min <= value < msg.range_max
                            for value in msg.ranges
                        )
                        sample["geometry"] = (msg.angle_min, msg.angle_max,
                                              msg.angle_increment, msg.range_min, msg.range_max)
                    elif name == "/team/perception/field_map":
                        sample["bases"] = tuple(base.name for base in msg.bases)
                        sample["self_count"] = sum(vehicle.role == vehicle.SELF
                                                   for vehicle in msg.vehicles)
                    elif name == "/team/server/state":
                        # Summarize assignment without storing run identifiers.
                        sample["assignment"] = (msg.camp, msg.slot, msg.phase)
                    samples.append(sample)

    results = {}
    for name in INPUT_TOPICS:
        samples = sorted(streams.get(name, []), key=lambda sample: sample["received"])
        result = {"count": len(samples)}
        if samples:
            stamps = [sample["stamped"] for sample in samples]
            delays = [(sample["received"] - sample["stamped"]) / 1e6
                      for sample in samples if sample["stamped"]]
            result.update(
                frames=dict(Counter(sample["frame"] for sample in samples)),
                zero_stamps=stamps.count(0),
                duplicate_or_reversed_stamps=sum(
                    stop <= start for start, stop in zip(stamps, stamps[1:])
                ),
                receive_minus_header_ms=(
                    {"min": min(delays), "median": median(delays), "max": max(delays)}
                    if delays else None
                ),
            )
            if name == "/scan":
                result["median_valid_beams"] = median(sample["valid"] for sample in samples)
                result["beam_counts"] = sorted({sample["beams"] for sample in samples})
                result["geometry"] = sorted({sample["geometry"] for sample in samples})
            elif name == "/team/perception/field_map":
                result["base_names"] = sorted({base for sample in samples for base in sample["bases"]})
                result["self_counts"] = dict(Counter(sample["self_count"] for sample in samples))
            elif name == "/team/server/state":
                result["assignments"] = sorted({sample["assignment"] for sample in samples})
        results[name] = result

    odometry = sorted(sample["stamped"] for sample in streams.get("/odom", [])
                      if sample["stamped"])
    for name in ("/scan", "/team/perception/field_map"):
        gaps = []
        for sample in streams.get(name, []):
            if not odometry or not sample["stamped"]:
                continue
            stamp = sample["stamped"]
            index = bisect_left(odometry, stamp)
            neighbors = odometry[max(0, index - 1):index + 1]
            gaps.append(min(abs(stamp - value) for value in neighbors) / 1e6)
        if gaps:
            results[name]["nearest_odometry_header_ms"] = {
                "median": median(gaps),
                "max": max(gaps),
                "over_100_ms": sum(gap > 100 for gap in gaps),
            }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    args = parser.parse_args()
    if not any(args.bag.glob("*.db3")):
        parser.error("expected a directory with a SQLite rosbag2 database")
    print(json.dumps(inspect(args.bag), indent=2))
