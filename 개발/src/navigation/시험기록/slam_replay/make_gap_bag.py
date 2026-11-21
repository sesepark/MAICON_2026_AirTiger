"""Copy SLAM inputs to a replay bag, dropping one topic in a time window."""

import argparse
from collections import Counter
from pathlib import Path

from rosbag2_py import ConverterOptions, SequentialReader, SequentialWriter, StorageOptions


INPUTS = ("/odom", "/imu/data", "/scan", "/team/perception/field_map", "/team/server/state")


def copy_inputs(source, destination, topic, start_ns, stop_ns):
    reader = SequentialReader()
    converter = ConverterOptions("", "")
    reader.open(StorageOptions(uri=str(source), storage_id="sqlite3"), converter)
    available = {item.name: item for item in reader.get_all_topics_and_types()}
    if topic not in available:
        raise ValueError(f"{topic} is absent from the source bag")
    writer = SequentialWriter()
    writer.open(StorageOptions(uri=str(destination), storage_id="sqlite3"), converter)
    for name in INPUTS:
        if name in available:
            writer.create_topic(available[name])

    kept = Counter()
    dropped = 0
    while reader.has_next():
        name, data, timestamp = reader.read_next()
        if name not in INPUTS:
            continue
        if name == topic and start_ns <= timestamp < stop_ns:
            dropped += 1
            continue
        writer.write(name, data, timestamp)
        kept[name] += 1
    del writer
    return kept, dropped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("start_ns", type=int)
    parser.add_argument("stop_ns", type=int)
    parser.add_argument("--topic", choices=INPUTS, default="/team/perception/field_map")
    args = parser.parse_args()
    if not args.source.is_dir() or args.destination.exists():
        parser.error("source must be a bag directory and destination must be new")
    if not 0 <= args.start_ns < args.stop_ns:
        parser.error("expected 0 <= start_ns < stop_ns")
    try:
        kept, dropped = copy_inputs(args.source, args.destination, args.topic,
                                    args.start_ns, args.stop_ns)
    except ValueError as error:
        parser.error(str(error))
    print(f"Dropped {dropped} {args.topic} messages; kept {dict(kept)}")
    if not dropped:
        parser.exit(1, "No messages matched the requested gap; check the time range.\n")
