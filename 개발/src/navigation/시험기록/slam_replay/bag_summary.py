"""Summarize receive-time gaps in a rosbag2 SQLite recording."""
import argparse
import json
import sqlite3
from pathlib import Path


def summarize(directory):
    streams = {}
    for path in sorted(directory.glob("*.db3")):
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            for topic, stamp in connection.execute(
                "SELECT topics.name, messages.timestamp FROM messages "
                "JOIN topics ON topics.id = messages.topic_id ORDER BY messages.timestamp"
            ):
                streams.setdefault(topic, []).append(stamp)
    if not streams:
        raise ValueError(f"No recorded messages in SQLite bag {directory}")

    summary = {}
    for topic, times in sorted(streams.items()):
        times.sort()
        duration = (times[-1] - times[0]) / 1e9
        summary[topic] = {
            "count": len(times),
            "first_receive_ns": times[0],
            "last_receive_ns": times[-1],
            "receive_hz": (len(times) - 1) / duration if duration else None,
            "max_receive_gap_s": max(
                ((end - start) / 1e9 for start, end in zip(times, times[1:])),
                default=None,
            ),
        }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.bag), indent=2))
