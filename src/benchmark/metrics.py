"""
Metric collection for the benchmark lab.

Two complementary sources:

1. SUMO's `--tripinfo-output` XML (written when the simulation ends): the gold
   standard per-vehicle stats — waiting time, time loss, travel time, plus the
   number of completed trips (throughput).
2. Online queue sampling: the total number of halting vehicles read once per
   macro-step, summarised into mean / max queue length.

All controllers are measured the exact same way on the exact same traffic, so
the comparison is fair even where absolute values exclude vehicles that never
finished within the horizon.
"""
import os
import xml.etree.ElementTree as ET


def parse_tripinfo(path):
    """
    Parse a SUMO tripinfo XML file into aggregate statistics.

    Returns a dict with mean per-vehicle metrics and the completed-trip count.
    Returns zeros if the file is missing or empty (e.g. no vehicle finished).
    """
    empty = {
        "completed_trips": 0,
        "mean_waiting_time": 0.0,
        "mean_time_loss": 0.0,
        "mean_travel_time": 0.0,
        "mean_depart_delay": 0.0,
    }
    if not path or not os.path.exists(path):
        return empty

    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return empty

    waits, losses, travels, delays = [], [], [], []
    for trip in root.findall("tripinfo"):
        waits.append(float(trip.get("waitingTime", 0.0)))
        losses.append(float(trip.get("timeLoss", 0.0)))
        travels.append(float(trip.get("duration", 0.0)))
        delays.append(float(trip.get("departDelay", 0.0)))

    n = len(waits)
    if n == 0:
        return empty

    return {
        "completed_trips": n,
        "mean_waiting_time": sum(waits) / n,
        "mean_time_loss": sum(losses) / n,
        "mean_travel_time": sum(travels) / n,
        "mean_depart_delay": sum(delays) / n,
    }


class QueueTracker:
    """Accumulates the total halting-vehicle count sampled once per macro-step."""
    def __init__(self):
        self.samples = []

    def record(self, total_queue):
        self.samples.append(float(total_queue))

    def summary(self):
        if not self.samples:
            return {"mean_queue": 0.0, "max_queue": 0.0}
        return {
            "mean_queue": sum(self.samples) / len(self.samples),
            "max_queue": max(self.samples),
        }


# Metrics where lower is better (used for pretty-printing improvements).
LOWER_IS_BETTER = {
    "mean_waiting_time",
    "mean_time_loss",
    "mean_travel_time",
    "mean_depart_delay",
    "mean_queue",
    "max_queue",
}
