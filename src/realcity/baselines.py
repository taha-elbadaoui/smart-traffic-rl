"""
Baselines on the real Cologne district — the bar the learned controllers must beat.

Currently implements the FIXED-TIME baseline: the real OSM-imported signal
programs running over the 07:00-08:00 peak. Aggregated per-vehicle stats are
parsed from SUMO's tripinfo output.

(Max-pressure — the classic adaptive baseline — will be added next; beating it,
not just fixed-time, is the real test of "RL helps".)

    python src/realcity/baselines.py
"""
import os
import sys
import subprocess
import tempfile
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cologne_env import SUMOCFG  # noqa: E402


def parse_tripinfo(path):
    root = ET.parse(path).getroot()
    waits, losses, durations = [], [], []
    for t in root.findall("tripinfo"):
        waits.append(float(t.get("waitingTime", 0.0)))
        losses.append(float(t.get("timeLoss", 0.0)))
        durations.append(float(t.get("duration", 0.0)))
    n = max(len(waits), 1)
    return {
        "completed_trips": len(waits),
        "mean_waiting_time": sum(waits) / n,
        "mean_time_loss": sum(losses) / n,
        "mean_travel_time": sum(durations) / n,
    }


def fixed_time():
    """Run the real signal programs over the peak hour and aggregate trip stats."""
    out = os.path.join(tempfile.gettempdir(), "cologne_fixedtime_tripinfo.xml")
    subprocess.run(
        ["sumo", "-c", SUMOCFG, "--tripinfo-output", out,
         "--no-step-log", "true", "--no-warnings", "true",
         "--duration-log.statistics", "true"],
        check=True, capture_output=True, text=True,
    )
    return parse_tripinfo(out)


def main():
    print("=== Baselines: real Cologne district, 07:00-08:00 peak ===\n")
    ft = fixed_time()
    print(f"{'controller':<16}{'wait(s)':>10}{'timeLoss(s)':>14}{'travel(s)':>12}{'trips':>8}")
    print("-" * 60)
    print(f"{'fixed_time':<16}{ft['mean_waiting_time']:>10.2f}"
          f"{ft['mean_time_loss']:>14.2f}{ft['mean_travel_time']:>12.2f}"
          f"{ft['completed_trips']:>8}")
    print("\n(lower is better for wait / timeLoss / travel)")


if __name__ == "__main__":
    main()
