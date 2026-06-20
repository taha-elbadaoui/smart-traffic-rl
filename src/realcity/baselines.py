"""
Baselines on the real Cologne district — the bar the learned controllers must beat.

  * fixed-time   : the real OSM-imported signal programs (no control)
  * max-pressure : the classic adaptive controller — each junction picks the phase
                   that maximizes pressure (incoming minus outgoing queue over the
                   movements that phase serves). Beating THIS, not just fixed-time,
                   is the real test of "RL helps".

Both are measured identically from SUMO tripinfo over the 07:00-08:00 peak.

    python src/realcity/baselines.py
"""
import os
import sys
import subprocess
import tempfile
import xml.etree.ElementTree as ET

# libsumo is much faster than socket TraCI for the control loop.
os.environ.setdefault("LIBSUMO_AS_TRACI", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cologne_env import SUMOCFG, make_env  # noqa: E402


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


def _phase_movements(env):
    """Per signal: for each green phase, the list of (in_lane, out_lane) it serves."""
    movements = {}
    for ts in env.ts_ids:
        sig = env.traffic_signals[ts]
        links = sig.sumo.trafficlight.getControlledLinks(ts)  # per link: [(in,out,via)]
        phases = []
        for ph in sig.green_phases:
            pairs = []
            for k, ch in enumerate(ph.state):
                if ch in "gG" and k < len(links) and links[k]:
                    in_lane, out_lane, _ = links[k][0]
                    pairs.append((in_lane, out_lane))
            phases.append(pairs)
        movements[ts] = phases
    return movements


def max_pressure():
    """Greedy max-pressure control over the peak hour; same tripinfo metrics."""
    out = os.path.join(tempfile.gettempdir(), "cologne_maxpressure_tripinfo.xml")
    env = make_env(use_gui=False, reward_fn="pressure", sumo_seed=42,
                   additional_sumo_cmd=f"--tripinfo-output {out}")
    env.reset()
    movements = _phase_movements(env)
    conn = env.traffic_signals[env.ts_ids[0]].sumo

    while True:
        # Query each relevant lane's queue once per decision (cache), then score phases.
        lanes = {l for ts in env.ts_ids for ph in movements[ts] for pair in ph for l in pair}
        q = {l: conn.lane.getLastStepHaltingNumber(l) for l in lanes}

        actions = {}
        for ts in env.ts_ids:
            best_i, best_p = 0, float("-inf")
            for i, pairs in enumerate(movements[ts]):
                pressure = sum(q[a] - q[b] for a, b in pairs)
                if pressure > best_p:
                    best_p, best_i = pressure, i
            actions[ts] = best_i

        _, _, dones, _ = env.step(actions)
        if dones.get("__all__", False):
            break

    env.close()
    return parse_tripinfo(out)


def _row(name, m):
    return (f"{name:<16}{m['mean_waiting_time']:>10.2f}{m['mean_time_loss']:>14.2f}"
            f"{m['mean_travel_time']:>12.2f}{m['completed_trips']:>8}")


def main():
    print("=== Baselines: real Cologne district, 07:00-08:00 peak ===\n")
    print(f"{'controller':<16}{'wait(s)':>10}{'timeLoss(s)':>14}{'travel(s)':>12}{'trips':>8}")
    print("-" * 60)
    print(_row("fixed_time", fixed_time()))
    print(_row("max_pressure", max_pressure()))
    print("\n(lower is better for wait / timeLoss / travel)")


if __name__ == "__main__":
    main()
