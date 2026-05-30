"""
Dynamic traffic generator for the Crossroad environment.

Called automatically at the start of every episode via CrossroadEnv.reset().
Randomises traffic flow probabilities across three realistic scenarios:
  - balanced   : similar volume on all approaches
  - ns_rush    : North/South axes are heavily loaded
  - ew_rush    : East/West axes are heavily loaded

The generated file overwrites envs/crossroad/dynamic.rou.xml each episode,
giving the agent a fresh traffic distribution to learn from.
"""
import random


def generate_dynamic_traffic(filepath: str, max_steps: int = 3600) -> None:
    """
    Overwrites `filepath` with a fresh randomised probabilistic route file.

    Args:
        filepath:  Absolute path to the .rou.xml file to overwrite.
        max_steps: Simulation duration in seconds (default 3600 = 1 hour).
    """
    # Base probability: ~1 car every 20–50 seconds per route
    p_base = random.uniform(0.02, 0.08)

    scenarios = ["balanced", "ns_rush", "ew_rush"]
    scenario = random.choice(scenarios)

    # All valid (in_edge, out_edge) pairs in the crossroad network
    routes = [
        ("in_N",  "out_S"), ("in_N",  "out_E"), ("in_N",  "out_W"),
        ("in_S",  "out_N"), ("in_S",  "out_E"), ("in_S",  "out_W"),
        ("in_W",  "out_E"), ("in_W",  "out_N"), ("in_W",  "out_S"),
        ("int_E", "out_W"), ("int_E", "out_N"), ("int_E", "out_S"),
    ]

    # Axes that receive a volume boost in each rush scenario
    ns_edges = {"in_N", "in_S"}
    ew_edges = {"in_W", "int_E"}

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<routes>",
        '    <vType id="standard_car" vClass="passenger" '
        'accel="2.6" decel="4.5" sigma="0.5" length="4.5" maxSpeed="15"/>',
    ]

    for i, (in_edge, out_edge) in enumerate(routes):
        prob = p_base

        if scenario == "ns_rush" and in_edge in ns_edges:
            prob += random.uniform(0.05, 0.15)
        elif scenario == "ew_rush" and in_edge in ew_edges:
            prob += random.uniform(0.05, 0.15)

        # Add per-route noise so flows aren't perfectly symmetric
        prob = max(0.01, min(prob * random.uniform(0.8, 1.2), 0.30))

        lines.append(
            f'    <flow id="flow_{i}" begin="0" end="{max_steps}" '
            f'probability="{prob:.3f}" from="{in_edge}" to="{out_edge}" '
            f'type="standard_car" departLane="best" departSpeed="max"/>'
        )

    lines.append("</routes>")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")