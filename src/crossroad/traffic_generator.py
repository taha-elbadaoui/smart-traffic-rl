"""
Dynamic traffic generator for the Crossroad environment.
Generates isolated randomized probabilistic routes to ensure process safety during parallel training.
"""
import random

def generate_dynamic_traffic(filepath: str, max_steps: int = 3600) -> None:
    """
    Overwrites `filepath` with a fresh randomised probabilistic route file.
    """
    p_base = random.uniform(0.02, 0.08)

    scenarios = ["balanced", "ns_rush", "ew_rush"]
    scenario = random.choice(scenarios)

    routes = [
        ("in_N",  "out_S"), ("in_N",  "out_E"), ("in_N",  "out_W"),
        ("in_S",  "out_N"), ("in_S",  "out_E"), ("in_S",  "out_W"),
        ("in_W",  "out_E"), ("in_W",  "out_N"), ("in_W",  "out_S"),
        ("int_E", "out_W"), ("int_E", "out_N"), ("int_E", "out_S"),
    ]

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

        prob = max(0.01, min(prob * random.uniform(0.8, 1.2), 0.30))

        lines.append(
            f'    <flow id="flow_{i}" begin="0" end="{max_steps}" '
            f'probability="{prob:.3f}" from="{in_edge}" to="{out_edge}" '
            f'type="standard_car" departLane="best" departSpeed="max"/>'
        )

    lines.append("</routes>")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")