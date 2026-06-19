"""
Dynamic traffic generator for the Crossroad environment.
Generates isolated randomized probabilistic routes to ensure process safety during parallel training.

Backward compatible: calling `generate_dynamic_traffic(path)` keeps the original
fully-random behaviour used during training. The optional `scenario`, `seed` and
`intensity` arguments make the output controllable and reproducible, which the
benchmark lab relies on to compare controllers on identical traffic.
"""
import random

ROUTES = [
    ("in_N",  "out_S"), ("in_N",  "out_E"), ("in_N",  "out_W"),
    ("in_S",  "out_N"), ("in_S",  "out_E"), ("in_S",  "out_W"),
    ("in_W",  "out_E"), ("in_W",  "out_N"), ("in_W",  "out_S"),
    ("int_E", "out_W"), ("int_E", "out_N"), ("int_E", "out_S"),
]
NS_EDGES = {"in_N", "in_S"}
EW_EDGES = {"in_W", "int_E"}
SCENARIOS = ["balanced", "ns_rush", "ew_rush"]

# Visual-only car palette: real car shape + distinct colours so the sumo-gui
# view is easy to read. Physics are identical across types, so dynamics and
# training are unaffected.
CAR_TYPES = [
    ("car_red",    "210,60,60"),
    ("car_blue",   "60,110,210"),
    ("car_white",  "235,235,235"),
    ("car_yellow", "230,200,60"),
    ("car_teal",   "60,190,180"),
]


def generate_dynamic_traffic(
    filepath: str,
    max_steps: int = 3600,
    scenario: str = None,
    seed: int = None,
    intensity: float = 1.0,
) -> str:
    """
    Overwrites `filepath` with a fresh randomised probabilistic route file.

    Args:
        filepath:  output .rou.xml path.
        max_steps: departure window (vehicles stop spawning after this).
        scenario:  one of SCENARIOS. None -> picked at random (training default).
        seed:      RNG seed for reproducible traffic. None -> non-deterministic.
        intensity: global multiplier on spawn probabilities (>1 = busier roads).

    Returns:
        The scenario actually used (handy for logging).
    """
    rng = random.Random(seed) if seed is not None else random

    if scenario is None:
        scenario = rng.choice(SCENARIOS)

    p_base = rng.uniform(0.02, 0.08) * intensity

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<routes>",
    ]
    # One vType per colour (same physics, just a different look in sumo-gui).
    for type_id, color in CAR_TYPES:
        lines.append(
            f'    <vType id="{type_id}" vClass="passenger" guiShape="passenger" '
            f'color="{color}" accel="2.6" decel="4.5" sigma="0.5" '
            f'length="4.5" maxSpeed="15"/>'
        )

    for i, (in_edge, out_edge) in enumerate(ROUTES):
        prob = p_base

        if scenario == "ns_rush" and in_edge in NS_EDGES:
            prob += rng.uniform(0.05, 0.15) * intensity
        elif scenario == "ew_rush" and in_edge in EW_EDGES:
            prob += rng.uniform(0.05, 0.15) * intensity

        # Clamp into a sane probability range (allow heavier flows when intensity is high).
        prob = max(0.01, min(prob * rng.uniform(0.8, 1.2), 0.5))

        car_type = CAR_TYPES[i % len(CAR_TYPES)][0]
        lines.append(
            f'    <flow id="flow_{i}" begin="0" end="{max_steps}" '
            f'probability="{prob:.3f}" from="{in_edge}" to="{out_edge}" '
            f'type="{car_type}" departLane="best" departSpeed="max"/>'
        )

    lines.append("</routes>")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return scenario
