"""
Generate a "city" of decorative polygons (buildings, parks, ground) around a
SUMO network so it looks like a real neighbourhood in sumo-gui.

The polygons are pure visuals (a SUMO *additional* file): they have no effect on
the simulation, the road network, the traffic lights or the trained models. The
script tiles the network's bounding box with building blocks and simply drops
any block that sits on top of a road, leaving the streets carved out.

Usage:
    python tools/decorate_network.py envs/crossroad/env.net.xml envs/crossroad/env.poly.xml
"""
import os
import sys
import math
import random
import argparse

sys.path.append(os.path.join(os.environ.get("SUMO_HOME", ""), "tools"))
import sumolib  # noqa: E402

# Muted building tones + a couple of greens for parks (RGB 0-255).
BUILDING_COLORS = [
    "196,188,176", "208,200,186", "184,176,164",
    "212,198,182", "175,167,155", "199,190,178",
]
PARK_COLORS = ["168,200,148", "152,190,138"]
GROUND_COLOR = "223,226,221"
WATER_COLOR = "150,190,214"


def _point_seg_dist(px, py, ax, ay, bx, by):
    """Shortest distance from point (px,py) to segment (a)-(b)."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def _collect_segments(net):
    """All lane-shape segments, used for road-proximity tests."""
    segs = []
    for edge in net.getEdges():
        for lane in edge.getLanes():
            shape = lane.getShape()
            for i in range(len(shape) - 1):
                ax, ay = shape[i]
                bx, by = shape[i + 1]
                segs.append((ax, ay, bx, by))
    return segs


def _near_road(px, py, segs, clearance):
    return any(_point_seg_dist(px, py, *s) < clearance for s in segs)


def decorate(net_file, out_file, seed=42,
             pitch=24.0, footprint=17.0, clearance=15.0, margin=35.0,
             park_ratio=0.12):
    net = sumolib.net.readNet(net_file)
    xmin, ymin, xmax, ymax = net.getBoundary()
    segs = _collect_segments(net)
    rng = random.Random(seed)

    polys = []

    # 1) A ground slab beneath everything so the canvas isn't blank white.
    gx0, gy0 = xmin - margin, ymin - margin
    gx1, gy1 = xmax + margin, ymax + margin
    ground = f"{gx0:.1f},{gy0:.1f} {gx1:.1f},{gy0:.1f} {gx1:.1f},{gy1:.1f} {gx0:.1f},{gy1:.1f} {gx0:.1f},{gy0:.1f}"
    polys.append(("ground", "ground", GROUND_COLOR, -10, ground))

    # 2) Tile the (padded) bounding box with building blocks; skip any on a road.
    half = footprint / 2.0
    n = 0
    x = gx0 + pitch / 2
    while x < gx1:
        y = gy0 + pitch / 2
        while y < gy1:
            # jitter so the blocks don't look like a perfect grid
            cx = x + rng.uniform(-2.0, 2.0)
            cy = y + rng.uniform(-2.0, 2.0)
            if not _near_road(cx, cy, segs, clearance):
                hw = half * rng.uniform(0.8, 1.0)
                hh = half * rng.uniform(0.8, 1.0)
                shape = (
                    f"{cx-hw:.1f},{cy-hh:.1f} {cx+hw:.1f},{cy-hh:.1f} "
                    f"{cx+hw:.1f},{cy+hh:.1f} {cx-hw:.1f},{cy+hh:.1f} {cx-hw:.1f},{cy-hh:.1f}"
                )
                if rng.random() < park_ratio:
                    polys.append((f"park_{n}", "park", rng.choice(PARK_COLORS), -6, shape))
                else:
                    polys.append((f"bldg_{n}", "building", rng.choice(BUILDING_COLORS), -5, shape))
                n += 1
            y += pitch
        x += pitch

    with open(out_file, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write("<additional>\n")
        for pid, ptype, color, layer, shape in polys:
            f.write(
                f'    <poly id="{pid}" type="{ptype}" color="{color}" '
                f'fill="1" layer="{layer}" shape="{shape}"/>\n'
            )
        f.write("</additional>\n")

    buildings = sum(1 for p in polys if p[1] == "building")
    parks = sum(1 for p in polys if p[1] == "park")
    print(f"{os.path.basename(os.path.dirname(net_file)):20s} -> "
          f"{buildings} buildings, {parks} parks  ({os.path.relpath(out_file)})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Decorate a SUMO network with a city of polygons")
    p.add_argument("net_file")
    p.add_argument("out_file")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pitch", type=float, default=24.0, help="grid spacing between blocks (m)")
    p.add_argument("--footprint", type=float, default=17.0, help="building block size (m)")
    p.add_argument("--clearance", type=float, default=15.0, help="min distance from a road (m)")
    args = p.parse_args()
    decorate(args.net_file, args.out_file, seed=args.seed,
             pitch=args.pitch, footprint=args.footprint, clearance=args.clearance)
