"""
Rebuild a SUMO network with MORE LANES plus sidewalks and pedestrian crossings,
regenerating the traffic-light program from the original phase *semantics* so the
RL wrappers keep their 2-action / N-phase structure (the policy must be retrained
because the road capacity changes, but observation and action spaces are intact).

Strategy:
  1. Extract the original network to plain XML (preserves edge IDs).
  2. From the original light program, learn — per phase — which incoming edge is
     green / yellow / red ("phase semantics"), independent of lane count.
  3. Widen every vehicle edge to the requested lane count.
  4. Trial-build (auto-connections + crossings) to discover the new links.
  5. Re-colour each new link from the phase semantics of its incoming edge
     (through/right = G, left = g), append red crossings, keep the same phase
     count and durations.
  6. Final build with the regenerated program.

Usage:
    python tools/rebuild_net.py <original_net.xml> <output_net.xml> --lanes 2
"""
import os
import sys
import argparse
import tempfile
import subprocess
import xml.etree.ElementTree as ET


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + "\n" + r.stderr + "\n")
        raise SystemExit("netconvert failed")


def _original_phase_semantics(tll_path):
    """Return durations and {tls: [ {edge: 'G'/'y'/'r'} per phase ]}."""
    root = ET.parse(tll_path).getroot()
    link_edge = {}            # (tls, linkIndex) -> incoming edge
    for c in root.iter("connection"):
        if c.get("tl") is not None:
            link_edge[(c.get("tl"), int(c.get("linkIndex")))] = c.get("from")

    durations, semantics = {}, {}
    for tl in root.iter("tlLogic"):
        tid = tl.get("id")
        durations[tid] = [p.get("duration") for p in tl.findall("phase")]
        sem = []
        for ph in tl.findall("phase"):
            state = ph.get("state")
            edge_char = {}
            for idx, ch in enumerate(state):
                edge = link_edge.get((tid, idx))
                if edge is None:
                    continue
                # Aggregate per edge: green dominates, then yellow, else red.
                prev = edge_char.get(edge, "r")
                rank = {"r": 0, "y": 1, "g": 2, "G": 2}
                cur = "G" if ch in "Gg" else ("y" if ch in "yY" else "r")
                if rank[cur] >= rank[prev]:
                    edge_char[edge] = cur
            sem.append(edge_char)
        semantics[tid] = sem
    return durations, semantics


def _new_links(trial_net):
    """Per TLS: ordered link info. Returns {tls: {idx: ('veh', from_edge, dir) | ('xing',)}, count}."""
    root = ET.parse(trial_net).getroot()
    info, count = {}, {}
    for c in root.iter("connection"):
        tid = c.get("tl")
        if tid is None:
            continue
        idx = int(c.get("linkIndex"))
        info.setdefault(tid, {})[idx] = ("veh", c.get("from"), c.get("dir") or "s")
        count[tid] = max(count.get(tid, -1), idx)
    for cr in root.iter("crossing"):
        if cr.get("linkIndex") is None:
            continue
        tid = cr.get("tl") or cr.get("node")
        idx = int(cr.get("linkIndex"))
        info.setdefault(tid, {})[idx] = ("xing",)
        count[tid] = max(count.get(tid, -1), idx)
    return info, count


def _widen_edges(edg_path, lanes):
    tree = ET.parse(edg_path)
    for e in tree.getroot().iter("edge"):
        e.set("numLanes", str(lanes))
    tree.write(edg_path)


def rebuild(original_net, output_net, lanes=2, sidewalk_width=2.0):
    work = tempfile.mkdtemp(prefix="rebuild_")
    p = os.path.join(work, "p")
    _run(["netconvert", "--sumo-net-file", original_net, "--plain-output-prefix", p])
    nod, edg, tll = p + ".nod.xml", p + ".edg.xml", p + ".tll.xml"

    durations, semantics = _original_phase_semantics(tll)
    _widen_edges(edg, lanes)

    cross = ["--sidewalks.guess", "--crossings.guess", "--walkingareas",
             "--default.sidewalk-width", str(sidewalk_width),
             "--no-turnarounds"]

    trial = os.path.join(work, "trial.net.xml")
    _run(["netconvert", "--node-files", nod, "--edge-files", edg,
          "--output-file", trial] + cross)
    links, count = _new_links(trial)

    # Regenerate the program from phase semantics.
    out = ['<tlLogics version="1.20" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
           'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/tllogic_file.xsd">']
    for tid, sem in semantics.items():
        n = count[tid] + 1
        out.append(f'    <tlLogic id="{tid}" type="static" programID="0" offset="0">')
        for p_idx, edge_char in enumerate(sem):
            chars = []
            for idx in range(n):
                li = links[tid].get(idx)
                if li is None or li[0] == "xing":
                    chars.append("r")
                    continue
                _, from_edge, direction = li
                st = edge_char.get(from_edge, "r")
                if st == "G":
                    chars.append("g" if direction[:1] in ("l", "t") else "G")
                elif st == "y":
                    chars.append("y")
                else:
                    chars.append("r")
            out.append(f'        <phase duration="{durations[tid][p_idx]}" state="{"".join(chars)}"/>')
        out.append("    </tlLogic>")
        veh = sum(1 for v in links[tid].values() if v[0] == "veh")
        print(f"  {tid}: {veh} vehicle links + {n - veh} crossings, {len(sem)} phases")
    out.append("</tlLogics>")

    custom = os.path.join(work, "custom.tll.xml")
    with open(custom, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")

    _run(["netconvert", "--node-files", nod, "--edge-files", edg,
          "--tllogic-files", custom, "--output-file", output_net] + cross)
    print(f"Rebuilt ({lanes} lanes + sidewalks + crossings): {output_net}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("original_net")
    ap.add_argument("output_net")
    ap.add_argument("--lanes", type=int, default=2)
    ap.add_argument("--sidewalk-width", type=float, default=2.0)
    args = ap.parse_args()
    rebuild(args.original_net, args.output_net, lanes=args.lanes,
            sidewalk_width=args.sidewalk_width)
