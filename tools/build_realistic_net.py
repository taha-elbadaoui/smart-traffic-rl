"""
Rebuild a SUMO network with realistic infrastructure — sidewalks and pedestrian
crossings — while preserving the existing traffic-light phase structure so the
RL wrappers and trained models keep working.

How it works:
  1. Extract the current network back into editable plain XML (keeps edge IDs).
  2. Read the existing traffic-light program(s): phases + vehicle link count V.
  3. Trial-build with crossings to learn the new total link count T per TLS.
     The extra C = T - V links are the appended pedestrian crossings.
  4. Write a custom program that reuses every original phase state and simply
     appends C red characters (crossings kept red — there are no pedestrians,
     and this leaves vehicle signalling byte-for-byte identical).
  5. Final build with sidewalks + crossings + the custom program.

Usage:
    python tools/build_realistic_net.py envs/crossroad/env.net.xml
        [--sidewalk-width 2.0] [--keep-source]
"""
import os
import sys
import argparse
import tempfile
import subprocess
import xml.etree.ElementTree as ET


def _run(cmd):
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stdout + "\n" + res.stderr + "\n")
        raise SystemExit(f"netconvert failed: {' '.join(cmd)}")


def _read_programs(path):
    """{tls_id: [(duration, state), ...]} from a .tll or .net file."""
    root = ET.parse(path).getroot()
    programs = {}
    for tl in root.iter("tlLogic"):
        programs[tl.get("id")] = [(p.get("duration"), p.get("state"))
                                  for p in tl.findall("phase")]
    return programs


def build(net_file, sidewalk_width=2.0, keep_source=False):
    work = tempfile.mkdtemp(prefix="realnet_")
    p = os.path.join(work, "p")

    # 1) Extract editable source (preserves edge IDs).
    _run(["netconvert", "--sumo-net-file", net_file, "--plain-output-prefix", p])
    nod, edg, con, tll = p + ".nod.xml", p + ".edg.xml", p + ".con.xml", p + ".tll.xml"

    # 2) Original vehicle programs (phase states + vehicle link count V).
    orig = _read_programs(tll)
    if not orig:
        raise SystemExit("No traffic light found in the source network.")

    cross_flags = ["--sidewalks.guess", "--crossings.guess", "--walkingareas",
                   "--default.sidewalk-width", str(sidewalk_width)]

    # 3) Trial build to learn the new total link count T per TLS.
    trial = os.path.join(work, "trial.net.xml")
    _run(["netconvert", "--node-files", nod, "--edge-files", edg,
          "--connection-files", con, "--output-file", trial] + cross_flags)
    trial_prog = _read_programs(trial)

    # 4) Custom program: original vehicle states + appended red crossings.
    lines = ['<tlLogics version="1.20" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
             'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/tllogic_file.xsd">']
    for tls_id, phases in orig.items():
        v = len(phases[0][1])
        t = len(trial_prog[tls_id][0][1])
        c = t - v
        if c < 0:
            raise SystemExit(f"Unexpected link shrink for {tls_id} ({v}->{t}).")
        lines.append(f'    <tlLogic id="{tls_id}" type="static" programID="0" offset="0">')
        for dur, state in phases:
            lines.append(f'        <phase duration="{dur}" state="{state + "r" * c}"/>')
        lines.append('    </tlLogic>')
        print(f"  {tls_id}: {v} vehicle links + {c} crossing links = {t}")
    lines.append('</tlLogics>')

    custom = os.path.join(work, "custom.tll.xml")
    with open(custom, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # 5) Final build with sidewalks + crossings + the preserved program.
    _run(["netconvert", "--node-files", nod, "--edge-files", edg,
          "--connection-files", con, "--tllogic-files", custom,
          "--output-file", net_file] + cross_flags)

    if keep_source:
        env_dir = os.path.dirname(net_file)
        for src, name in [(nod, "env.src.nod.xml"), (edg, "env.src.edg.xml"),
                          (con, "env.src.con.xml"), (custom, "env.src.tll.xml")]:
            with open(src, encoding="utf-8") as fr, \
                 open(os.path.join(env_dir, name), "w", encoding="utf-8") as fw:
                fw.write(fr.read())

    print(f"Rebuilt (sidewalks + crossings): {net_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rebuild a SUMO net with sidewalks + crossings")
    ap.add_argument("net_file")
    ap.add_argument("--sidewalk-width", type=float, default=2.0)
    ap.add_argument("--keep-source", action="store_true",
                    help="also write the plain source files into the env folder")
    args = ap.parse_args()
    build(args.net_file, sidewalk_width=args.sidewalk_width, keep_source=args.keep_source)
