"""
Record a short GIF of MAX-PRESSURE control working on the real Cologne network,
zoomed onto one busy intersection (traffic flowing, lights switching with proper
yellow transitions).

    python tools/record_demo.py        # writes docs/maxpressure_demo.gif

Uses raw TraCI + sumo-gui directly (reliable), implementing max-pressure control
on the real signal programs. Opens sumo-gui briefly to render the frames.
"""
import os
import sys
import glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci          # noqa: E402
import sumolib        # noqa: E402
from PIL import Image  # noqa: E402

CFG = os.path.join(ROOT, "envs", "cologne8", "cologne8.sumocfg")
NET = os.path.join(ROOT, "envs", "cologne8", "cologne8.net.xml")
OUT = os.path.join(ROOT, "docs", "maxpressure_demo.gif")
FRAME_DIR = os.path.join(ROOT, "results", "frames")
MIN_GREEN, YELLOW = 6, 3

os.makedirs(FRAME_DIR, exist_ok=True)
for f in glob.glob(os.path.join(FRAME_DIR, "*.png")):
    os.remove(f)

# Busiest signalized junction, to zoom on.
net = sumolib.net.readNet(NET)
busy = max(net.getTrafficLights(), key=lambda t: len(t.getConnections()))
cx, cy = net.getNode(busy.getID()).getCoord()

traci.start(["sumo-gui", "-c", CFG, "--start", "--quit-on-end", "--delay", "0",
             "--window-size", "920,660", "--no-warnings", "--no-step-log"])


def yellow_of(state):
    return "".join("y" if c in "Gg" else c for c in state)


# Per traffic light: the green phase states + the (in,out) lane pairs each serves.
sig = {}
for tid in traci.trafficlight.getIDList():
    phases = traci.trafficlight.getAllProgramLogics(tid)[0].phases
    links = traci.trafficlight.getControlledLinks(tid)
    greens, pairs = [], []
    for ph in phases:
        st = ph.state
        if "y" in st.lower() or not ("G" in st or "g" in st):
            continue
        greens.append(st)
        pairs.append([(links[k][0][0], links[k][0][1]) for k, c in enumerate(st)
                      if c in "Gg" and k < len(links) and links[k]])
    sig[tid] = {"states": greens, "pairs": pairs, "cur": 0, "tig": 0, "yt": -1, "tgt": 0}
    traci.trafficlight.setRedYellowGreenState(tid, greens[0])

all_lanes = {l for d in sig.values() for pr in d["pairs"] for p in pr for l in p}


def control():
    t = traci.simulation.getTime()
    q = {l: traci.lane.getLastStepHaltingNumber(l) for l in all_lanes}
    for tid, d in sig.items():
        if d["yt"] >= 0:                                   # mid yellow
            if t >= d["yt"]:
                traci.trafficlight.setRedYellowGreenState(tid, d["states"][d["tgt"]])
                d["cur"], d["yt"], d["tig"] = d["tgt"], -1, 0
        else:
            d["tig"] += 1
            if d["tig"] >= MIN_GREEN:
                best = max(range(len(d["states"])),
                           key=lambda i: sum(q[a] - q[b] for a, b in d["pairs"][i]))
                if best != d["cur"]:
                    traci.trafficlight.setRedYellowGreenState(tid, yellow_of(d["states"][d["cur"]]))
                    d["tgt"], d["yt"] = best, t + YELLOW


# Warm up so traffic builds, then zoom in and record one frame per second.
for _ in range(120):
    control(); traci.simulationStep()
traci.gui.setBoundary("View #0", cx - 92, cy - 66, cx + 92, cy + 66)

frames = []
for i in range(70):
    fp = os.path.join(FRAME_DIR, f"f{i:03d}.png")
    traci.gui.screenshot("View #0", fp)
    control(); traci.simulationStep()
    frames.append(fp)

traci.close()

imgs = [Image.open(f).convert("RGB") for f in frames if os.path.exists(f)]
imgs[0].save(OUT, save_all=True, append_images=imgs[1:], duration=160, loop=0, optimize=True)
print(f"saved {OUT} ({len(imgs)} frames)")
