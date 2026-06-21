"""
Visualize WHY the custom IPPO fails — and contrast with max-pressure.

Runs a controller on the Cologne network and reports, per junction, how often it
picks each phase. The failed IPPO policy collapses to (almost) one phase per
junction -> the cross directions never go green -> gridlock. Max-pressure keeps
switching to serve whatever is queued -> traffic flows.

    python src/realcity/visualize_policy.py --controller ippo --gui
    python src/realcity/visualize_policy.py --controller maxpressure --gui
    # add --screenshot to save a PNG into results/ (works headless too)
"""
import os
import sys
import argparse
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cologne_env import make_env, NET, ROOT  # noqa: E402
from ippo import Spaces, ActorCritic  # noqa: E402
from baselines import _phase_movements  # noqa: E402


def ippo_controller(env, sp):
    policy = ActorCritic(sp.MAXO, sp.MAXA)
    policy.load_state_dict(torch.load(os.path.join(ROOT, "models", "ippo_cologne_best.pt"),
                                      map_location="cpu"))
    policy.eval()

    def act(obs):
        obs_t = torch.stack([torch.from_numpy(sp.pad(ts, obs[ts])) for ts in sp.ids])
        mask_t = torch.stack([sp.mask[ts] for ts in sp.ids])
        with torch.no_grad():
            logits, _ = policy(obs_t, mask_t)
            a = torch.argmax(logits, dim=-1)
        return {ts: int(a[i]) for i, ts in enumerate(sp.ids)}
    return act


def maxpressure_controller(env, sp):
    movements = _phase_movements(env)
    conn = env.traffic_signals[sp.ids[0]].sumo

    def act(obs):
        lanes = {l for ts in sp.ids for ph in movements[ts] for pair in ph for l in pair}
        q = {l: conn.lane.getLastStepHaltingNumber(l) for l in lanes}
        out = {}
        for ts in sp.ids:
            out[ts] = max(range(len(movements[ts])),
                          key=lambda i: sum(q[a] - q[b] for a, b in movements[ts][i]))
        return out
    return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["ippo", "maxpressure"], default="ippo")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--screenshot", action="store_true")
    args = ap.parse_args()

    # libsumo can't drive the GUI; baselines.py sets LIBSUMO_AS_TRACI, so clear it.
    if args.gui:
        os.environ.pop("LIBSUMO_AS_TRACI", None)

    env = make_env(use_gui=args.gui, num_seconds=args.steps * 5)
    obs = env.reset()
    sp = Spaces(env)
    act = (ippo_controller if args.controller == "ippo" else maxpressure_controller)(env, sp)

    phase_counts = {ts: Counter() for ts in sp.ids}
    total_queue = []
    conn = env.traffic_signals[sp.ids[0]].sumo
    all_lanes = {l for ts in sp.ids for l in env.traffic_signals[ts].lanes}

    done = False
    while not done:
        actions = act(obs)
        for ts in sp.ids:
            phase_counts[ts][actions[ts]] += 1
        obs, _, dones, _ = env.step(actions)
        total_queue.append(sum(conn.lane.getLastStepHaltingNumber(l) for l in all_lanes))
        done = dones.get("__all__", False)

    if args.screenshot:
        out = os.path.join(ROOT, "results", f"cologne_{args.controller}.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        try:
            conn.gui.screenshot("View #0", out)
            env.step({ts: 0 for ts in sp.ids})
            print(f"screenshot -> {out}")
        except Exception as e:
            print(f"(screenshot needs --gui) {e}")

    print(f"\n=== {args.controller} on Cologne ===")
    print(f"network-wide queue: mean={sum(total_queue)/len(total_queue):.0f} "
          f"max={max(total_queue)} halted vehicles")
    print("\nper-junction phase usage (collapsed = one phase dominates = gridlock):")
    for ts in sp.ids:
        c = phase_counts[ts]
        tot = sum(c.values())
        dist = " ".join(f"p{p}:{100*n//tot}%" for p, n in sorted(c.items()))
        top = max(c.values()) / tot
        flag = "  <-- COLLAPSED" if top > 0.9 else ""
        print(f"  {ts:<34} {dist}{flag}")
    env.close()


if __name__ == "__main__":
    main()
