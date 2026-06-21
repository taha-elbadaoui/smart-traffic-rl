"""
Final comparison on the real Cologne district — the result table.

Evaluates every controller on the same 07:00-08:00 peak with the same
throughput-aware metric and prints one table:

    fixed-time   (real OSM signals)
    max-pressure (classic adaptive)
    IPPO         (shared-param learned, no coordination)   [if trained]
    CoLight      (graph-attention learned, coordinated)    [if trained]

    python src/realcity/compare.py
"""
import os
import sys
import torch

os.environ.setdefault("LIBSUMO_AS_TRACI", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cologne_env import make_env, ROOT, PEAK_SECONDS, EXPECTED_TRIPS  # noqa: E402
from baselines import fixed_time, max_pressure  # noqa: E402
from ippo import Spaces, ActorCritic, evaluate  # noqa: E402
from colight import CoLightPolicy  # noqa: E402

MODELS = os.path.join(ROOT, "models")


def get_spaces():
    env = make_env(num_seconds=PEAK_SECONDS)
    env.reset()
    sp = Spaces(env)
    env.close()
    return sp


def score(m):
    return m["mean_time_loss"] + 1000.0 * (1.0 - m["completed_trips"] / EXPECTED_TRIPS)


def eval_model(kind, sp):
    path = os.path.join(MODELS, f"{kind}_cologne_best.pt")
    if not os.path.exists(path):
        return None
    net = (ActorCritic if kind == "ippo" else CoLightPolicy)(sp.MAXO, sp.MAXA)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    return evaluate(net, sp, tag=kind)


def row(name, m):
    if m is None:
        return f"{name:<14}{'(not trained yet)':>46}"
    return (f"{name:<14}{m['mean_waiting_time']:>10.2f}{m['mean_time_loss']:>14.2f}"
            f"{m['mean_travel_time']:>12.2f}{m['completed_trips']:>8}{score(m):>10.1f}")


def main():
    print("Evaluating controllers on real Cologne (07:00-08:00 peak)...\n")
    results = [
        ("fixed_time", fixed_time()),
        ("max_pressure", max_pressure()),
    ]
    sp = get_spaces()
    results.append(("IPPO", eval_model("ippo", sp)))
    results.append(("CoLight", eval_model("colight", sp)))

    print(f"{'controller':<14}{'wait(s)':>10}{'timeLoss(s)':>14}{'travel(s)':>12}"
          f"{'trips':>8}{'score':>10}")
    print("-" * 68)
    for name, m in results:
        print(row(name, m))
    print("\nLower is better everywhere (score = timeLoss penalized by unfinished trips).")


if __name__ == "__main__":
    main()
