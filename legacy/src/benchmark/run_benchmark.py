"""
Benchmark lab for the Crossroad intersection.

Compares traffic-light control strategies on identical, reproducible traffic and
writes statistics you can analyse later:

    * fixed-time  -> a conventional "normal" signal on a fixed cycle (baseline)
    * ppo         -> the trained reinforcement-learning policy
    * random      -> a noisy lower bound (optional)

Each strategy is run across several traffic regimes (e.g. "normal" vs "rush")
and several random seeds. For every run we record per-vehicle waiting time,
time loss, travel time, completed-trip throughput, and queue lengths.

Outputs (written to <root>/results/):
    benchmark_runs.csv      one row per (controller, regime, seed)
    benchmark_summary.csv   mean +/- std aggregated over seeds
    benchmark_<metric>.png  grouped bar charts (if matplotlib is available)

Examples:
    python src/benchmark/run_benchmark.py
    python src/benchmark/run_benchmark.py --seeds 10 --model 2M --include-random
    python src/benchmark/run_benchmark.py --regimes normal rush ew_rush
"""
import os
import sys
import csv
import argparse
import statistics
import warnings

# Quiet, clean console.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
CROSSROAD_DIR = os.path.join(ROOT_DIR, "src", "crossroad")
sys.path.insert(0, CROSSROAD_DIR)   # so wrapper_crossroad + traffic_generator import
sys.path.insert(0, SCRIPT_DIR)

CONFIG_PATH = os.path.join(ROOT_DIR, "envs", "crossroad", "env.sumocfg")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

from wrapper_crossroad import CrossroadEnv          # noqa: E402
from controllers import (                           # noqa: E402
    FixedTimeController, RandomController, PPOController,
)
from metrics import parse_tripinfo, QueueTracker, LOWER_IS_BETTER  # noqa: E402

# Traffic regimes: how busy and which axis is rushed.
REGIMES = {
    "normal":  {"scenario": "balanced", "intensity": 1.0},
    "rush":    {"scenario": "ns_rush",  "intensity": 1.8},
    "ew_rush": {"scenario": "ew_rush",  "intensity": 1.8},
}

# Metrics summarised in the printed report / charts.
REPORT_METRICS = [
    "mean_waiting_time", "mean_time_loss", "mean_queue",
    "max_queue", "completed_trips", "total_reward",
]


def load_ppo_controller(model_tag):
    """Load the trained PPO model and its VecNormalize statistics."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model_path = os.path.join(MODEL_DIR, f"ppo_crossroad_{model_tag}_final")
    norm_path = os.path.join(MODEL_DIR, f"ppo_crossroad_{model_tag}_vecnorm.pkl")

    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model not found: {model_path}.zip")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Normalizer not found: {norm_path}")

    model = PPO.load(model_path, device="cpu")

    # A throwaway vec env (never reset, so no SUMO is launched) lets us load the
    # saved normalization statistics for use on single observations.
    dummy = DummyVecEnv([lambda: CrossroadEnv(CONFIG_PATH)])
    vecnorm = VecNormalize.load(norm_path, dummy)
    vecnorm.training = False
    vecnorm.norm_reward = False

    return PPOController(model, vecnorm, name=f"ppo_{model_tag}")


def run_episode(controller, regime, seed, horizon):
    """Run one episode and return its metric row."""
    cfg = REGIMES[regime]
    tripinfo_path = os.path.join(
        RESULTS_DIR, f"_tmp_tripinfo_{controller.name}_{regime}_{seed}.xml"
    ).replace("(", "").replace(")", "").replace("=", "")

    env = CrossroadEnv(
        CONFIG_PATH,
        use_gui=False,
        rank=0,
        scenario=cfg["scenario"],
        traffic_seed=seed,
        traffic_intensity=cfg["intensity"],
        tripinfo_path=tripinfo_path,
        max_episode_steps=horizon,
    )

    controller.reset()
    queue = QueueTracker()
    obs, _ = env.reset(seed=seed)

    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        action = controller.act(obs)
        obs, reward, done, _, _ = env.step(action)
        total_reward += float(reward)
        queue.record(sum(env._get_queue_lengths()))
        steps += 1

    env.close()

    trip = parse_tripinfo(tripinfo_path)
    if os.path.exists(tripinfo_path):
        try:
            os.remove(tripinfo_path)
        except OSError:
            pass

    row = {
        "controller": controller.name,
        "regime": regime,
        "scenario": cfg["scenario"],
        "intensity": cfg["intensity"],
        "seed": seed,
        "steps": steps,
        "total_reward": round(total_reward, 3),
    }
    row.update({k: round(v, 3) for k, v in trip.items()})
    row.update({k: round(v, 3) for k, v in queue.summary().items()})
    return row


def aggregate(rows):
    """Group rows by (controller, regime) and compute mean / std per metric."""
    metric_keys = [k for k in rows[0]
                   if k not in ("controller", "regime", "scenario", "intensity", "seed")]
    groups = {}
    for r in rows:
        groups.setdefault((r["controller"], r["regime"]), []).append(r)

    summary = []
    for (controller, regime), grp in sorted(groups.items()):
        entry = {"controller": controller, "regime": regime, "n_seeds": len(grp)}
        for k in metric_keys:
            vals = [g[k] for g in grp]
            entry[f"{k}_mean"] = round(statistics.mean(vals), 3)
            entry[f"{k}_std"] = round(statistics.pstdev(vals), 3) if len(vals) > 1 else 0.0
        summary.append(entry)
    return summary, metric_keys


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_report(summary):
    print("\n" + "=" * 78)
    print("BENCHMARK SUMMARY (mean over seeds; lower is better except throughput/reward)")
    print("=" * 78)
    regimes = sorted({s["regime"] for s in summary})
    for regime in regimes:
        print(f"\n--- Regime: {regime} ---")
        header = f"{'controller':<22}" + "".join(f"{m.replace('mean_',''):>16}" for m in REPORT_METRICS)
        print(header)
        print("-" * len(header))
        for s in sorted(summary, key=lambda x: x["controller"]):
            if s["regime"] != regime:
                continue
            line = f"{s['controller']:<22}"
            for m in REPORT_METRICS:
                line += f"{s.get(m + '_mean', 0.0):>16.2f}"
            print(line)


def make_charts(summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[charts skipped] matplotlib not installed.")
        return

    controllers = sorted({s["controller"] for s in summary})
    regimes = sorted({s["regime"] for s in summary})
    lookup = {(s["controller"], s["regime"]): s for s in summary}

    import numpy as np
    for metric in REPORT_METRICS:
        x = np.arange(len(regimes))
        width = 0.8 / max(1, len(controllers))
        fig, ax = plt.subplots(figsize=(1.6 * len(regimes) + 3, 4.5))
        for i, ctrl in enumerate(controllers):
            means = [lookup.get((ctrl, rg), {}).get(metric + "_mean", 0.0) for rg in regimes]
            errs = [lookup.get((ctrl, rg), {}).get(metric + "_std", 0.0) for rg in regimes]
            ax.bar(x + i * width, means, width, yerr=errs, capsize=3, label=ctrl)
        ax.set_xticks(x + width * (len(controllers) - 1) / 2)
        ax.set_xticklabels(regimes)
        better = "lower is better" if metric in LOWER_IS_BETTER else "higher is better"
        ax.set_title(f"{metric}  ({better})")
        ax.set_ylabel(metric)
        ax.legend()
        fig.tight_layout()
        out = os.path.join(RESULTS_DIR, f"benchmark_{metric}.png")
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  chart: {os.path.relpath(out, ROOT_DIR)}")


def main():
    parser = argparse.ArgumentParser(description="Crossroad control-strategy benchmark")
    parser.add_argument("--seeds", type=int, default=5, help="number of seeds (0..N-1)")
    parser.add_argument("--model", type=str, default="2M", choices=["2M", "500k"])
    parser.add_argument("--regimes", nargs="+", default=["normal", "rush"],
                        choices=list(REGIMES.keys()))
    parser.add_argument("--hold", type=int, default=2,
                        help="fixed-time green duration in macro-steps")
    parser.add_argument("--horizon", type=int, default=5400,
                        help="max simulated steps per episode (let traffic clear)")
    parser.add_argument("--include-random", action="store_true",
                        help="also benchmark a random controller")
    parser.add_argument("--no-charts", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    seeds = list(range(args.seeds))

    controllers = [FixedTimeController(hold=args.hold)]
    if args.include_random:
        controllers.append(RandomController(seed=0))
    try:
        controllers.append(load_ppo_controller(args.model))
    except FileNotFoundError as e:
        print(f"[WARNING] PPO controller unavailable: {e}")
        print("          Continuing with baselines only.")

    print(f"Controllers: {[c.name for c in controllers]}")
    print(f"Regimes:     {args.regimes}")
    print(f"Seeds:       {seeds}")
    print(f"Horizon:     {args.horizon} sim steps\n")

    rows = []
    total = len(controllers) * len(args.regimes) * len(seeds)
    done = 0
    for controller in controllers:
        for regime in args.regimes:
            for seed in seeds:
                row = run_episode(controller, regime, seed, args.horizon)
                rows.append(row)
                done += 1
                print(f"[{done:>3}/{total}] {controller.name:<22} {regime:<8} "
                      f"seed={seed} | wait={row['mean_waiting_time']:7.2f} "
                      f"queue={row['mean_queue']:6.2f} trips={row['completed_trips']}")

    runs_csv = os.path.join(RESULTS_DIR, "benchmark_runs.csv")
    write_csv(runs_csv, rows)

    summary, _ = aggregate(rows)
    summary_csv = os.path.join(RESULTS_DIR, "benchmark_summary.csv")
    write_csv(summary_csv, summary)

    print_report(summary)
    print(f"\nPer-run rows : {os.path.relpath(runs_csv, ROOT_DIR)}")
    print(f"Summary      : {os.path.relpath(summary_csv, ROOT_DIR)}")
    if not args.no_charts:
        make_charts(summary)
    print("\nDone.")


if __name__ == "__main__":
    main()
