"""
Watch a single benchmark scenario play out live in sumo-gui.

Runs ONE controller on ONE traffic regime + seed, with the SUMO GUI open, so you
can visually compare strategies. Because traffic is seed-controlled, running the
two controllers on the same regime+seed shows them handling the *identical* cars.

Examples:
    # Watch the trained policy handle a rush-hour pattern
    python src/benchmark/watch.py --controller ppo --regime rush --seed 0

    # Watch the conventional fixed-time light on the same exact traffic
    python src/benchmark/watch.py --controller fixed_time --regime rush --seed 0

    # Slower / faster playback (ms between sim steps in the GUI)
    python src/benchmark/watch.py --controller ppo --regime normal --delay 150
"""
import os
import sys
import argparse

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
CROSSROAD_DIR = os.path.join(ROOT_DIR, "src", "crossroad")
sys.path.insert(0, CROSSROAD_DIR)
sys.path.insert(0, SCRIPT_DIR)

CONFIG_PATH = os.path.join(ROOT_DIR, "envs", "crossroad", "env.sumocfg")

from wrapper_crossroad import CrossroadEnv               # noqa: E402
from controllers import FixedTimeController, RandomController  # noqa: E402
from metrics import parse_tripinfo, QueueTracker         # noqa: E402
from run_benchmark import REGIMES, load_ppo_controller   # noqa: E402


def build_controller(kind, model_tag, hold):
    if kind == "fixed_time":
        return FixedTimeController(hold=hold)
    if kind == "random":
        return RandomController(seed=0)
    if kind == "ppo":
        return load_ppo_controller(model_tag)
    raise ValueError(kind)


def main():
    p = argparse.ArgumentParser(description="Watch one benchmark scenario in sumo-gui")
    p.add_argument("--controller", choices=["fixed_time", "ppo", "random"], default="ppo")
    p.add_argument("--regime", choices=list(REGIMES.keys()), default="rush")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", choices=["2M", "500k"], default="2M")
    p.add_argument("--hold", type=int, default=2, help="fixed-time green length (macro-steps)")
    p.add_argument("--horizon", type=int, default=5400)
    p.add_argument("--delay", type=int, default=120,
                   help="ms between simulation steps in the GUI (bigger = slower)")
    args = p.parse_args()

    cfg = REGIMES[args.regime]
    controller = build_controller(args.controller, args.model, args.hold)
    tripinfo_path = os.path.join(ROOT_DIR, "results", "_watch_tripinfo.xml")
    os.makedirs(os.path.dirname(tripinfo_path), exist_ok=True)

    env = CrossroadEnv(
        CONFIG_PATH,
        use_gui=True,
        scenario=cfg["scenario"],
        traffic_seed=args.seed,
        traffic_intensity=cfg["intensity"],
        tripinfo_path=tripinfo_path,
        max_episode_steps=args.horizon,
        # Auto-start playing, close when finished, and slow down so it's watchable.
        extra_sumo_args=["--start", "--quit-on-end", "--delay", str(args.delay)],
    )

    print(f"\n=== Watching: {controller.name} | regime={args.regime} "
          f"(scenario={cfg['scenario']}, intensity={cfg['intensity']}) | seed={args.seed} ===")
    print("The sumo-gui window will open and start automatically.\n")

    controller.reset()
    queue = QueueTracker()
    obs, _ = env.reset(seed=args.seed)
    total_reward, steps, done = 0.0, 0, False

    while not done:
        action = controller.act(obs)
        obs, reward, done, _, _ = env.step(action)
        total_reward += float(reward)
        q = sum(env._get_queue_lengths())
        queue.record(q)
        steps += 1
        phase = "N/S green" if int(action) == 0 else "E/W green"
        print(f"step {steps:4d} | {phase:<9} | queue={q:3d} | reward={reward:7.2f}")

    env.close()

    trip = parse_tripinfo(tripinfo_path)
    qs = queue.summary()
    if os.path.exists(tripinfo_path):
        try:
            os.remove(tripinfo_path)
        except OSError:
            pass

    print("\n=== Episode summary ===")
    print(f"  controller        : {controller.name}")
    print(f"  completed trips   : {trip['completed_trips']}")
    print(f"  mean waiting time : {trip['mean_waiting_time']:.2f} s")
    print(f"  mean time loss    : {trip['mean_time_loss']:.2f} s")
    print(f"  mean queue        : {qs['mean_queue']:.2f}")
    print(f"  max queue         : {qs['max_queue']:.0f}")
    print("\nTip: run the other controller with the SAME --regime and --seed to")
    print("     compare them on identical traffic.")


if __name__ == "__main__":
    main()
