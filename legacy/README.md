# Legacy — Original synthetic-intersection project

This folder is the **original version of the project**: three hand-built
synthetic SUMO intersections and the PPO training / evaluation / benchmark code
written around them. It is the work the **initial report was based on** and is
preserved here for reproducibility.

In the current direction (real-city networks, see the top-level `README.md`),
these synthetic environments are kept as **controlled unit tests / debugging
fixtures** — small, fast, fully understood scenarios — rather than as the
headline result.

## Contents

```
legacy/
├── src/
│   ├── T_junction/      train, evaluate, wrapper (single agent, 3-way)
│   ├── crossroad/       train, evaluate, wrapper, traffic_generator (single agent, 4-way)
│   ├── multi_agent/     train, evaluate, wrapper (centralized PPO, 2-light corridor)
│   └── benchmark/       fixed-time vs PPO vs random comparison + watch.py
├── envs/
│   ├── T_junction/      net + routes + sumocfg + buildings
│   ├── crossroad/       net + dynamic route generation + sumocfg + buildings
│   ├── boulevard_coordonne/   2-light corridor
│   └── gui_settings.xml shared GUI theme
├── models/              trained policies + VecNormalize stats (git-ignored)
└── tensorboard_logs/    training curves (git-ignored)
```

The three networks were rebuilt with realistic 2-lane roads, sidewalks and
pedestrian crossings (see `tools/` at the repo root). They render as proper
neighbourhoods in `sumo-gui`.

## Running (from the repo root)

```bash
# Train
python legacy/src/T_junction/train_T_junction.py --mode train
python legacy/src/crossroad/train_crossroad.py --mode train
python legacy/src/multi_agent/train_marl.py

# Evaluate (opens sumo-gui)
python legacy/src/T_junction/evaluate_T_junction.py --mode final
python legacy/src/crossroad/evaluate_crossroad.py --mode 2M
python legacy/src/multi_agent/evaluate_marl.py

# Benchmark: fixed-time vs trained PPO across traffic regimes
python legacy/src/benchmark/run_benchmark.py
# Watch one scenario live in sumo-gui
python legacy/src/benchmark/watch.py --controller ppo --regime rush --seed 0
```

> The scripts resolve paths relative to this `legacy/` folder, so models and
> environments stay self-contained here.

## Known result / caveat

The benchmark showed the trained PPO policies are **roughly tied with a tuned
fixed-time baseline** — they push more throughput in rush but don't reduce
average delay, because the reward is throughput-weighted. This (and the move to
realistic, demand-driven networks) is what motivated the current real-city
direction. See the top-level `README.md`.
