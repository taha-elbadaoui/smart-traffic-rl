# 🧪 Benchmark Lab — Crossroad

Compare traffic-light control strategies on **identical, reproducible traffic**
and produce statistics you can analyse later.

## What it compares

| Controller    | Description                                                        |
| ------------- | ------------------------------------------------------------------ |
| `fixed_time`  | A conventional signal on a fixed cycle (the "normal traffic light" baseline). Ignores traffic. |
| `ppo`         | The trained PPO policy (`models/ppo_crossroad_<tag>_final`).        |
| `random`      | Random phase each step — a noisy lower bound (optional).            |

Each controller runs across **traffic regimes** and several **seeds**:

| Regime    | Scenario   | Intensity | Meaning                  |
| --------- | ---------- | --------- | ------------------------ |
| `normal`  | balanced   | 1.0       | calm, even traffic       |
| `rush`    | ns_rush    | 1.8       | heavy North/South demand |
| `ew_rush` | ew_rush    | 1.8       | heavy East/West demand   |

Fairness: for a given seed, **all controllers see the exact same traffic**
(same generated routes + same SUMO `--seed`), so differences are due to the
control strategy alone.

## Metrics

Per-vehicle stats come from SUMO's `--tripinfo-output`; queue stats are sampled
online each macro-step:

- `mean_waiting_time` — avg seconds stopped per vehicle *(lower is better)*
- `mean_time_loss` — avg delay vs free-flow *(lower is better)*
- `mean_travel_time` — avg trip duration *(lower is better)*
- `mean_queue` / `max_queue` — halting vehicles across all approaches *(lower is better)*
- `completed_trips` — throughput within the horizon *(higher is better)*
- `total_reward` — cumulative env reward *(higher is better)*

## Usage

```bash
# default: fixed_time vs ppo (2M), regimes {normal, rush}, 5 seeds, with charts
python src/benchmark/run_benchmark.py

# more seeds, add the random lower bound, evaluate the 500k model
python src/benchmark/run_benchmark.py --seeds 10 --include-random --model 500k

# add the East/West rush regime, longer fixed-time green, no charts
python src/benchmark/run_benchmark.py --regimes normal rush ew_rush --hold 3 --no-charts
```

### Options

| Flag               | Default        | Meaning                                       |
| ------------------ | -------------- | --------------------------------------------- |
| `--seeds N`        | 5              | run seeds `0..N-1`                             |
| `--model {2M,500k}`| 2M             | which trained crossroad model to load         |
| `--regimes ...`    | normal rush    | any of `normal rush ew_rush`                  |
| `--hold N`         | 2              | fixed-time green duration in macro-steps (~25 s each) |
| `--horizon N`      | 5400           | max simulated steps per episode               |
| `--include-random` | off            | also benchmark a random controller            |
| `--no-charts`      | off            | skip PNG generation                           |

## Outputs (`results/`)

- `benchmark_runs.csv` — one row per (controller, regime, seed); raw data for your own analysis.
- `benchmark_summary.csv` — mean ± std aggregated over seeds.
- `benchmark_<metric>.png` — grouped bar charts (controllers × regimes, with error bars).

> `results/` is git-ignored — regenerate locally any time.

## Extending

- **More regimes:** add an entry to `REGIMES` in `run_benchmark.py`.
- **A new controller:** subclass `Controller` in `controllers.py` (implement
  `reset()` and `act(obs) -> 0|1`) and add it to the list in `main()`.
- **Other intersections:** the same pattern applies to the T-junction once its
  wrapper exposes the same `scenario` / `traffic_seed` / `tripinfo_path` hooks.
