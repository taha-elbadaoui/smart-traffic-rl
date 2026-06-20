# Project Log — Decisions & Results

A running record of **architectural decisions** (why we chose X over Y) and
**measured results** (baselines, model performance). Update it whenever you make
a non-trivial choice or get a number worth keeping.

> **How to use:** add a new `## ADR-NN` entry for each decision (newest at the
> bottom), and append rows to the **Results** tables as experiments finish. Keep
> entries short — decision, why, alternatives, status.

---

## 🧭 Current direction (one-liner)

Multi-agent traffic-signal control on a **real Cologne district** (`sumo-rl`),
reward = **pressure**, working toward a **CoLight** graph-attention policy,
benchmarked against fixed-time and max-pressure. Synthetic envs archived in
`legacy/`.

---

## 📋 Decision log (ADRs)

### ADR-01 — Two repositories (graded vs dev)
- **Date:** 2026-06
- **Decision:** Keep the graded repo (`origin`) untouched; do all new work on a
  private `dev` remote.
- **Why:** Protect the submitted/graded version while experimenting freely.
- **Status:** Active. Local `main` tracks `dev`.

### ADR-02 — Train on CPU with in-process libsumo
- **Decision:** `device="cpu"`, libsumo (not socket TraCI), `torch.set_num_threads(1)`,
  8 parallel envs.
- **Why:** Training is bound by SUMO simulation, not the tiny MLP. GPU transfer
  overhead outweighs the gain; CPU + libsumo + subscriptions is fastest.
- **Alternatives:** GPU (rejected — no benefit for small nets).
- **Status:** Active.

### ADR-03 — Realistic networks (sidewalks, crossings, multi-lane)
- **Decision:** Rebuild nets with sidewalks + pedestrian crossings, preserving
  the traffic-light phase structure (`tools/build_realistic_net.py`,
  `rebuild_net.py`).
- **Why:** Realism for visualization/credibility without breaking trained models.
- **Status:** Done for the legacy synthetic envs.

### ADR-04 — Pivot to real-city networks
- **Date:** 2026-06-20
- **Decision:** Move from synthetic toy intersections to **real OSM-imported city
  networks**; archive the synthetic envs to `legacy/` as unit tests.
- **Why:** Separated T-junction/crossroad/boulevard is scaffolding, not a result.
  Real networks + real demand are the credible story.
- **Status:** Active.

### ADR-05 — City & data source: Cologne (RESCO / TAPAS)
- **Date:** 2026-06-20
- **Decision:** Use the **Cologne 8-intersection** scenario (RESCO benchmark):
  real net + signals + **2046 real TAPAS activity-based trips** over the 07:00–08:00
  peak + OSM buildings.
- **Why:** Real network *and* real-derived demand; it's the standard RL
  signal-control benchmark (comparable to published work). Kaggle traffic data
  (METR-LA/PEMS) is forecasting sensor data, not SUMO demand — not usable here.
- **Alternatives:** Luxembourg (LuST), Monaco (MoST) — also good; Cologne chosen
  for RL-benchmark credibility.
- **Status:** Active.

### ADR-06 — Reuse `sumo-rl` instead of a hand-built wrapper
- **Date:** 2026-06-20
- **Decision:** Build the RL environment on **`sumo-rl`** rather than hand-rolling
  a generic multi-phase TLS wrapper.
- **Why:** It handles each real junction's variable phase count generically, is
  battle-tested, SB3/PettingZoo-ready, and gives credibility/comparability. Our
  effort goes into the *experiment*, not the plumbing.
- **Alternatives:** Build from scratch (slower, reinvents); pure RESCO harness
  (more opinionated).
- **Status:** Active. Factory: `src/realcity/cologne_env.py`.

### ADR-07 — Pressure-based reward
- **Date:** 2026-06-20
- **Decision:** Reward = **pressure** (incoming − outgoing queues), not throughput.
- **Why:** The earlier throughput-weighted reward made PPO only *tie* fixed-time
  (it maximized throughput, not delay). Pressure is the proven objective.
- **Status:** Active.

### ADR-08 — Core model: CoLight (via an IPPO foundation)
- **Date:** 2026-06-20
- **Decision:** Target a **CoLight-style graph-attention** multi-agent policy;
  reach it through a shared-parameter **IPPO** baseline first.
- **Why:** CoLight is the SOTA frontier (agents coordinate over the road graph) —
  the headline novelty. IPPO first because graph MARL can't be debugged without a
  working non-graph baseline.
- **Status:** In progress (IPPO next).

---

## 📊 Results

### Baselines & models — real Cologne district, 07:00–08:00 peak (2046 trips)

| Controller | Mean wait (s) | Mean time loss (s) | Mean travel (s) | Trips done | Notes |
|-----------|--------------:|-------------------:|----------------:|-----------:|-------|
| fixed-time (real signals) | 29.27 | 49.26 | 114.39 | 1995 | floor |
| **max-pressure** | **6.46** | **24.69** | **90.28** | **2015** | **−78% wait vs fixed-time — the real bar** |
| IPPO (shared param) | _todo_ | | | | learned baseline |
| CoLight | _todo_ | | | | graph-attention (goal) |

*Lower is better for wait / time loss / travel. Reproduce with
`python src/realcity/baselines.py`. **Key finding:** classic max-pressure already
beats the real fixed-time signals massively — so the meaningful research question
is whether learned control (CoLight) can beat **max-pressure**, which is hard.*

### Legacy synthetic envs (for reference)

| Env | Result | Notes |
|-----|--------|-------|
| T-junction | PPO converged ~3910, EV ~0.03 | saturated/simple; ≈ tied fixed-time |
| Crossroad | PPO EV 0.2–0.34 | ≈ tied fixed-time; +5.7% throughput in rush, slightly worse delay → motivated the pressure-reward + real-city pivot |

---

## 🧱 Milestones

- 2026-06 — Optimized synthetic envs (libsumo, subscriptions), benchmark lab, realistic networks.
- 2026-06-20 — Real OSM import pipeline (Rabat proof); real Cologne district + demand; restructure into `legacy/` + `realcity/`; sumo-rl foundation + fixed-time baseline.
