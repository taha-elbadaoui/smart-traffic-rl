# 🚦 Reinforcement Learning for Urban Traffic Signal Control

> **Institution:** Institut National des Postes et Télécommunications (INPT), Rabat
> **Specialization:** Advanced Software Engineering for Digital Services (ASEDS)
> **Authors:** Taha El Badaoui & Walid Hazzam

Adaptive traffic-signal control with **Proximal Policy Optimization (PPO)** on the
**SUMO** simulator. The project replaces fixed-timing signal plans with learned
policies that react to live traffic.

## 🧭 Project direction

The project moved from **synthetic toy intersections** to **real city networks
with realistic, demand-driven traffic**:

- **Real networks** imported from OpenStreetMap (real roads, lanes, signals, buildings).
- **Realistic demand** — activity-based trips (e.g. the TAPAS model for Cologne),
  not random spawns.
- **Goal:** coordinated multi-intersection control on a real district, compared
  against the real fixed-time signals.

The original three synthetic environments (T-junction, crossroad, boulevard) that
the first report was built on now live in [`legacy/`](legacy/README.md) and serve
as small, controlled unit tests.

> 📒 **Decisions & results are tracked in [`PROJECT_LOG.md`](PROJECT_LOG.md)** —
> architecture choices (with rationale) and the benchmark table. Update it as the
> project evolves.

## 📂 Repository structure

```text
envs/
├── cologne8/         Real 8-intersection district of Cologne (RESCO + TAPAS demand)
├── rabat_real/       Real central-Rabat area imported from OSM (proof of concept)
└── gui_settings.xml  Shared sumo-gui theme

tools/                Network-building utilities
├── decorate_network.py     procedural "city" of building/park polygons
├── build_realistic_net.py  add sidewalks + crossings to a net
└── rebuild_net.py          widen lanes + regenerate signal program

docs/                 Rendered previews
legacy/               Original synthetic-intersection project (see legacy/README.md)
```

## 🌍 Real-city scenarios

### Cologne — 8 intersections (current focus)

A real district of Cologne, Germany: real road network and signals, **2,046 real
TAPAS activity-based trips** over the 07:00–08:00 peak, and **5,202 OSM building
footprints**.

![Cologne district in sumo-gui](docs/preview_cologne_full.png)

```bash
sumo-gui -c envs/cologne8/cologne8.sumocfg
# Press Play; sim starts at 25200s (07:00). Give it a few sim-seconds to fill up.
```

### Rabat — OSM proof of concept

A real area of central Rabat imported from OpenStreetMap (real roads + buildings +
plausible synthetic demand). See [`envs/rabat_real/BUILD.md`](envs/rabat_real/BUILD.md)
for the full `OSM → netconvert → polyconvert → randomTrips` pipeline.

```bash
sumo-gui -c envs/rabat_real/rabat.sumocfg
```

> **On "real traffic":** road networks, lanes, signals and buildings are real
> (OSM). Vehicle *demand* is real activity-based data where available (Cologne /
> TAPAS) and otherwise realistic synthetic demand (`randomTrips`) — there is no
> open dataset of measured counts for Rabat.

## ⚙️ Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

Requires **SUMO** installed with `SUMO_HOME` set:

```bash
setx SUMO_HOME "C:\Program Files (x86)\Eclipse\Sumo"   # Windows
export SUMO_HOME=/usr/share/sumo                        # Linux/macOS
```

> **Performance note:** training is bound by the SUMO simulation, not the small
> policy network — it runs on **CPU** with in-process **libsumo**; a GPU does not
> help.

## 🧠 RL formulation

| Element | Description |
| ------- | ----------- |
| **State** | per-approach queue lengths, current phase, phase duration (normalized) |
| **Action** | which movement group gets green (synthetic envs) / per-junction phase choice (real, planned) |
| **Reward** | − waiting time − queue − imbalance + throughput − switching penalty |
| **Algorithm** | PPO (Stable-Baselines3), parallel `SubprocVecEnv`, `VecNormalize` |

## 🗺️ Roadmap

- [x] Synthetic single- and multi-agent environments (now in `legacy/`)
- [x] Parallel PPO, benchmark lab, realistic rebuilt networks
- [x] Real OSM import pipeline (Rabat proof)
- [x] Real demand-driven district (Cologne 8 intersections)
- [ ] **Generic multi-phase traffic-light wrapper** for real intersections
- [ ] Multi-agent PPO control on the Cologne district
- [ ] Benchmark vs real fixed-time + actuated baselines

## 📄 License & authors

Developed for the **ASEDS** curriculum at **INPT Rabat** for academic and research
purposes. Authors: **Taha El Badaoui** and **Walid Hazzam**.
