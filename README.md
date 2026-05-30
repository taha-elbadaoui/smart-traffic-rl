# 🚦 Traffic Flow Optimization via Reinforcement Learning

> **Institution:** INPT Rabat (ASEDS)
> **Team:** Taha El Badaoui & Walid Hazzam

---

## 📝 Project Overview

This project designs a **Multi-Objective Reinforcement Learning (RL)** system to optimize traffic signal control at various intersection types (T-Junctions and Crossroads).

### Goals

* Maximize traffic throughput
* Minimize waiting time
* Reduce environmental impact (CO₂ emissions & fuel consumption)

A **single-agent architecture** is used to deeply optimize state-space representation and reward shaping, avoiding the complexity of multi-agent coordination while providing a strong custom baseline.

---

## 🛠 Tech Stack

| Component       | Technology                                                       |
| --------------- | ---------------------------------------------------------------- |
| Simulator       | SUMO (Simulation of Urban MObility)                              |
| RL Framework    | Stable-Baselines3 (DQN → PPO)                                    |
| Environment     | Custom `gymnasium.Env` wrappers (`TJunctionEnv`, `CrossroadEnv`) |
| Training Mode   | `libsumo` — headless C++ bindings (high-speed)                   |
| Evaluation Mode | `traci` — GUI via `sumo-gui` (visual verification)               |

---

## 🚀 Key Features

* ⚡ **High-Speed Training** — Bypasses GUI and TCP overhead via C++ bindings (`libsumo`) during the training loop.
* 👁️ **Visual Verification** — Dynamically switches to `traci` post-training for human observation of agent behavior.
* 🚗 **Queue-Based Rewards** — Minimizes the normalized sum of halting vehicles on incoming lanes.
* 🌱 **Environmental Awareness** *(Upcoming)* — Real-time CO₂ and fuel tracking to penalize harsh braking and inefficient flow patterns.

---

## 💻 Getting Started

### 1. Setup

Ensure your virtual environment is active, then install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Headless Training (Maximum Speed)

Training defaults to `libsumo` to prevent GUI overhead. The repository currently supports two intersection environments.

#### T-Junction

```bash
python src/T_junction/train_T_junction.py --mode train
```

**Outputs:**

```text
models/ppo_t_junction_final.zip
models/vecnormalize_t_junction.pkl
```

#### Crossroad

```bash
python src/crossroad/train_crossroad.py --mode train
```

**Outputs:**

```text
models/ppo_crossroad_final.zip
models/vecnormalize_crossroad.pkl
```

> **Note:** You can also run the scripts with `--mode random` to generate an untrained baseline model for comparison.

---

### 3. Visual Evaluation (GUI Mode)

To load a saved model and observe its behavior via `sumo-gui`, execute the corresponding evaluation script:

```bash
# Evaluate T-Junction
python src/T_junction/evaluate_T_junction.py

# Evaluate Crossroad
python src/crossroad/evaluate_crossroad.py
```

---

## 📊 Understanding Training Logs

Stable-Baselines3 outputs a metrics table during training.

| Metric             | Description                                                   | Target                    |
| ------------------ | ------------------------------------------------------------- | ------------------------- |
| `ep_rew_mean`      | Average penalty per episode (negative sum of waiting cars)    | → 0 over time             |
| `ep_len_mean`      | Average steps until all scheduled cars clear the intersection | Lower is generally better |
| `exploration_rate` | Epsilon-greedy decay (starts around 0.50)                     | Decreasing                |
| `loss`             | Q-value prediction error                                      | Stabilizing               |

---

## 📅 Implementation Roadmap

### Phase 1 — Foundational Environments ✅ Completed

* [x] Build minimal intersection layouts (T-Junction & Crossroad) in NetEdit
* [x] Implement custom `gymnasium.Env` with dynamic `libsumo/traci` switching
* [x] Train and evaluate baseline agents (queue-based control) for both environments

### Phase 2 — Scaling & Realism 🚧 In Progress

* [ ] Simulate rush-hour traffic using `randomTrips.py` for dynamic `.rou.xml` generation
* [ ] Switch to PPO for improved stability in larger state spaces
* [ ] Centralize configuration and hyperparameters

### Phase 3 — Multi-Objective Optimization

* [ ] Integrate CO₂-based reward signals
* [ ] Benchmark against fixed-time and actuated traffic signals
* [ ] Visualize results (waiting time vs. emissions trade-offs)

---

## 📊 Expected Outcomes

* Reduced average waiting time at intersections
* Improved traffic throughput
* Lower emissions compared to traditional fixed-signal systems

---

## 📌 Future Work

* Multi-agent traffic networks
* Real-world sensor data integration
* Deployment on smart city infrastructure

---

## 📄 License

This project is developed as part of the ASEDS engineering curriculum at INPT Rabat for academic and research purposes.
