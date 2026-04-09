# 🚦 Traffic Flow Optimization via Reinforcement Learning

> **Institution:** INPT Rabat (ASEDS)
> **Team:** Taha El Badaoui & Walid
> **Supervisor:** Prof. Zineb El Akkaoui
> **Timeline:** 3 Months (Part-Time)

---

## 📝 Project Overview

This project designs a **Multi-Objective Reinforcement Learning (RL)** system to optimize traffic signal control at intersections.

**Goals:**
- Maximize traffic throughput
- Minimize waiting time
- Reduce environmental impact (CO₂ emissions & fuel consumption)

A **single-agent architecture** is used to deeply optimize state-space representation and reward shaping — avoiding the complexity of multi-agent coordination while providing a strong, custom baseline.

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Simulator | [SUMO](https://eclipse.dev/sumo/) — Simulation of Urban MObility |
| RL Framework | [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (DQN → PPO) |
| Environment | Custom `gymnasium.Env` wrapper |
| Training Mode | `libsumo` — headless C++ bindings (high-speed) |
| Evaluation Mode | `traci` — GUI via `sumo-gui` (visual verification) |

---

## 🚀 Key Features

- ⚡ **High-Speed Training** — Bypasses GUI and TCP overhead via C++ bindings (`libsumo`) during the training loop
- 👁️ **Visual Verification** — Dynamically switches to `traci` post-training for human observation of agent behavior
- 🚗 **Queue-Based Rewards** — Minimizes the normalized sum of halting vehicles on incoming lanes
- 🌱 **Environmental Awareness** *(Upcoming)* — Real-time CO₂ and fuel tracking to penalize harsh braking and inefficient flow patterns

---

## 💻 Getting Started

### 1. Setup

Ensure your virtual environment is active, then install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Headless Training (Maximum Speed)

Defaults to `libsumo` to prevent GUI overhead during thousands of rapid iterations.

```bash
cd src
python train.py
```

**Output:** Saves trained model to `models/dqn_1x1_baseline.zip`

### 3. Visual Evaluation (GUI Mode)

Loads the saved model and opens `sumo-gui` via `traci`.

```bash
cd src
python evaluate.py
```

> **Note:** Press the **Play** button in the SUMO interface to start the simulation.

---

## 📊 Understanding Training Logs

Stable-Baselines3 outputs a metrics table during training. Key fields:

| Metric | Description | Target |
|---|---|---|
| `ep_rew_mean` | Average penalty per episode (negative sum of waiting cars) | → 0 over time |
| `ep_len_mean` | Avg. steps until all scheduled cars clear the intersection | — |
| `exploration_rate` | Epsilon-greedy decay (starts ~0.70, decays to 0.05) | Decreasing |
| `fps` | Simulation speed — should stay 500–1000+ FPS with `libsumo` | High |
| `loss` | Q-value prediction error; fluctuates, general stabilization is healthy | Stabilizing |

---

## 📅 Implementation Roadmap

### Phase 1 — "Hello World" Pipeline *(Month 1)*
- [x] Build a minimal 1×1 intersection (NetEdit)
- [x] Implement custom `gymnasium.Env` with dynamic `libsumo`/`traci` switching
- [x] Train and evaluate baseline DQN agent (queue-based control)

### Phase 2 — Scaling & Realism *(Month 2)*
- [ ] Expand to 4-way intersection with dedicated lanes
- [ ] Simulate rush-hour traffic using `randomTrips.py`
- [ ] Switch to PPO for improved stability in larger state spaces

### Phase 3 — Multi-Objective Optimization *(Month 3)*
- [ ] Integrate CO₂-based reward signals
- [ ] Benchmark vs fixed-time & actuated signals
- [ ] Visualize results (wait time vs emissions tradeoffs)

---

## 📊 Expected Outcomes

- Reduced average waiting time at intersections
- Improved traffic throughput
- Lower emissions compared to traditional fixed-signal systems

---

## 📌 Future Work

- Multi-agent traffic networks
- Real-world sensor data integration
- Deployment on smart city infrastructure
