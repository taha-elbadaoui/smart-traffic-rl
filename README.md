# 🚦 Traffic Flow Optimization via Reinforcement Learning

**Institution:** INPT Rabat (ASEDS)  
**Team:** Taha El Badaoui & Walid  
**Supervisor:** Prof. Zineb El Akkaoui  
**Timeline:** 3 Months (Part-Time)

---

## 📝 Project Overview

This project designs a **Multi-Objective Reinforcement Learning (RL)** system to optimize traffic signal control at intersections.

The goal is to:
- Maximize traffic throughput  
- Minimize waiting time  
- Reduce environmental impact (CO₂ emissions & fuel consumption)

A **single-agent architecture** is used to deeply optimize:
- State-space representation  
- Reward shaping  

This avoids the complexity and overhead of multi-agent coordination while providing a strong, custom baseline.

---

## 🛠 Tech Stack

- **Simulator:** SUMO (Simulation of Urban MObility)  
- **RL Framework:** Stable-Baselines3 (DQN baseline, transitioning to PPO)  
- **Environment:** Custom `gymnasium.Env` wrapper  
- **Dynamic Interface Switching:**
  - `libsumo`: Used for high-speed, headless execution during training.
  - `traci`: Used with `sumo-gui` for real-time visual evaluation of the trained agent.

---

## 🚀 Key Features

- ⚡ **High-Speed Training** Bypasses GUI and TCP overhead by utilizing C++ bindings (`libsumo`) during the training loop.
  
- 👁️ **Visual Verification** Dynamically switches to `traci` for post-training evaluation, allowing human observation of the agent's logic.

- 🚗 **Queue-Based Rewards** Optimizes traffic flow by minimizing the normalized sum of halting vehicles on incoming lanes.

- 🌱 **Environmental Awareness (Upcoming)** Tracks CO₂ emissions and fuel consumption in real time to penalize harsh braking and inefficient flow patterns.

---

## 💻 Execution Guide

### 1. Setup
Ensure your virtual environment is active and dependencies are installed:
```bash
pip install -r requirements.txt
```

### 2. Headless Training (Maximum Speed)
To train the neural network, run the training script. This automatically defaults to `libsumo` (headless mode) to prevent the GUI from choking the CPU during thousands of rapid iterations.
```bash
cd src
python train.py
```
*Output: Saves the trained model to `models/dqn_1x1_baseline.zip`*

### 3. Visual Evaluation (GUI Mode)
To visually verify the agent's behavior, run the evaluation script. This loads the saved model and forces `sumo-gui` to open using `traci`.
```bash
cd src
python evaluate.py
```
*Note: Press the "Play" button in the SUMO interface to start the simulation.*

---

## 📅 Implementation Roadmap

### Phase 1 — "Hello World" Pipeline (Month 1)
- [x] Build a minimal 1×1 intersection (NetEdit)  
- [x] Implement custom `gymnasium.Env` with dynamic `libsumo`/`traci` switching  
- [ ] Train and evaluate baseline DQN agent (queue-based control)  

---

### Phase 2 — Scaling & Realism (Month 2)
- [ ] Expand to 4-way intersection with dedicated lanes  
- [ ] Simulate rush-hour traffic using `randomTrips.py`  
- [ ] Switch to PPO for improved stability in larger state spaces  

---

### Phase 3 — Multi-Objective Optimization (Month 3)
- [ ] Integrate CO₂-based reward signals  
- [ ] Benchmark vs fixed-time & actuated signals  
- [ ] Visualize results (wait time vs emissions)  

---

## 📊 Expected Outcomes

- Reduced average waiting time  
- Improved traffic throughput  
- Lower emissions compared to traditional systems  

---

## 📌 Future Work

- Multi-agent traffic networks  
- Real-world data integration  
- Deployment on smart city infrastructure
