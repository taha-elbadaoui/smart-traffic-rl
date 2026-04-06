# 🚦 Traffic Flow Optimization via Reinforcement Learning

**Institution:** INPT Rabat (ASEDS)  
**Team:** Taha & Walid  
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

This avoids the complexity and overhead of multi-agent coordination.

---

## 🛠 Tech Stack

- **Simulator:** SUMO (Simulation of Urban MObility)  
- **Performance Interface:** `libsumo` (high-speed, headless execution)  
- **Debugging Interface:** TraCI (Python API with GUI support)  
- **RL Framework:** Stable-Baselines3 (PPO, DQN)  
- **Environment:** Gymnasium wrapper via `sumo-rl`  

---

## 🚀 Key Features

- ⚡ **High-Speed Training**  
  Uses `libsumo` to eliminate TCP overhead and accelerate training  

- 🚗 **Pressure-Based Rewards**  
  Optimizes traffic flow using lane pressure (incoming vs outgoing vehicles)  

- 🌱 **Environmental Awareness**  
  Tracks CO₂ emissions and fuel consumption in real time  

- ⚖️ **Multi-Objective Optimization**  
  Penalizes harsh braking and inefficient flow patterns  

---

## 📅 Implementation Roadmap

### Phase 1 — "Hello World" Pipeline (Month 1)
- [ ] Build a minimal 1×1 intersection (NetEdit)  
- [ ] Implement custom `gymnasium.Env` with `libsumo`  
- [ ] Train baseline DQN agent (queue-based control)  

---

### Phase 2 — Scaling & Realism (Month 2)
- [ ] Expand to 4-way intersection with dedicated lanes  
- [ ] Simulate rush-hour traffic using `randomTrips`  
- [ ] Switch to PPO for improved stability  

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

---
