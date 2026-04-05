# Traffic Flow Optimization via Reinforcement Learning

## Project Overview
[cite_start]This project designs an RL system to improve intersection throughput by dynamically selecting traffic signal phases instead of using fixed-time cycles. [cite: 3, 20]

## Tech Stack
* [cite_start]**Simulator:** SUMO (Simulation of Urban MObility) [cite: 8]
* [cite_start]**Interface:** TraCI (Python API) [cite: 9]
* [cite_start]**RL Framework:** Stable-Baselines3 / Ray RLlib [cite: 58, 75]
* [cite_start]**Environment:** SUMO-RL (Gym-compatible) [cite: 15, 52]

## Features
* [cite_start]**Phase-Based Control:** The agent selects optimal green/yellow phases for North-South and East-West corridors. [cite: 20, 46]
* [cite_start]**Multi-Objective Rewards:** Optimization based on mobility (waiting time), environment (CO2), and safety. [cite: 62, 63, 65]
* [cite_start]**State Space:** Observations include vehicle counts, queue lengths, and lane occupancy. [cite: 34, 35, 36, 39]

## Roadmap
- [ ] [cite_start]Phase 1: 4-way intersection setup in SUMO [cite: 70]
- [ ] [cite_start]Phase 2: Traffic generation with `randomTrips` [cite: 71]
- [ ] [cite_start]Phase 3: Integration with SUMO-RL [cite: 73]
- [ ] [cite_start]Phase 4: Training PPO/DQN baseline agents [cite: 75]
- [ ] [cite_start]Phase 5: Multi-Agent (MARL) scaling [cite: 77]
