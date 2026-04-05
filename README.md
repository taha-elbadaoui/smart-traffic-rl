# Traffic Flow Optimization via Reinforcement Learning

## Project Overview
[cite_start]This project designs a Reinforcement Learning (RL) system to improve intersection throughput by dynamically selecting traffic signal phases instead of using fixed-time cycles[cite: 3]. [cite_start]The system progresses from a single-agent controller to a collaborative multi-agent system[cite: 4].

## Tech Stack
* [cite_start]**Simulator:** SUMO (Simulation of Urban MObility) [cite: 8]
* [cite_start]**Interface:** TraCI (Python API) [cite: 9]
* [cite_start]**RL Framework:** Stable-Baselines3 / Ray RLlib [cite: 58, 75]
* [cite_start]**Environment:** SUMO-RL (Gym-compatible) [cite: 15, 52]

## Features
* [cite_start]**Phase-Based Control:** The agent selects optimal green/yellow phases for North-South and East-West corridors[cite: 20, 46].
* [cite_start]**Multi-Objective Rewards:** Optimization based on mobility (waiting time), environment ($CO_{2}$), and safety[cite: 62, 63, 65].
* [cite_start]**State Space:** Observations include vehicle counts, queue lengths, waiting times, and lane occupancy[cite: 34, 35, 36, 39].

## Implementation Roadmap
- [ ] [cite_start]**Phase 1:** 4-way intersection setup in SUMO [cite: 19, 70]
- [ ] [cite_start]**Phase 2:** Traffic generation using `randomTrips` or manual `.rou.xml` [cite: 13, 14, 71]
- [ ] [cite_start]**Phase 3:** Integration with SUMO-RL wrapper [cite: 52, 73]
- [ ] [cite_start]**Phase 4:** Training baseline agents (PPO/DQN) [cite: 54, 55, 75]
- [ ] [cite_start]**Phase 5:** Comparison against fixed-time signal baseline [cite: 76]
- [ ] [cite_start]**Phase 6:** Scaling to Multi-Agent RL (MARL) with RLlib [cite: 58, 77]
