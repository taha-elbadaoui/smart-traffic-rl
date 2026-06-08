# 🚦 Multi-Agent Traffic Flow Optimization via Reinforcement Learning

> **Institution:** Institut National des Postes et Télécommunications (INPT), Rabat  
> **Specialization:** Advanced Software Engineering for Digital Services (ASEDS)  
> **Authors:** Taha El Badaoui & Walid Hazzam  

---

## 📝 Project Architecture & Real State

This project implements high-performance, single-intersection, and coordinated multi-intersection traffic control policies using **Proximal Policy Optimization (PPO)**. By wrapping the **Simulation of Urban MObility (SUMO)** engine inside custom `gymnasium.Env` interfaces, the system transforms rigid signal control schedules into dynamic, state-reflective policies.

Moving beyond simple baselines, the architecture features:
1. **Parallelized Scaling (`SubprocVecEnv`):** Overcomes SUMO's single-threaded limitations by mapping environments across isolated OS worker processes. Each process dynamically generates its own rank-suffixed traffic layout to prevent multi-core write collisions.
2. **Deterministic Evaluation Safety:** Evaluation pipelines enforce strict structural guardrails (`sys.exit(1)`) preventing unnormalized state vectors from corrupting inference validation if `VecNormalize` statistics are missing.
3. **Multi-Agent Coordination Corridor (CTCE):** Implements Centralized Training Centralized Execution across a 1x2 sequential intersection corridor to naturally discover "Green Wave" synchronization and prevent regional queue starvation.

---

## 🛠 Tech Stack

| Component       | Technology                                                                 |
| --------------- | -------------------------------------------------------------------------- |
| Simulator       | SUMO (Simulation of Urban MObility)                                        |
| RL Framework    | Stable-Baselines3 (PPO)                                                    |
| Environment     | Custom `gymnasium.Env` wrappers (Single & Flattened Multi-Agent)           |
| Vectorization   | `SubprocVecEnv` & `VecNormalize` (Multi-CPU Training)                      |
| Training Mode   | `libsumo` — Headless C++ bindings for maximum parallel throughput          |
| Evaluation Mode | `traci` — GUI via `sumo-gui` for visual verification and policy inspection |

---

## 📂 Supported Environments

1. **T-Junction (`1x1`):** A standard 3-way intersection baseline.
2. **Crossroad (`1x1`):** A complex 4-way intersection handling dynamic, probabilistic traffic generation per parallel worker.
3. **Coordinated Boulevard (`1x2`):** Two sequential traffic lights (`B0`, `C0`). Trained via a Centralized PPO wrapper to synchronize phases and optimize corridor throughput.

---

## 💻 Getting Started

### 1. Setup

Ensure your virtual environment is active, then install dependencies:

```bash
pip install -r requirements.txt
