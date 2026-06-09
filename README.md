# 🚦 Multi-Agent Traffic Flow Optimization via Reinforcement Learning

> **Institution:** Institut National des Postes et Télécommunications (INPT), Rabat
> **Specialization:** Advanced Software Engineering for Digital Services (ASEDS)
> **Authors:** Taha El Badaoui & Walid Hazzam

---

## 📝 Project Architecture & Current State

This project implements high-performance reinforcement learning policies for adaptive traffic signal control using **Proximal Policy Optimization (PPO)** and the **Simulation of Urban MObility (SUMO)** framework.

Traditional traffic lights operate using static timing plans that cannot react effectively to changing traffic conditions. This work replaces predefined schedules with learned policies capable of adapting in real time based on observed traffic states.

The project currently supports both **single-intersection optimization** and **coordinated multi-intersection control**, enabling experimentation with centralized traffic management strategies and scalable reinforcement learning architectures.

### Key Features

#### Parallelized PPO Training

To accelerate training, environments are vectorized using **SubprocVecEnv**, allowing multiple SUMO simulations to run simultaneously across CPU cores.

Each worker process dynamically generates its own traffic configuration and route files, preventing file collisions and enabling robust randomized training.

#### Observation Normalization

Training and evaluation use **VecNormalize** to stabilize learning and improve policy generalization.

Evaluation scripts enforce strict consistency between trained models and normalization statistics. Missing normalization files automatically trigger execution safeguards to prevent invalid inference results.

#### Multi-Agent Corridor Optimization

The project includes a coordinated corridor composed of two consecutive intersections.

A centralized PPO agent controls both traffic lights simultaneously using a flattened global observation space, enabling the emergence of synchronization strategies commonly referred to as a **Green Wave**, where vehicle platoons experience consecutive green signals along a corridor.

---

## 🛠 Technology Stack

| Component                 | Technology                          |
| ------------------------- | ----------------------------------- |
| Simulator                 | SUMO (Simulation of Urban MObility) |
| Reinforcement Learning    | Stable-Baselines3 PPO               |
| Environment API           | Gymnasium                           |
| Parallelization           | SubprocVecEnv                       |
| Observation Normalization | VecNormalize                        |
| Training Backend          | libsumo                             |
| Visualization             | traci + sumo-gui                    |
| Language                  | Python 3                            |

---

## 📂 Repository Structure

```text
src/
├── T_junction/
│   ├── train_T_junction.py
│   └── evaluate_T_junction.py
│
├── crossroad/
│   ├── train_crossroad.py
│   └── evaluate_crossroad.py
│
└── multi_agent/
    ├── train_marl.py
    └── evaluate_marl.py

envs/
├── T_junction/
├── crossroad/
└── boulevard_coordonne/
```

---

## 🚥 Supported Environments

### 1. T-Junction (Single Agent)

A three-way intersection serving as the project's baseline environment.

Characteristics:

* Single traffic light
* PPO optimization
* Fixed road topology
* Suitable for rapid experimentation and debugging

---

### 2. Crossroad (Single Agent)

A four-way intersection with dynamic traffic generation.

Characteristics:

* Parallel PPO training
* Randomized route generation
* Higher traffic complexity
* Multi-core training support

---

### 3. Coordinated Boulevard (Multi-Agent)

A corridor composed of two sequential intersections:

* B0
* C0

Characteristics:

* Centralized PPO control
* Flattened joint observation space
* Joint action selection
* Corridor-level optimization
* Green Wave emergence

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/taha-elbadaoui/Smart-traffic-management.git
cd Smart-traffic-management
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training

### T-Junction

```bash
python src/T_junction/train_T_junction.py --mode train
```

---

### Crossroad

```bash
python src/crossroad/train_crossroad.py
```

Features:

* Parallel environments
* Dynamic route generation
* PPO optimization

---

### Multi-Agent Boulevard

```bash
python src/multi_agent/train_marl.py
```

Features:

* Centralized training
* Coordinated traffic-light control
* Corridor throughput optimization

---

## 🎮 Evaluation

### T-Junction

```bash
python src/T_junction/evaluate_T_junction.py --mode final
```

---

### Crossroad

```bash
python src/crossroad/evaluate_crossroad.py --mode 2M
```

**Important:** Evaluation requires the matching `vecnorm.pkl` generated during training.

---

### Multi-Agent Boulevard

```bash
python src/multi_agent/evaluate_marl.py
```

The evaluation runs through `sumo-gui`, allowing visual inspection of learned coordination strategies.

---

## 📊 Monitoring Training

TensorBoard can be used to visualize PPO training metrics.

Start TensorBoard:

```bash
tensorboard --logdir=tensorboard_logs/
```

Open:

```text
http://localhost:6006
```

### Important Metrics

| Metric               | Description              |
| -------------------- | ------------------------ |
| ep_rew_mean          | Average episode reward   |
| ep_len_mean          | Average episode duration |
| value_loss           | Critic prediction error  |
| policy_gradient_loss | PPO optimization signal  |
| explained_variance   | Critic quality indicator |

---

## 🧠 Reinforcement Learning Formulation

### State Space

Observations may include:

* Queue lengths
* Waiting times
* Current signal phase
* Vehicle occupancy information
* Multi-intersection aggregated states

### Action Space

Traffic-light phase selection:

```text
0 → Phase A
1 → Phase B
...
```

For the coordinated corridor:

```text
Action = [phase_B0, phase_C0]
```

### Reward Function

The reward encourages:

* Reduced waiting times
* Reduced queue lengths
* Increased throughput
* Fewer unnecessary signal switches

General form:

```text
Reward =
+ Throughput
- Waiting Time
- Queue Length
- Switching Penalty
```

---

## 📅 Development Roadmap

### Phase 1 — Foundational Environments ✅

* [x] Build T-Junction environment
* [x] Build Crossroad environment
* [x] Implement Gymnasium wrappers
* [x] Integrate SUMO

### Phase 2 — Parallel PPO & MARL ✅

* [x] SubprocVecEnv parallelization
* [x] Dynamic route generation
* [x] VecNormalize integration
* [x] Multi-agent corridor implementation
* [x] Centralized PPO controller

### Phase 3 — Advanced Optimization 🚧

* [ ] CO₂ emission minimization
* [ ] Fuel consumption optimization
* [ ] Benchmark against SUMO actuated traffic lights
* [ ] Comparative statistical analysis
* [ ] Extended traffic-network scaling

---

## 🎯 Research Objectives

This project investigates whether reinforcement learning can outperform traditional traffic signal control methods by:

1. Minimizing average vehicle waiting time.
2. Reducing queue congestion.
3. Increasing corridor throughput.
4. Learning coordinated signal synchronization.
5. Scaling traffic optimization through parallel simulation.

---

## 📄 License

This repository was developed as part of the **ASEDS Engineering Curriculum** at **INPT Rabat** for academic and research purposes.

Educational and research use is permitted.
Commercial use requires prior authorization from the authors.

---

## 👨‍💻 Authors

**Taha El Badaoui**
Software Engineering Student — INPT Rabat

**Walid Hazzam**
Software Engineering Student — INPT Rabat
