# FlowMind AI - Technical Documentation

> AI-Adaptive Traffic Signal Control for Vietnamese Smart Cities

**Version:** Level 1 (Pure Timing Optimization)
**Target:** Hai Chau District, Da Nang, Vietnam
**Stack:** SUMO Simulator + Gymnasium + PyTorch + TraCI

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [SUMO Network](#3-sumo-network)
4. [TLS Discovery & Geometry](#4-tls-discovery--geometry)
5. [RL Environment](#5-rl-environment)
6. [Reward Function](#6-reward-function)
7. [MAPPO Agent](#7-mappo-agent)
8. [Training](#8-training)
9. [Evaluation](#9-evaluation)
10. [Visualization](#10-visualization)
11. [Vietnamese Traffic Adaptations](#11-vietnamese-traffic-adaptations)

---

## 1. Project Overview

FlowMind AI is a multi-agent reinforcement learning system that controls traffic signals in real-time to reduce congestion, wait times, and queue lengths in Vietnamese urban areas. The system targets Hai Chau District --- the downtown core of Da Nang --- a dense urban area characterized by mixed traffic dominated by motorbikes, narrow streets, and complex intersections.

**Core approach:**
- **Multi-Agent Proximal Policy Optimization (MAPPO)** with shared parameters controls 34 non-trivial traffic light systems (TLS) simultaneously.
- Each TLS is an independent agent that observes local traffic conditions and selects the optimal green phase.
- All agents share a single neural network (parameter sharing), enabling knowledge transfer between intersections of different sizes.
- The system is built on the SUMO (Simulation of Urban Mobility) microsimulator, interfaced via TraCI, and wrapped in a Gymnasium-compatible environment for standard RL training.

**Key results (Level 1):**
- **-82% average wait time** (974.5s to 175.9s)
- **-77% average queue length** (5.2 to 1.2 vehicles)
- **+82% throughput** (591 to 1,078 arrived vehicles)

---

## 2. System Architecture

### High-Level Data Flow

```text
+-------------------+         +-------------------+         +-------------------+
|                   |  TraCI  |                   |  obs    |                   |
|   SUMO Simulator  |<------->|  SumoTrafficEnv   |-------->|   MAPPO Agent     |
|   (Hai Chau map)  |         |  (Gymnasium env)  |<--------|   (ActorCritic)   |
|                   |         |                   | actions |                   |
+-------------------+         +-------------------+         +-------------------+
        ^                             |                             |
        |                             v                             v
  Vehicle dynamics            Reward computation              Policy update
  Signal states               Observation build               GAE + PPO clip
  Arrival tracking             Min-green enforce              Shared parameters
```

### Detailed Component Interaction

```text
+-------------------------------------------------------------------+
|                        TRAINING LOOP                               |
|                                                                    |
|   rl_dashboard.py (Tkinter GUI)                                   |
|     |                                                              |
|     v                                                              |
|   train.py                                                         |
|     |                                                              |
|     +---> SumoTrafficEnv.reset()                                   |
|     |       |-> TLSMetadata(net_file)  -- discover TLS             |
|     |       |-> _start_sumo()          -- launch SUMO via TraCI    |
|     |       |-> _set_trivial_tls_green() -- force trivial to green |
|     |       |-> _get_observations()    -- initial obs              |
|     |                                                              |
|     +---> for each decision step:                                  |
|     |       |-> MAPPO.select_action(obs, global_obs, valid_mask)   |
|     |       |-> SumoTrafficEnv.step(actions)                       |
|     |       |     |-> Resolve target phases per TLS                |
|     |       |     |-> Min-green enforcement (per-TLS threshold)    |
|     |       |     |-> Tick-by-tick transition (Y -> AR -> G)       |
|     |       |     |-> Advance remaining green time                 |
|     |       |     |-> _get_observations()                          |
|     |       |     |-> _compute_rewards() via reward.py             |
|     |       |-> Store in RolloutBuffer                             |
|     |                                                              |
|     +---> MAPPO.update()  -- PPO clipped surrogate + GAE           |
|     +---> Save best model checkpoint                               |
+-------------------------------------------------------------------+
```

### Module Map

| Module | Purpose |
|--------|---------|
| `src/ai/traffic_env.py` | Gymnasium multi-agent environment wrapping SUMO |
| `src/ai/reward.py` | Per-TLS reward computation (6 weighted terms) |
| `src/ai/mappo_agent.py` | MAPPO agent with shared ActorCritic network |
| `src/ai/train.py` | Training loop with DQN/MAPPO support |
| `src/simulation/tls_metadata.py` | Dynamic TLS discovery, geometry, and timing |
| `src/tools/compare.py` | Baseline vs AI comparison framework |
| `src/tools/visualize.py` | SUMO-GUI visualization with live stats panel |
| `src/tools/rl_dashboard.py` | Tkinter training management GUI |
| `sumo/danang/build_danang.py` | OSM-to-SUMO network build pipeline |

---

## 3. SUMO Network

### Map Source

The simulation network covers **Hai Chau District** --- Da Nang's downtown core, extracted from OpenStreetMap. The area includes the Bach Dang riverside, Han Market, Da Nang Cathedral, and major arterials such as Tran Phu, Le Duan, Nguyen Van Linh, Hai Phong, and Hung Vuong.

**Bounding box:** 16.047--16.065 N, 108.205--108.224 E

### Build Pipeline (`build_danang.py`)

The network is constructed through a multi-stage pipeline:

```text
map.osm (OpenStreetMap)
    |
    v
[1] netconvert: OSM -> SUMO .net.xml
    - TLS guessing with threshold=100
    - Junction join at 50m distance
    - Roundabout and ramp guessing
    - UTM projection
    - Keep only passenger/motorcycle/bus/truck edges
    |
    v
[2] Speed cap: all lanes capped to 50 km/h (13.89 m/s)
    - OSM tags many roads as trunk (100 km/h default)
    - Real speed limit in Hai Chau is 40-50 km/h
    |
    v
[3] polyconvert: buildings, water, parks for GUI rendering
    |
    v
[4a] Vehicle types: 8 Vietnamese vehicle types
[4b] Edge weights: lane-count-based weighting for trip generation
    |
    v
[5] randomTrips.py + duarouter: demand generation
    |
    v
[6] .sumocfg with step-length=0.5, lateral-resolution=0.8
```

### Vehicle Types (Vietnamese Traffic Mix)

| Type | Class | Count | Fraction | Length | Max Speed |
|------|-------|-------|----------|--------|-----------|
| `motorbike` | motorcycle | 750 | 45.3% | 2.2m | 40 km/h |
| `motorbike2` | motorcycle | 380 | 23.0% | 2.0m | 45 km/h |
| `car` | passenger | 200 | 12.1% | 4.5m | 50 km/h |
| `car_suv` | passenger | 80 | 4.8% | 4.8m | 50 km/h |
| `taxi` | passenger | 80 | 4.8% | 4.5m | 50 km/h |
| `bus` | bus | 65 | 3.9% | 12.0m | 40 km/h |
| `truck` | truck | 50 | 3.0% | 10.0m | 30 km/h |
| `delivery` | truck | 50 | 3.0% | 6.0m | 40 km/h |
| **Total** | | **1,655** | **100%** | | |

Aggregate split: ~68% motorbike, ~20% car, ~6% bus, ~6% truck --- consistent with Vietnamese urban traffic surveys.

### Traffic Demand

- **1,655 vehicles** spawned over the first 600 seconds of simulated time
- Remaining 300 seconds allow the network to drain
- Edge weights direct traffic toward arterials (5+ lane roads get weight 20, single-lane alleys get weight 0.5)
- Fringe factors bias buses and trucks toward edge-of-network origins (realistic through-traffic)

### Network Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `step-length` | 0.5s | Fine-grained vehicle dynamics |
| `lateral-resolution` | 0.8m | Enables sublane model for motorcycle filtering |
| `junctions.join-dist` | 50m | Merge closely-spaced TLS clusters (e.g., Duy Tan corridor) |
| `time-to-teleport` | 300s | Remove deadlocked vehicles after 5 minutes |
| `collision.action` | none | Ignore collisions (focus on signal timing) |

---

## 4. TLS Discovery & Geometry

### Dynamic Discovery (`tls_metadata.py`)

The system dynamically discovers all traffic light systems from any SUMO `.net.xml` file --- no hard-coded intersection metadata. This enables portability to other Vietnamese cities (Hanoi, Ho Chi Minh City, etc.).

**Discovery process:**
1. Parse `.net.xml` via `sumolib` with phase programs
2. Extract per-TLS: phase states, durations, incoming edges/lanes, connection count
3. Classify phases as green (contains `G` or `g`, not all-yellow/red)
4. Compute per-intersection geometry and engineering-based timing

### Non-Trivial Filtering

Not every TLS in the SUMO network warrants RL optimization. The system filters to **non-trivial** TLS using three criteria:

| Criterion | Threshold | Purpose |
|-----------|-----------|---------|
| Green phases | >= 2 | Skip single-phase pedestrian crossings |
| Incoming edges | >= 2 | Skip median breaks and channelized turns |
| Roundabout distance | > 200m | Roundabouts work better with yield-based flow |

**Hai Chau statistics:** ~107 total TLS in the network, **34 non-trivial** TLS selected for AI control.

### Trivial TLS Handling

Trivial TLS (single-phase or uniform-phase signals) are forced to **permanent green** at simulation reset. This is critical because:
- Nguyen Van Linh alone has ~25 trivial TLS (pedestrian crossings) that were randomly cycling and blocking arterial flow
- Roundabout-entry TLS with uniform phases (all `GGG`/`yyy`/`rrr`) serve no traffic separation purpose

Two categories are forced green:
1. **Single green phase:** Only one non-yellow/red phase exists (pedestrian crossings, median breaks)
2. **Uniform phase:** Multiple phases but all have the same green/red pattern (roundabout entries)

### Per-Intersection Geometry (`TLSGeometry`)

Each non-trivial TLS receives individualized timing parameters computed from engineering formulas:

#### Tier Classification

| Tier | Junction Width | Vehicle Min Green | Max Green |
|------|---------------|-------------------|-----------|
| Small | < 25m | 10s | 25s |
| Medium | 25--50m | 15s | 45s |
| Large | >= 50m | 20s | 60s |

#### Yellow Time (ITE / Kell-Fullerton Formula)

```text
yellow = max(3.0, 1.0 + v / (2 * a))
```

Where:
- `v` = approach speed (max incoming edge speed, capped at 13.89 m/s = 50 km/h)
- `a` = comfortable deceleration rate (3.05 m/s^2, ITE standard)
- Minimum 3.0 seconds (MUTCD requirement)

#### All-Red Clearance (FHWA Formula)

```text
allred = max(1.5, (W + L) / v)
```

Where:
- `W` = junction width from node bounding box
- `L` = clearance vehicle length (6.0m)
- `v` = approach speed
- Minimum 1.5 seconds

#### Minimum Green Time

The minimum green time is the **maximum** of two constraints:

1. **Vehicle minimum:** Tier-based (10/15/20 seconds for small/medium/large)
2. **Pedestrian crossing time:** `7.0 + W / 1.0` seconds
   - 7.0s = MUTCD walk interval
   - W / 1.0 = crossing time at Vietnamese walking speed (1.0 m/s)

The pedestrian constraint is typically the **binding constraint** for medium and large intersections (a 40m intersection requires 7 + 40 = 47s pedestrian crossing time).

#### Max Cycle Constraint

Total cycle length is constrained to <= 120 real seconds:

```text
if n_phases * (min_green + yellow + allred) > 120s:
    min_green = max((120 - n * (yellow + allred)) / n, 7.0)
```

#### Webster's Optimal Cycle (Baseline)

Used for the default fixed-timing baseline:

```text
L = n * (1.5 + yellow + allred)        # total lost time
Y = min(n * 0.30, 0.90)                # flow ratio estimate
C_opt = (1.5 * L + 5) / (1 - Y)       # Webster's formula
C_opt = clamp(C_opt, 60, 120)          # practical bounds
green_splits = (C_opt - L) / n         # equal splits
```

---

## 5. RL Environment

### `SumoTrafficEnv` (Gymnasium Multi-Agent Environment)

The environment wraps SUMO via TraCI and presents a standard Gymnasium interface for multi-agent RL. Each non-trivial TLS is an independent agent sharing the same observation and action space definitions.

### Observation Space

**Per-TLS observation:** 39-dimensional float vector, all values normalized to [0, 1].

| Slots | Feature | Normalization | Description |
|-------|---------|---------------|-------------|
| 0--11 | Queue per edge | `/50` vehicles | Halting vehicles on each incoming edge |
| 12--23 | Wait time per edge | `/300` seconds | Cumulative waiting time on each incoming edge |
| 24--35 | Lane density per edge | `veh/capacity` | Vehicle count / edge capacity (lanes * length / 7.5m) |
| 36 | Phase ratio | `phase_idx / num_phases` | Current phase index normalized by total phases |
| 37 | Elapsed green time | `/max_green` | Actual green elapsed, normalized by per-TLS max green |
| 38 | Min-green satisfied | Binary (0/1) | Whether elapsed green >= per-TLS min_green threshold |

Edges are padded/truncated to a fixed 12 slots (`MAX_INCOMING_EDGES = 12`) to enable parameter sharing across intersections with different numbers of approaches.

### Action Space

**Discrete(7):** Select green phase index 0 through 6. Each action maps to a specific green phase in the TLS program. If a TLS has fewer than 7 green phases, excess actions map to the first valid phase.

There is **no OFF action** --- the agent must always select a valid green phase. The OFF action (originally action index 7) was removed after it caused all-green collisions in early development.

### Step Function: Tick-by-Tick State Machine

The `step()` function implements a per-TLS state machine that processes yellow, all-red, and green transitions independently for each intersection:

```text
For each decision step (delta_time = 30 sim ticks = 15 real seconds):

1. RESOLVE target phases from actions
   - Min-green enforcement: if elapsed < per-TLS min_green, override to keep current phase

2. DETECT phase changes
   - For each TLS requesting a new phase: set yellow state

3. TICK-BY-TICK TRANSITION (per-TLS independent)
   For each sim tick up to max_transition:
     For each changing TLS:
       if yellow_remaining > 0:  decrement yellow
       elif allred_remaining > 0: decrement all-red
       else: set green, mark done
     simulationStep()  -- advance SUMO

4. SET GREEN for non-changing TLS (they stay green throughout)

5. ADVANCE remaining green time
   green_ticks = delta_time - max_transition
   for each remaining tick: simulationStep()

INVARIANT: total sim ticks per step = exactly delta_time
```

This design means:
- **Small intersections** (shorter yellow/allred) finish transition faster and get more green time per step
- **Large intersections** consume more of the step budget for clearance
- All TLS are synchronized to the same delta_time decision interval

### Episode Structure

| Parameter | Value | Notes |
|-----------|-------|-------|
| `delta_time` | 30 ticks | 15 real seconds per decision |
| `sim_length` | 3600 ticks | 1800 real seconds total |
| Steps per episode | 120 | 3600 / 30 |
| Warm-up | 30 ticks | Before first observation |

### Early Termination

The episode ends early if `simulation.getMinExpectedNumber() <= 0` (no vehicles remain in the network or scheduled to depart).

---

## 6. Reward Function

### Design Philosophy

The reward function follows a **pressure-primary** design, inspired by the **PressLight** research (Wei et al., KDD 2019). Pressure --- the imbalance between incoming and outgoing traffic at an intersection --- is the most theoretically grounded signal for throughput maximization in traffic networks.

### Reward Terms

The per-TLS reward is a weighted sum of six terms:

```text
reward = wait_term + queue_term + fairness_term + throughput_term + pressure_term + switch_term
```

**Overall range:** approximately -1.0 to +0.5 per step.

| Term | Weight | Formula | Purpose |
|------|--------|---------|---------|
| **Pressure** | 0.30 | `+w * clip(pressure / 20, -1, 1)` | Primary signal. Positive when more vehicles exit than enter (good flow). |
| **Wait improvement** | 0.25 | `-w * clip(delta_wait / 100, -2, 2)` | Reward decrease in total waiting time on incoming edges. |
| **Queue penalty** | 0.15 | `-w * (avg_queue / 50)` | Penalize average queue length across incoming edges. |
| **Switch penalty** | 0.15 | `-w * transition_cost` (if changed) | Discourage unnecessary phase changes. Scaled by per-TLS transition cost. |
| **Throughput** | 0.10 | `+w * clip(delta_throughput / 10, -1, 1)` | Reward net increase in vehicles flowing through. |
| **Fairness** | 0.05 | `-w * (max_queue / 50)` | Penalize worst-case queue to prevent starvation of any approach. |

### Term Details

**Pressure (w=0.30):**
Computed as `outgoing_vehicles - incoming_vehicles` for the intersection node. Positive pressure means the intersection is clearing vehicles faster than they arrive. This term directly targets throughput maximization and is the primary learning signal.

**Wait Improvement (w=0.25):**
Tracks the change in total waiting time across all incoming edges between consecutive steps. Rewards the agent for reducing cumulative wait, penalizes for increasing it. Clipped to [-2, 2] to prevent large spikes from dominating.

**Queue Penalty (w=0.15):**
Average queue length (halting vehicles) across all incoming edges, normalized by a cap of 50 vehicles. Provides a continuous penalty proportional to congestion severity.

**Switch Penalty (w=0.15):**
Applied only when the agent switches to a different phase. Scaled by the **per-TLS transition cost**:

```text
transition_cost = (yellow_steps + allred_steps) / avg_transition_across_all_TLS
```

This means large intersections (which have longer yellow + all-red clearance) are penalized more for switching, reflecting the real cost of lost green time during transitions.

**Throughput (w=0.10):**
Change in vehicle count on incoming edges between steps. Rewards when fewer vehicles remain (they have passed through).

**Fairness (w=0.05):**
Maximum queue across all incoming edges (worst-case approach). Prevents the agent from permanently starving one direction to optimize the overall average.

---

## 7. MAPPO Agent

### Algorithm: Multi-Agent PPO with Shared Parameters

The MAPPO (Multi-Agent Proximal Policy Optimization) agent implements the **Centralized Training, Decentralized Execution (CTDE)** paradigm:

- **Training:** The critic sees both local and global information for better value estimation
- **Execution:** The actor only sees local observations, enabling decentralized real-time control

### Network Architecture (`ActorCritic`)

```text
ACTOR (local obs -> action logits):
    Linear(39, 256) -> ReLU
    Linear(256, 256) -> ReLU
    Linear(256, 7)   -> action logits
    + action masking (invalid actions set to -1e8)

CRITIC (local obs + global obs -> value):
    Linear(78, 256) -> ReLU       # 39 local + 39 global = 78
    Linear(256, 256) -> ReLU
    Linear(256, 1)   -> state value
```

**Global observation:** Mean of all agent observations (aggregated view of network-wide traffic state).

### Action Masking

Each TLS may have fewer than 7 green phases. Invalid action indices are masked to `-1e8` before the softmax, ensuring the agent only selects valid phases. The valid action mask is constructed per-TLS and stored in the rollout buffer for PPO updates.

### Rollout Buffer and GAE

Transitions are stored in an on-policy `RolloutBuffer` containing per-agent tuples: `(obs, global_obs, action, log_prob, reward, value, done, valid_mask)`.

**Generalized Advantage Estimation (GAE):**

```text
delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
A_t = delta_t + gamma * lambda * (1 - done_{t+1}) * A_{t+1}
returns_t = A_t + V(s_t)
```

Advantages are normalized (zero mean, unit variance) before PPO updates.

### PPO Update

For `ppo_epochs` iterations over shuffled mini-batches:

1. Recompute action distribution and value under current parameters
2. Compute probability ratio: `r = exp(log_pi_new - log_pi_old)`
3. Clipped surrogate loss: `L_actor = -min(r * A, clip(r, 1-eps, 1+eps) * A)`
4. Value loss: `L_critic = MSE(V, returns)`
5. Total loss: `L = L_actor + 0.5 * L_critic - 0.03 * entropy`
6. Gradient clipping at norm 0.5

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hidden` | 256 | Hidden layer size |
| `lr` | 3e-4 | Adam learning rate |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_eps` | 0.2 | PPO clip range |
| `entropy_coef` | 0.03 | Entropy bonus coefficient |
| `value_coef` | 0.5 | Value loss coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping norm |
| `ppo_epochs` | 10 | PPO optimization epochs per rollout |
| `mini_batch_size` | 256 | Mini-batch size for PPO updates |

### Model Persistence

Checkpoints save: model state dict, optimizer state dict, obs_dim, act_dim, and algorithm identifier ("mappo"). Loading automatically detects algorithm type for backward compatibility with DQN checkpoints.

---

## 8. Training

### Training Dashboard (`rl_dashboard.py`)

A Tkinter-based GUI provides training management:
- **Configuration panel:** Network/route file selection, hyperparameter tuning
- **Live charts:** Reward, wait time, and epsilon per episode (Matplotlib embedded via TkAgg)
- **Training controls:** Start/stop/pause, save checkpoints
- **Color scheme:** Catppuccin Mocha dark theme

### Training Loop (`train.py`)

The training loop supports both DQN and MAPPO algorithms:

1. Initialize `SumoTrafficEnv` with network files
2. Print banner with TLS count, obs/action dims, steps per episode
3. For each episode:
   - Reset environment, get initial observations
   - For each step: select actions (with exploration), step environment, store transitions
   - At episode end: run PPO update (MAPPO) or batch DQN update
   - Log metrics, save checkpoint if best reward

### Key Training Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `lr` | 3e-4 | Learning rate for Adam optimizer |
| `gamma` | 0.99 | Discount factor |
| `delta_time` | 30 | Sim ticks per decision (15 real seconds) |
| `sim_length` | 3600 | Sim ticks per episode (1800 real seconds) |
| `hidden` | 256 | MLP hidden layer size |
| `batch_size` | 64 | DQN replay batch size |
| `episodes` | 100 | Default training episodes |

### Training Log Header

At startup, the environment prints per-TLS timing information (tier, yellow/allred/min_green/max_green in real seconds), enabling verification of engineering formula outputs.

---

## 9. Evaluation

### Baseline Comparison (`compare.py`)

The evaluation framework runs two simulations through the **same** `SumoTrafficEnv`:

#### Baseline Run

The baseline uses a **"do nothing" policy** --- each TLS keeps its current phase at every decision step. This means:
- Same environment initialization (warm-up, trivial TLS forced green, same vehicle demand)
- Same per-TLS timing constraints (yellow, all-red, min-green)
- Only difference: no intelligent phase switching

This is a fairer comparison than running SUMO's default fixed-timing programs directly, because the baseline and AI experience identical environmental conditions.

#### AI Model Run

The trained model runs greedily (no exploration), selecting the highest-probability action at each step. The global observation (mean of all agent observations) feeds the centralized critic, though only the actor is needed for execution.

#### Metrics

| Metric | Description |
|--------|-------------|
| Average Wait Time (s) | Mean waiting time across all incoming edges of all TLS |
| Average Queue Length | Mean halting vehicle count across all incoming edges |
| Throughput | Total vehicles that arrived at their destination |

#### Results (Level 1)

| Metric | Baseline | AI Model | Improvement |
|--------|----------|----------|-------------|
| Avg Wait Time | 974.5s | 175.9s | **-82%** |
| Avg Queue Length | 5.2 | 1.2 | **-77%** |
| Throughput | 591 | 1,078 | **+82%** |

---

## 10. Visualization

### SUMO-GUI Visualization (`visualize.py`)

The visualization tool runs the trained model in SUMO's graphical interface with an accompanying Tkinter stats panel.

#### Modes

| Mode | Description |
|------|-------------|
| `--ai` | Run trained model with live visualization |
| `--baseline` | Run baseline (do nothing) policy |
| `--both` | Side-by-side comparison |

#### Features

- **Live metrics panel:** Real-time wait time, queue length, throughput, vehicle count
- **Baseline vs AI comparison:** Side-by-side metric display when running `--both`
- **TLS action log:** Per-TLS timing display showing tier classification, green/yellow/red durations
- **Congestion monitor:** Automatic camera fly-to for congested intersections
- **Visual overlays:**
  - Green circles: active TLS under AI control
  - Color-coded signal states in SUMO-GUI

#### Integration

The visualizer uses the same `SumoTrafficEnv` as training, ensuring behavioral consistency. It supports both MAPPO and DQN model checkpoints with automatic algorithm detection and observation dimension remapping for backward compatibility.

---

## 11. Vietnamese Traffic Adaptations

FlowMind AI incorporates several adaptations specific to Vietnamese urban traffic that distinguish it from Western traffic signal control systems:

### Motorcycle-Dominant Traffic

Vietnamese cities have a fundamentally different traffic composition than Western cities. In Hai Chau District, **68% of vehicles are motorbikes**, which have distinct dynamics:

- **PCU (Passenger Car Unit):** Motorbike = 0.25 PCU (vs 1.0 for cars). Four motorbikes occupy roughly the same road space as one car.
- **Saturation flow:** 4,092 PCU/hr/lane (measured Vietnamese value), compared to the Highway Capacity Manual standard of 1,900 PCU/hr/lane. Vietnamese intersections process far more vehicles per hour due to the high motorbike fraction.
- **Sublane model:** SUMO's lateral resolution is set to 0.8m (`lateral-resolution=0.8`), enabling motorbikes to filter between cars and utilize lane space more efficiently --- a hallmark of Vietnamese traffic behavior.

### Pedestrian Timing

- **Walking speed:** 1.0 m/s (Vietnamese standard), slower than the typical 1.2 m/s used in Western engineering manuals
- **Pedestrian crossing time:** `7s walk + junction_width / 1.0 m/s`
- This is frequently the **binding constraint** on minimum green time, especially for large intersections

### Speed Limits

- **Downtown cap:** All lanes capped to 50 km/h (13.89 m/s)
- OSM data often tags roads as `highway=trunk` with 100 km/h defaults, but Hai Chau is a dense urban district with actual limits of 40--50 km/h
- The speed cap is applied post-netconvert by directly editing the `.net.xml`

### Vehicle Type Calibration

Eight vehicle types are calibrated for Vietnamese conditions:
- Motorbike dimensions: 2.0--2.2m length, 0.7--0.8m width
- Higher acceleration/deceleration than Western equivalents (aggressive Vietnamese driving style)
- Speed deviation factors reflect the variability observed in Vietnamese traffic
- Fringe factors bias buses and trucks toward network edges (through-traffic patterns)

### Junction Handling

- **Merge distance 50m:** Vietnamese urban blocks are small, and closely-spaced intersections (especially along corridors like Duy Tan) are merged into single TLS to prevent coordination conflicts
- **Roundabout exclusion 200m:** Vietnamese roundabouts function on yield-based flow; adding signal control typically worsens performance

---

## Appendix A: File Structure

```text
WeSaveTime-ASEAN/
  src/
    ai/
      traffic_env.py      # Gymnasium multi-agent environment
      reward.py            # Per-TLS reward computation
      mappo_agent.py       # MAPPO agent (ActorCritic + PPO)
      dqn_agent.py         # DQN agent (legacy, still supported)
      train.py             # Training loop
    simulation/
      tls_metadata.py      # TLS discovery, geometry, timing
      engine.py            # Simulation engine abstraction
      events.py            # Event system (Level 2, currently disabled)
      scenarios.py         # Scenario definitions
      sumo_engine.py       # SUMO-specific engine implementation
    tools/
      compare.py           # Baseline vs AI comparison
      visualize.py         # SUMO-GUI visualization
      rl_dashboard.py      # Tkinter training dashboard
      event_dashboard.py   # Event dashboard (Level 2, currently disabled)
  sumo/
    danang/
      build_danang.py      # Network build pipeline
      map.osm              # OpenStreetMap extract (Hai Chau District)
      danang.net.xml       # SUMO network
      danang.rou.xml       # Vehicle routes
      danang.sumocfg       # SUMO configuration
      danang.vtypes.xml    # Vietnamese vehicle types
      danang.poly.xml      # Polygon overlays (buildings, water)
  checkpoints/
    best_model.pt          # Trained model checkpoint
    training_log.json      # Training metrics history
    comparison.json        # Baseline vs AI results
  docs/
    TECHNICAL.md           # This document
    CHANGELOG.md           # Development changelog
```

## Appendix B: Engineering Formula Reference

| Formula | Source | Application |
|---------|--------|-------------|
| Yellow time: `max(3.0, 1.0 + v/(2*a))` | ITE / Kell-Fullerton | Per-TLS yellow duration |
| All-red: `max(1.5, (W+L)/v)` | FHWA | Per-TLS clearance interval |
| Ped crossing: `7.0 + W/1.0` | MUTCD + Vietnamese walking speed | Min green constraint |
| Webster's cycle: `(1.5L+5)/(1-Y)` | Webster (1958) | Baseline optimal cycle |
| Saturation flow: 4,092 PCU/hr/lane | Vietnamese field measurements | Capacity estimation |

## Appendix C: Constants Reference

| Constant | Value | Location |
|----------|-------|----------|
| `OBS_DIM` | 39 | `traffic_env.py` |
| `ACT_DIM` | 7 | `traffic_env.py` |
| `MAX_INCOMING_EDGES` | 12 | `traffic_env.py` |
| `MIN_GREEN_STEPS` | 60 | `traffic_env.py` (30s real) |
| `_DOWNTOWN_SPEED_CAP_MS` | 13.89 | `tls_metadata.py` (50 km/h) |
| `_DECEL_RATE` | 3.05 | `tls_metadata.py` (m/s^2) |
| `_PED_WALK_S` | 7.0 | `tls_metadata.py` (seconds) |
| `_PED_SPEED_MS` | 1.0 | `tls_metadata.py` (m/s) |
| `_VEH_LENGTH_M` | 6.0 | `tls_metadata.py` (meters) |
| `_MAX_CYCLE_STEPS` | 240 | `tls_metadata.py` (120 real seconds) |
| `_VN_SATURATION_FLOW` | 4,092 | `tls_metadata.py` (PCU/hr/lane) |
| `_SIM_STEP_LENGTH` | 0.5 | `tls_metadata.py` (seconds) |
