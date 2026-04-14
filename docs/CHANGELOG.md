# FlowMind AI - Changelog

> Development history of the AI-adaptive traffic signal control system.

---

## Level 1: Pure Timing Optimization

Level 1 focuses exclusively on optimizing traffic signal timing for all 83 TLS in Hai Chau District, Da Nang. No structural changes to the road network (adding/removing TLS) are made at this level.

---

### Phase 1: Initial Setup

**Goal:** Establish the foundational multi-agent RL system for traffic signal control.

- **MAPPO agent implementation** (`mappo_agent.py`)
  - Multi-Agent PPO with shared parameters across all TLS agents
  - Centralized Training, Decentralized Execution (CTDE) paradigm
  - Actor-Critic architecture with 2-layer MLP (256 hidden units)
  - PPO clipped surrogate objective with GAE advantage estimation
  - Action masking for valid traffic phases per intersection

- **Dynamic TLS discovery** (`tls_metadata.py`)
  - Automatic extraction of TLS metadata from any SUMO `.net.xml` file
  - Phase program parsing, incoming edge/lane detection, connection counting
  - Green phase classification (exclude yellow/all-red transition phases)
  - Portable across different SUMO networks (Hanoi, Da Nang, etc.)

- **Da Nang network construction** (`build_danang.py`)
  - Hai Chau District map extracted from OpenStreetMap
  - `netconvert` pipeline with TLS guessing, junction joining, roundabout detection
  - Vietnamese vehicle type definitions (8 types, 68% motorbike mix)
  - Edge-weight-based demand generation directing traffic to arterials

- **Gymnasium environment** (`traffic_env.py`)
  - Multi-agent wrapper around SUMO via TraCI
  - Per-TLS observation vectors and discrete action spaces
  - Basic reward function with wait time and queue penalties

---

### Phase 2: Critical Fixes

**Goal:** Resolve safety-critical issues discovered during initial training runs.

- **Removed OFF action (ACT_DIM 8 -> 7)**
  - The OFF action (index 7) set all signals to green simultaneously, causing all-direction conflicts
  - Agents must now always select a valid green phase --- no "traffic light off" option
  - `ACT_OFF` constant set to -1 (disabled sentinel)

- **Increased MIN_GREEN_STEPS (5 -> 15)**
  - Initial value of 5 ticks (2.5 real seconds) allowed rapid phase toggling
  - Agents were switching phases every 2-3 seconds, creating dangerous conditions
  - Raised to 15 ticks (7.5 real seconds) as an initial fix

- **Added all-red clearance between phase switches**
  - Previously, transitions went directly from one green phase to another
  - Added mandatory all-red interval after yellow to clear the intersection
  - Default: 6 ticks (3 real seconds)

- **Roundabout exclusion radius (80m -> 200m)**
  - Initial 80m radius was too small --- TLS near roundabouts were still being selected for optimization
  - Roundabout-adjacent signals perform better with yield-based flow
  - Increased to 200m to fully exclude the Hai Chau roundabout cluster

- **Trivial TLS detection**
  - Identified that many single-phase TLS (pedestrian crossings) were cycling randomly
  - Added `num_green_phases` property to classify TLS complexity
  - Only uniform-phase signals (all phases identical pattern) forced to permanent green
  - Multi-phase intersections with distinct green phases left on default programs

---

### Phase 3: Anti-Oscillation

**Goal:** Prevent the agent from rapidly switching phases, which wastes green time in yellow/all-red transitions.

- **Phase-switch penalty in reward (w_switch = 0.15)**
  - Added explicit negative reward term when the agent changes phases
  - Weight of 0.15 makes switching costly enough to discourage frivolous changes
  - But not so costly that the agent refuses to switch when necessary

- **Increased delta_time (10 -> 30 ticks, i.e., 15 real seconds)**
  - At delta_time=10, the agent made decisions every 5 seconds --- too frequent
  - At delta_time=30, each decision covers 15 real seconds of simulation
  - Fewer decisions per episode forces each decision to be more meaningful
  - Steps per episode: 1800/30 = 60 decisions

- **Increased MIN_GREEN_STEPS to 60 (30 real seconds)**
  - Aligns with real-world minimum green standards for pedestrian safety
  - Prevents sub-30-second phase durations that cause driver confusion
  - Combined with delta_time=30, ensures at least 2 decision steps per phase

- **Rebalanced reward weights**
  - Adjusted relative importance of wait, queue, and switch terms
  - Goal: smooth, stable phase patterns rather than reactive switching

---

### Phase 4: Simulation Accuracy

**Goal:** Fix measurement bugs and ensure fair baseline comparison.

- **Fixed throughput tracking (was missing 99.8% of arrivals)**
  - Original implementation only checked arrivals at decision boundaries
  - Between decision steps, hundreds of vehicles arrived uncounted
  - Added `_sim_step_and_track()` helper that accumulates `getArrivedNumber()` on every simulation tick
  - Throughput jumped from near-zero to realistic values (500--1000+ per episode)

- **Removed collision tracking**
  - Set `collision.action=none` in SUMO configuration
  - Collisions in SUMO are mostly artifacts of network geometry, not signal timing
  - Removing collision handling simplifies the simulation and prevents teleportation artifacts

- **Fixed sim_length interpretation**
  - Clarified: `sim_length=1800` means 1800 simulation steps
  - At `step_length=0.5`, this equals 900 real seconds (15 minutes)
  - Previously ambiguous whether the value represented steps or seconds

- **Baseline comparison through same SumoTrafficEnv**
  - Previously, baseline ran SUMO's default programs directly (different initialization)
  - Now baseline uses the exact same `SumoTrafficEnv` with a "do nothing" policy
  - "Do nothing" = each TLS keeps its current phase at every decision step
  - Same warm-up, same trivial TLS handling, same metric collection
  - Only difference: no intelligent phase switching

- **Reduced traffic demand (5000 -> 1656 vehicles, depart_end 900 -> 600s)**
  - Initial 5000 vehicles caused gridlock regardless of signal timing
  - Reduced to 1656 vehicles (realistic for Hai Chau district size)
  - Shortened departure window to 600s, leaving 300s for network drainage
  - This demand level creates meaningful congestion without guaranteed gridlock

---

### Phase 5: Per-Intersection Realistic Timing

**Goal:** Replace fixed timing constants with engineering-formula-based per-TLS parameters.

- **TLSGeometry dataclass** (`tls_metadata.py`)
  - New dataclass storing per-intersection computed timing values
  - All values derived from junction geometry (width, approach speed, lane count)
  - Tier classification: Small (<25m), Medium (25-50m), Large (>=50m)

- **Per-TLS yellow from ITE formula**
  - `yellow = max(3.0, 1.0 + v / (2 * 3.05))`
  - Approach speed capped at 50 km/h for downtown
  - Small intersections: ~3.0s, Large intersections: ~3.3s

- **Per-TLS all-red from junction width clearance**
  - `allred = max(1.5, (W + 6.0) / v)`
  - Width measured from node bounding box in SUMO network
  - Small: ~2.2s, Medium: ~3.5s, Large: ~5.0s+

- **Per-TLS min_green from pedestrian crossing time**
  - `ped_min = 7.0 + width / 1.0` (Vietnamese walking speed 1.0 m/s)
  - Pedestrian constraint is the binding constraint for most medium/large intersections
  - A 40m intersection requires 47s minimum green for safe pedestrian crossing
  - Vehicle minimum (tier-based 10/15/20s) is secondary

- **Max cycle constraint <= 120 seconds**
  - Total cycle (all phases * min_green + yellow + allred) capped at 120 real seconds
  - If constraint violated, min_green is reduced proportionally (floor at 7s)
  - Prevents absurdly long cycles at complex intersections

- **Webster's optimal cycle for baseline**
  - `C_opt = (1.5L + 5) / (1 - Y)` clamped to [60, 120] seconds
  - Used as the default fixed-timing cycle for comparison
  - Green time split equally among phases

- **Tick-by-tick step() state machine**
  - Replaced batch transition with per-TLS independent state machine
  - Each TLS transitions through yellow -> all-red -> green at its own pace
  - Small intersections finish transition faster, getting proportionally more green time
  - Total sim ticks per step remains exactly delta_time (invariant maintained)

- **Observations normalized by per-TLS max_green**
  - Elapsed green time normalized by each TLS's own max_green (not a global constant)
  - Min-green satisfied flag uses each TLS's own min_green threshold
  - Enables the agent to learn size-appropriate behavior

- **Speed cap: all lanes 50 km/h**
  - Post-netconvert XML editing to cap all lane speeds to 13.89 m/s
  - Many OSM-sourced lanes had 100+ km/h speeds (highway=trunk tags)
  - Fixed in `build_danang.py` step 2b

- **Junction merge distance (20m -> 50m)**
  - Increased `junctions.join-dist` from 20m to 50m
  - Successfully merged the Duy Tan corridor cluster (3 closely-spaced TLS -> 1)
  - Prevents conflicting signal programs at adjacent intersections

---

### Phase 6: Trivial TLS Fix

**Goal:** Prevent non-AI-controlled trivial TLS from randomly blocking traffic.

- **Force ALL non-AI single-phase TLS to permanent green**
  - Discovery: 73 trivial TLS (single green phase) were cycling through yellow/red for no reason
  - These pedestrian crossings and median breaks randomly stopped traffic on major arterials
  - Fix: at environment reset, all non-AI TLS with <= 1 green phase are set to "GGG...G"

- **Fixed Nguyen Van Linh arterial**
  - Nguyen Van Linh (major north-south arterial) had 25+ trivial TLS
  - Each was a pedestrian crossing cycling independently on fixed timers
  - Combined effect: vehicles rarely got an uninterrupted green wave
  - After fix: continuous flow on the arterial, dramatic throughput improvement

- **Two-category trivial detection**
  - Category 1: Single green phase (pedestrian crossings, median breaks) --- always force green
  - Category 2: Uniform phase (roundabout entries where all phases have same G/r pattern) --- force green
  - Multi-phase intersections with 2+ distinct green phases are NOT forced green (left on default programs)

---

### Phase 7: Reward Optimization

**Goal:** Rebalance reward weights based on PressLight research and empirical training results.

- **Pressure-primary reward (w_pressure: 0.10 -> 0.40)**
  - Inspired by PressLight (Wei et al., KDD 2019)
  - Pressure (outgoing - incoming vehicles) is the most theoretically grounded signal for throughput
  - Increased to primary weight (0.40)

- **Removed broken global baseline bonus**
  - A global bonus compared network-wide wait against `baseline_wait=25.0`
  - Actual wait was always >>25 so every agent received **-0.5 penalty every step** (~-2400 total per episode)
  - This constant penalty made the reward signal completely uninformative
  - Removed entirely --- reward now depends only on the agent's own traffic state

- **Switch penalty scaled by per-TLS transition cost**
  - `transition_cost = (yellow_steps + allred_steps) / avg_transition`
  - Large intersections (longer yellow + all-red) have higher transition cost
  - Small intersections can switch more freely (shorter lost time)

- **Final reward weight distribution:**

  | Term | Weight | Direction |
  |------|--------|-----------|
  | Pressure | 0.40 | + (higher is better) |
  | Wait improvement | 0.20 | + (lower wait is better) |
  | Queue penalty | 0.25 | - (lower queue is better) |
  | Throughput | 0.10 | + (more throughput is better) |
  | Switch penalty | 0.05 | - (fewer switches is better) |

---

### Phase 8: Continuous Action Space + All-TLS Control

**Goal:** Switch from discrete 7-level actions to continuous duration control, and extend AI control to all 83 TLS.

- **TanhNormal distribution for continuous PPO** (`mappo_agent.py`)
  - Replaced discrete 7-level action space with a continuous TanhNormal policy
  - Actor outputs `mean` and `log_std`; action = `(tanh(u) + 1) / 2` where `u ~ Normal(mean, std)`
  - Correct log_prob via inverse tanh + Jacobian correction: `log P(a) = log P(u) - log(1 - tanh²(u))`
  - Without the Jacobian, gradient signal is wrong and entropy gets stuck at ~1.7 (near-uniform)
  - With TanhNormal: entropy stabilizes at ~0.41 (healthy exploration without randomness)
  - `log_std` initialized to -1.0 (std ≈ 0.37) for tighter initial distribution

- **Per-TLS duration decode fix** (`traffic_env.py`)
  - Continuous action [0,1] was mapped to the **global** min/max across all 83 TLS
  - Pedestrian crossings have max_green=123s → global range was 10.5s–123s
  - action=0.5 → 66.5s green, causing 200s+ cycles and massive wait times
  - Fix: each TLS maps its own action to its own per-TLS bounds
  - Multi-phase intersections: action=0.5 → 23-35s (appropriate for real intersections)

- **All 83 TLS under AI duration control**
  - Previously: 10 multi-phase TLS had AI phase selection, 73 pedestrian crossings forced to permanent green
  - Now: all 83 TLS receive AI-chosen green durations via continuous action [0,1]
  - Pedestrian crossings still have single phases (no phase switching), but AI controls how long each green lasts
  - Enables the network to dynamically lengthen/shorten pedestrian cycles based on demand

- **MASAC agent added** (`masac_agent.py`)
  - Multi-Agent SAC with shared `SACActorNetwork` and twin centralized `SACCriticNetwork`
  - 500K replay buffer for off-policy experience reuse
  - Auto-tuned entropy temperature `log_alpha` with target_entropy=-1.0
  - Soft target updates (tau=0.005)
  - Available via `--algorithm masac` CLI flag
  - Note: MASAC underperformed in practice due to alpha collapse on low-traffic episodes filling the buffer

- **Parallel MAPPO workers** (`train.py`)
  - `--workers N` flag spawns N SUMO instances collecting episodes simultaneously
  - Combined transitions feed a single PPO update (~N× faster training)
  - Recommended: `--workers 4` on typical hardware (8 workers causes CUDA OOM)

- **Best model saved by wait/vehicle ratio**
  - Previously saved by highest training reward — picked lucky low-traffic episodes
  - Now: only saves when `vehicles >= 200` and `wait_per_vehicle` is a new minimum
  - Ensures saved checkpoint reflects genuine policy quality, not favorable random seeds

- **`--resume-from` loads actor weights only**
  - When resuming from a checkpoint, only actor weights are restored
  - Critic starts fresh to avoid value miscalibration when reward function has changed
  - Loading the full checkpoint (actor + critic + optimizer) with a different reward caused wrong PPO advantages and policy collapse within 10 episodes

---

### Phase 9: Level 2 Neighbor-GAT Refinement

**Goal:** Give Level 2 richer local-network context and make parallel MAPPO checkpoint selection match actual rollout quality.

- **Neighbor feature vector expanded from 2D to 5D**
  - Previously, each neighbor only exposed queue and wait
  - Now each neighbor slot carries:
    - queue ratio
    - wait ratio
    - density ratio
    - current phase ratio
    - elapsed green ratio
  - This allows Level 2 to reason about both traffic state and signal state in nearby intersections

- **GAT updated to consume richer neighbor context**
  - The actor/critic now build an ego summary in the same feature space as the neighbors
  - Attention can rank neighbors using congestion and signal-cycle progress, not just queue/wait
  - Older checkpoints still load because neighbor feature dimensionality is auto-detected at load time

- **Parallel MAPPO best-checkpoint saving fixed**
  - Previously, parallel training could save post-update weights using pre-update rollout metrics
  - Now `best_model.pt` is saved from the rollout policy snapshot that actually produced the winning batch
  - Selection uses lowest batch mean wait, with throughput as a tie-breaker

- **Training/evaluation defaults aligned with the stable Level 2 setup**
  - Added `--worker-device` and `--curriculum` to the training CLI
  - Stable recommendation for Da Nang: `workers=2`, CPU rollout workers, `sim_length=1800`, `delta_time=30`
  - Compare/eval defaults now match the same `sim_length=1800`, `delta_time=30` setup

- **Retraining required for the richer 5D GAT features**
  - Older checkpoints remain compatible
  - A new Level 2 retrain is required for the model to benefit from the added neighbor signals

---

### Results

Final performance comparison after all Level 1 optimizations, averaged over 3 evaluation episodes at sim_length=3600:

| Metric | Baseline | AI Model | Change |
|--------|----------|----------|--------|
| Average Wait Time | 32.1s | **0.8s** | **-98%** |
| Average Queue Length | 0.2 vehicles | **0.0 vehicles** | **-88%** |
| Throughput (arrived) | 1,479 vehicles | 1,468 vehicles | -1% |

**Key observations:**
- -98% wait time is the result of the combined fix stack: removing the broken reward bonus, per-TLS duration decode, and continuous TanhNormal policy
- Queue drops to near-zero: intersections clear faster than vehicles arrive
- Throughput is roughly unchanged (-1%) --- vehicles are not lost, they simply wait far less

---

## Future Work

Potential next steps beyond the current Level 2 system:

- **Random traffic events** (accidents, road closures, VIP convoys, weather) via `EventManager`
- **Adaptive event response** --- agent learns to reroute traffic around disruptions  
- **Multi-city transfer** --- test whether the Hai Chau-trained policy generalizes to Hanoi/Ho Chi Minh City networks
- **Green wave coordination** --- explicit neighbor communication between adjacent TLS agents
- **Demand prediction** --- short-horizon vehicle count forecasting to improve anticipatory switching
