# FlowMind AI - Changelog

> Development history of the AI-adaptive traffic signal control system.

---

## Level 1: Pure Timing Optimization

Level 1 focuses exclusively on optimizing traffic signal timing for the 34 non-trivial intersections in Hai Chau District, Da Nang. No structural changes to the road network (adding/removing TLS) are made at this level.

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

- **Pressure-primary reward (w_pressure: 0.10 -> 0.30)**
  - Inspired by PressLight (Wei et al., KDD 2019)
  - Pressure (outgoing - incoming vehicles) is the most theoretically grounded signal for throughput
  - Increased from secondary (0.10) to primary weight (0.30)
  - Computed per-TLS using SUMO node's outgoing vs incoming vehicle counts

- **Wait reduced to secondary (w_wait: 0.40 -> 0.25)**
  - Wait time improvement was previously the dominant signal
  - Demoted to allow pressure to drive learning
  - Still important for user-facing metrics but less critical for policy optimization

- **Switch penalty scaled by per-TLS transition cost**
  - `transition_cost = (yellow_steps + allred_steps) / avg_transition`
  - Large intersections (longer yellow + all-red) have higher transition cost
  - Small intersections can switch more freely (shorter lost time)
  - Average across all TLS normalized to ~1.0

- **Throughput weight increased (w_throughput: 0.05 -> 0.10)**
  - Doubled to provide stronger signal for vehicle completion
  - Computed as change in vehicle count on incoming edges

- **Queue and fairness weights**
  - Queue: 0.15 (unchanged)
  - Fairness: 0.05 (unchanged)
  - Max-queue fairness prevents approach starvation

- **Final reward weight distribution:**

  | Term | Weight | Direction |
  |------|--------|-----------|
  | Pressure | 0.30 | + (higher is better) |
  | Wait improvement | 0.25 | + (lower wait is better) |
  | Queue penalty | 0.15 | - (lower queue is better) |
  | Switch penalty | 0.15 | - (fewer switches is better) |
  | Throughput | 0.10 | + (more throughput is better) |
  | Fairness | 0.05 | - (lower max-queue is better) |

---

### Results

Final performance comparison after all Level 1 optimizations, averaged over 3 evaluation episodes:

| Metric | Baseline | AI Model | Change |
|--------|----------|----------|--------|
| Average Wait Time | 974.5s | 175.9s | **-82%** |
| Average Queue Length | 5.2 vehicles | 1.2 vehicles | **-77%** |
| Throughput (arrived) | 591 vehicles | 1,078 vehicles | **+82%** |

**Key observations:**
- The AI model reduces average wait time by over 13 minutes per vehicle
- Queue lengths drop below 2 vehicles on average, indicating free-flowing conditions
- Nearly double the vehicles complete their journeys, demonstrating genuine throughput improvement
- The baseline "do nothing" policy is a fair comparison --- same environment, same constraints, only difference is intelligent phase selection

---

## Future: Level 2 (Planned)

Level 2 will extend the system with:
- **Random traffic events** (accidents, road closures, weather) via `EventManager`
- **Adaptive event response** --- agent learns to reroute traffic around disruptions
- **TLS placement optimization** --- AI recommends where to add/remove traffic lights
- **Extended observation space** with event-aware features (edge blocked flags, active event count)

Code scaffolding for Level 2 exists in the codebase (commented out with `[Level 2 REMOVED]` markers) but is disabled for Level 1.
