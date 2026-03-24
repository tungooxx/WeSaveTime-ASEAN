# FlowMind AI - Level 2: Advanced Traffic Intelligence

> Building on Level 1's per-intersection timing optimization with coordination, pedestrians, and adaptive features.

---

## Level 1 Recap

Level 1 delivered:
- MAPPO multi-agent RL controlling 10 non-trivial TLS in Hai Chau district
- Per-intersection timing from engineering formulas (ITE, FHWA, MUTCD)
- Vietnamese countdown timer model (committed green durations)
- Pressure-primary reward (PressLight-inspired)
- 73 trivial TLS forced to permanent green
- Speed cap 50 km/h, junction merge 50m
- **Results: -82% wait time, -77% queue, +82% throughput vs baseline**

---

## Level 2 Roadmap

### Phase 1: Pressure State + Observation Enhancement
**Priority: HIGH | Impact: HIGH | Effort: LOW**

Add pressure (incoming - outgoing vehicles per movement) to the observation space. Currently pressure is only used in the reward — the agent can't SEE it. PressLight (KDD 2019) proved that pressure as state representation is the most effective signal.

Changes:
- Add per-edge pressure to observation vector (12 slots)
- OBS_DIM: 39 → 51
- No architecture change needed (just input size)

### Phase 2: Variable Green Duration
**Priority: HIGH | Impact: HIGH | Effort: MEDIUM**

Currently the AI picks WHICH phase but green duration is fixed at min_green. Real Vietnamese signals show a countdown timer — the AI should choose the duration too.

Options:
- A) Extend action space: 7 phases × 3 durations = 21 actions
- B) Two-head network: phase head + duration head
- C) Keep phase-only, let AI extend by choosing same phase again

### Phase 3: Pedestrian Simulation
**Priority: MEDIUM | Impact: MEDIUM | Effort: MEDIUM**

Add pedestrians crossing roads at signalized intersections. Validates the min_green pedestrian timing constraints from Level 1.

Changes:
- Sidewalks via `--sidewalks.guess`
- Pedestrian crossings via `--crossings.guess`
- 400 pedestrians (regular + elderly)
- SUMO striping model
- Pedestrian delay in reward function

### Phase 4: Neighbor Coordination (CoLight)
**Priority: HIGH | Impact: VERY HIGH | Effort: HIGH**

Enable green wave coordination between adjacent intersections. Currently each TLS decides independently — no corridor-level optimization.

Changes:
- Graph attention network (GAT) for neighbor influence
- Each agent sees its own obs + weighted neighbor observations
- Adjacency matrix from network topology
- Enables green wave patterns along arterials

### Phase 5: CityLight Phase Reindexing
**Priority: MEDIUM | Impact: MEDIUM | Effort: MEDIUM**

Standardize phase ordering so phase 0 always means "NS through" regardless of intersection geometry. Improves parameter sharing across heterogeneous intersections.

### Phase 6: Random Traffic Events
**Priority: LOW | Impact: MEDIUM | Effort: MEDIUM**

Accidents, road closures, weather events that the agent must adapt to. Code scaffold exists (commented out with `[Level 2 REMOVED]` markers).

### Phase 7: Demand-Based Green Splits
**Priority: LOW | Impact: LOW | Effort: LOW**

Use route file demand to compute per-approach flow ratios for Webster's formula. Currently using equal splits for baseline timing.

---

## Recommended Implementation Order

```
Phase 1 (Pressure State)     → quick win, proven by research
Phase 3 (Pedestrians)        → visual impact, validates timing
Phase 2 (Variable Duration)  → makes countdown timer meaningful
Phase 4 (Coordination)       → biggest performance gain
Phase 5 (Phase Reindexing)   → improves learning efficiency
Phase 6 (Events)             → resilience testing
Phase 7 (Demand Splits)      → baseline improvement
```

---

## Implementation Status

### Phase 1: Pressure State — IMPLEMENTED
- OBS_DIM: 38 → 50 (added 12 pressure slots at [36..47])
- Pressure per edge = (avg_outgoing - incoming) / capacity, normalized [-1, 1]
- Vietnamese countdown timer (committed green durations)
- rl_dashboard: Level 1 / Level 2 training selector
- Level 1 training uses 38-dim obs (auto-remap strips pressure)
- Level 2 training uses 50-dim obs (full PressLight state)
- Level 1 results saved to `checkpoints/history/level1_comparison.json`

### Phase 2: Variable Green Duration — IMPLEMENTED
- ACT_DIM: 7 → 21 (7 phases × 3 duration levels)
- Duration levels: SHORT (ped minimum), MEDIUM (midpoint), LONG (max green)
- AI chooses both WHICH phase and HOW LONG
- Vietnamese countdown timer shows committed duration
- Max green extended: small=45s, medium=60s, large=90s
- Visualizer shows duration label: G=45s(LONG), G=22s(SHORT)

### Phase 4: Neighbor Coordination — IMPLEMENTED
- Distance-based neighbor graph (500m radius, max 4 neighbors)
- Mean neighbor observation passed to MAPPO actor
- Actor input: [own_obs(50), neighbor_mean_obs(50)] = 100 dimensions
- Critic unchanged: [own_obs, global_mean_obs] = 100 dimensions
- 6/10 TLS have neighbors, 4 isolated (too far from other non-trivial TLS)
- Enables green wave coordination along corridors
- Training, compare, and visualize all updated for neighbor obs

### Baseline: Real SUMO Default Timing
- Baseline now uses SUMO's auto-generated TLS programs from netconvert
- No AI override — raw SUMO default cycling
- Real comparison target (not fake "do nothing" policy)
