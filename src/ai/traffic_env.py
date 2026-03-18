"""
FlowMind AI - Gymnasium environment for multi-agent traffic signal RL.

Each non-trivial TLS (traffic light system) in the SUMO network is an
independent agent that shares the same neural network (parameter sharing).
The environment provides per-TLS observations and per-TLS rewards.

Observation per TLS (fixed 26-dim vector):
    [queue_0 ... queue_11, wait_0 ... wait_11, phase_ratio, elapsed_ratio]

Action per TLS:
    Index 0..6 -> mapped to a green phase of that TLS.
    Index 7    -> "TLS OFF" (all-green / yield) — the AI can learn that
                  some intersections work better without signal control.
"""

from __future__ import annotations

import os
import shutil
from typing import Optional

import gymnasium as gym
import numpy as np
import traci

from ..simulation.tls_metadata import TLSMetadata
from ..simulation.events import EventManager, make_event, EVENT_TYPES
from .reward import compute_tls_reward

# ── Constants ─────────────────────────────────────────────────────────
MAX_INCOMING_EDGES = 12     # pad/truncate incoming edges to this size
MAX_GREEN_PHASES = 7        # actions 0-6: choose a green phase
ACT_OFF = 7                 # action 7: turn TLS off (all-green / yield)
MIN_GREEN_STEPS = 5         # minimum green time before allowing phase change

# Observation layout per TLS (fixed size for parameter sharing):
#   [0..11]   queue per edge          (12)
#   [12..23]  wait per edge           (12)
#   [24..35]  lane density per edge   (12)
#   [36..47]  edge blocked by event   (12)  ← NEW: 1.0 if event active on this edge
#   [48]      current phase ratio     (1)
#   [49]      elapsed time ratio      (1)
#   [50]      min_green satisfied     (1)
#   [51]      num active events       (1)   ← NEW: how many events near this TLS
OBS_DIM = MAX_INCOMING_EDGES * 4 + 4   # 52
OLD_OBS_DIM = MAX_INCOMING_EDGES * 2 + 2  # 26 (v1 layout)
ACT_DIM = MAX_GREEN_PHASES + 1         # 8 (7 phases + 1 off)


V2_OBS_DIM = MAX_INCOMING_EDGES * 3 + 3  # 39 (v2 layout without event features)


def remap_obs_for_old_model(obs_new: np.ndarray, target_dim: int = OLD_OBS_DIM
                            ) -> np.ndarray:
    """Convert current observation to an older layout for backward compat.

    Supported target dims:
      26 (v1): [queue(12), wait(12), phase_ratio, elapsed]
      39 (v2): [queue(12), wait(12), density(12), phase_ratio, elapsed, min_green]
    """
    if target_dim == OLD_OBS_DIM:  # 26
        obs = np.zeros(OLD_OBS_DIM, dtype=np.float32)
        obs[:12] = obs_new[:12]                             # queue
        obs[12:24] = obs_new[12:24]                         # wait
        obs[24] = obs_new[MAX_INCOMING_EDGES * 4]           # phase_ratio (slot 48)
        obs[25] = obs_new[MAX_INCOMING_EDGES * 4 + 1]       # elapsed (slot 49)
        return obs
    elif target_dim == V2_OBS_DIM:  # 39
        obs = np.zeros(V2_OBS_DIM, dtype=np.float32)
        obs[:12] = obs_new[:12]                             # queue
        obs[12:24] = obs_new[12:24]                         # wait
        obs[24:36] = obs_new[24:36]                         # density
        obs[36] = obs_new[MAX_INCOMING_EDGES * 4]           # phase_ratio
        obs[37] = obs_new[MAX_INCOMING_EDGES * 4 + 1]       # elapsed
        obs[38] = obs_new[MAX_INCOMING_EDGES * 4 + 2]       # min_green
        return obs
    else:
        return obs_new[:target_dim]


class SumoTrafficEnv(gym.Env):
    """Multi-agent SUMO traffic signal environment."""

    metadata = {"render_modes": ["sumo-gui"]}

    def __init__(
        self,
        net_file: str,
        route_file: str,
        sumo_cfg: Optional[str] = None,
        delta_time: int = 10,
        sim_length: int = 3600,
        gui: bool = False,
        seed: int = 42,
        min_green_phases: int = 2,
        min_incoming: int = 2,
        yellow_time: int = 3,
        collision_penalty: float = 5.0,
        ebrake_penalty: float = 0.5,
        random_events: bool = True,
        max_concurrent_events: int = 2,
        event_probability: float = 0.01,
    ) -> None:
        super().__init__()

        self.net_file = os.path.abspath(net_file)
        self.route_file = os.path.abspath(route_file)
        self.sumo_cfg = os.path.abspath(sumo_cfg) if sumo_cfg else None
        self.delta_time = delta_time
        self.sim_length = sim_length
        self.gui = gui
        self.seed_val = seed
        self.yellow_time = yellow_time
        self.collision_penalty = collision_penalty
        self.ebrake_penalty = ebrake_penalty

        # ── TLS discovery ────────────────────────────────────────────
        self._tls_meta = TLSMetadata(self.net_file)
        self._tls_list = self._tls_meta.get_non_trivial(
            min_green_phases, min_incoming
        )
        self.tls_ids: list[str] = [t.id for t in self._tls_list]
        self._tls_map = {t.id: t for t in self._tls_list}

        # Cache sumolib network for pressure + density computation
        import sumolib
        self._net = sumolib.net.readNet(self.net_file)

        # Pre-compute edge capacities for lane density observation
        # capacity = num_lanes * edge_length / 7.5m (avg vehicle spacing)
        self._edge_capacity: dict[str, float] = {}
        for tls in self._tls_list:
            for eid in tls.incoming_edges:
                if eid not in self._edge_capacity:
                    try:
                        edge = self._net.getEdge(eid)
                        n_lanes = edge.getLaneNumber()
                        length = edge.getLength()
                        self._edge_capacity[eid] = max(n_lanes * length / 7.5, 1.0)
                    except Exception:
                        self._edge_capacity[eid] = 10.0

        # Green-phase indices per TLS (for action mapping)
        self._green_phases: dict[str, list[int]] = {}
        for tls in self._tls_list:
            gp = tls.green_phase_indices()
            self._green_phases[tls.id] = gp if gp else [0]

        # ── Candidate TLS tracking ─────────────────────────────────
        self.candidate_tls_ids: set[str] = set()
        self.existing_tls_ids: set[str] = set()
        self._load_candidate_info()

        # Per-TLS action statistics (accumulated across episodes)
        self.action_stats: dict[str, dict[int, int]] = {
            tid: {} for tid in self.tls_ids
        }

        # ── Gym spaces (per-agent, shared definition) ────────────────
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(ACT_DIM)

        # ── Random event config ────────────────────────────────────
        self.random_events = random_events
        self.max_concurrent_events = max_concurrent_events
        self.event_probability = event_probability
        self._event_manager = None
        self._active_event_log: list[str] = []

        # Build list of non-internal edges for random event placement
        self._all_edges: list[str] = [
            e.getID() for e in self._net.getEdges()
            if not e.getID().startswith(":") and e.getLaneNumber() >= 2
        ]

        # ── Runtime state ────────────────────────────────────────────
        self._conn: Optional[traci.Connection] = None
        self._conn_label = "rl_training"
        self._sim_step: int = 0
        self._max_steps: int = sim_length // delta_time

        # Per-TLS tracking for reward deltas
        self._prev_waiting: dict[str, float] = {}
        self._prev_throughput: dict[str, int] = {}
        self._current_phases: dict[str, int] = {}
        self._phase_start_step: dict[str, int] = {}

    # ── Properties ────────────────────────────────────────────────────

    @property
    def num_agents(self) -> int:
        return len(self.tls_ids)

    def get_valid_actions(self, tls_id: str) -> list[int]:
        """Valid action indices for *tls_id* (0..N-1 = green phases, ACT_OFF = TLS off)."""
        phase_actions = list(range(len(self._green_phases.get(tls_id, [0]))))
        return phase_actions + [ACT_OFF]

    # ── Candidate TLS helpers ──────────────────────────────────────

    def _load_candidate_info(self) -> None:
        """Load candidate_tls.json to identify which TLS are candidates vs existing."""
        import json
        candidate_file = os.path.join(
            os.path.dirname(self.net_file), "candidate_tls.json")
        if os.path.isfile(candidate_file):
            with open(candidate_file) as f:
                data = json.load(f)
            existing = set(data.get("existing_tls", []))
            candidate_ids = {c["id"] for c in data.get("candidates", [])}
        else:
            existing = set(self.tls_ids)
            candidate_ids = set()

        self.existing_tls_ids = {tid for tid in self.tls_ids if tid in existing}
        self.candidate_tls_ids = {tid for tid in self.tls_ids if tid in candidate_ids}
        # TLS added by netconvert may get a different ID — catch stragglers
        for tid in self.tls_ids:
            if tid not in self.existing_tls_ids and tid not in self.candidate_tls_ids:
                self.candidate_tls_ids.add(tid)

    def record_actions(self, actions: dict[str, int]) -> None:
        """Track per-TLS action counts for recommendation analysis."""
        for tid, act in actions.items():
            self.action_stats[tid][act] = self.action_stats[tid].get(act, 0) + 1

    def get_tls_snapshot(self) -> dict:
        """Return a lightweight snapshot of TLS add/remove decisions so far.

        Returns dict with:
            n_add: candidate TLS the agent keeps active (OFF < 40%)
            n_remove: existing TLS the agent wants OFF (OFF > 60%)
            n_candidates: total candidate TLS
            n_existing: total existing TLS
        """
        n_add = n_remove = 0
        for tid in self.tls_ids:
            stats = self.action_stats.get(tid, {})
            total = sum(stats.values())
            if total == 0:
                continue
            off_pct = stats.get(ACT_OFF, 0) / total
            if tid in self.candidate_tls_ids and off_pct < 0.4:
                n_add += 1
            elif tid in self.existing_tls_ids and off_pct > 0.6:
                n_remove += 1
        return {
            "n_add": n_add,
            "n_remove": n_remove,
            "n_candidates": len(self.candidate_tls_ids),
            "n_existing": len(self.existing_tls_ids),
        }

    def get_tls_details(self) -> list[dict]:
        """Return per-TLS detail snapshot for the current episode.

        Each entry has: tls_id, type (existing/candidate), off_pct,
        top_phase, wait, queue, action_counts, decision.
        """
        details = []
        for tid in self.tls_ids:
            stats = self.action_stats.get(tid, {})
            total = sum(stats.values())
            if total == 0:
                continue

            off_pct = stats.get(ACT_OFF, 0) / total * 100
            is_candidate = tid in self.candidate_tls_ids

            # Get current wait/queue for this TLS
            tls_info = self._tls_map.get(tid)
            wait = 0.0
            queue = 0
            if tls_info and self._conn:
                for eid in tls_info.incoming_edges:
                    try:
                        wait += self._conn.edge.getWaitingTime(eid)
                        queue += self._conn.edge.getLastStepHaltingNumber(eid)
                    except Exception:
                        pass

            # Get road name from first incoming edge
            road_name = tid
            if tls_info and tls_info.incoming_edges:
                try:
                    edge = self._net.getEdge(tls_info.incoming_edges[0])
                    name = edge.getName()
                    if name:
                        road_name = name
                except Exception:
                    pass

            # Top phase (most chosen non-OFF action)
            phase_counts = {k: v for k, v in stats.items() if k != ACT_OFF}
            top_phase = max(phase_counts, key=phase_counts.get) if phase_counts else -1

            # Decision label
            if is_candidate:
                decision = "ADD" if off_pct < 40 else "NO TLS"
            else:
                decision = "REMOVE" if off_pct > 60 else "KEEP"

            details.append({
                "tls_id": tid,
                "road_name": road_name,
                "type": "candidate" if is_candidate else "existing",
                "decision": decision,
                "off_pct": round(off_pct, 1),
                "top_phase": top_phase,
                "wait": round(wait, 1),
                "queue": queue,
                "actions": dict(stats),
                "n_incoming": len(tls_info.incoming_edges) if tls_info else 0,
            })

        details.sort(key=lambda d: d["off_pct"])
        return details

    def get_recommendations(self) -> dict:
        """Analyze action statistics to generate TLS recommendations.

        Returns dict with keys:
            - remove: list of existing TLS the agent wants OFF (>60% OFF actions)
            - add: list of candidate TLS the agent wants active (<40% OFF actions)
            - keep_off: list of candidate TLS the agent keeps OFF
            - timing: dict of tls_id -> phase distribution for all active TLS
        """
        remove = []
        add = []
        keep_off = []
        timing = {}

        for tid in self.tls_ids:
            stats = self.action_stats.get(tid, {})
            total = sum(stats.values())
            if total == 0:
                continue

            off_pct = stats.get(ACT_OFF, 0) / total

            if tid in self.candidate_tls_ids:
                if off_pct < 0.4:
                    add.append({"id": tid, "off_pct": round(off_pct * 100, 1),
                                "action_dist": stats})
                else:
                    keep_off.append({"id": tid, "off_pct": round(off_pct * 100, 1)})
            elif tid in self.existing_tls_ids:
                if off_pct > 0.6:
                    remove.append({"id": tid, "off_pct": round(off_pct * 100, 1),
                                   "action_dist": stats})

            # Phase timing for all TLS
            timing[tid] = {
                "action_dist": {str(k): v for k, v in stats.items()},
                "total_actions": total,
                "off_pct": round(off_pct * 100, 1),
                "is_candidate": tid in self.candidate_tls_ids,
            }

        return {
            "remove": remove,
            "add": add,
            "keep_off": keep_off,
            "timing": timing,
        }

    # ── SUMO lifecycle ────────────────────────────────────────────────

    def _find_binary(self) -> str:
        name = "sumo-gui" if self.gui else "sumo"
        found = shutil.which(name)
        if found:
            return found
        sumo_home = os.environ.get("SUMO_HOME", "")
        if sumo_home:
            p = os.path.join(sumo_home, "bin", name)
            if os.path.isfile(p) or os.path.isfile(p + ".exe"):
                return p
        raise FileNotFoundError(f"Cannot find '{name}'. Install SUMO or set SUMO_HOME.")

    def _start_sumo(self) -> None:
        # Close stale connection
        try:
            traci.getConnection(self._conn_label).close()
        except (traci.TraCIException, KeyError):
            pass

        binary = self._find_binary()

        if self.sumo_cfg:
            cmd = [binary, "-c", self.sumo_cfg]
        else:
            cmd = [binary, "-n", self.net_file, "-r", self.route_file]

        cmd += [
            "--seed", str(self.seed_val),
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--time-to-teleport", "300",
            "--waiting-time-memory", "1000",
            "--error-log", os.devnull,
        ]

        traci.start(cmd, label=self._conn_label)
        self._conn = traci.getConnection(self._conn_label)

    # ── Gym interface ─────────────────────────────────────────────────

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict[str, np.ndarray], dict]:
        if seed is not None:
            self.seed_val = seed

        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass

        self._start_sumo()
        self._sim_step = 0
        self._prev_waiting.clear()
        self._prev_throughput.clear()
        self._current_phases.clear()
        self._phase_start_step.clear()

        # Initialise per-TLS tracking and validate green phases at runtime
        for tls_id in self.tls_ids:
            self._prev_waiting[tls_id] = 0.0
            self._prev_throughput[tls_id] = 0
            try:
                self._current_phases[tls_id] = self._conn.trafficlight.getPhase(tls_id)
            except traci.TraCIException:
                self._current_phases[tls_id] = 0
            self._phase_start_step[tls_id] = 0

            # Re-validate green phase indices against SUMO's runtime program
            try:
                logic = self._conn.trafficlight.getAllProgramLogics(tls_id)
                if logic:
                    runtime_num_phases = len(logic[0].phases)
                    static_gp = self._green_phases.get(tls_id, [0])
                    valid_gp = [p for p in static_gp if p < runtime_num_phases]
                    self._green_phases[tls_id] = valid_gp if valid_gp else [0]
            except traci.TraCIException:
                self._green_phases[tls_id] = [0]

        # ── Event manager ──────────────────────────────────────────
        self._active_event_log = []
        self._rng = np.random.RandomState(self.seed_val)
        if self.random_events and self._rng.random() > 0.4:
            # 60% of episodes have events, 40% are event-free
            self._event_manager = EventManager(self._conn, self.net_file)
            self._maybe_inject_events()
        else:
            self._event_manager = None

        # Warm-up so some vehicles enter the network
        for _ in range(self.delta_time):
            self._conn.simulationStep()
            self._sim_step += 1
            if self._event_manager:
                msgs = self._event_manager.tick(float(self._sim_step))
                self._active_event_log.extend(msgs)

        obs = self._get_observations()
        return obs, {"step": self._sim_step, "sim_time": self._sim_step}

    def step(
        self, actions: dict[str, int]
    ) -> tuple[dict[str, np.ndarray], dict[str, float], bool, bool, dict]:
        """Apply phase actions, advance SUMO by *delta_time* seconds, return
        (obs, rewards, terminated, truncated, info)."""

        # ── 1. Resolve target phases and cache state strings ─────────
        # We use setRedYellowGreenState exclusively (not setPhase) because
        # setRedYellowGreenState switches the TLS to a custom 1-phase program,
        # making a subsequent setPhase fail with "not in allowed range".
        #
        # Action ACT_OFF = "turn off TLS" → set all links to green (G).
        targets: dict[str, tuple[int, str]] = {}  # tls_id -> (phase_idx, state_str)
        off_tls: set[str] = set()                  # TLS the AI wants turned off

        for tls_id, action_idx in actions.items():
            # ── Min-green enforcement (from sumo-rl) ──────────────
            # If current phase hasn't been green long enough, keep it
            elapsed = self._sim_step - self._phase_start_step.get(tls_id, 0)
            current_phase = self._current_phases.get(tls_id, 0)
            if elapsed < MIN_GREEN_STEPS and action_idx != ACT_OFF:
                # Force keep current phase
                green_phases = self._green_phases.get(tls_id, [0])
                for ai, gp in enumerate(green_phases):
                    if gp == current_phase:
                        action_idx = ai
                        break

            if action_idx == ACT_OFF:
                # "Off" = all-green (every link gets 'G')
                off_tls.add(tls_id)
                try:
                    cur = self._conn.trafficlight.getRedYellowGreenState(tls_id)
                    targets[tls_id] = (-1, "G" * len(cur))
                except traci.TraCIException:
                    targets[tls_id] = (-1, "")
            else:
                green_phases = self._green_phases.get(tls_id, [0])
                target_phase = (
                    green_phases[action_idx]
                    if action_idx < len(green_phases)
                    else green_phases[0]
                )
                tls_info = self._tls_map.get(tls_id)
                if tls_info and target_phase < len(tls_info.phases):
                    targets[tls_id] = (target_phase, tls_info.phases[target_phase].state)
                else:
                    targets[tls_id] = (0, "")

        # ── 2. Apply yellow where phase changes ──────────────────────
        for tls_id, (target_phase, _) in targets.items():
            current_phase = self._current_phases.get(tls_id, 0)
            if target_phase != current_phase:
                if tls_id not in off_tls:
                    self._set_yellow(tls_id, current_phase)
                self._current_phases[tls_id] = target_phase
                self._phase_start_step[tls_id] = self._sim_step + self.yellow_time

        # ── 3. Advance yellow period ─────────────────────────────────
        for _ in range(self.yellow_time):
            self._conn.simulationStep()
            self._sim_step += 1

        # ── 4. Set target green states (using state strings, not setPhase) ─
        for tls_id, (_, state_str) in targets.items():
            if state_str:
                try:
                    self._conn.trafficlight.setRedYellowGreenState(tls_id, state_str)
                except traci.TraCIException:
                    pass

        # ── 5. Advance remaining green time ──────────────────────────
        green_time = max(self.delta_time - self.yellow_time, 1)
        for _ in range(green_time):
            self._conn.simulationStep()
            self._sim_step += 1
            # Tick events each sim step
            if self._event_manager:
                msgs = self._event_manager.tick(float(self._sim_step))
                self._active_event_log.extend(msgs)

        # ── 5b. Maybe inject new random events ─────────────────────
        if self._event_manager:
            self._maybe_inject_events()

        # ── 6. Observe & reward ───────────────────────────────────────
        obs = self._get_observations()
        rewards = self._compute_rewards()

        terminated = self._sim_step >= self.sim_length
        truncated = False

        # End early if no vehicles remain
        try:
            if self._conn.simulation.getMinExpectedNumber() <= 0:
                terminated = True
        except traci.TraCIException:
            pass

        info = {
            "step": self._sim_step,
            "sim_time": self._sim_step,
        }
        try:
            info["vehicles_in_network"] = self._conn.vehicle.getIDCount()
        except traci.TraCIException:
            pass

        return obs, rewards, terminated, truncated, info

    # ── Yellow helper ─────────────────────────────────────────────────

    def _set_yellow(self, tls_id: str, _current_phase: int) -> None:
        """Turn all current greens to yellow for the transition period."""
        try:
            state = self._conn.trafficlight.getRedYellowGreenState(tls_id)
            yellow = "".join("y" if c in ("G", "g") else c for c in state)
            self._conn.trafficlight.setRedYellowGreenState(tls_id, yellow)
        except traci.TraCIException:
            pass

    # ── Observations ──────────────────────────────────────────────────

    def _get_observations(self) -> dict[str, np.ndarray]:
        return {tls_id: self._obs_for(tls_id) for tls_id in self.tls_ids}

    def _obs_for(self, tls_id: str) -> np.ndarray:
        """Build a fixed-size (OBS_DIM,) observation vector for one TLS.

        Layout: [queue(12), wait(12), density(12), event_blocked(12),
                 phase_ratio, elapsed, min_green, num_events]
        """
        vec = np.zeros(OBS_DIM, dtype=np.float32)
        tls_info = self._tls_map.get(tls_id)
        if tls_info is None:
            return vec

        off_q = 0                        # queue slots
        off_w = MAX_INCOMING_EDGES       # wait slots
        off_d = MAX_INCOMING_EDGES * 2   # density slots
        off_e = MAX_INCOMING_EDGES * 3   # event blocked slots

        # Collect edges affected by active events
        affected_edges: set[str] = set()
        n_events = 0
        if self._event_manager:
            for evt in self._event_manager.get_active():
                affected_edges.update(evt.affected_edges)
                n_events += 1

        for i, edge_id in enumerate(tls_info.incoming_edges[:MAX_INCOMING_EDGES]):
            try:
                q = self._conn.edge.getLastStepHaltingNumber(edge_id)
                w = self._conn.edge.getWaitingTime(edge_id)
                veh_count = self._conn.edge.getLastStepVehicleNumber(edge_id)
                capacity = self._edge_capacity.get(edge_id, 10.0)
                density = veh_count / capacity
            except traci.TraCIException:
                q, w, density = 0, 0.0, 0.0
            vec[off_q + i] = min(q / 50.0, 1.0)
            vec[off_w + i] = min(w / 300.0, 1.0)
            vec[off_d + i] = min(density, 1.0)
            # Event flag: 1.0 if this edge is affected by an active event
            vec[off_e + i] = 1.0 if edge_id in affected_edges else 0.0

        # Current phase ratio
        cur_phase = self._current_phases.get(tls_id, 0)
        num_phases = max(tls_info.num_phases, 1)
        vec[MAX_INCOMING_EDGES * 4] = cur_phase / num_phases

        # Elapsed time in current phase
        elapsed = self._sim_step - self._phase_start_step.get(tls_id, 0)
        vec[MAX_INCOMING_EDGES * 4 + 1] = min(elapsed / 60.0, 1.0)

        # Min-green satisfied flag
        vec[MAX_INCOMING_EDGES * 4 + 2] = 1.0 if elapsed >= MIN_GREEN_STEPS else 0.0

        # Number of active events (normalized)
        vec[MAX_INCOMING_EDGES * 4 + 3] = min(n_events / 5.0, 1.0)

        return vec

    # ── Rewards ───────────────────────────────────────────────────────

    def _compute_rewards(self) -> dict[str, float]:
        rewards: dict[str, float] = {}

        # ── Collect collision data (global, then attribute to nearest TLS) ──
        tls_collisions: dict[str, int] = {tid: 0 for tid in self.tls_ids}
        tls_ebrakes: dict[str, int] = {tid: 0 for tid in self.tls_ids}
        try:
            collisions = self._conn.simulation.getCollisions()
            for col in collisions:
                # Attribute collision to the nearest TLS by matching edge
                lane = getattr(col, "lane", "")
                edge = lane.rsplit("_", 1)[0] if "_" in lane else lane
                for tid in self.tls_ids:
                    tls_info = self._tls_map.get(tid)
                    if tls_info and edge in tls_info.incoming_edges:
                        tls_collisions[tid] += 1
                        break
        except Exception:
            pass

        # ── Collect emergency braking (vehicles on incoming edges) ──
        for tls_id in self.tls_ids:
            tls_info = self._tls_map.get(tls_id)
            if tls_info is None:
                continue
            ebrakes = 0
            for edge_id in tls_info.incoming_edges:
                try:
                    for vid in self._conn.edge.getLastStepVehicleIDs(edge_id):
                        try:
                            # Emergency brake: deceleration > 4.5 m/s^2
                            accel = self._conn.vehicle.getAcceleration(vid)
                            if accel < -4.5:
                                ebrakes += 1
                        except traci.TraCIException:
                            pass
                except traci.TraCIException:
                    pass
            tls_ebrakes[tls_id] = ebrakes

        # ── Per-TLS reward ──────────────────────────────────────────────
        for tls_id in self.tls_ids:
            tls_info = self._tls_map.get(tls_id)
            if tls_info is None:
                rewards[tls_id] = 0.0
                continue

            total_wait = 0.0
            queues: list[float] = []
            incoming_veh = 0

            for edge_id in tls_info.incoming_edges:
                try:
                    q = self._conn.edge.getLastStepHaltingNumber(edge_id)
                    w = self._conn.edge.getWaitingTime(edge_id)
                    t = self._conn.edge.getLastStepVehicleNumber(edge_id)
                except traci.TraCIException:
                    q, w, t = 0, 0.0, 0
                queues.append(float(q))
                total_wait += w
                incoming_veh += t

            # Pressure: vehicles on outgoing - incoming
            outgoing_veh = 0
            try:
                node = self._net.getNode(tls_id)
                for e in node.getOutgoing():
                    eid = e.getID()
                    if not eid.startswith(":"):
                        try:
                            outgoing_veh += self._conn.edge.getLastStepVehicleNumber(eid)
                        except traci.TraCIException:
                            pass
                pressure = float(outgoing_veh - incoming_veh)
            except Exception:
                pressure = 0.0

            rewards[tls_id] = compute_tls_reward(
                old_waiting=self._prev_waiting.get(tls_id, 0.0),
                new_waiting=total_wait,
                queue_lengths=queues,
                old_throughput=self._prev_throughput.get(tls_id, 0),
                new_throughput=incoming_veh,
                pressure=pressure,
                collisions=tls_collisions.get(tls_id, 0),
                emergency_brakes=tls_ebrakes.get(tls_id, 0),
                collision_penalty=self.collision_penalty,
                ebrake_penalty=self.ebrake_penalty,
            )

            self._prev_waiting[tls_id] = total_wait
            self._prev_throughput[tls_id] = incoming_veh

        return rewards

    # ── Metrics (for logging) ─────────────────────────────────────────

    def get_metrics(self) -> dict:
        if self._conn is None:
            return {}
        total_wait = 0.0
        total_queue = 0
        count = 0
        for tls_id in self.tls_ids:
            tls_info = self._tls_map.get(tls_id)
            if tls_info is None:
                continue
            for edge_id in tls_info.incoming_edges:
                try:
                    total_queue += self._conn.edge.getLastStepHaltingNumber(edge_id)
                    total_wait += self._conn.edge.getWaitingTime(edge_id)
                    count += 1
                except traci.TraCIException:
                    pass
        n = max(count, 1)
        vehicles = 0
        collisions = 0
        try:
            vehicles = self._conn.vehicle.getIDCount()
        except traci.TraCIException:
            pass
        try:
            collisions = self._conn.simulation.getCollidingVehiclesNumber()
        except traci.TraCIException:
            pass
        return {
            "avg_wait_time": round(total_wait / n, 2),
            "avg_queue_length": round(total_queue / n, 2),
            "total_vehicles": vehicles,
            "collisions": collisions,
            "sim_time": self._sim_step,
        }

    # ── Random event injection ─────────────────────────────────────

    def _maybe_inject_events(self) -> None:
        """Randomly inject traffic events to train the agent for disruptions."""
        if not self._event_manager or not self._rng:
            return

        n_active = len(self._event_manager.get_active())
        if n_active >= self.max_concurrent_events:
            return

        if self._rng.random() > self.event_probability:
            return

        # Pick random event type (weighted: rain and accidents more common)
        weights = {
            "accident": 0.25,
            "heavy_rain": 0.20,
            "flood": 0.15,
            "construction": 0.15,
            "concert": 0.15,
            "vip": 0.10,
        }
        types = list(weights.keys())
        probs = np.array([weights[t] for t in types])
        probs /= probs.sum()
        event_type = self._rng.choice(types, p=probs)

        # Pick random edges
        if event_type == "heavy_rain":
            affected = []  # city-wide, no specific edges
        else:
            n_edges = self._rng.randint(1, 4)
            idx = self._rng.randint(0, len(self._all_edges), size=n_edges)
            affected = [self._all_edges[i] for i in idx]

        # Random intensity and duration
        intensity = float(self._rng.uniform(0.3, 0.9))
        duration = float(self._rng.uniform(120, 600))  # 2-10 minutes

        event = make_event(
            event_type=event_type,
            affected_edges=affected,
            intensity=intensity,
            start_time=float(self._sim_step),
            duration=duration,
        )
        self._event_manager.add_event(event)

    def get_active_events(self) -> list[dict]:
        """Return active events for logging/display."""
        if not self._event_manager:
            return []
        return [
            {
                "type": e.event_type,
                "edges": e.affected_edges[:3],
                "intensity": e.intensity,
                "remaining": e.time_remaining(float(self._sim_step)),
            }
            for e in self._event_manager.get_active()
        ]

    # ── Cleanup ───────────────────────────────────────────────────────

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
