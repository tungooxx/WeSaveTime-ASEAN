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
# [Level 2 REMOVED] from ..simulation.events import EventManager, make_event, EVENT_TYPES
from .reward import compute_tls_reward

# ── Constants ─────────────────────────────────────────────────────────
MAX_INCOMING_EDGES = 12     # pad/truncate incoming edges to this size
MAX_GREEN_PHASES = 7        # phases 0-6
NUM_DURATION_LEVELS = 3     # SHORT / MEDIUM / LONG green duration
ACT_OFF = -1                # OFF action DISABLED
MIN_GREEN_STEPS = 60        # fallback minimum green (30s real @ step_length=0.5)

# Observation layout per TLS (fixed size for parameter sharing):
#   [0..11]   queue per edge          (12)
#   [12..23]  wait per edge           (12)
#   [24..35]  lane density per edge   (12)
#   [36..47]  pressure per edge       (12)  — Level 2: PressLight state
#   [48]      current phase ratio     (1)
#   [49]      countdown remaining     (1)
OBS_DIM = MAX_INCOMING_EDGES * 4 + 2   # 50 (Level 2: with pressure)
OLD_OBS_DIM = MAX_INCOMING_EDGES * 2 + 2  # 26 (v1 layout)
L1_OBS_DIM = MAX_INCOMING_EDGES * 3 + 2  # 38 (Level 1 layout: no pressure, no free flag)
# Action = phase × duration: 7 phases × 3 durations = 21 actions
# action // 3 = phase index (0-6)
# action % 3  = duration (0=SHORT/ped_min, 1=MEDIUM, 2=LONG/max_green)
ACT_DIM = MAX_GREEN_PHASES * NUM_DURATION_LEVELS  # 21


def remap_obs_for_old_model(obs_new: np.ndarray, target_dim: int = OLD_OBS_DIM
                            ) -> np.ndarray:
    """Convert current observation to an older layout for backward compat.

    Supported target dims:
      26 (v1): [queue(12), wait(12), phase_ratio, elapsed]
      39 (L1): [queue(12), wait(12), density(12), phase_ratio, countdown, flag]
      51 (L2): [queue(12), wait(12), density(12), pressure(12), phase, countdown, flag]
    """
    if target_dim == OLD_OBS_DIM:  # 26
        obs = np.zeros(OLD_OBS_DIM, dtype=np.float32)
        obs[:12] = obs_new[:12]                             # queue
        obs[12:24] = obs_new[12:24]                         # wait
        obs[24] = obs_new[MAX_INCOMING_EDGES * 4]           # phase_ratio (slot 48)
        obs[25] = obs_new[MAX_INCOMING_EDGES * 4 + 1]       # countdown (slot 49)
        return obs
    elif target_dim == L1_OBS_DIM:  # 38 (Level 1 — drop pressure)
        obs = np.zeros(L1_OBS_DIM, dtype=np.float32)
        obs[:36] = obs_new[:36]                             # queue + wait + density
        obs[36] = obs_new[MAX_INCOMING_EDGES * 4]           # phase_ratio
        obs[37] = obs_new[MAX_INCOMING_EDGES * 4 + 1]       # countdown
        return obs
    elif target_dim == OBS_DIM:  # 51 (Level 2 — current)
        return obs_new.copy()
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
        delta_time: int = 30,
        sim_length: int = 1800,  # sim steps (= 900 real seconds at step_length=0.5)
        gui: bool = False,
        seed: int = 42,
        min_green_phases: int = 2,
        min_incoming: int = 2,
        yellow_time: int = 3,
        allred_time: int = 6,  # 3 real seconds at step_length=0.5
        # [Level 1] Pure timing optimization
        # [Level 2 REMOVED] random_events: bool = True,
        # [Level 2 REMOVED] max_concurrent_events: int = 2,
        # [Level 2 REMOVED] event_probability: float = 0.01,
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
        self.allred_time = allred_time
        # [Level 1] Pure timing optimization

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

        # ── Per-TLS timing from engineering formulas ───────────────
        self._yellow_steps: dict[str, int] = {}
        self._allred_steps: dict[str, int] = {}
        self._min_green_steps: dict[str, int] = {}
        self._max_green_steps: dict[str, int] = {}
        for tls in self._tls_list:
            g = tls.geometry
            if g:
                self._yellow_steps[tls.id] = g.yellow_steps
                self._allred_steps[tls.id] = g.allred_steps
                self._min_green_steps[tls.id] = g.min_green_steps
                self._max_green_steps[tls.id] = g.max_green_steps
            else:
                self._yellow_steps[tls.id] = yellow_time
                self._allred_steps[tls.id] = allred_time
                self._min_green_steps[tls.id] = 30
                self._max_green_steps[tls.id] = 90

        # Per-TLS duration levels: SHORT, MEDIUM, LONG (in sim steps)
        # SHORT = min_green (pedestrian minimum — just enough to cross)
        # MEDIUM = midpoint between min and max
        # LONG = max_green (full capacity serving)
        self._duration_levels: dict[str, list[int]] = {}
        for tid in self.tls_ids:
            mn = self._min_green_steps[tid]
            mx = self._max_green_steps[tid]
            mid = (mn + mx) // 2
            self._duration_levels[tid] = [mn, mid, mx]

        # Precompute avg transition for reward normalization
        transitions = [
            self._yellow_steps[t] + self._allred_steps[t] for t in self.tls_ids
        ]
        self._avg_transition = sum(transitions) / max(len(transitions), 1)

        # ── Neighbor graph (for coordination) ──────────────────────
        self._neighbor_graph = self._tls_meta.get_neighbor_graph(
            self.tls_ids, radius=500.0, max_neighbors=4,
        )

        # [TLS CANDIDATE COMMENTED OUT] ── Candidate TLS tracking ──
        # self.candidate_tls_ids: set[str] = set()
        # self.existing_tls_ids: set[str] = set()
        # self._load_candidate_info()

        # [TLS CANDIDATE COMMENTED OUT] Per-TLS action statistics
        # self.action_stats: dict[str, dict[int, int]] = {
        #     tid: {} for tid in self.tls_ids
        # }

        # ── Gym spaces (per-agent, shared definition) ────────────────
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(ACT_DIM)

        # [Level 2 REMOVED] ── Random event config ──────────────────
        # self.random_events = random_events
        # self.max_concurrent_events = max_concurrent_events
        # self.event_probability = event_probability
        # self._event_manager = None
        self._active_event_log: list[str] = []
        #
        # # Build list of non-internal edges for random event placement
        # self._all_edges: list[str] = [
        #     e.getID() for e in self._net.getEdges()
        #     if not e.getID().startswith(":") and e.getLaneNumber() >= 2
        # ]

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
        self._green_start_step: dict[str, int] = {}  # when actual green began
        self._countdown: dict[str, int] = {}  # Vietnamese countdown timer (sim steps remaining)
        self._committed_green: dict[str, int] = {}  # initial countdown when phase changed (for logging)

    # ── Properties ────────────────────────────────────────────────────

    @property
    def num_agents(self) -> int:
        return len(self.tls_ids)

    def get_valid_actions(self, tls_id: str) -> list[int]:
        """Valid action indices: phase × duration.

        action // 3 = phase index, action % 3 = duration level.
        E.g. for 4 green phases: actions [0..11] are valid (4 × 3).
        """
        n_phases = len(self._green_phases.get(tls_id, [0]))
        return list(range(n_phases * NUM_DURATION_LEVELS))

    def decode_action(self, action: int) -> tuple[int, int]:
        """Decode combined action into (phase_index, duration_level)."""
        return action // NUM_DURATION_LEVELS, action % NUM_DURATION_LEVELS

    # [TLS CANDIDATE COMMENTED OUT] ── Candidate TLS helpers ──────
    # All candidate TLS tracking, action stats, snapshots, details,
    # and recommendations have been commented out.
    # These features determine whether to ADD new or REMOVE existing
    # traffic lights. Level 1 only optimizes timing of current TLS.

    # def _load_candidate_info(self) -> None: ...
    # def record_actions(self, actions) -> None: ...
    # def get_tls_snapshot(self) -> dict: ...
    # def get_tls_details(self) -> list[dict]: ...
    # def get_recommendations(self) -> dict: ...

    def record_actions(self, actions: dict[str, int]) -> None:
        """Stub — TLS candidate tracking disabled."""
        pass

    def get_tls_snapshot(self) -> dict:
        """Stub — TLS candidate tracking disabled."""
        return {"n_add": 0, "n_remove": 0, "n_candidates": 0, "n_existing": 0}

    def get_tls_details(self) -> list[dict]:
        """Stub — TLS candidate tracking disabled."""
        return []

    def get_recommendations(self) -> dict:
        """Stub — TLS candidate tracking disabled."""
        return {"remove": [], "add": [], "keep_off": [], "timing": {}}

    # ── SUMO lifecycle ────────────────────────────────────────────────

    def _find_binary(self) -> str:
        name = "sumo-gui" if self.gui else "sumo"
        found = shutil.which(name)
        if found:
            return found
        sumo_home = os.environ.get("SUMO_HOME", "")
        # Auto-detect common Windows SUMO install paths
        search_paths = [sumo_home] if sumo_home else []
        search_paths += [
            r"C:\Program Files\Eclipse\Sumo",
            r"C:\Program Files (x86)\Eclipse\Sumo",
            r"C:\Sumo",
        ]
        for base in search_paths:
            if not base:
                continue
            p = os.path.join(base, "bin", name)
            if os.path.isfile(p) or os.path.isfile(p + ".exe"):
                os.environ["SUMO_HOME"] = base  # set for traci/sumolib
                return p
        raise FileNotFoundError(f"Cannot find '{name}'. Install SUMO or set SUMO_HOME.")

    def _sim_step_and_track(self) -> None:
        """Advance one sim step and accumulate throughput count."""
        self._conn.simulationStep()
        self._sim_step += 1
        try:
            self._total_throughput += self._conn.simulation.getArrivedNumber()
        except Exception:
            pass

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
            "--collision.action", "none",
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
        self._total_throughput = 0
        self._prev_waiting.clear()
        self._prev_throughput.clear()
        self._current_phases.clear()
        self._phase_start_step.clear()
        self._green_start_step.clear()
        self._countdown.clear()
        self._committed_green.clear()

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

        # [Level 2 REMOVED] ── Event manager ──────────────────────
        self._active_event_log = []
        # self._rng = np.random.RandomState(self.seed_val)
        # if self.random_events and self._rng.random() > 0.4:
        #     self._event_manager = EventManager(self._conn, self.net_file)
        #     self._maybe_inject_events()
        # else:
        #     self._event_manager = None

        # Warm-up so some vehicles enter the network
        for _ in range(self.delta_time):
            self._sim_step_and_track()

        # Force roundabout-entry TLS (trivial, not AI-controlled) to permanent
        # green so they don't randomly block traffic with their default programs
        self._set_trivial_tls_green()

        obs = self._get_observations()
        return obs, {"step": self._sim_step, "sim_time": self._sim_step}

    def step(
        self, actions: dict[str, int]
    ) -> tuple[dict[str, np.ndarray], dict[str, float], bool, bool, dict]:
        """Apply phase actions with Vietnamese countdown timer model.

        Each TLS commits to a full green duration when switching phases.
        Like real Vietnamese signals with countdown displays, the green
        time is set upfront and runs to completion before the AI decides
        again.  TLS with active countdowns are locked — the AI's action
        is ignored until the countdown expires.

        Advances SUMO by *delta_time* sim steps per call."""

        # ── 1. Decode actions: phase + duration (countdown lock) ────
        targets: dict[str, tuple[int, str]] = {}
        changed_tls: list[str] = []
        chosen_duration: dict[str, int] = {}  # tid -> duration in sim steps

        for tls_id, action_idx in actions.items():
            current_phase = self._current_phases.get(tls_id, 0)

            # If countdown active, force keep current phase (locked)
            if self._countdown.get(tls_id, 0) > 0:
                green_phases = self._green_phases.get(tls_id, [0])
                for ai, gp in enumerate(green_phases):
                    if gp == current_phase:
                        action_idx = ai * NUM_DURATION_LEVELS  # keep same, SHORT
                        break

            # Decode: phase index + duration level
            phase_idx, dur_level = self.decode_action(action_idx)
            green_phases = self._green_phases.get(tls_id, [0])
            target_phase = (
                green_phases[phase_idx]
                if phase_idx < len(green_phases)
                else green_phases[0]
            )

            # Get duration from per-TLS levels
            levels = self._duration_levels.get(tls_id, [30, 60, 90])
            dur_level = min(dur_level, len(levels) - 1)
            chosen_duration[tls_id] = levels[dur_level]

            tls_info = self._tls_map.get(tls_id)
            if tls_info and target_phase < len(tls_info.phases):
                targets[tls_id] = (target_phase, tls_info.phases[target_phase].state)
            else:
                targets[tls_id] = (0, "")

            # Detect phase change (only if countdown expired)
            if target_phase != current_phase and self._countdown.get(tls_id, 0) <= 0:
                changed_tls.append(tls_id)

        # ── 2. Apply transitions for changed TLS ────────────────────
        tls_transition: dict[str, dict] = {}
        for tls_id in changed_tls:
            self._set_yellow(tls_id, self._current_phases.get(tls_id, 0))
            target_phase = targets[tls_id][0]
            self._current_phases[tls_id] = target_phase
            yw = self._yellow_steps.get(tls_id, self.yellow_time)
            ar = self._allred_steps.get(tls_id, self.allred_time)
            tls_transition[tls_id] = {
                "yellow_rem": yw, "allred_rem": ar, "done": False,
            }
            # Set countdown = AI's chosen duration (SHORT/MEDIUM/LONG)
            dur = chosen_duration.get(tls_id, self._min_green_steps.get(tls_id, 30))
            self._countdown[tls_id] = dur
            self._committed_green[tls_id] = dur  # snapshot for logging

        # ── 3. Tick-by-tick transition (per-TLS yellow → allred → green)
        max_transition = max(
            (s["yellow_rem"] + s["allred_rem"] for s in tls_transition.values()),
            default=0,
        )

        for _tick in range(max_transition):
            for tid, st in tls_transition.items():
                if st["done"]:
                    continue
                if st["yellow_rem"] > 0:
                    st["yellow_rem"] -= 1
                    if st["yellow_rem"] == 0:
                        try:
                            cur = self._conn.trafficlight.getRedYellowGreenState(tid)
                            self._conn.trafficlight.setRedYellowGreenState(
                                tid, "r" * len(cur))
                        except traci.TraCIException:
                            pass
                elif st["allred_rem"] > 0:
                    st["allred_rem"] -= 1
                    if st["allred_rem"] == 0:
                        state_str = targets[tid][1]
                        if state_str:
                            try:
                                self._conn.trafficlight.setRedYellowGreenState(
                                    tid, state_str)
                            except traci.TraCIException:
                                pass
                        self._green_start_step[tid] = self._sim_step
                        self._phase_start_step[tid] = self._sim_step
                        st["done"] = True
            self._sim_step_and_track()

        # ── 4. Finalize green states ─────────────────────────────────
        for tid, st in tls_transition.items():
            if not st["done"]:
                state_str = targets[tid][1]
                if state_str:
                    try:
                        self._conn.trafficlight.setRedYellowGreenState(
                            tid, state_str)
                    except traci.TraCIException:
                        pass
                self._green_start_step[tid] = self._sim_step
                self._phase_start_step[tid] = self._sim_step

        for tls_id, (_, state_str) in targets.items():
            if tls_id not in tls_transition:
                if state_str:
                    try:
                        self._conn.trafficlight.setRedYellowGreenState(
                            tls_id, state_str)
                    except traci.TraCIException:
                        pass
                if tls_id not in self._green_start_step:
                    self._green_start_step[tls_id] = self._sim_step

        # ── 5. Advance green time + decrement countdowns ─────────────
        green_time = max(self.delta_time - max_transition, 1)
        for _ in range(green_time):
            self._sim_step_and_track()

        # Decrement countdowns by full delta_time (total sim steps this call)
        for tid in self.tls_ids:
            if self._countdown.get(tid, 0) > 0:
                self._countdown[tid] = max(0, self._countdown[tid] - self.delta_time)

        # ── 6. Observe & reward ───────────────────────────────────────
        obs = self._get_observations()
        rewards = self._compute_rewards(changed_tls_set=set(changed_tls))

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

    def _set_trivial_tls_green(self) -> None:
        """Force non-AI trivial TLS to permanent green — ONLY on straight roads.

        Only forces single-phase TLS with 1 incoming edge (pedestrian
        crossings on a single road). Skips:
        - TLS near roundabouts (they regulate entry flow)
        - TLS with 2+ incoming edges (real intersections)
        - Multi-phase TLS with conflicting directions
        """
        import math

        try:
            all_tls_ids = set(self._conn.trafficlight.getIDList())
        except traci.TraCIException:
            return

        # Find roundabout centers to exclude nearby TLS
        ra_centers: list[tuple[float, float]] = []
        for ra in self._net.getRoundabouts():
            xs, ys = [], []
            for nid in ra.getNodes():
                try:
                    n = self._net.getNode(nid)
                    x, y = n.getCoord()
                    xs.append(x); ys.append(y)
                except Exception:
                    pass
            if xs:
                ra_centers.append((sum(xs) / len(xs), sum(ys) / len(ys)))

        def _near_roundabout(tid: str, radius: float = 300.0) -> bool:
            if not ra_centers:
                return False
            try:
                node = self._net.getNode(tid)
                tx, ty = node.getCoord()
            except Exception:
                return False
            for cx, cy in ra_centers:
                if math.sqrt((tx - cx) ** 2 + (ty - cy) ** 2) < radius:
                    return True
            return False

        agent_tls = set(self.tls_ids)
        count = 0
        for tid in all_tls_ids - agent_tls:
            tls_info = self._tls_meta.get(tid)
            if tls_info is None:
                continue

            # Skip TLS near roundabouts — let them cycle normally
            if _near_roundabout(tid):
                continue

            # Only force green on single-edge, single-phase TLS (ped crossings)
            if tls_info.num_green_phases <= 1 and len(tls_info.incoming_edges) <= 1:
                try:
                    state = self._conn.trafficlight.getRedYellowGreenState(tid)
                    self._conn.trafficlight.setRedYellowGreenState(
                        tid, "G" * len(state))
                    count += 1
                except traci.TraCIException:
                    pass

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

    def get_neighbor_obs(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Compute mean neighbor observation for each TLS.

        Returns a dict of (OBS_DIM,) arrays. If a TLS has no neighbors,
        returns zeros (the agent must rely on its own observation only).
        """
        neighbor_obs: dict[str, np.ndarray] = {}
        for tid in self.tls_ids:
            nbrs = self._neighbor_graph.get(tid, [])
            if nbrs:
                nbr_vecs = [obs[n] for n in nbrs if n in obs]
                if nbr_vecs:
                    neighbor_obs[tid] = np.mean(nbr_vecs, axis=0).astype(np.float32)
                else:
                    neighbor_obs[tid] = np.zeros(OBS_DIM, dtype=np.float32)
            else:
                neighbor_obs[tid] = np.zeros(OBS_DIM, dtype=np.float32)
        return neighbor_obs

    def _obs_for(self, tls_id: str) -> np.ndarray:
        """Build a fixed-size (OBS_DIM,) observation vector for one TLS.

        Layout (Level 2 — 51 dim):
          [0..11]   queue per edge          (12)
          [12..23]  wait per edge           (12)
          [24..35]  lane density per edge   (12)
          [36..47]  pressure per edge       (12)  — PressLight state
          [48]      current phase ratio     (1)
          [49]      countdown remaining     (1)
          [50]      countdown expired flag  (1)
        """
        vec = np.zeros(OBS_DIM, dtype=np.float32)
        tls_info = self._tls_map.get(tls_id)
        if tls_info is None:
            return vec

        off_q = 0                        # queue slots [0..11]
        off_w = MAX_INCOMING_EDGES       # wait slots [12..23]
        off_d = MAX_INCOMING_EDGES * 2   # density slots [24..35]
        off_p = MAX_INCOMING_EDGES * 3   # pressure slots [36..47]

        # Collect outgoing vehicle counts for pressure computation
        outgoing_by_edge: dict[str, int] = {}
        try:
            node = self._net.getNode(tls_id)
            for e in node.getOutgoing():
                eid = e.getID()
                if not eid.startswith(":"):
                    try:
                        outgoing_by_edge[eid] = self._conn.edge.getLastStepVehicleNumber(eid)
                    except traci.TraCIException:
                        pass
        except Exception:
            pass
        total_outgoing = sum(outgoing_by_edge.values())
        n_out_edges = max(len(outgoing_by_edge), 1)
        avg_outgoing = total_outgoing / n_out_edges

        for i, edge_id in enumerate(tls_info.incoming_edges[:MAX_INCOMING_EDGES]):
            try:
                q = self._conn.edge.getLastStepHaltingNumber(edge_id)
                w = self._conn.edge.getWaitingTime(edge_id)
                veh_count = self._conn.edge.getLastStepVehicleNumber(edge_id)
                capacity = self._edge_capacity.get(edge_id, 10.0)
                density = veh_count / capacity
            except traci.TraCIException:
                q, w, veh_count, density = 0, 0.0, 0, 0.0
            vec[off_q + i] = min(q / 50.0, 1.0)
            vec[off_w + i] = min(w / 300.0, 1.0)
            vec[off_d + i] = min(density, 1.0)
            # Pressure: (avg outgoing - incoming) per edge, normalized
            # Positive = flowing well, negative = congestion building
            pressure = (avg_outgoing - veh_count) / max(capacity, 1.0)
            vec[off_p + i] = max(-1.0, min(pressure, 1.0))

        # Current phase ratio
        cur_phase = self._current_phases.get(tls_id, 0)
        num_phases = max(tls_info.num_phases, 1)
        vec[MAX_INCOMING_EDGES * 4] = cur_phase / num_phases

        # Countdown remaining (Vietnamese timer display)
        # 1.0 = full countdown, 0.0 = expired (free to switch)
        countdown = self._countdown.get(tls_id, 0)
        max_g = self._max_green_steps.get(tls_id, 120)
        vec[MAX_INCOMING_EDGES * 4 + 1] = min(countdown / max_g, 1.0)

        # [Level 2 REMOVED] Number of active events (normalized)
        # vec[MAX_INCOMING_EDGES * 4 + 3] = min(n_events / 5.0, 1.0)

        return vec

    # ── Rewards ───────────────────────────────────────────────────────

    def _compute_rewards(self, changed_tls_set: set[str] | None = None) -> dict[str, float]:
        rewards: dict[str, float] = {}

        # [Level 1] Pure timing optimization

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

            # Per-TLS transition cost for switch penalty scaling
            yw = self._yellow_steps.get(tls_id, 6)
            ar = self._allred_steps.get(tls_id, 4)
            tc = (yw + ar) / max(self._avg_transition, 1)

            rewards[tls_id] = compute_tls_reward(
                old_waiting=self._prev_waiting.get(tls_id, 0.0),
                new_waiting=total_wait,
                queue_lengths=queues,
                old_throughput=self._prev_throughput.get(tls_id, 0),
                new_throughput=incoming_veh,
                pressure=pressure,
                phase_changed=(tls_id in changed_tls_set) if changed_tls_set else False,
                transition_cost=tc,
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
        try:
            vehicles = self._conn.vehicle.getIDCount()
        except traci.TraCIException:
            pass
        return {
            "avg_wait_time": round(total_wait / n, 2),
            "avg_queue_length": round(total_queue / n, 2),
            "total_vehicles": vehicles,
            "throughput": self._total_throughput,
            "sim_time": self._sim_step,
        }

    # [Level 2 REMOVED] Random event injection and active events — deleted

    def get_active_events(self) -> list[dict]:
        """Return active events for logging/display. (Level 2 removed — always empty)"""
        return []

    # ── Cleanup ───────────────────────────────────────────────────────

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
