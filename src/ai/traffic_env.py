"""
FlowMind AI - Gymnasium environment for multi-agent traffic signal RL.

Each TLS (traffic light system) in the SUMO network is an independent
agent that shares the same neural network (parameter sharing).
The environment provides per-TLS observations and per-TLS rewards.

Observation per TLS (fixed 39-dim vector):
    [queue(12), wait(12), density(12), phase_ratio, elapsed, min_green]

Action per TLS (duration-only):
    Phases auto-cycle in fixed SUMO order — the AI only picks how long
    each phase lasts.  Like real Vietnamese signals with countdown displays,
    the phase order is predictable and the duration is committed upfront.

    discrete mode: index 0..N-1 → N duration levels from min to max green
    continuous mode: float 0.0-1.0 → mapped to [min_green, max_green]
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
DEFAULT_DURATION_LEVELS = 7  # default number of discrete duration levels
ACT_OFF = -1                 # OFF action DISABLED
MIN_GREEN_STEPS = 60         # fallback minimum green (30s real @ step_length=0.5)

# Action mode: "discrete" or "continuous"
# discrete: N levels evenly spread between min and max green (ACT_DIM = N)
# continuous: single float 0.0-1.0 mapped to min-max range (ACT_DIM = 1)
ACTION_MODE = "discrete"     # default, overridden per-env

# Observation layout per TLS (fixed size for parameter sharing):
#   [0..11]   queue per edge          (12)
#   [12..23]  wait per edge           (12)
#   [24..35]  lane density per edge   (12)
#   [36]      current phase ratio     (1)
#   [37]      elapsed time ratio      (1)
#   [38]      min_green satisfied     (1)
OBS_DIM = MAX_INCOMING_EDGES * 3 + 3   # 39 (Level 1: no pressure, no events)
OLD_OBS_DIM = MAX_INCOMING_EDGES * 2 + 2  # 26 (v1 layout)
# Duration-only action space (phases auto-cycle):
# discrete mode: ACT_DIM = num_duration_levels (default 7)
# continuous mode: ACT_DIM = 1 (single float)
ACT_DIM = DEFAULT_DURATION_LEVELS  # overridden by env __init__


V2_OBS_DIM = MAX_INCOMING_EDGES * 3 + 3  # 39 (same as OBS_DIM now)


def remap_obs_for_old_model(obs_new: np.ndarray, target_dim: int = OLD_OBS_DIM
                            ) -> np.ndarray:
    """Convert current observation to an older layout for backward compat.

    Supported target dims:
      26 (v1): [queue(12), wait(12), phase_ratio, elapsed]
      39 (current): [queue(12), wait(12), density(12), phase_ratio, elapsed, min_green]
    """
    if target_dim == OLD_OBS_DIM:  # 26
        obs = np.zeros(OLD_OBS_DIM, dtype=np.float32)
        obs[:12] = obs_new[:12]                             # queue
        obs[12:24] = obs_new[12:24]                         # wait
        obs[24] = obs_new[MAX_INCOMING_EDGES * 3]           # phase_ratio (slot 36)
        obs[25] = obs_new[MAX_INCOMING_EDGES * 3 + 1]       # elapsed (slot 37)
        return obs
    elif target_dim == V2_OBS_DIM:  # 39 (same as current OBS_DIM)
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
        action_mode: str = "discrete",  # "discrete" or "continuous"
        num_duration_levels: int = DEFAULT_DURATION_LEVELS,  # for discrete mode
    ) -> None:
        super().__init__()
        self.action_mode = action_mode
        self.num_duration_levels = num_duration_levels

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

        # ── TLS discovery (ALL TLS are AI-controlled) ──────────────
        self._tls_meta = TLSMetadata(self.net_file)
        self._tls_list = self._tls_meta.all_tls
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

            # For single-phase TLS (ped crossings), allow max_green up to
            # 150% of their SUMO default — the engineering cap is too low
            if tls.num_green_phases <= 1 and tls.phases:
                default_green_steps = max(
                    int(p.duration / 0.5)
                    for p in tls.phases
                    if any(c in ('G', 'g') for c in p.state)
                ) if tls.phases else 60
                self._max_green_steps[tls.id] = max(
                    self._max_green_steps[tls.id],
                    int(default_green_steps * 1.5)
                )

        # ── Per-TLS duration levels (in sim steps) ────────────────────
        # N levels evenly spread from min_green to max_green for each TLS.
        self._duration_levels: dict[str, list[int]] = {}
        n = self.num_duration_levels
        for tid in self.tls_ids:
            mn = self._min_green_steps[tid]
            mx = self._max_green_steps[tid]
            if n <= 1:
                self._duration_levels[tid] = [mn]
            elif n == 2:
                self._duration_levels[tid] = [mn, mx]
            else:
                levels = []
                for i in range(n):
                    t = i / (n - 1)
                    val = int(mn + t * (mx - mn))
                    levels.append(val)
                self._duration_levels[tid] = levels
        self._flex_min_steps: int = min(self._min_green_steps.values())
        self._flex_max_steps: int = max(self._max_green_steps.values())

        # ── Reward importance weights ─────────────────────────────────
        # Weight by complexity: multi-phase intersections matter more
        # than single-phase pedestrian crossings.
        raw_weights = {}
        for tls in self._tls_list:
            gp = tls.num_green_phases
            ie = max(len(tls.incoming_edges), 1)
            if gp >= 2:
                w = gp * ie
            else:
                total_lanes = sum(
                    self._net.getEdge(eid).getLaneNumber()
                    for eid in tls.incoming_edges
                    if not eid.startswith(":")
                ) if tls.incoming_edges else 1
                w = total_lanes * 0.3
            raw_weights[tls.id] = float(w)
        mean_w = sum(raw_weights.values()) / max(len(raw_weights), 1)
        self._reward_weight: dict[str, float] = {
            tid: w / max(mean_w, 0.01) for tid, w in raw_weights.items()
        }

        # Precompute avg transition for reward normalization
        transitions = [
            self._yellow_steps[t] + self._allred_steps[t] for t in self.tls_ids
        ]
        self._avg_transition = sum(transitions) / max(len(transitions), 1)

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
        if self.action_mode == "continuous":
            self.act_dim = 1
            self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,))
        else:
            self.act_dim = self.num_duration_levels
            self.action_space = gym.spaces.Discrete(self.act_dim)

        # [Level 2 REMOVED] ── Random event config ──────────────────
        # self.random_events = random_events
        # self.max_concurrent_events = max_concurrent_events
        # self.event_probability = event_probability
        # self._event_manager = None
        self._active_event_log: list[str] = []
        self.baseline_active: bool = True  # set False during curriculum Phase 1-2
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
        self._cycle_index: dict[str, int] = {}  # index into green_phases for auto-cycling

    # ── Properties ────────────────────────────────────────────────────

    @property
    def num_agents(self) -> int:
        return len(self.tls_ids)

    def get_valid_actions(self, tls_id: str) -> list[int]:
        """Valid action indices for discrete mode.

        Returns a single-element list [0] when the TLS is locked (countdown > 0)
        so the trainer doesn't sample actions that step() will ignore.
        """
        if self.action_mode == "continuous":
            return [0]  # continuous has 1 "action" (the float)
        if self._countdown.get(tls_id, 0) > 0:
            return [0]  # locked — only no-op/min-duration valid
        return list(range(self.num_duration_levels))

    def decode_duration_steps(self, tls_id: str, action) -> int:
        """Convert action to duration in sim steps.

        Discrete mode: action is an int index into duration_levels.
        Continuous mode: action is a float 0.0-1.0 mapped to min-max range.
        """
        levels = self._duration_levels.get(tls_id, [60])
        mn = self._flex_min_steps
        mx = self._flex_max_steps

        if self.action_mode == "continuous":
            t = float(action) if not hasattr(action, '__len__') else float(action[0])
            t = max(0.0, min(1.0, t))
            # Use per-TLS bounds so ped crossings don't inflate the range
            per_mn = self._min_green_steps.get(tls_id, mn)
            per_mx = self._max_green_steps.get(tls_id, mx)
            return int(per_mn + t * (per_mx - per_mn))
        else:
            idx = min(int(action), len(levels) - 1)
            return levels[idx]

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
            cmd = [binary, "-c", self.sumo_cfg,
                   "-r", self.route_file]  # override route from cfg
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
        self._cycle_index.clear()

        # Initialise per-TLS tracking and validate green phases at runtime
        for tls_id in self.tls_ids:
            self._prev_waiting[tls_id] = 0.0
            self._prev_throughput[tls_id] = 0
            try:
                self._current_phases[tls_id] = self._conn.trafficlight.getPhase(tls_id)
            except traci.TraCIException:
                self._current_phases[tls_id] = 0
            self._phase_start_step[tls_id] = 0
            self._countdown[tls_id] = 0  # all TLS free at start
            self._cycle_index[tls_id] = 0  # start at first green phase

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

        # Resync internal phase state to SUMO's live phase after warm-up
        for tls_id in self.tls_ids:
            try:
                live_phase = self._conn.trafficlight.getPhase(tls_id)
                self._current_phases[tls_id] = live_phase
                # Map live phase to cycle index
                gp = self._green_phases.get(tls_id, [0])
                if live_phase in gp:
                    self._cycle_index[tls_id] = gp.index(live_phase)
                # Reset phase timers so elapsed-green in _obs_for() starts from 0
                self._phase_start_step[tls_id] = self._sim_step
                self._green_start_step[tls_id] = self._sim_step
            except traci.TraCIException:
                pass

        obs = self._get_observations()
        return obs, {"step": self._sim_step, "sim_time": self._sim_step}

    def step(
        self, actions: dict  # dict[str, int] for discrete, dict[str, float] for continuous
    ) -> tuple[dict[str, np.ndarray], dict[str, float], bool, bool, dict]:
        """Apply duration actions with Vietnamese countdown timer model.

        Phases auto-cycle in fixed SUMO order -- the AI only picks how long
        each phase lasts (SHORT / MEDIUM / LONG).  Like real Vietnamese
        signals with countdown displays, the phase order is predictable
        and the duration is committed upfront.

        When a countdown expires, the next phase in cycle starts automatically.
        The AI then picks a duration for that new phase.

        Advances SUMO by *delta_time* sim steps per call."""

        # ── 1. Auto-cycle phases + apply AI duration ──────────────────
        targets: dict[str, tuple[int, str]] = {}
        changed_tls: list[str] = []
        chosen_duration: dict[str, int] = {}

        for tls_id, action_idx in actions.items():
            current_phase = self._current_phases.get(tls_id, 0)

            # If countdown active, TLS is locked -- skip
            if self._countdown.get(tls_id, 0) > 0:
                continue

            # Countdown expired -> auto-advance to next green phase in cycle
            green_phases = self._green_phases.get(tls_id, [0])
            ci = self._cycle_index.get(tls_id, 0)
            next_ci = (ci + 1) % len(green_phases)
            target_phase = green_phases[next_ci]
            self._cycle_index[tls_id] = next_ci

            # Decode AI action -> duration in sim steps
            chosen_duration[tls_id] = self.decode_duration_steps(tls_id, action_idx)

            # Set target phase state
            tls_info = self._tls_map.get(tls_id)
            if tls_info and target_phase < len(tls_info.phases):
                targets[tls_id] = (target_phase, tls_info.phases[target_phase].state)
            else:
                targets[tls_id] = (0, "")

            # Every countdown expiry triggers a phase change
            if target_phase != current_phase:
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
            # Set countdown = AI's chosen duration
            dur = chosen_duration.get(tls_id, self._min_green_steps.get(tls_id, 30))
            self._countdown[tls_id] = dur
            self._committed_green[tls_id] = dur

        # ── 3. Tick-by-tick transition (per-TLS yellow -> allred -> green)
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
        """Force all non-AI trivial TLS to permanent green.

        Targets two categories:
        1. Single-phase TLS (pedestrian crossings, median breaks) — only
           1 green phase, cycling serves no purpose. These are common on
           arterials like Nguyễn Văn Linh (25+ trivial TLS).
        2. Uniform-phase TLS (roundabout entries) — all phases have the
           same pattern (GGG/yyy/rrr), no conflicting directions.

        Multi-phase intersections with 2+ distinct green phases and
        conflicting directions are left on default programs.
        """
        try:
            all_tls_ids = set(self._conn.trafficlight.getIDList())
        except traci.TraCIException:
            return

        agent_tls = set(self.tls_ids)
        count = 0
        for tid in all_tls_ids - agent_tls:
            tls_info = self._tls_meta.get(tid)
            if tls_info is None:
                continue

            # Category 1: Single green phase — always force green
            if tls_info.num_green_phases <= 1:
                try:
                    state = self._conn.trafficlight.getRedYellowGreenState(tid)
                    self._conn.trafficlight.setRedYellowGreenState(
                        tid, "G" * len(state))
                    count += 1
                except traci.TraCIException:
                    pass
                continue

            # Category 2: Multi-phase but all uniform (roundabout entries)
            is_uniform = True
            for p in tls_info.phases:
                has_green = any(c in ('G', 'g') for c in p.state)
                has_red = any(c in ('r', 'R') for c in p.state)
                if has_green and has_red:
                    is_uniform = False
                    break
            if is_uniform:
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

    def _obs_for(self, tls_id: str) -> np.ndarray:
        """Build a fixed-size (OBS_DIM,) observation vector for one TLS.

        Layout: [queue(12), wait(12), density(12),
                 phase_ratio, elapsed, min_green]
        """
        vec = np.zeros(OBS_DIM, dtype=np.float32)
        tls_info = self._tls_map.get(tls_id)
        if tls_info is None:
            return vec

        off_q = 0                        # queue slots
        off_w = MAX_INCOMING_EDGES       # wait slots
        off_d = MAX_INCOMING_EDGES * 2   # density slots
        # [Level 2 REMOVED] off_e = MAX_INCOMING_EDGES * 3   # event blocked slots

        # [Level 2 REMOVED] Collect edges affected by active events
        # affected_edges: set[str] = set()
        # n_events = 0
        # if self._event_manager:
        #     for evt in self._event_manager.get_active():
        #         affected_edges.update(evt.affected_edges)
        #         n_events += 1

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
            # [Level 2 REMOVED] Event flag
            # vec[off_e + i] = 1.0 if edge_id in affected_edges else 0.0

        # Current phase ratio
        cur_phase = self._current_phases.get(tls_id, 0)
        num_phases = max(tls_info.num_phases, 1)
        vec[MAX_INCOMING_EDGES * 3] = cur_phase / num_phases

        # Elapsed actual green time (per-TLS normalization)
        green_start = self._green_start_step.get(
            tls_id, self._phase_start_step.get(tls_id, 0))
        actual_green_elapsed = self._sim_step - green_start
        max_g = self._max_green_steps.get(tls_id, 120)
        vec[MAX_INCOMING_EDGES * 3 + 1] = min(actual_green_elapsed / max_g, 1.0)

        # Slot [38]: unused (was min-green flag, removed — no fixed constraints)
        vec[MAX_INCOMING_EDGES * 3 + 2] = 0.0

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

            raw_reward = compute_tls_reward(
                old_waiting=self._prev_waiting.get(tls_id, 0.0),
                new_waiting=total_wait,
                queue_lengths=queues,
                old_throughput=self._prev_throughput.get(tls_id, 0),
                new_throughput=incoming_veh,
                pressure=pressure,
                phase_changed=(tls_id in changed_tls_set) if changed_tls_set else False,
                transition_cost=tc,
            )

            # Apply importance weighting
            rewards[tls_id] = raw_reward * self._reward_weight.get(tls_id, 1.0)

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
