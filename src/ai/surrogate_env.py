"""
FlowMind AI - Surrogate environment (drop-in replacement for SumoTrafficEnv).

Uses the trained neural surrogate instead of SUMO for fast RL training.
Same Gymnasium interface so the training loop works without modification.

Speed: ~1000x faster than SUMO (~0.5s per episode vs ~600s).
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

from .traffic_env import OBS_DIM, ACT_DIM, ACT_OFF, MAX_INCOMING_EDGES
from .surrogate_model import SurrogateTrainer
from .transition_buffer import TransitionBuffer
from .reward import compute_tls_reward


class SurrogateEnv(gym.Env):
    """Fast surrogate environment using a trained neural network."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        surrogate: SurrogateTrainer,
        buffer: TransitionBuffer,
        tls_ids: list[str],
        candidate_tls_ids: set[str] = None,
        existing_tls_ids: set[str] = None,
        green_phases: dict[str, list[int]] = None,
        delta_time: int = 10,
        sim_length: int = 3600,
        seed: int = 42,
    ):
        super().__init__()

        self._surrogate = surrogate
        self._buffer = buffer
        self.tls_ids = list(tls_ids)
        self.candidate_tls_ids = candidate_tls_ids or set()
        self.existing_tls_ids = existing_tls_ids or set(tls_ids)
        self._green_phases = green_phases or {tid: [0] for tid in tls_ids}
        self.delta_time = delta_time
        self.sim_length = sim_length
        self.seed_val = seed
        self._max_steps = sim_length // delta_time

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(ACT_DIM)

        # Runtime state
        self._obs: dict[str, np.ndarray] = {}
        self._step_count = 0
        self._rng = np.random.RandomState(seed)
        self._prev_waiting: dict[str, float] = {}
        self._prev_throughput: dict[str, int] = {}

        # Action stats (same interface as SumoTrafficEnv)
        self.action_stats: dict[str, dict[int, int]] = {
            tid: {} for tid in self.tls_ids
        }

    @property
    def num_agents(self) -> int:
        return len(self.tls_ids)

    def get_valid_actions(self, tls_id: str) -> list[int]:
        phase_actions = list(range(len(self._green_phases.get(tls_id, [0]))))
        return phase_actions + [ACT_OFF]

    def record_actions(self, actions: dict[str, int]) -> None:
        for tid, act in actions.items():
            self.action_stats[tid][act] = self.action_stats[tid].get(act, 0) + 1

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None
              ) -> tuple[dict[str, np.ndarray], dict]:
        if seed is not None:
            self.seed_val = seed
            self._rng = np.random.RandomState(seed)

        self._step_count = 0
        self._prev_waiting.clear()
        self._prev_throughput.clear()

        # Sample initial observations from the real data buffer
        for tid in self.tls_ids:
            self._obs[tid] = self._buffer.sample_initial_obs(self._rng)
            # Extract initial wait/throughput from obs for reward computation
            obs = self._obs[tid]
            self._prev_waiting[tid] = float(
                np.sum(obs[MAX_INCOMING_EDGES:MAX_INCOMING_EDGES*2]) * 300)
            self._prev_throughput[tid] = int(
                np.sum(obs[MAX_INCOMING_EDGES*2:MAX_INCOMING_EDGES*3]) * 50)

        return dict(self._obs), {"step": 0, "sim_time": 0, "source": "surrogate"}

    def step(self, actions: dict[str, int]
             ) -> tuple[dict[str, np.ndarray], dict[str, float], bool, bool, dict]:
        """Predict next state and reward using surrogate model."""
        # Batch predict for all TLS at once (fast)
        n = len(self.tls_ids)
        obs_batch = np.stack([self._obs[tid] for tid in self.tls_ids])
        act_batch = np.array([actions.get(tid, 0) for tid in self.tls_ids],
                             dtype=np.int64)

        pred_obs, pred_rewards = self._surrogate.predict_batch(obs_batch, act_batch)

        # Unpack into dicts, override deterministic features analytically
        next_obs: dict[str, np.ndarray] = {}
        rewards: dict[str, float] = {}
        for i, tid in enumerate(self.tls_ids):
            o = pred_obs[i]
            action = actions.get(tid, 0)
            gp = self._green_phases.get(tid, [0])
            n_phases = max(len(gp), 1)

            # Feature 36: phase_ratio — deterministic from action
            if action == ACT_OFF:
                o[MAX_INCOMING_EDGES * 3] = 0.0
            else:
                phase_idx = gp[action] if action < len(gp) else gp[0]
                o[MAX_INCOMING_EDGES * 3] = phase_idx / max(n_phases, 1)

            # Feature 37: elapsed — resets on phase change, otherwise increments
            # NOTE: After per-TLS timing rewrite, the real env normalizes slot 37
            # by per-TLS max_green_steps (varies by intersection size) and slot 38
            # uses per-TLS min_green_steps. Surrogate uses global defaults as an
            # approximation — this is acceptable for fast pre-training only.
            prev_phase = obs_batch[i][MAX_INCOMING_EDGES * 3]
            if abs(o[MAX_INCOMING_EDGES * 3] - prev_phase) > 0.01:
                o[MAX_INCOMING_EDGES * 3 + 1] = 0.0  # phase changed, reset
            else:
                elapsed = obs_batch[i][MAX_INCOMING_EDGES * 3 + 1]
                o[MAX_INCOMING_EDGES * 3 + 1] = min(elapsed + self.delta_time / 90.0, 1.0)

            # Feature 38: min_green flag (approx: ~25 real seconds = 50 steps)
            elapsed_val = o[MAX_INCOMING_EDGES * 3 + 1] * 90.0  # denormalize
            o[MAX_INCOMING_EDGES * 3 + 2] = 1.0 if elapsed_val >= 50 else 0.0

            next_obs[tid] = o

            # Compute reward from predicted obs using REAL reward function
            # (not surrogate reward head — keeps metrics and reward consistent)
            queues = [float(o[j]) * 50.0
                      for j in range(MAX_INCOMING_EDGES)]
            new_wait = float(
                np.sum(o[MAX_INCOMING_EDGES:MAX_INCOMING_EDGES*2]) * 300)
            new_tp = int(
                np.sum(o[MAX_INCOMING_EDGES*2:MAX_INCOMING_EDGES*3]) * 50)
            rewards[tid] = compute_tls_reward(
                old_waiting=self._prev_waiting.get(tid, 0.0),
                new_waiting=new_wait,
                queue_lengths=queues,
                old_throughput=self._prev_throughput.get(tid, 0),
                new_throughput=new_tp,
            )
            self._prev_waiting[tid] = new_wait
            self._prev_throughput[tid] = new_tp

        self._obs = next_obs
        self._step_count += 1

        terminated = self._step_count >= self._max_steps
        truncated = False

        info = {
            "step": self._step_count,
            "sim_time": self._step_count * self.delta_time,
            "source": "surrogate",
        }
        return next_obs, rewards, terminated, truncated, info

    def get_metrics(self) -> dict:
        """Derive synthetic metrics from predicted observations."""
        total_wait = 0.0
        total_queue = 0.0
        count = 0
        for tid in self.tls_ids:
            obs = self._obs.get(tid)
            if obs is None:
                continue
            # obs[0:12] = queue (normalized by 50), obs[12:24] = wait (normalized by 300)
            total_queue += float(np.sum(obs[:MAX_INCOMING_EDGES]) * 50)
            total_wait += float(np.sum(obs[MAX_INCOMING_EDGES:MAX_INCOMING_EDGES*2]) * 300)
            count += MAX_INCOMING_EDGES
        n = max(count, 1)
        return {
            "avg_wait_time": round(total_wait / n, 2),
            "avg_queue_length": round(total_queue / n, 2),
            "total_vehicles": 0,  # not available from surrogate
            "collisions": 0,
            "sim_time": self._step_count * self.delta_time,
        }

    def get_tls_snapshot(self) -> dict:
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
            "n_add": n_add, "n_remove": n_remove,
            "n_candidates": len(self.candidate_tls_ids),
            "n_existing": len(self.existing_tls_ids),
        }

    def get_tls_details(self) -> list[dict]:
        return []  # Not available from surrogate

    # [Level 2 REMOVED] Event stubs kept for API compatibility
    def get_active_events(self) -> list[dict]:
        return []

    def get_recommendations(self) -> dict:
        return {"add": [], "remove": [], "keep_off": [], "timing": {}}

    def close(self) -> None:
        pass  # No SUMO process to close

    # [Level 2 REMOVED] For compatibility
    _active_event_log: list[str] = []
