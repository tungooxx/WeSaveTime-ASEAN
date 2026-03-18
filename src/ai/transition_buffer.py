"""
FlowMind AI - Transition buffer for surrogate model training.

Collects (obs, action, next_obs, reward, done) transitions during real
SUMO episodes and persists them as compressed numpy arrays.

Usage:
    buf = TransitionBuffer(capacity=3_000_000)
    # During training:
    buf.add(obs, action, next_obs, reward, done)
    # After training:
    buf.save("checkpoints/transitions.npz")
    # Load later:
    buf = TransitionBuffer.load("checkpoints/transitions.npz")
"""

from __future__ import annotations

import numpy as np

from .traffic_env import OBS_DIM


class TransitionBuffer:
    """Ring buffer storing (obs, action, next_obs, reward, done) tuples
    as pre-allocated numpy arrays for fast I/O and surrogate training."""

    def __init__(self, capacity: int = 3_000_000, obs_dim: int = OBS_DIM):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self._pos = 0
        self._size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int8)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def __len__(self) -> int:
        return self._size

    def add(self, obs: np.ndarray, action: int, next_obs: np.ndarray,
            reward: float, done: bool) -> None:
        """Add a single transition."""
        self.obs[self._pos] = obs
        self.actions[self._pos] = action
        self.next_obs[self._pos] = next_obs
        self.rewards[self._pos] = reward
        self.dones[self._pos] = done
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_batch(self, obs_dict: dict[str, np.ndarray],
                  actions: dict[str, int],
                  next_obs_dict: dict[str, np.ndarray],
                  rewards: dict[str, float],
                  done: bool) -> None:
        """Add transitions for all TLS agents from one env step."""
        for tid in obs_dict:
            self.add(
                obs_dict[tid],
                actions.get(tid, 0),
                next_obs_dict[tid],
                rewards.get(tid, 0.0),
                done,
            )

    def get_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                np.ndarray, np.ndarray]:
        """Return all valid data as contiguous arrays."""
        n = self._size
        return (
            self.obs[:n].copy(),
            self.actions[:n].copy(),
            self.next_obs[:n].copy(),
            self.rewards[:n].copy(),
            self.dones[:n].copy(),
        )

    def sample_initial_obs(self, rng: np.random.RandomState = None
                           ) -> np.ndarray:
        """Sample a random observation (for surrogate env reset)."""
        if self._size == 0:
            return np.zeros(self.obs_dim, dtype=np.float32)
        if rng is None:
            rng = np.random.RandomState()
        idx = rng.randint(0, self._size)
        return self.obs[idx].copy()

    def save(self, path: str) -> None:
        """Save to compressed .npz file."""
        n = self._size
        np.savez_compressed(
            path,
            obs=self.obs[:n],
            actions=self.actions[:n],
            next_obs=self.next_obs[:n],
            rewards=self.rewards[:n],
            dones=self.dones[:n],
            obs_dim=np.array([self.obs_dim]),
        )

    @classmethod
    def load(cls, path: str) -> "TransitionBuffer":
        """Load from .npz file."""
        data = np.load(path)
        obs_dim = int(data["obs_dim"][0])
        n = len(data["obs"])
        buf = cls(capacity=max(n * 2, 1_000_000), obs_dim=obs_dim)
        buf.obs[:n] = data["obs"]
        buf.actions[:n] = data["actions"]
        buf.next_obs[:n] = data["next_obs"]
        buf.rewards[:n] = data["rewards"]
        buf.dones[:n] = data["dones"]
        buf._pos = n % buf.capacity
        buf._size = n
        return buf
