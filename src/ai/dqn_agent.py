"""
FlowMind AI - Double DQN agent with shared parameters.

A single Q-network is shared across all TLS agents (parameter sharing).
Each TLS feeds its own observation through the shared network and receives
Q-values for its action space.  Invalid actions are masked to -inf before
argmax.

Features:
  - Double DQN (online net selects action, target net evaluates)
  - Smooth L1 (Huber) loss
  - Exponential epsilon decay
  - Periodic hard target-network sync
  - Gradient clipping (max norm 10)
"""

from __future__ import annotations

import random
from collections import deque, namedtuple
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


# ──────────────────────────────────────────────────────────────────────
# Q-Network
# ──────────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """Two-hidden-layer feedforward Q-network."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────
# Replay buffer
# ──────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Fixed-capacity FIFO experience replay buffer."""

    def __init__(self, capacity: int = 100_000) -> None:
        self._buf: deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append(
            Transition(
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self._buf, batch_size)

    def __len__(self) -> int:
        return len(self._buf)


# ──────────────────────────────────────────────────────────────────────
# DQN Agent
# ──────────────────────────────────────────────────────────────────────

class TrafficDQNAgent:
    """Shared-parameter Double DQN agent for multi-agent traffic control."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: int = 256,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 500_000,
        target_update_freq: int = 1000,
        batch_size: int = 64,
        buffer_capacity: int = 200_000,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Exploration schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self._step_count: int = 0

        # Networks
        self.q_net = QNetwork(obs_dim, act_dim, hidden).to(self.device)
        self.target_net = QNetwork(obs_dim, act_dim, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

    # ── epsilon ───────────────────────────────────────────────────────

    @property
    def epsilon(self) -> float:
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-self._step_count / self.epsilon_decay)

    # ── action selection ──────────────────────────────────────────────

    def select_action(
        self,
        obs: np.ndarray,
        valid_actions: Optional[list[int]] = None,
        greedy: bool = False,
    ) -> int:
        """Epsilon-greedy with action masking."""
        if valid_actions is None:
            valid_actions = list(range(self.act_dim))

        if not greedy and random.random() < self.epsilon:
            return random.choice(valid_actions)

        with torch.no_grad():
            t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            q = self.q_net(t).cpu().numpy()[0]

        masked = np.full(self.act_dim, -np.inf)
        for a in valid_actions:
            masked[a] = q[a]
        return int(np.argmax(masked))

    # ── experience storage ────────────────────────────────────────────

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.push(state, action, reward, next_state, done)
        self._step_count += 1

    # ── learning step ─────────────────────────────────────────────────

    def update(self) -> Optional[float]:
        """One gradient step.  Returns loss, or None if buffer too small."""
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)

        states = torch.FloatTensor(np.array([t.state for t in batch])).to(self.device)
        actions = torch.LongTensor([t.action for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).to(self.device)
        next_s = torch.FloatTensor(np.array([t.next_state for t in batch])).to(self.device)
        dones = torch.BoolTensor([t.done for t in batch]).to(self.device)

        # Current Q(s, a)
        q_vals = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            best_next_actions = self.q_net(next_s).argmax(dim=1)
            next_q = self.target_net(next_s).gather(
                1, best_next_actions.unsqueeze(1)
            ).squeeze(1)
            next_q[dones] = 0.0
            target = rewards + self.gamma * next_q

        loss = nn.functional.smooth_l1_loss(q_vals, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # Hard target-network sync
        if self._step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    # ── persistence ───────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step_count": self._step_count,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._step_count = ckpt.get("step_count", 0)
