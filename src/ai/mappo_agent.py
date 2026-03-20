"""
FlowMind AI - MAPPO (Multi-Agent PPO) agent with shared parameters.

A single ActorCritic network is shared across all TLS agents (parameter sharing).
The critic receives a global state (mean of all agent observations) for centralized
value estimation, while the actor only sees local observations (CTDE paradigm).

Features:
  - Shared actor-critic network for all TLS agents
  - Centralized critic with global state aggregation
  - PPO clipped surrogate objective
  - Generalized Advantage Estimation (GAE)
  - Action masking for valid traffic phases
  - Entropy bonus for exploration
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ──────────────────────────────────────────────────────────────────────
# Actor-Critic Network
# ──────────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """Shared actor-critic with centralized critic."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256) -> None:
        super().__init__()
        # Actor: local obs -> action logits
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
        # Critic: local obs + global obs -> value
        self.critic = nn.Sequential(
            nn.Linear(obs_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward_actor(self, obs: torch.Tensor,
                      valid_mask: Optional[torch.Tensor] = None) -> Categorical:
        logits = self.actor(obs)
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, -1e8)
        return Categorical(logits=logits)

    def forward_critic(self, obs: torch.Tensor,
                       global_obs: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, global_obs], dim=-1)
        return self.critic(x).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────
# Rollout Buffer
# ──────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """On-policy buffer storing one episode of multi-agent transitions."""

    def __init__(self) -> None:
        self.obs: list[np.ndarray] = []
        self.global_obs: list[np.ndarray] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []
        self.valid_masks: list[np.ndarray] = []

    def add(self, obs: np.ndarray, global_obs: np.ndarray, action: int,
            log_prob: float, reward: float, value: float, done: bool,
            valid_mask: np.ndarray) -> None:
        self.obs.append(obs)
        self.global_obs.append(global_obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.valid_masks.append(valid_mask)

    def compute_returns(self, last_values: list[float], gamma: float = 0.99,
                        gae_lambda: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and discounted returns.

        last_values: bootstrapped V(s_T+1) for each agent at episode end.
        """
        n = len(self.rewards)
        if n == 0:
            return np.array([]), np.array([])

        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        # We need to handle multi-agent: transitions are interleaved
        # (agent0_step0, agent1_step0, ..., agent0_step1, agent1_step1, ...)
        # For GAE, we compute per-agent then interleave back.
        # But since all agents share the same done signal and we store
        # transitions sequentially per step, we can use a simpler approach:
        # group by agent within each step.

        # Since rewards/values are stored flat (all agents per step consecutively),
        # we compute GAE in the flat order. At episode boundaries (done=True),
        # advantage resets.

        # For simplicity, use flat GAE (treating it as single-agent with
        # interleaved samples). This works because all agents share the same
        # episode boundary (done flag).
        last_gae = 0.0
        # Assign last_values cyclically
        n_agents = len(last_values)
        for i in reversed(range(n)):
            if i == n - 1:
                next_val = last_values[i % n_agents] if last_values else 0.0
                next_done = False
            else:
                next_val = self.values[i + 1]
                next_done = self.dones[i + 1]

            if self.dones[i]:
                last_gae = 0.0
                next_val = 0.0

            delta = self.rewards[i] + gamma * next_val * (1 - float(self.dones[i])) - self.values[i]
            last_gae = delta + gamma * gae_lambda * (1 - float(next_done)) * last_gae
            advantages[i] = last_gae
            returns[i] = advantages[i] + self.values[i]

        return advantages, returns

    def get_batches(self, batch_size: int, advantages: np.ndarray,
                    returns: np.ndarray):
        """Yield shuffled mini-batches."""
        n = len(self.obs)
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            yield {
                "obs": np.array([self.obs[i] for i in idx], dtype=np.float32),
                "global_obs": np.array([self.global_obs[i] for i in idx], dtype=np.float32),
                "actions": np.array([self.actions[i] for i in idx], dtype=np.int64),
                "old_log_probs": np.array([self.log_probs[i] for i in idx], dtype=np.float32),
                "advantages": advantages[idx],
                "returns": returns[idx],
                "valid_masks": np.array([self.valid_masks[i] for i in idx], dtype=np.bool_),
            }

    def clear(self) -> None:
        self.obs.clear()
        self.global_obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.valid_masks.clear()

    def __len__(self) -> int:
        return len(self.obs)


# ──────────────────────────────────────────────────────────────────────
# MAPPO Agent
# ──────────────────────────────────────────────────────────────────────

class MAPPOAgent:
    """Multi-Agent PPO with shared parameters and centralized critic."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        mini_batch_size: int = 256,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        self.network = ActorCritic(obs_dim, act_dim, hidden).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    def select_action(
        self,
        obs: np.ndarray,
        global_obs: np.ndarray,
        valid_actions: Optional[list[int]] = None,
        greedy: bool = False,
    ) -> tuple[int, float, float]:
        """Select action, return (action, log_prob, value)."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            global_t = torch.FloatTensor(global_obs).unsqueeze(0).to(self.device)

            # Build valid mask
            mask = None
            if valid_actions is not None:
                mask = torch.zeros(1, self.act_dim, dtype=torch.bool, device=self.device)
                for a in valid_actions:
                    mask[0, a] = True

            dist = self.network.forward_actor(obs_t, mask)
            value = self.network.forward_critic(obs_t, global_t)

            if greedy:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

        return int(action.item()), float(log_prob.item()), float(value.item())

    def get_valid_mask(self, valid_actions: list[int]) -> np.ndarray:
        """Create boolean mask array for valid actions."""
        mask = np.zeros(self.act_dim, dtype=np.bool_)
        for a in valid_actions:
            mask[a] = True
        return mask

    def update(self) -> dict:
        """Run PPO update over collected rollout buffer. Returns loss stats."""
        if len(self.buffer) == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}

        # Bootstrap last values (0 since episode ended)
        # For multi-agent, all agents end at the same time
        last_values = [0.0]  # episode terminated

        advantages, returns = self.buffer.compute_returns(
            last_values, self.gamma, self.gae_lambda)

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            for batch in self.buffer.get_batches(self.mini_batch_size, advantages, returns):
                obs_t = torch.FloatTensor(batch["obs"]).to(self.device)
                global_t = torch.FloatTensor(batch["global_obs"]).to(self.device)
                actions_t = torch.LongTensor(batch["actions"]).to(self.device)
                old_lp_t = torch.FloatTensor(batch["old_log_probs"]).to(self.device)
                adv_t = torch.FloatTensor(batch["advantages"]).to(self.device)
                ret_t = torch.FloatTensor(batch["returns"]).to(self.device)
                mask_t = torch.BoolTensor(batch["valid_masks"]).to(self.device)

                # Recompute
                dist = self.network.forward_actor(obs_t, mask_t)
                new_lp = dist.log_prob(actions_t)
                entropy = dist.entropy().mean()
                values = self.network.forward_critic(obs_t, global_t)

                # PPO clipped surrogate
                ratio = torch.exp(new_lp - old_lp_t)
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                critic_loss = nn.functional.mse_loss(values, ret_t)

                # Total loss
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        self.buffer.clear()

        n = max(n_updates, 1)
        return {
            "actor_loss": total_actor_loss / n,
            "critic_loss": total_critic_loss / n,
            "entropy": total_entropy / n,
            "total_loss": (total_actor_loss + total_critic_loss) / n,
        }

    # ── persistence ───────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "model": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "algorithm": "mappo",
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
