"""
FlowMind AI - Multi-Agent SAC (MASAC) with shared parameters.

Off-policy SAC for continuous traffic signal duration control.
All 83 TLS agents share one actor network (parameter sharing).
Twin Q-networks use centralized global_obs (CTDE).
Auto-tuned entropy temperature alpha.

Key advantages over MAPPO:
  - Replay buffer: reuses all past experience (sample efficient)
  - Off-policy: no on-policy staleness issues
  - Auto-entropy: no manual entropy_coef tuning
  - Consistent: deterministic greedy policy at deployment
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .mappo_agent import TanhNormal


# ──────────────────────────────────────────────────────────────────────
# Networks
# ──────────────────────────────────────────────────────────────────────

class SACActorNetwork(nn.Module):
    """Shared stochastic actor: obs -> TanhNormal(mean, std)."""

    def __init__(self, obs_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.log_std_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> TanhNormal:
        h = self.backbone(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(-4, 2)
        return TanhNormal(mean, torch.exp(log_std))


class SACCriticNetwork(nn.Module):
    """Centralized Q-network: (obs, global_obs, action) -> Q-value."""

    def __init__(self, obs_dim: int, hidden: int = 256) -> None:
        super().__init__()
        # Input: local obs + global obs + action
        self.net = nn.Sequential(
            nn.Linear(obs_dim * 2 + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, global_obs: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, global_obs, action], dim=-1)
        return self.net(x).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────
# Replay Buffer
# ──────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Ring buffer for off-policy transitions."""

    def __init__(self, capacity: int = 500_000, obs_dim: int = 39) -> None:
        self.capacity = capacity
        self._pos = 0
        self._size = 0
        self.obs          = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.global_obs   = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions      = np.zeros((capacity, 1),       dtype=np.float32)
        self.rewards      = np.zeros(capacity,            dtype=np.float32)
        self.next_obs     = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_global  = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones        = np.zeros(capacity,            dtype=np.float32)

    def add(self, obs: np.ndarray, global_obs: np.ndarray, action: float,
            reward: float, next_obs: np.ndarray, next_global: np.ndarray,
            done: bool) -> None:
        i = self._pos
        self.obs[i]        = obs
        self.global_obs[i] = global_obs
        self.actions[i]    = action
        self.rewards[i]    = reward
        self.next_obs[i]   = next_obs
        self.next_global[i]= next_global
        self.dones[i]      = float(done)
        self._pos  = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, self._size, size=batch_size)
        return {
            "obs":         self.obs[idx],
            "global_obs":  self.global_obs[idx],
            "actions":     self.actions[idx],
            "rewards":     self.rewards[idx],
            "next_obs":    self.next_obs[idx],
            "next_global": self.next_global[idx],
            "dones":       self.dones[idx],
        }

    def __len__(self) -> int:
        return self._size


# ──────────────────────────────────────────────────────────────────────
# MASAC Agent
# ──────────────────────────────────────────────────────────────────────

class MASACAgent:
    """Multi-Agent SAC with shared actor and centralized twin critics.

    All TLS agents share one actor. Twin Q-networks take local obs +
    global obs (mean of all agents) + action as input.
    Entropy temperature alpha is auto-tuned.
    """

    def __init__(
        self,
        obs_dim: int = 39,
        hidden: int = 256,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_capacity: int = 500_000,
        batch_size: int = 256,
        warmup_steps: int = 1000,
        updates_per_step: int = 1,
        device: Optional[str] = None,
    ) -> None:
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.updates_per_step = updates_per_step
        self.obs_dim = obs_dim

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Networks
        self.actor   = SACActorNetwork(obs_dim, hidden).to(self.device)
        self.critic1 = SACCriticNetwork(obs_dim, hidden).to(self.device)
        self.critic2 = SACCriticNetwork(obs_dim, hidden).to(self.device)
        self.target1 = SACCriticNetwork(obs_dim, hidden).to(self.device)
        self.target2 = SACCriticNetwork(obs_dim, hidden).to(self.device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_opt   = optim.Adam(self.actor.parameters(),   lr=lr_actor)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=lr_critic)

        # Auto-tuned entropy temperature
        self.target_entropy = -1.0  # -act_dim for continuous scalar action
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr_alpha)

        self.buffer = ReplayBuffer(buffer_capacity, obs_dim)
        self._total_steps = 0

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().item())

    def select_action(self, obs: np.ndarray, global_obs: np.ndarray,
                      greedy: bool = False) -> float:
        with torch.no_grad():
            obs_t    = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            dist     = self.actor(obs_t)
            action   = dist.mean if greedy else dist.sample()
            return float(action.item())

    def store(self, obs, global_obs, action, reward,
              next_obs, next_global, done) -> None:
        self.buffer.add(obs, global_obs, action, reward, next_obs, next_global, done)
        self._total_steps += 1

    def update(self) -> dict:
        if len(self.buffer) < self.warmup_steps:
            return {}

        stats = {"critic_loss": 0.0, "actor_loss": 0.0,
                 "alpha_loss": 0.0, "alpha": self.alpha}

        for _ in range(self.updates_per_step):
            batch = self.buffer.sample(self.batch_size)
            obs_t       = torch.FloatTensor(batch["obs"]).to(self.device)
            global_t    = torch.FloatTensor(batch["global_obs"]).to(self.device)
            act_t       = torch.FloatTensor(batch["actions"]).to(self.device)
            rew_t       = torch.FloatTensor(batch["rewards"]).to(self.device)
            next_obs_t  = torch.FloatTensor(batch["next_obs"]).to(self.device)
            next_glob_t = torch.FloatTensor(batch["next_global"]).to(self.device)
            done_t      = torch.FloatTensor(batch["dones"]).to(self.device)

            # ── Critic update ──────────────────────────────────────────
            with torch.no_grad():
                next_dist   = self.actor(next_obs_t)
                next_act    = next_dist.sample()            # (batch, 1)
                next_lp     = next_dist.log_prob(next_act)  # (batch,)
                q1_next = self.target1(next_obs_t, next_glob_t, next_act)
                q2_next = self.target2(next_obs_t, next_glob_t, next_act)
                q_next  = torch.min(q1_next, q2_next) - self.alpha * next_lp
                target_q = rew_t + self.gamma * (1 - done_t) * q_next

            q1 = self.critic1(obs_t, global_t, act_t)
            q2 = self.critic2(obs_t, global_t, act_t)
            c1_loss = nn.functional.mse_loss(q1, target_q)
            c2_loss = nn.functional.mse_loss(q2, target_q)

            self.critic1_opt.zero_grad(); c1_loss.backward(); self.critic1_opt.step()
            self.critic2_opt.zero_grad(); c2_loss.backward(); self.critic2_opt.step()

            # ── Actor update ───────────────────────────────────────────
            dist    = self.actor(obs_t)
            new_act = dist.sample()                         # (batch, 1)
            new_lp  = dist.log_prob(new_act)                # (batch,)
            new_act_t = new_act
            q1_new  = self.critic1(obs_t, global_t, new_act_t)
            q2_new  = self.critic2(obs_t, global_t, new_act_t)
            q_new   = torch.min(q1_new, q2_new)
            actor_loss = (self.alpha * new_lp - q_new).mean()

            self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

            # ── Alpha update ───────────────────────────────────────────
            alpha_loss = -(self.log_alpha * (new_lp + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

            # ── Soft target update ─────────────────────────────────────
            for p, tp in zip(self.critic1.parameters(), self.target1.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.critic2.parameters(), self.target2.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

            stats["critic_loss"] += (c1_loss.item() + c2_loss.item()) / 2
            stats["actor_loss"]  += actor_loss.item()
            stats["alpha_loss"]  += alpha_loss.item()
            stats["alpha"]        = self.alpha

        for k in ("critic_loss", "actor_loss", "alpha_loss"):
            stats[k] /= self.updates_per_step

        return stats

    def save(self, path: str) -> None:
        torch.save({
            "actor":   self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target1": self.target1.state_dict(),
            "target2": self.target2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "obs_dim": self.obs_dim,
            "algorithm": "masac",
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.target1.load_state_dict(ckpt["target1"])
        self.target2.load_state_dict(ckpt["target2"])
        self.log_alpha = ckpt["log_alpha"].to(self.device).requires_grad_(True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=3e-4)
