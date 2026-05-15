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
  - Neighbor-aware GAT for upstream awareness
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
# Tanh-Normal Distribution (Squashed Gaussian)
# ──────────────────────────────────────────────────────────────────────

class TanhNormal:
    """Squashed Gaussian: action = (tanh(u) + 1) / 2, u ~ Normal(mean, std).

    Naturally bounded to (0, 1) without clipping.  Log-prob includes the
    Jacobian correction for the tanh squashing so PPO ratios are correct.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self._normal = torch.distributions.Normal(mean, std)

    def sample(self) -> torch.Tensor:
        u = self._normal.rsample()
        return (torch.tanh(u) + 1.0) / 2.0

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Log-prob of a squashed action ∈ (0,1)."""
        tanh_val = (action * 2.0 - 1.0).clamp(-1 + 1e-6, 1 - 1e-6)
        u = torch.atanh(tanh_val)
        lp = self._normal.log_prob(u)
        lp = lp - torch.log(0.5 * (1.0 - tanh_val.pow(2)) + 1e-6)
        return lp.sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """Approximate TanhNormal entropy (Normal entropy + Jacobian correction)."""
        u = self._normal.rsample()
        tanh_u = torch.tanh(u)
        # Jacobian correction is negative (squashing reduces entropy)
        correction = torch.log(0.5 * (1.0 - tanh_u.pow(2)) + 1e-6).sum(dim=-1)
        return self._normal.entropy().sum(dim=-1) + correction

    @property
    def mean(self) -> torch.Tensor:
        return (torch.tanh(self._normal.mean) + 1.0) / 2.0


# ──────────────────────────────────────────────────────────────────────
# Neighbor Graph Attention (GAT)
# ──────────────────────────────────────────────────────────────────────

class NeighborGAT(nn.Module):
    def __init__(self, feat_dim=2, out_dim=16):
        super().__init__()
        self.feat_dim = feat_dim
        self.attn = nn.Linear(feat_dim * 2, 1)
        self.ego_proj = nn.Linear(feat_dim, feat_dim)
        self.out = nn.Linear(feat_dim, out_dim)

    def forward(self, ego_summary, neighbors, mask, return_weights: bool = False):
        # ego_summary: (B, feat_dim), neighbors: (B, N, feat_dim), mask: (B, N) bool
        B, N, F = neighbors.shape
        ego_exp = self.ego_proj(ego_summary).unsqueeze(1).expand(B, N, F)
        scores = self.attn(torch.cat([ego_exp, neighbors], dim=-1)).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        all_masked = (~mask).all(dim=-1, keepdim=True)
        scores = scores.masked_fill(all_masked.expand_as(scores), 0.0)
        weights = torch.softmax(scores, dim=-1) * mask.float()
        agg = (weights.unsqueeze(-1) * neighbors).sum(dim=1)
        out = self.out(agg)
        if return_weights:
            return out, weights
        return out


# ──────────────────────────────────────────────────────────────────────
# Actor-Critic Network
# ──────────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """Shared actor-critic with centralized critic and optional neighbor GAT."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256,
                 gat_out: int = 16,
                 neighbor_feat_dim: int = 5) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gat_out = gat_out
        self.neighbor_feat_dim = neighbor_feat_dim

        if gat_out > 0:
            self.gat = NeighborGAT(feat_dim=neighbor_feat_dim, out_dim=gat_out)
        else:
            self.gat = None

        actor_in = obs_dim + gat_out
        critic_in = obs_dim + gat_out + obs_dim

        self.actor_backbone = nn.Sequential(
            nn.Linear(actor_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden, act_dim)
        self.actor_log_std = nn.Parameter(torch.full((act_dim,), -0.5))

        self.critic = nn.Sequential(
            nn.Linear(critic_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _build_ego_summary(self, obs: torch.Tensor) -> torch.Tensor:
        """Map local TLS obs into the same feature space used for neighbors."""
        batch = obs.shape[0]
        summary = torch.zeros(batch, self.neighbor_feat_dim, device=obs.device)

        summary[:, 0] = obs[:, :12].mean(dim=-1)
        summary[:, 1] = obs[:, 12:24].mean(dim=-1)

        if self.neighbor_feat_dim >= 3 and obs.shape[1] >= 39:
            summary[:, 2] = obs[:, 24:36].mean(dim=-1)
        if self.neighbor_feat_dim >= 4:
            if obs.shape[1] >= 39:
                summary[:, 3] = obs[:, 36]
            elif obs.shape[1] >= 25:
                summary[:, 3] = obs[:, 24]
        if self.neighbor_feat_dim >= 5:
            if obs.shape[1] >= 39:
                summary[:, 4] = obs[:, 37]
            elif obs.shape[1] >= 26:
                summary[:, 4] = obs[:, 25]

        return summary

    def _coerce_neighbor_feats(self, neighbor_feats: torch.Tensor) -> torch.Tensor:
        """Trim or zero-pad neighbor features to match the checkpoint GAT size."""
        feat_dim = neighbor_feats.shape[-1]
        if feat_dim == self.neighbor_feat_dim:
            return neighbor_feats
        if feat_dim > self.neighbor_feat_dim:
            return neighbor_feats[..., :self.neighbor_feat_dim]

        pad_shape = list(neighbor_feats.shape)
        pad_shape[-1] = self.neighbor_feat_dim - feat_dim
        pad = torch.zeros(
            pad_shape, dtype=neighbor_feats.dtype, device=neighbor_feats.device
        )
        return torch.cat([neighbor_feats, pad], dim=-1)

    def _apply_gat(self, obs, neighbor_feats, neighbor_mask, return_weights: bool = False):
        if self.gat_out == 0 or self.gat is None:
            if return_weights:
                return obs, None
            return obs
        neighbor_feats = self._coerce_neighbor_feats(neighbor_feats)
        ego_summary = self._build_ego_summary(obs)
        gat_out = self.gat(
            ego_summary, neighbor_feats, neighbor_mask,
            return_weights=return_weights,
        )
        if return_weights:
            gat_emb, weights = gat_out
            return torch.cat([obs, gat_emb], dim=-1), weights
        return torch.cat([obs, gat_out], dim=-1)

    def forward_actor(self, obs: torch.Tensor,
                      valid_mask: Optional[torch.Tensor] = None,
                      neighbor_feats: Optional[torch.Tensor] = None,
                      neighbor_mask: Optional[torch.Tensor] = None):
        if self.gat_out > 0 and neighbor_feats is not None and neighbor_mask is not None:
            x = self._apply_gat(obs, neighbor_feats, neighbor_mask)
        else:
            if self.gat_out > 0:
                pad = torch.zeros(obs.shape[0], self.gat_out, device=obs.device)
                x = torch.cat([obs, pad], dim=-1)
            else:
                x = obs
        h = self.actor_backbone(x)
        mean = self.actor_mean(h)
        std = torch.exp(self.actor_log_std.clamp(-4, 2)).expand_as(mean)
        return TanhNormal(mean, std)

    def explain_actor(self, obs: torch.Tensor,
                      valid_mask: Optional[torch.Tensor] = None,
                      neighbor_feats: Optional[torch.Tensor] = None,
                      neighbor_mask: Optional[torch.Tensor] = None):
        """Return actor distribution plus optional neighbor attention weights."""
        attn_weights = None
        if self.gat_out > 0 and neighbor_feats is not None and neighbor_mask is not None:
            x, attn_weights = self._apply_gat(
                obs, neighbor_feats, neighbor_mask, return_weights=True,
            )
        else:
            if self.gat_out > 0:
                pad = torch.zeros(obs.shape[0], self.gat_out, device=obs.device)
                x = torch.cat([obs, pad], dim=-1)
            else:
                x = obs
        h = self.actor_backbone(x)
        mean = self.actor_mean(h)
        std = torch.exp(self.actor_log_std.clamp(-4, 2)).expand_as(mean)
        return TanhNormal(mean, std), attn_weights

    def forward_critic(self, obs: torch.Tensor,
                       global_obs: torch.Tensor,
                       neighbor_feats: Optional[torch.Tensor] = None,
                       neighbor_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.gat_out > 0 and neighbor_feats is not None and neighbor_mask is not None:
            x = self._apply_gat(obs, neighbor_feats, neighbor_mask)
        else:
            if self.gat_out > 0:
                pad = torch.zeros(obs.shape[0], self.gat_out, device=obs.device)
                x = torch.cat([obs, pad], dim=-1)
            else:
                x = obs
        inp = torch.cat([x, global_obs], dim=-1)
        return self.critic(inp).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────
# Rollout Buffer
# ──────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """On-policy buffer storing one episode of multi-agent transitions."""

    def __init__(self) -> None:
        self.obs: list[np.ndarray] = []
        self.global_obs: list[np.ndarray] = []
        self.actions: list = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []
        self.valid_masks: list[np.ndarray] = []
        self.neighbor_feats: list = []
        self.neighbor_masks: list = []

    def add(self, obs: np.ndarray, global_obs: np.ndarray,
            action,
            log_prob: float, reward: float, value: float, done: bool,
            valid_mask: np.ndarray,
            neighbor_feat=None, neighbor_mask=None) -> None:
        self.obs.append(obs)
        self.global_obs.append(global_obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.valid_masks.append(valid_mask)
        self.neighbor_feats.append(neighbor_feat)
        self.neighbor_masks.append(neighbor_mask)

    def compute_returns(self, last_values: list[float], gamma: float = 0.99,
                        gae_lambda: float = 0.95,
                        n_agents: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and discounted returns per-agent.

        Transitions are stored interleaved:
          [agent0_t0, agent1_t0, ..., agentK_t0, agent0_t1, ...]
        GAE must be computed along each agent's own trajectory (stride n_agents),
        not sequentially across the flat buffer.
        """
        n = len(self.rewards)
        if n == 0:
            return np.array([]), np.array([])

        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        for agent_offset in range(min(n_agents, n)):
            agent_idx = list(range(agent_offset, n, n_agents))
            last_gae = 0.0
            for j in reversed(range(len(agent_idx))):
                i = agent_idx[j]
                if j == len(agent_idx) - 1:
                    next_val = 0.0
                    next_done = False
                else:
                    i_next = agent_idx[j + 1]
                    next_val = self.values[i_next]
                    next_done = self.dones[i_next]

                if self.dones[i]:
                    last_gae = 0.0
                    next_val = 0.0

                delta = (self.rewards[i]
                         + gamma * next_val * (1 - float(self.dones[i]))
                         - self.values[i])
                last_gae = delta + gamma * gae_lambda * (1 - float(next_done)) * last_gae
                advantages[i] = last_gae
                returns[i] = advantages[i] + self.values[i]

        return advantages, returns

    def get_batches(self, batch_size: int, advantages: np.ndarray,
                    returns: np.ndarray, continuous: bool = False):
        """Yield shuffled mini-batches."""
        n = len(self.obs)
        indices = np.random.permutation(n)
        act_dtype = np.float32 if continuous else np.int64
        has_neighbor = any(f is not None for f in self.neighbor_feats)
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            batch = {
                "obs": np.array([self.obs[i] for i in idx], dtype=np.float32),
                "global_obs": np.array([self.global_obs[i] for i in idx], dtype=np.float32),
                "actions": np.array([self.actions[i] for i in idx], dtype=act_dtype),
                "old_log_probs": np.array([self.log_probs[i] for i in idx], dtype=np.float32),
                "advantages": advantages[idx],
                "returns": returns[idx],
                "valid_masks": np.array([self.valid_masks[i] for i in idx], dtype=np.bool_),
            }
            if has_neighbor:
                nf_list = [self.neighbor_feats[i] for i in idx]
                nm_list = [self.neighbor_masks[i] for i in idx]
                if all(f is not None for f in nf_list):
                    batch["neighbor_feats"] = np.array(nf_list, dtype=np.float32)
                    batch["neighbor_masks"] = np.array(nm_list, dtype=np.bool_)
            yield batch

    def clear(self) -> None:
        self.obs.clear()
        self.global_obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.valid_masks.clear()
        self.neighbor_feats.clear()
        self.neighbor_masks.clear()

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
        action_mode: str = "continuous",  # kept for backward compat, ignored
        gat_out: int = 16,
        neighbor_feat_dim: int = 5,
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
        self.neighbor_feat_dim = neighbor_feat_dim

        self.network = ActorCritic(
            obs_dim, act_dim, hidden,
            gat_out=gat_out,
            neighbor_feat_dim=neighbor_feat_dim,
        ).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    def select_action(
        self,
        obs: np.ndarray,
        global_obs: np.ndarray,
        valid_actions: Optional[list[int]] = None,
        greedy: bool = False,
        neighbor_feats: Optional[np.ndarray] = None,
        neighbor_mask: Optional[np.ndarray] = None,
    ) -> tuple:
        """Select action, return (action_array, log_prob, value)."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            global_t = torch.FloatTensor(global_obs).unsqueeze(0).to(self.device)

            nf_t = None
            nm_t = None
            if neighbor_feats is not None and neighbor_mask is not None:
                nf_t = torch.FloatTensor(neighbor_feats).unsqueeze(0).to(self.device)
                nm_t = torch.BoolTensor(neighbor_mask).unsqueeze(0).to(self.device)

            dist = self.network.forward_actor(obs_t, None, nf_t, nm_t)
            value = self.network.forward_critic(obs_t, global_t, nf_t, nm_t)

            action = dist.mean if greedy else dist.sample()
            log_prob = dist.log_prob(action)
            return action.squeeze(0).cpu().numpy(), float(log_prob.item()), float(value.item())

    def explain_action(
        self,
        obs: np.ndarray,
        global_obs: np.ndarray,
        valid_actions: Optional[list[int]] = None,
        greedy: bool = True,
        neighbor_feats: Optional[np.ndarray] = None,
        neighbor_mask: Optional[np.ndarray] = None,
    ) -> dict:
        """Inference-only explanation path returning action, value, and attention."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            global_t = torch.FloatTensor(global_obs).unsqueeze(0).to(self.device)

            nf_t = None
            nm_t = None
            if neighbor_feats is not None and neighbor_mask is not None:
                nf_t = torch.FloatTensor(neighbor_feats).unsqueeze(0).to(self.device)
                nm_t = torch.BoolTensor(neighbor_mask).unsqueeze(0).to(self.device)

            dist, attn_weights = self.network.explain_actor(obs_t, None, nf_t, nm_t)
            value = self.network.forward_critic(obs_t, global_t, nf_t, nm_t)

            action = dist.mean if greedy else dist.sample()
            log_prob = dist.log_prob(action)

            weights = None
            if attn_weights is not None:
                weights = attn_weights.squeeze(0).detach().cpu().numpy()

            return {
                "action": action.squeeze(0).cpu().numpy(),
                "log_prob": float(log_prob.item()),
                "value": float(value.item()),
                "attention_weights": weights,
            }

    def get_valid_mask(self, valid_actions: list[int]) -> np.ndarray:
        """Create boolean mask array for valid actions."""
        mask = np.zeros(self.act_dim, dtype=np.bool_)
        for a in valid_actions:
            if a < self.act_dim:
                mask[a] = True
        return mask

    def update(self, n_agents: int = 1) -> dict:
        """Run PPO update over collected rollout buffer. Returns loss stats."""
        if len(self.buffer) == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}

        last_values = [0.0]

        advantages, returns = self.buffer.compute_returns(
            last_values, self.gamma, self.gae_lambda, n_agents=n_agents)

        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            for batch in self.buffer.get_batches(
                    self.mini_batch_size, advantages, returns, continuous=True):
                obs_t = torch.nan_to_num(
                    torch.FloatTensor(batch["obs"]).to(self.device))
                global_t = torch.nan_to_num(
                    torch.FloatTensor(batch["global_obs"]).to(self.device))
                actions_t = torch.FloatTensor(batch["actions"]).to(self.device)
                old_lp_t = torch.FloatTensor(batch["old_log_probs"]).to(self.device)
                adv_t = torch.FloatTensor(batch["advantages"]).to(self.device)
                ret_t = torch.FloatTensor(batch["returns"]).to(self.device)

                nf_t = None
                nm_t = None
                if "neighbor_feats" in batch:
                    nf_t = torch.FloatTensor(batch["neighbor_feats"]).to(self.device)
                    nm_t = torch.BoolTensor(batch["neighbor_masks"]).to(self.device)

                dist = self.network.forward_actor(obs_t, None, nf_t, nm_t)
                new_lp = dist.log_prob(actions_t)
                entropy = dist.entropy().mean()
                values = self.network.forward_critic(obs_t, global_t, nf_t, nm_t)

                ratio = torch.exp(new_lp - old_lp_t)
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.functional.mse_loss(values, ret_t)

                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                if not torch.isfinite(loss):
                    continue

                self.optimizer.zero_grad()
                loss.backward()

                all_finite = all(
                    p.grad is None or torch.isfinite(p.grad).all()
                    for p in self.network.parameters()
                )
                if all_finite:
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
            "total_loss": (total_actor_loss + self.value_coef * total_critic_loss
                          - self.entropy_coef * total_entropy) / n,
        }

    # ── persistence ───────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "model": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "algorithm": "mappo",
            "neighbor_feat_dim": self.network.neighbor_feat_dim,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        model_keys = list(ckpt["model"].keys())

        # Auto-detect gat_out from backbone input weight shape
        backbone_key = next((k for k in model_keys if "actor_backbone.0.weight" in k), None)
        if backbone_key is not None:
            backbone_in = ckpt["model"][backbone_key].shape[1]
            gat_out = backbone_in - self.obs_dim
        else:
            gat_out = 16

        feat_dim = ckpt.get("neighbor_feat_dim")
        if feat_dim is None:
            attn_key = next((k for k in model_keys if "gat.attn.weight" in k), None)
            if attn_key is not None:
                feat_dim = ckpt["model"][attn_key].shape[1] // 2
            else:
                feat_dim = self.neighbor_feat_dim

        if (
            self.network.gat_out != gat_out
            or self.network.neighbor_feat_dim != feat_dim
        ):
            hidden = ckpt["model"]["critic.0.weight"].shape[0]
            self.network = ActorCritic(
                self.obs_dim, self.act_dim, hidden,
                gat_out=gat_out,
                neighbor_feat_dim=feat_dim,
            ).to(self.device)
            lr = self.optimizer.defaults["lr"]
            self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.neighbor_feat_dim = feat_dim

        self.network.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except (ValueError, KeyError):
                pass
