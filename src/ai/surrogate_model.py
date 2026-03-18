"""
FlowMind AI - Neural surrogate model for traffic simulation.

Replaces SUMO for fast RL training. Given (obs, action) for a single TLS,
predicts (next_obs, reward) using a residual MLP.

Architecture:
    Input:  obs(39) + action_onehot(8) = 47
    Hidden: 256 -> 256 (ReLU)
    Output: delta_obs(39) via Tanh*0.5 + reward(1) via Tanh

    next_obs = clip(obs + delta_obs, 0, 1)

Usage:
    trainer = SurrogateTrainer(obs_dim=39, act_dim=8)
    stats = trainer.train(transition_buffer, epochs=50)
    trainer.save("checkpoints/surrogate.pt")
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from .traffic_env import OBS_DIM, ACT_DIM


class SurrogateNetwork(nn.Module):
    """Residual MLP: predicts (delta_obs, reward) from (obs, action_onehot)."""

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        inp = obs_dim + act_dim
        self.shared = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Residual obs prediction: small deltas clamped by tanh
        self.obs_head = nn.Sequential(
            nn.Linear(hidden, obs_dim),
            nn.Tanh(),
        )
        # Reward prediction
        self.reward_head = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor, action_onehot: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (batch, obs_dim)
            action_onehot: (batch, act_dim)
        Returns:
            pred_next_obs: (batch, obs_dim)
            pred_reward: (batch, 1)
        """
        x = torch.cat([obs, action_onehot], dim=-1)
        h = self.shared(x)

        # Residual prediction: next = obs + delta * 0.5
        delta = self.obs_head(h) * 0.5
        pred_next = torch.clamp(obs + delta, 0.0, 1.0)

        pred_reward = self.reward_head(h)

        return pred_next, pred_reward


class SurrogateTrainer:
    """Train and manage the surrogate model."""

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden: int = 256):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SurrogateNetwork(obs_dim, act_dim, hidden).to(self.device)
        self.train_stats: dict = {}

    def train(self, buffer, epochs: int = 15, lr: float = 1e-3,
              batch_size: int = 512, val_split: float = 0.1,
              patience: int = 5,
              obs_loss_weight: float = 1.0,
              reward_loss_weight: float = 0.1) -> dict:
        """Train surrogate on transition buffer data.

        Args:
            buffer: TransitionBuffer instance
            epochs: training epochs
            lr: learning rate
            batch_size: batch size
            val_split: fraction for validation
            obs_loss_weight: weight for observation MSE
            reward_loss_weight: weight for reward MSE

        Returns:
            dict with training stats (train_loss, val_loss, etc.)
        """
        obs_arr, act_arr, next_obs_arr, rew_arr, _ = buffer.get_data()
        n = len(obs_arr)
        if n < batch_size:
            return {"error": f"Not enough data ({n} < {batch_size})"}

        # One-hot encode actions
        act_onehot = np.zeros((n, self.act_dim), dtype=np.float32)
        act_onehot[np.arange(n), act_arr.astype(int)] = 1.0

        # Train/val split
        indices = np.random.permutation(n)
        val_n = max(int(n * val_split), batch_size)
        val_idx, train_idx = indices[:val_n], indices[val_n:]

        def make_loader(idx):
            return DataLoader(
                TensorDataset(
                    torch.FloatTensor(obs_arr[idx]),
                    torch.FloatTensor(act_onehot[idx]),
                    torch.FloatTensor(next_obs_arr[idx]),
                    torch.FloatTensor(rew_arr[idx].reshape(-1, 1)),
                ),
                batch_size=batch_size, shuffle=True,
            )

        train_loader = make_loader(train_idx)
        val_loader = make_loader(val_idx)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        best_val = float("inf")
        best_state = None
        no_improve = 0  # early stopping counter

        history = {"train_loss": [], "val_loss": [], "val_obs_mse": [],
                   "val_reward_mae": []}

        for epoch in range(1, epochs + 1):
            # Train
            self.model.train()
            train_loss_sum = 0.0
            train_n = 0
            for obs_b, act_b, nobs_b, rew_b in train_loader:
                obs_b = obs_b.to(self.device)
                act_b = act_b.to(self.device)
                nobs_b = nobs_b.to(self.device)
                rew_b = rew_b.to(self.device)

                pred_obs, pred_rew = self.model(obs_b, act_b)
                loss_obs = nn.functional.mse_loss(pred_obs, nobs_b)
                loss_rew = nn.functional.mse_loss(pred_rew, rew_b)
                loss = obs_loss_weight * loss_obs + reward_loss_weight * loss_rew

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()

                train_loss_sum += loss.item() * len(obs_b)
                train_n += len(obs_b)

            # Validate
            self.model.eval()
            val_loss_sum = val_obs_sum = val_rew_sum = 0.0
            val_n_total = 0
            with torch.no_grad():
                for obs_b, act_b, nobs_b, rew_b in val_loader:
                    obs_b = obs_b.to(self.device)
                    act_b = act_b.to(self.device)
                    nobs_b = nobs_b.to(self.device)
                    rew_b = rew_b.to(self.device)

                    pred_obs, pred_rew = self.model(obs_b, act_b)
                    loss_obs = nn.functional.mse_loss(pred_obs, nobs_b)
                    loss_rew = nn.functional.mse_loss(pred_rew, rew_b)
                    loss = obs_loss_weight * loss_obs + reward_loss_weight * loss_rew

                    val_loss_sum += loss.item() * len(obs_b)
                    val_obs_sum += loss_obs.item() * len(obs_b)
                    val_rew_sum += (pred_rew - rew_b).abs().sum().item()
                    val_n_total += len(obs_b)

            train_loss = train_loss_sum / max(train_n, 1)
            val_loss = val_loss_sum / max(val_n_total, 1)
            val_obs_mse = val_obs_sum / max(val_n_total, 1)
            val_reward_mae = val_rew_sum / max(val_n_total, 1)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_obs_mse"].append(val_obs_mse)
            history["val_reward_mae"].append(val_reward_mae)

            # Early stopping: save best, stop if no improvement
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone()
                              for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break  # stop training, val loss isn't improving

        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)

        self.train_stats = {
            "samples": n,
            "epochs": epochs,
            "best_val_loss": best_val,
            "final_val_obs_mse": history["val_obs_mse"][-1],
            "final_val_reward_mae": history["val_reward_mae"][-1],
            "history": history,
        }
        return self.train_stats

    def predict(self, obs: np.ndarray, action: int
                ) -> tuple[np.ndarray, float]:
        """Single-sample prediction (for SurrogateEnv)."""
        self.model.eval()
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            act_oh = torch.zeros(1, self.act_dim, device=self.device)
            act_oh[0, action] = 1.0
            pred_obs, pred_rew = self.model(obs_t, act_oh)
        return pred_obs.cpu().numpy()[0], float(pred_rew.cpu().item())

    def predict_batch(self, obs_batch: np.ndarray, actions: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
        """Batch prediction (for fast surrogate episodes)."""
        self.model.eval()
        n = len(obs_batch)
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs_batch).to(self.device)
            act_oh = torch.zeros(n, self.act_dim, device=self.device)
            act_oh[torch.arange(n), torch.LongTensor(actions)] = 1.0
            pred_obs, pred_rew = self.model(obs_t, act_oh)
        return pred_obs.cpu().numpy(), pred_rew.cpu().numpy().flatten()

    def save(self, path: str) -> None:
        torch.save({
            "model": self.model.state_dict(),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "train_stats": self.train_stats,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.train_stats = ckpt.get("train_stats", {})
