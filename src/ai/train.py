"""
FlowMind AI - DQN training pipeline for traffic signal optimization.

Usage (Da Nang):
    python -m src.ai.train \
        --net  sumo/danang/danang.net.xml \
        --route sumo/danang/danang.rou.xml \
        --cfg  sumo/danang/danang.sumocfg \
        --episodes 100

Evaluation only:
    python -m src.ai.train \
        --net  sumo/danang/danang.net.xml \
        --route sumo/danang/danang.rou.xml \
        --cfg  sumo/danang/danang.sumocfg \
        --eval-only checkpoints/best_model.pt --gui
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path so ``python -m src.ai.train`` works
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.ai.traffic_env import SumoTrafficEnv, OBS_DIM, ACT_DIM
from src.ai.dqn_agent import TrafficDQNAgent
from src.ai.mappo_agent import MAPPOAgent


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────

def train(
    net_file: str,
    route_file: str,
    sumo_cfg: str | None = None,
    episodes: int = 100,
    delta_time: int = 10,
    sim_length: int = 3600,
    hidden: int = 256,
    lr: float = 1e-3,
    gamma: float = 0.99,
    batch_size: int = 64,
    buffer_capacity: int = 200_000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: int = 500_000,
    target_update: int = 1000,
    save_dir: str = "checkpoints",
    save_every: int = 10,
    seed: int = 42,
    gui: bool = False,
) -> tuple[TrafficDQNAgent, dict]:
    """Run the full DQN training loop and return (agent, log)."""

    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "training_log.json")

    # ── banner ────────────────────────────────────────────────────────
    print("=" * 60)
    print("  FlowMind AI — DQN Traffic Signal Optimization")
    print("=" * 60)
    print(f"  Network : {net_file}")
    print(f"  Routes  : {route_file}")
    print(f"  Episodes: {episodes}")
    print(f"  dt      : {delta_time}s   Sim length: {sim_length}s")
    print()

    # ── environment ───────────────────────────────────────────────────
    env = SumoTrafficEnv(
        net_file=net_file,
        route_file=route_file,
        sumo_cfg=sumo_cfg,
        delta_time=delta_time,
        sim_length=sim_length,
        gui=gui,
        seed=seed,
    )

    print(f"  TLS agents      : {env.num_agents}")
    print(f"  Observation dim : {OBS_DIM}")
    print(f"  Action dim      : {ACT_DIM}")
    print(f"  Steps / episode : {env._max_steps}")
    print()

    # ── agent ─────────────────────────────────────────────────────────
    agent = TrafficDQNAgent(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        hidden=hidden,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        target_update_freq=target_update,
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
    )

    print(f"  Device  : {agent.device}")
    print(f"  Params  : {sum(p.numel() for p in agent.q_net.parameters()):,}")
    print()

    # ── log structure ─────────────────────────────────────────────────
    log: dict = {
        "config": {
            "net_file": net_file,
            "route_file": route_file,
            "episodes": episodes,
            "delta_time": delta_time,
            "sim_length": sim_length,
            "hidden": hidden,
            "lr": lr,
            "gamma": gamma,
            "batch_size": batch_size,
            "num_agents": env.num_agents,
            "obs_dim": OBS_DIM,
            "act_dim": ACT_DIM,
        },
        "episodes": [],
    }

    best_reward = -float("inf")

    # ── episode loop ──────────────────────────────────────────────────
    for ep in range(1, episodes + 1):
        t0 = time.time()
        obs, _ = env.reset(seed=seed + ep)

        ep_rewards: dict[str, float] = {tid: 0.0 for tid in env.tls_ids}
        ep_losses: list[float] = []
        step_count = 0
        terminated = truncated = False

        while not (terminated or truncated):
            # ── select actions ────────────────────────────────────────
            actions: dict[str, int] = {}
            for tid in env.tls_ids:
                valid = env.get_valid_actions(tid)
                actions[tid] = agent.select_action(obs[tid], valid)

            # ── env step ──────────────────────────────────────────────
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            env.record_actions(actions)

            # ── store transitions (shared buffer) ─────────────────────
            for tid in env.tls_ids:
                agent.store_transition(
                    obs[tid],
                    actions[tid],
                    rewards[tid],
                    next_obs[tid],
                    terminated,
                )
                ep_rewards[tid] += rewards[tid]

            # ── train ─────────────────────────────────────────────────
            loss = agent.update()
            if loss is not None:
                ep_losses.append(loss)

            obs = next_obs
            step_count += 1

        # ── episode stats ─────────────────────────────────────────────
        elapsed = time.time() - t0
        mean_r = float(np.mean(list(ep_rewards.values())))
        mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        metrics = env.get_metrics()

        ep_log = {
            "episode": ep,
            "mean_reward": round(mean_r, 4),
            "total_reward": round(sum(ep_rewards.values()), 4),
            "mean_loss": round(mean_loss, 6),
            "epsilon": round(agent.epsilon, 4),
            "steps": step_count,
            "time_s": round(elapsed, 1),
            "buffer_size": len(agent.buffer),
            "avg_wait": metrics.get("avg_wait_time", 0),
            "avg_queue": metrics.get("avg_queue_length", 0),
            "vehicles": metrics.get("total_vehicles", 0),
        }
        log["episodes"].append(ep_log)

        print(
            f"  Ep {ep:4d}/{episodes} | "
            f"R={mean_r:+.3f} | "
            f"Loss={mean_loss:.4f} | "
            f"eps={agent.epsilon:.3f} | "
            f"Wait={metrics.get('avg_wait_time', 0):.1f}s | "
            f"Queue={metrics.get('avg_queue_length', 0):.1f} | "
            f"Steps={step_count} | "
            f"{elapsed:.0f}s"
        )

        # ── checkpointing ─────────────────────────────────────────────
        if mean_r > best_reward:
            best_reward = mean_r
            agent.save(os.path.join(save_dir, "best_model.pt"))

        if ep % save_every == 0:
            agent.save(os.path.join(save_dir, f"model_ep{ep}.pt"))
            with open(log_path, "w") as f:
                json.dump(log, f, indent=2)

    # ── final save ────────────────────────────────────────────────────
    agent.save(os.path.join(save_dir, "final_model.pt"))

    # [TLS CANDIDATE COMMENTED OUT] recommendations
    # recs = env.get_recommendations()
    # log["recommendations"] = recs

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    env.close()

    print()
    print("=" * 60)
    print(f"  Training complete!  Best mean reward: {best_reward:.4f}")
    print(f"  Models saved to : {save_dir}/")
    print(f"  Training log    : {log_path}")
    print("=" * 60)

    return agent, log


# [TLS CANDIDATE COMMENTED OUT] _print_recommendations removed


# ──────────────────────────────────────────────────────────────────────
# Callback-based training (for GUI integration)
# ──────────────────────────────────────────────────────────────────────

def train_with_callbacks(
    net_file: str,
    route_file: str,
    sumo_cfg: str | None = None,
    episodes: int = 100,
    delta_time: int = 10,
    sim_length: int = 3600,
    hidden: int = 256,
    lr: float = 1e-3,
    gamma: float = 0.99,
    batch_size: int = 64,
    buffer_capacity: int = 200_000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: int = 500_000,
    target_update: int = 1000,
    save_dir: str = "checkpoints",
    save_every: int = 10,
    seed: int = 42,
    gui: bool = False,
    on_episode=None,      # callback(ep_log_dict) after each episode
    on_status=None,       # callback(str) for status updates
    stop_check=None,      # callable() -> bool, checked before each episode
) -> tuple[TrafficDQNAgent, dict]:
    """Same as train() but with callbacks for GUI integration.

    *on_episode*: called with episode metrics dict after each episode.
    *on_status*: called with status string messages (e.g. "Starting SUMO...").
    *stop_check*: if returns True, training stops gracefully.
    """
    def _status(msg):
        if on_status:
            on_status(msg)

    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "training_log.json")

    _status(f"Creating environment (starting SUMO{'-gui' if gui else ''})...")
    env = SumoTrafficEnv(
        net_file=net_file, route_file=route_file, sumo_cfg=sumo_cfg,
        delta_time=delta_time, sim_length=sim_length, gui=gui, seed=seed,
    )
    _status(f"Environment ready: {env.num_agents} TLS agents")

    agent = TrafficDQNAgent(
        obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=hidden, lr=lr,
        gamma=gamma, epsilon_start=epsilon_start, epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay, target_update_freq=target_update,
        batch_size=batch_size, buffer_capacity=buffer_capacity,
    )
    _status(f"Agent ready ({sum(p.numel() for p in agent.q_net.parameters()):,} params)")

    log: dict = {
        "config": {
            "net_file": net_file, "episodes": episodes,
            "num_agents": env.num_agents, "obs_dim": OBS_DIM, "act_dim": ACT_DIM,
        },
        "episodes": [],
    }

    best_reward = -float("inf")

    for ep in range(1, episodes + 1):
        if stop_check and stop_check():
            break

        t0 = time.time()
        obs, _ = env.reset(seed=seed + ep)
        ep_rewards: dict[str, float] = {tid: 0.0 for tid in env.tls_ids}
        ep_losses: list[float] = []
        step_count = 0
        terminated = truncated = False

        while not (terminated or truncated):
            actions = {
                tid: agent.select_action(obs[tid], env.get_valid_actions(tid))
                for tid in env.tls_ids
            }
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            env.record_actions(actions)
            for tid in env.tls_ids:
                agent.store_transition(obs[tid], actions[tid], rewards[tid],
                                       next_obs[tid], terminated)
                ep_rewards[tid] += rewards[tid]
            loss = agent.update()
            if loss is not None:
                ep_losses.append(loss)
            obs = next_obs
            step_count += 1

        elapsed = time.time() - t0
        mean_r = float(np.mean(list(ep_rewards.values())))
        mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        metrics = env.get_metrics()

        # [TLS CANDIDATE COMMENTED OUT] tls_snap = env.get_tls_snapshot()
        # [TLS CANDIDATE COMMENTED OUT] tls_details = env.get_tls_details()
        ep_log = {
            "episode": ep, "total_episodes": episodes,
            "mean_reward": round(mean_r, 4),
            "total_reward": round(sum(ep_rewards.values()), 4),
            "mean_loss": round(mean_loss, 6),
            "epsilon": round(agent.epsilon, 4),
            "steps": step_count,
            "time_s": round(elapsed, 1),
            "buffer_size": len(agent.buffer),
            "avg_wait": metrics.get("avg_wait_time", 0),
            "avg_queue": metrics.get("avg_queue_length", 0),
            "vehicles": metrics.get("total_vehicles", 0),
            "collisions": metrics.get("collisions", 0),
            "best_reward": round(max(best_reward, mean_r), 4),
            # [TLS CANDIDATE COMMENTED OUT] "tls_add": tls_snap["n_add"],
            # [TLS CANDIDATE COMMENTED OUT] "tls_remove": tls_snap["n_remove"],
            # [TLS CANDIDATE COMMENTED OUT] "tls_details": tls_details,
        }
        log["episodes"].append(ep_log)

        if mean_r > best_reward:
            best_reward = mean_r
            agent.save(os.path.join(save_dir, "best_model.pt"))

        if ep % save_every == 0:
            agent.save(os.path.join(save_dir, f"model_ep{ep}.pt"))
            with open(log_path, "w") as f:
                json.dump(log, f, indent=2)

        if on_episode:
            on_episode(ep_log)

    agent.save(os.path.join(save_dir, "final_model.pt"))

    # [TLS CANDIDATE COMMENTED OUT] recs = env.get_recommendations()
    # log["recommendations"] = recs

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    _status("Training complete!")

    env.close()

    return agent, log


# ──────────────────────────────────────────────────────────────────────
# MAPPO Training (callback-based for GUI integration)
# ──────────────────────────────────────────────────────────────────────

def train_mappo_with_callbacks(
    net_file: str,
    route_file: str,
    sumo_cfg: str | None = None,
    episodes: int = 100,
    delta_time: int = 10,
    sim_length: int = 3600,
    hidden: int = 256,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    ppo_epochs: int = 10,
    mini_batch_size: int = 256,
    save_dir: str = "checkpoints",
    save_every: int = 10,
    seed: int = 42,
    gui: bool = False,
    on_episode=None,
    on_status=None,
    stop_check=None,
) -> tuple[MAPPOAgent, dict]:
    """MAPPO training loop with callbacks for GUI integration."""
    def _status(msg):
        if on_status:
            on_status(msg)

    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "training_log.json")

    _status(f"Creating environment (starting SUMO{'-gui' if gui else ''})...")
    env = SumoTrafficEnv(
        net_file=net_file, route_file=route_file, sumo_cfg=sumo_cfg,
        delta_time=delta_time, sim_length=sim_length, gui=gui, seed=seed,
    )
    _status(f"Environment ready: {env.num_agents} TLS agents")

    # Print per-TLS timing from engineering formulas
    _status("Per-TLS timing (engineering formulas):")
    for tid in env.tls_ids:
        tls_info = env._tls_map.get(tid)
        g = tls_info.geometry if tls_info else None
        if g:
            yw_s = g.yellow_s
            ar_s = g.allred_s
            mg_s = g.min_green_s
            xg_s = g.max_green_s
            _status(
                f"  {tid[:40]:<42s} [{g.tier[:3].upper()}] "
                f"G={mg_s:.0f}-{xg_s:.0f}s Y={yw_s:.1f}s R={ar_s:.1f}s "
                f"W={g.width_m:.0f}m L={g.total_lanes}"
            )

    agent = MAPPOAgent(
        obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=hidden, lr=lr,
        gamma=gamma, gae_lambda=gae_lambda, clip_eps=clip_eps,
        entropy_coef=entropy_coef, value_coef=value_coef,
        ppo_epochs=ppo_epochs, mini_batch_size=mini_batch_size,
    )
    n_params = sum(p.numel() for p in agent.network.parameters())
    _status(f"MAPPO agent ready ({n_params:,} params, device={agent.device})")

    # Build TLS timing info for log
    tls_timing_lines = []
    for tid in env.tls_ids:
        tls_info = env._tls_map.get(tid)
        g = tls_info.geometry if tls_info else None
        if g:
            tls_timing_lines.append(
                f"  {tid[:40]:<42s} [{g.tier[:3].upper()}] "
                f"G={g.min_green_s:.0f}-{g.max_green_s:.0f}s "
                f"Y={g.yellow_s:.1f}s R={g.allred_s:.1f}s "
                f"W={g.width_m:.0f}m L={g.total_lanes}"
            )

    log: dict = {
        "config": {
            "algorithm": "mappo",
            "net_file": net_file, "episodes": episodes,
            "num_agents": env.num_agents, "obs_dim": OBS_DIM, "act_dim": ACT_DIM,
            "lr": lr, "gamma": gamma, "clip_eps": clip_eps,
            "ppo_epochs": ppo_epochs, "entropy_coef": entropy_coef,
        },
        "tls_timing": tls_timing_lines,
        "episodes": [],
    }

    best_reward = -float("inf")

    for ep in range(1, episodes + 1):
        if stop_check and stop_check():
            break

        t0 = time.time()
        obs, _ = env.reset(seed=seed + ep)
        agent.buffer.clear()

        ep_rewards: dict[str, float] = {tid: 0.0 for tid in env.tls_ids}
        step_count = 0
        terminated = truncated = False

        while not (terminated or truncated):
            # Global state = mean of all agent observations
            all_obs = [obs[tid] for tid in env.tls_ids]
            global_obs = np.mean(all_obs, axis=0).astype(np.float32)

            actions = {}
            for tid in env.tls_ids:
                valid = env.get_valid_actions(tid)
                a, lp, v = agent.select_action(obs[tid], global_obs, valid)
                actions[tid] = a

                # Store transition
                mask = agent.get_valid_mask(valid)
                agent.buffer.add(obs[tid], global_obs, a, lp, 0.0, v,
                                 False, mask)  # reward filled after step

            next_obs, rewards, terminated, truncated, info = env.step(actions)
            env.record_actions(actions)

            # Fill in rewards for the transitions we just stored
            buf = agent.buffer
            n_agents = len(env.tls_ids)
            for i, tid in enumerate(env.tls_ids):
                idx = len(buf.rewards) - n_agents + i
                buf.rewards[idx] = rewards[tid]
                buf.dones[idx] = terminated
                ep_rewards[tid] += rewards[tid]

            obs = next_obs
            step_count += 1

        # PPO update at end of episode
        loss_stats = agent.update()

        elapsed = time.time() - t0
        mean_r = float(np.mean(list(ep_rewards.values())))
        metrics = env.get_metrics()

        ep_log = {
            "episode": ep, "total_episodes": episodes,
            "mean_reward": round(mean_r, 4),
            "total_reward": round(sum(ep_rewards.values()), 4),
            "mean_loss": round(loss_stats["total_loss"], 6),
            "epsilon": 0.0,  # PPO has no epsilon
            "entropy": round(loss_stats["entropy"], 4),
            "actor_loss": round(loss_stats["actor_loss"], 6),
            "critic_loss": round(loss_stats["critic_loss"], 6),
            "steps": step_count,
            "time_s": round(elapsed, 1),
            "buffer_size": 0,  # on-policy, no persistent buffer
            "avg_wait": metrics.get("avg_wait_time", 0),
            "avg_queue": metrics.get("avg_queue_length", 0),
            "vehicles": metrics.get("total_vehicles", 0),
            "collisions": metrics.get("collisions", 0),
            "best_reward": round(max(best_reward, mean_r), 4),
            "algorithm": "mappo",
        }
        log["episodes"].append(ep_log)

        if mean_r > best_reward:
            best_reward = mean_r
            agent.save(os.path.join(save_dir, "best_model.pt"))

        if ep % save_every == 0:
            agent.save(os.path.join(save_dir, f"model_ep{ep}.pt"))
            with open(log_path, "w") as f:
                json.dump(log, f, indent=2)

        if on_episode:
            on_episode(ep_log)

    agent.save(os.path.join(save_dir, "final_model.pt"))

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    _status("MAPPO training complete!")
    env.close()

    return agent, log


# ──────────────────────────────────────────────────────────────────────
# Dyna-style training (real SUMO + surrogate)
# ──────────────────────────────────────────────────────────────────────

def train_dyna_with_callbacks(
    net_file: str,
    route_file: str,
    sumo_cfg: str | None = None,
    episodes: int = 100,
    delta_time: int = 10,
    sim_length: int = 3600,
    hidden: int = 256,
    lr: float = 1e-3,
    gamma: float = 0.99,
    batch_size: int = 64,
    buffer_capacity: int = 200_000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: int = 500_000,
    target_update: int = 1000,
    save_dir: str = "checkpoints",
    save_every: int = 10,
    seed: int = 42,
    gui: bool = False,
    surrogate_ratio: int = 2,
    min_real_episodes: int = 5,
    surrogate_retrain_freq: int = 5,
    surrogate_epochs: int = 15,
    on_episode=None,
    on_status=None,
    stop_check=None,
) -> tuple[TrafficDQNAgent, dict]:
    """Dyna-style training: real SUMO episodes + fast surrogate episodes.

    For every real SUMO episode, runs *surrogate_ratio* surrogate episodes.
    The surrogate is retrained every *surrogate_retrain_freq* real episodes.
    """
    from .transition_buffer import TransitionBuffer
    from .surrogate_model import SurrogateTrainer
    from .surrogate_env import SurrogateEnv

    def _status(msg):
        if on_status:
            on_status(msg)

    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "training_log.json")
    buf_path = os.path.join(save_dir, "transitions.npz")
    sur_path = os.path.join(save_dir, "surrogate.pt")

    # ── Real SUMO env ──────────────────────────────────────────────
    _status(f"Creating SUMO environment{' (gui)' if gui else ''}...")
    env = SumoTrafficEnv(
        net_file=net_file, route_file=route_file, sumo_cfg=sumo_cfg,
        delta_time=delta_time, sim_length=sim_length, gui=gui, seed=seed,
    )
    _status(f"Environment ready: {env.num_agents} TLS agents")

    # ── Agent ──────────────────────────────────────────────────────
    agent = TrafficDQNAgent(
        obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=hidden, lr=lr,
        gamma=gamma, epsilon_start=epsilon_start, epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay, target_update_freq=target_update,
        batch_size=batch_size, buffer_capacity=buffer_capacity,
    )
    _status(f"Agent ready ({sum(p.numel() for p in agent.q_net.parameters()):,} params)")

    # ── Transition buffer + surrogate ──────────────────────────────
    trans_buf = TransitionBuffer(capacity=3_000_000, obs_dim=OBS_DIM)
    surrogate = SurrogateTrainer(OBS_DIM, ACT_DIM)
    surrogate_ready = False

    log: dict = {
        "config": {
            "net_file": net_file, "episodes": episodes, "mode": "dyna",
            "surrogate_ratio": surrogate_ratio,
            "num_agents": env.num_agents, "obs_dim": OBS_DIM, "act_dim": ACT_DIM,
        },
        "episodes": [],
    }
    best_reward = -float("inf")
    real_ep_count = 0
    total_ep = 0  # Total episodes (real + surrogate)

    # ── Helper: run one episode on any env ─────────────────────────
    def _run_episode(run_env, ep_seed, source="sumo", collect=False):
        nonlocal best_reward, total_ep
        total_ep += 1

        t0 = time.time()
        obs, _ = run_env.reset(seed=ep_seed)
        ep_rewards = {tid: 0.0 for tid in run_env.tls_ids}
        ep_losses = []
        step_count = 0
        terminated = truncated = False

        while not (terminated or truncated):
            actions = {
                tid: agent.select_action(obs[tid], run_env.get_valid_actions(tid))
                for tid in run_env.tls_ids
            }
            next_obs, rewards, terminated, truncated, info = run_env.step(actions)
            run_env.record_actions(actions)

            for tid in run_env.tls_ids:
                agent.store_transition(obs[tid], actions[tid], rewards[tid],
                                       next_obs[tid], terminated)
                ep_rewards[tid] += rewards[tid]

            # Collect transitions for surrogate training
            if collect:
                trans_buf.add_batch(obs, actions, next_obs, rewards, terminated)

            loss = agent.update()
            if loss is not None:
                ep_losses.append(loss)
            obs = next_obs
            step_count += 1

        elapsed = time.time() - t0
        mean_r = float(np.mean(list(ep_rewards.values())))
        mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        metrics = run_env.get_metrics()
        # [TLS CANDIDATE COMMENTED OUT] tls_snap = run_env.get_tls_snapshot()

        ep_log = {
            "episode": total_ep, "total_episodes": episodes,
            "mean_reward": round(mean_r, 4),
            "total_reward": round(sum(ep_rewards.values()), 4),
            "mean_loss": round(mean_loss, 6),
            "epsilon": round(agent.epsilon, 4),
            "steps": step_count,
            "time_s": round(elapsed, 1),
            "buffer_size": len(agent.buffer),
            "avg_wait": metrics.get("avg_wait_time", 0),
            "avg_queue": metrics.get("avg_queue_length", 0),
            "vehicles": metrics.get("total_vehicles", 0),
            "collisions": metrics.get("collisions", 0),
            "best_reward": round(max(best_reward, mean_r), 4),
            # [TLS CANDIDATE COMMENTED OUT] "tls_add": tls_snap["n_add"],
            # [TLS CANDIDATE COMMENTED OUT] "tls_remove": tls_snap["n_remove"],
            # [TLS CANDIDATE COMMENTED OUT] "tls_details": ...,
            "source": source,
            "surrogate_buf": len(trans_buf),
        }
        log["episodes"].append(ep_log)

        if mean_r > best_reward:
            best_reward = mean_r
            agent.save(os.path.join(save_dir, "best_model.pt"))

        if on_episode:
            on_episode(ep_log)

        return mean_r

    # ── Main Dyna loop ─────────────────────────────────────────────
    for real_ep in range(1, episodes + 1):
        if stop_check and stop_check():
            break

        real_ep_count += 1

        # ── 1. Real SUMO episode ──────────────────────────────────
        _status(f"SUMO ep {real_ep}/{episodes} "
                f"(buf={len(trans_buf):,})")
        _run_episode(env, seed + real_ep, source="sumo", collect=True)

        # ── 2. Retrain surrogate periodically ─────────────────────
        if (real_ep_count >= min_real_episodes and
                real_ep_count % surrogate_retrain_freq == 0):
            _status(f"Training surrogate on {len(trans_buf):,} transitions...")
            stats = surrogate.train(trans_buf, epochs=surrogate_epochs)
            val_loss = stats.get("best_val_loss", 999)
            _status(f"Surrogate trained: val_loss={val_loss:.5f}")

            surrogate_ready = val_loss < 0.1  # quality gate
            if surrogate_ready:
                surrogate.save(sur_path)
                trans_buf.save(buf_path)

        # ── 3. Surrogate episodes ─────────────────────────────────
        if surrogate_ready:
            sur_env = SurrogateEnv(
                surrogate=surrogate,
                buffer=trans_buf,
                tls_ids=env.tls_ids,
                candidate_tls_ids=env.candidate_tls_ids,
                existing_tls_ids=env.existing_tls_ids,
                green_phases=env._green_phases,
                delta_time=delta_time,
                sim_length=sim_length,
                seed=seed + real_ep * 1000,
            )
            for s in range(surrogate_ratio):
                if stop_check and stop_check():
                    break
                _status(f"Surrogate ep {s+1}/{surrogate_ratio} "
                        f"(after SUMO ep {real_ep})")
                _run_episode(sur_env, seed + real_ep * 1000 + s,
                             source="surrogate", collect=False)

        # ── 4. Checkpoint ─────────────────────────────────────────
        if real_ep % save_every == 0:
            agent.save(os.path.join(save_dir, f"model_ep{real_ep}.pt"))
            with open(log_path, "w") as f:
                json.dump(log, f, indent=2)

    # ── Final ──────────────────────────────────────────────────────
    agent.save(os.path.join(save_dir, "final_model.pt"))
    # [TLS CANDIDATE COMMENTED OUT] recs = env.get_recommendations()
    # log["recommendations"] = recs
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    _status(f"Done! {real_ep_count} real + "
            f"{total_ep - real_ep_count} surrogate episodes")
    env.close()
    return agent, log


# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate(
    agent: TrafficDQNAgent,
    net_file: str,
    route_file: str,
    sumo_cfg: str | None = None,
    episodes: int = 5,
    delta_time: int = 10,
    sim_length: int = 3600,
    gui: bool = False,
    seed: int = 1000,
) -> dict:
    """Run *episodes* with a greedy policy (no exploration)."""

    env = SumoTrafficEnv(
        net_file=net_file,
        route_file=route_file,
        sumo_cfg=sumo_cfg,
        delta_time=delta_time,
        sim_length=sim_length,
        gui=gui,
        seed=seed,
    )

    results: list[dict] = []

    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            actions = {
                tid: agent.select_action(obs[tid], env.get_valid_actions(tid), greedy=True)
                for tid in env.tls_ids
            }
            obs, rewards, terminated, truncated, _ = env.step(actions)
            ep_reward += sum(rewards.values())

        metrics = env.get_metrics()
        results.append({"episode": ep, "total_reward": ep_reward, **metrics})
        print(
            f"  Eval {ep}/{episodes}: "
            f"reward={ep_reward:.2f}, "
            f"wait={metrics.get('avg_wait_time', 0):.1f}s"
        )

    env.close()

    return {
        "mean_reward": float(np.mean([r["total_reward"] for r in results])),
        "mean_wait": float(np.mean([r.get("avg_wait_time", 0) for r in results])),
        "mean_queue": float(np.mean([r.get("avg_queue_length", 0) for r in results])),
        "episodes": results,
    }


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="FlowMind AI — Train DQN traffic signal optimizer"
    )
    ap.add_argument("--net", required=True, help="SUMO .net.xml")
    ap.add_argument("--route", required=True, help="SUMO .rou.xml")
    ap.add_argument("--cfg", default=None, help="SUMO .sumocfg (optional)")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--delta-time", type=int, default=10)
    ap.add_argument("--sim-length", type=int, default=3600)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--buffer", type=int, default=200_000)
    ap.add_argument("--epsilon-start", type=float, default=1.0)
    ap.add_argument("--epsilon-end", type=float, default=0.05)
    ap.add_argument("--epsilon-decay", type=int, default=500_000)
    ap.add_argument("--target-update", type=int, default=1000)
    ap.add_argument("--save-dir", default="checkpoints")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--eval-only", default=None, help="Checkpoint path for eval")
    ap.add_argument("--algorithm", choices=["dqn", "mappo"], default="dqn",
                    help="RL algorithm: dqn or mappo (default: dqn)")

    args = ap.parse_args()

    if args.eval_only:
        agent = TrafficDQNAgent(OBS_DIM, ACT_DIM, args.hidden)
        agent.load(args.eval_only)
        results = evaluate(
            agent, args.net, args.route, args.cfg, gui=args.gui, seed=args.seed
        )
        print(f"\n  Results: {json.dumps(results, indent=2)}")
    elif args.algorithm == "mappo":
        train_mappo_with_callbacks(
            net_file=args.net,
            route_file=args.route,
            sumo_cfg=args.cfg,
            episodes=args.episodes,
            delta_time=args.delta_time,
            sim_length=args.sim_length,
            hidden=args.hidden,
            lr=args.lr,
            gamma=args.gamma,
            save_dir=args.save_dir,
            save_every=args.save_every,
            seed=args.seed,
            gui=args.gui,
        )
    else:
        train(
            net_file=args.net,
            route_file=args.route,
            sumo_cfg=args.cfg,
            episodes=args.episodes,
            delta_time=args.delta_time,
            sim_length=args.sim_length,
            hidden=args.hidden,
            lr=args.lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            buffer_capacity=args.buffer,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            target_update=args.target_update,
            save_dir=args.save_dir,
            save_every=args.save_every,
            seed=args.seed,
            gui=args.gui,
        )


if __name__ == "__main__":
    main()
