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

from src.ai.traffic_env import SumoTrafficEnv, OBS_DIM, ACT_DIM, remap_obs_for_old_model
from src.ai.dqn_agent import TrafficDQNAgent
from src.ai.mappo_agent import MAPPOAgent
from src.ai.masac_agent import MASACAgent


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
            loss = agent.update(n_agents=n_agents)
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
            loss = agent.update(n_agents=n_agents)
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
    entropy_coef: float = 0.0,
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

    act_dim = env.act_dim
    agent = MAPPOAgent(
        obs_dim=OBS_DIM, act_dim=act_dim, hidden=hidden, lr=lr,
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
            "num_agents": env.num_agents, "obs_dim": OBS_DIM, "act_dim": act_dim,
            "lr": lr, "gamma": gamma, "clip_eps": clip_eps,
            "ppo_epochs": ppo_epochs, "entropy_coef": entropy_coef,
            "action_mode": "continuous",
        },
        "tls_timing": tls_timing_lines,
        "episodes": [],
    }

    best_reward = -float("inf")
    best_wait = float("inf")

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
            neighbor_feats, neighbor_masks = env.get_neighbor_obs()

            actions = {}
            for tid in env.tls_ids:
                valid = env.get_valid_actions(tid)
                nf = neighbor_feats.get(tid)
                nm = neighbor_masks.get(tid)
                a, lp, v = agent.select_action(obs[tid], global_obs, valid,
                                               neighbor_feats=nf, neighbor_mask=nm)
                actions[tid] = a

                # Store transition
                mask = agent.get_valid_mask(valid)
                agent.buffer.add(obs[tid], global_obs, a, lp, 0.0, v,
                                 False, mask,
                                 neighbor_feat=nf, neighbor_mask=nm)

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
        loss_stats = agent.update(n_agents=n_agents)

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

        ep_wait = metrics.get("avg_wait_time", 9999)
        ep_vehicles = metrics.get("total_vehicles", 0)
        if ep_vehicles >= 50 and ep_wait < best_wait:
            best_wait = ep_wait
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
# Parallel episode worker (for multi-CPU training)
# ──────────────────────────────────────────────────────────────────────

def _collect_episode_worker(args):
    """Run one SUMO episode and return collected transitions.

    This runs in a separate process -- each gets its own SUMO instance.
    Returns a dict with transitions and metrics (no torch objects).
    """
    (net_file, route_file, sumo_cfg, delta_time, sim_length,
     seed, obs_dim, worker_id, agent_state_dict, hidden, act_dim,
     worker_device, baseline_active) = args

    import torch as _torch

    env = SumoTrafficEnv(
        net_file=net_file, route_file=route_file, sumo_cfg=sumo_cfg,
        delta_time=delta_time, sim_length=sim_length, gui=False, seed=seed,
    )
    env.baseline_active = baseline_active

    # Create local agent copy with shared weights
    agent = MAPPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden=hidden,
        device=worker_device,
    )
    agent.network.load_state_dict(agent_state_dict)
    agent.network.eval()

    obs, _ = env.reset(seed=seed)

    # Collect transitions
    transitions = []
    ep_rewards = {tid: 0.0 for tid in env.tls_ids}
    terminated = truncated = False

    while not (terminated or truncated):
        all_obs = [obs[tid] for tid in env.tls_ids]
        global_obs = np.mean(all_obs, axis=0).astype(np.float32)
        neighbor_feats, neighbor_masks = env.get_neighbor_obs()

        actions = {}
        step_transitions = []
        for tid in env.tls_ids:
            valid = env.get_valid_actions(tid)
            nf = neighbor_feats.get(tid)
            nm = neighbor_masks.get(tid)
            with _torch.no_grad():
                a, lp, v = agent.select_action(obs[tid], global_obs, valid,
                                               neighbor_feats=nf, neighbor_mask=nm)
            actions[tid] = a
            mask = agent.get_valid_mask(valid)
            step_transitions.append({
                "obs": obs[tid].copy(),
                "global_obs": global_obs.copy(),
                "action": a, "log_prob": lp, "value": v,
                "mask": mask.copy(),
                "tid": tid,
                "neighbor_feat": nf.copy() if nf is not None else None,
                "neighbor_mask": nm.copy() if nm is not None else None,
            })

        next_obs, rewards, terminated, truncated, info = env.step(actions)

        for i, tid in enumerate(env.tls_ids):
            step_transitions[i]["reward"] = rewards[tid]
            step_transitions[i]["done"] = terminated
            ep_rewards[tid] += rewards[tid]

        transitions.extend(step_transitions)
        obs = next_obs

    metrics = env.get_metrics()
    mean_r = float(np.mean(list(ep_rewards.values())))
    env.close()

    return {
        "transitions": transitions,
        "mean_reward": mean_r,
        "total_reward": sum(ep_rewards.values()),
        "metrics": metrics,
        "worker_id": worker_id,
        "seed": seed,
    }


def _create_scaled_routes(route_file: str, fraction: float, tag: str) -> str:
    """Create a route file with only *fraction* of the original vehicles.

    Returns the path to the new scaled route file.  Vehicles are sampled
    deterministically (every Nth) so the spatial distribution is preserved.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(route_file)
    root = tree.getroot()

    vehicles = [e for e in root if e.tag in ("vehicle", "trip", "person")]
    keep_every = max(1, round(1.0 / fraction))
    remove = [v for i, v in enumerate(vehicles) if i % keep_every != 0]
    for v in remove:
        root.remove(v)

    kept = len(vehicles) - len(remove)
    out_path = route_file.replace(".rou.xml", f".cl_{tag}.rou.xml")
    tree.write(out_path, encoding="unicode", xml_declaration=True)
    return out_path


def _save_mappo_snapshot(path: str, model_state_dict: dict, obs_dim: int, act_dim: int) -> None:
    """Persist a MAPPO policy snapshot that matches the rollout weights."""
    import torch

    torch.save({
        "model": model_state_dict,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "algorithm": "mappo",
    }, path)


def train_mappo_parallel(
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
    entropy_coef: float = 0.0,
    value_coef: float = 0.5,
    ppo_epochs: int = 10,
    mini_batch_size: int = 256,
    save_dir: str = "checkpoints",
    save_every: int = 10,
    seed: int = 42,
    gui: bool = False,
    num_workers: int = 4,
    worker_device: str = "cpu",
    curriculum: bool = False,
    resume_from: str | None = None,
    on_episode=None,
    on_status=None,
    stop_check=None,
) -> tuple[MAPPOAgent, dict]:
    """MAPPO training with parallel SUMO workers.

    Runs *num_workers* SUMO instances in parallel, each collecting
    one episode. Combined transitions feed a single PPO update.
    ~N times faster than sequential training on N CPU cores.
    """
    import multiprocessing as mp
    import torch

    def _status(msg):
        if on_status:
            on_status(msg)

    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "training_log.json")

    # Create env once to get TLS info
    _status("Initializing environment...")
    env = SumoTrafficEnv(
        net_file=net_file, route_file=route_file, sumo_cfg=sumo_cfg,
        delta_time=delta_time, sim_length=sim_length, gui=False, seed=seed,
    )
    tls_ids = list(env.tls_ids)
    n_agents = len(tls_ids)
    act_dim = env.act_dim
    _status(f"Environment: {n_agents} TLS agents, {num_workers} parallel workers")

    # Print TLS timing
    for tid in tls_ids:
        tls_info = env._tls_map.get(tid)
        g = tls_info.geometry if tls_info else None
        if g:
            _status(
                f"  {tid[:40]:<42s} [{g.tier[:3].upper()}] "
                f"G={g.min_green_s:.0f}-{g.max_green_s:.0f}s Y={g.yellow_s:.1f}s R={g.allred_s:.1f}s "
                f"W={g.width_m:.0f}m L={g.total_lanes}"
            )
    env.close()

    # Create agent
    agent = MAPPOAgent(
        obs_dim=OBS_DIM, act_dim=act_dim, hidden=hidden, lr=lr,
        gamma=gamma, gae_lambda=gae_lambda, clip_eps=clip_eps,
        entropy_coef=entropy_coef, value_coef=value_coef,
        ppo_epochs=ppo_epochs, mini_batch_size=mini_batch_size,
    )
    if resume_from and os.path.isfile(resume_from):
        import torch as _torch
        ckpt = _torch.load(resume_from, map_location="cpu", weights_only=True)
        full_sd = ckpt.get("model", {})
        # Load only actor weights — critic starts fresh to avoid value miscalibration
        # when reward function differs from checkpoint's training reward.
        actor_sd = {k: v for k, v in full_sd.items()
                    if k.startswith("actor")}
        if actor_sd:
            missing, unexpected = agent.network.load_state_dict(actor_sd, strict=False)
            _status(f"Resumed actor weights from {resume_from} "
                    f"(critic starts fresh; missing={len(missing)} keys)")
        else:
            _status(f"No actor keys found in {resume_from}, starting fresh")
    _status(f"MAPPO agent ready, obs_dim={OBS_DIM}, act_dim={act_dim}")

    log = {
        "config": {
            "algorithm": "mappo_parallel", "episodes": episodes,
            "num_agents": n_agents, "num_workers": num_workers,
            "obs_dim": OBS_DIM, "act_dim": act_dim,
            "action_mode": "continuous",
            "curriculum": curriculum,
            "worker_device": worker_device,
        },
        "episodes": [],
    }

    best_reward = -float("inf")
    best_wait = float("inf")
    best_throughput = -float("inf")
    ep_count = 0

    # Resolve absolute paths for workers
    abs_net = os.path.abspath(net_file)
    abs_route = os.path.abspath(route_file)
    abs_cfg = os.path.abspath(sumo_cfg) if sumo_cfg else None

    # ── Curriculum Learning ──────────────────────────────────────────
    # Phase 1 (0-20%):  33% traffic -> learn basic duration selection
    # Phase 2 (20-50%): 66% traffic -> learn coordination
    # Phase 3 (50%+):   100% traffic -> handle full demand
    if curriculum:
        cl_phases = [
            (0.20, 0.33, "phase1"),
            (0.50, 0.66, "phase2"),
            (1.00, 1.00, "phase3"),
        ]
        _status("Curriculum Learning enabled:")
        cl_routes = {}
        for _, frac, tag in cl_phases:
            if frac < 1.0:
                scaled = _create_scaled_routes(abs_route, frac, tag)
                veh_count = int(3000 * frac)
                cl_routes[tag] = os.path.abspath(scaled)
                _status(f"  {tag}: {frac:.0%} traffic (~{veh_count} vehicles)")
            else:
                cl_routes[tag] = abs_route
                _status(f"  {tag}: 100% traffic (full demand)")
    else:
        cl_phases = [(1.0, 1.0, "full")]
        cl_routes = {"full": abs_route}

    current_cl_phase = None

    while ep_count < episodes:
        if stop_check and stop_check():
            break

        t0 = time.time()
        batch_size = min(num_workers, episodes - ep_count)

        # Determine curriculum phase route file
        progress = ep_count / max(episodes, 1)
        active_route = abs_route
        for threshold, frac, tag in cl_phases:
            if progress < threshold:
                active_route = cl_routes[tag]
                if tag != current_cl_phase:
                    current_cl_phase = tag
                    _status(f"  >> Curriculum: {tag} ({frac:.0%} traffic)")
                break

        # Baseline bonus only active during full-traffic phase
        is_full_traffic = (active_route == abs_route)

        # Prepare worker args
        state_dict = {k: v.cpu() for k, v in agent.network.state_dict().items()}
        worker_args = []
        for w in range(batch_size):
            ep_seed = seed + ep_count + w + 1
            worker_args.append((
                abs_net, active_route, abs_cfg, delta_time, sim_length,
                ep_seed, OBS_DIM, w, state_dict, hidden, act_dim,
                worker_device,
                is_full_traffic,
            ))

        # Run workers in parallel
        _status(f"Ep {ep_count+1}-{ep_count+batch_size}/{episodes}: collecting {batch_size} episodes...")
        with mp.Pool(batch_size) as pool:
            results = pool.map(_collect_episode_worker, worker_args)

        # Load all transitions into agent buffer
        agent.buffer.clear()
        batch_rewards = []
        batch_metrics = []

        for result in results:
            for t in result["transitions"]:
                agent.buffer.add(
                    t["obs"], t["global_obs"],
                    t["action"], t["log_prob"], t["reward"],
                    t["value"], t["done"], t["mask"],
                    neighbor_feat=t.get("neighbor_feat"),
                    neighbor_mask=t.get("neighbor_mask"),
                )
            batch_rewards.append(result["mean_reward"])
            batch_metrics.append(result["metrics"])

        batch_mean_wait = float(np.mean([
            m.get("avg_wait_time", 0.0) for m in batch_metrics
        ]))
        batch_mean_throughput = float(np.mean([
            m.get("throughput", 0.0) for m in batch_metrics
        ]))

        # Save the policy that actually generated this rollout batch.
        # The older flow saved post-update weights using pre-update rollout
        # metrics, which could label the wrong checkpoint as "best".
        if (
            batch_mean_wait < best_wait
            or (
                np.isclose(batch_mean_wait, best_wait)
                and batch_mean_throughput > best_throughput
            )
        ):
            best_wait = batch_mean_wait
            best_throughput = batch_mean_throughput
            _save_mappo_snapshot(
                os.path.join(save_dir, "best_model.pt"),
                state_dict,
                OBS_DIM,
                act_dim,
            )
            _status(
                "  >> New best rollout policy: "
                f"mean wait={batch_mean_wait:.1f}s "
                f"tp={batch_mean_throughput:.1f} "
                f"(eps {ep_count + 1}-{ep_count + batch_size})"
            )

        # PPO update on combined data
        loss_stats = agent.update(n_agents=n_agents)

        elapsed = time.time() - t0

        # Log each episode from this batch
        for i, result in enumerate(results):
            ep_count += 1
            mean_r = result["mean_reward"]
            metrics = result["metrics"]

            ep_log = {
                "episode": ep_count, "total_episodes": episodes,
                "mean_reward": round(mean_r, 4),
                "total_reward": round(result["total_reward"], 4),
                "mean_loss": round(loss_stats["total_loss"], 6),
                "entropy": round(loss_stats.get("entropy", 0.0), 4),
                "epsilon": 0.0,
                "steps": len(result["transitions"]) // n_agents,
                "time_s": round(elapsed / batch_size, 1),
                "avg_wait": metrics.get("avg_wait_time", 0),
                "avg_queue": metrics.get("avg_queue_length", 0),
                "vehicles": metrics.get("total_vehicles", 0),
                "vehicles_end": metrics.get("total_vehicles", 0),
                "throughput": metrics.get("throughput", 0),
                "collisions": 0,
                "best_reward": round(max(best_reward, mean_r), 4),
                "algorithm": "mappo_parallel",
            }
            log["episodes"].append(ep_log)

            if mean_r > best_reward:
                best_reward = mean_r

            if on_episode:
                on_episode(ep_log)

        # Save periodically
        if ep_count % save_every < batch_size:
            agent.save(os.path.join(save_dir, f"model_ep{ep_count}.pt"))
            with open(log_path, "w") as f:
                json.dump(log, f, indent=2)

        avg_r = np.mean(batch_rewards)
        _status(f"Batch done: {batch_size} eps in {elapsed:.0f}s, avg R={avg_r:.3f}")

    agent.save(os.path.join(save_dir, "final_model.pt"))
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    _status(f"Parallel MAPPO complete! {ep_count} episodes, best R={best_reward:.4f}")
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

            loss = agent.update(n_agents=n_agents)
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


def evaluate_mappo(
    agent: MAPPOAgent,
    net_file: str,
    route_file: str,
    sumo_cfg: str | None = None,
    episodes: int = 5,
    delta_time: int = 10,
    sim_length: int = 3600,
    gui: bool = False,
    seed: int = 1000,
) -> dict:
    """Run *episodes* with a greedy MAPPO policy."""
    env = SumoTrafficEnv(
        net_file=net_file, route_file=route_file, sumo_cfg=sumo_cfg,
        delta_time=delta_time, sim_length=sim_length, gui=gui, seed=seed,
    )
    results: list[dict] = []
    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        global_obs = np.mean(list(obs.values()), axis=0) if obs else np.zeros(OBS_DIM)
        ep_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            global_obs = np.mean(list(obs.values()), axis=0)
            actions = {
                tid: agent.select_action(obs[tid], global_obs, greedy=True)[0]
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
# MASAC Training
# ──────────────────────────────────────────────────────────────────────

def train_masac(
    net_file: str,
    route_file: str,
    sumo_cfg: str | None = None,
    episodes: int = 1000,
    delta_time: int = 30,
    sim_length: int = 3600,
    hidden: int = 256,
    lr: float = 3e-4,
    gamma: float = 0.99,
    tau: float = 0.005,
    buffer_capacity: int = 500_000,
    batch_size: int = 256,
    warmup_steps: int = 1000,
    updates_per_step: int = 1,
    save_dir: str = "checkpoints",
    save_every: int = 10,
    seed: int = 42,
    gui: bool = False,
    on_episode=None,
    on_status=None,
    stop_check=None,
) -> tuple[MASACAgent, dict]:
    """MASAC training loop — off-policy, continuous actions, replay buffer."""

    def _status(msg):
        if on_status:
            on_status(msg)
        else:
            print(msg)

    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "training_log.json")

    _status("Creating environment...")
    env = SumoTrafficEnv(
        net_file=net_file, route_file=route_file, sumo_cfg=sumo_cfg,
        delta_time=delta_time, sim_length=sim_length, gui=gui, seed=seed,
    )
    n_agents = env.num_agents
    _status(f"Environment ready: {n_agents} TLS agents")

    agent = MASACAgent(
        obs_dim=OBS_DIM, hidden=hidden,
        lr_actor=lr, lr_critic=lr, lr_alpha=lr,
        gamma=gamma, tau=tau,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        updates_per_step=updates_per_step,
    )
    _status(f"MASAC agent ready | buffer={buffer_capacity} | warmup={warmup_steps}")

    log = {
        "config": {
            "algorithm": "masac", "episodes": episodes,
            "num_agents": n_agents, "obs_dim": OBS_DIM,
            "action_mode": "continuous", "delta_time": delta_time,
        },
        "episodes": [],
    }

    best_reward = -float("inf")
    best_wait = float("inf")

    for ep in range(1, episodes + 1):
        if stop_check and stop_check():
            break

        t0 = time.time()
        obs, _ = env.reset(seed=seed + ep)
        terminated = truncated = False
        ep_rewards = {tid: 0.0 for tid in env.tls_ids}
        step_count = 0

        while not (terminated or truncated):
            all_obs = [obs[tid] for tid in env.tls_ids]
            global_obs = np.mean(all_obs, axis=0).astype(np.float32)

            actions = {
                tid: agent.select_action(obs[tid], global_obs)
                for tid in env.tls_ids
            }

            next_obs, rewards, terminated, truncated, _ = env.step(actions)

            next_all = [next_obs[tid] for tid in env.tls_ids]
            next_global = np.mean(next_all, axis=0).astype(np.float32)

            for tid in env.tls_ids:
                agent.store(
                    obs[tid], global_obs, actions[tid], rewards[tid],
                    next_obs[tid], next_global, terminated,
                )
                ep_rewards[tid] += rewards[tid]

            # Update after every step (off-policy)
            agent.update(n_agents=n_agents)

            obs = next_obs
            step_count += 1

        elapsed = time.time() - t0
        mean_r = float(np.mean(list(ep_rewards.values())))
        metrics = env.get_metrics()

        avg_wait = metrics.get("avg_wait_time", 9999)
        vehicles = metrics.get("total_vehicles", 0)
        ep_log = {
            "episode": ep, "total_episodes": episodes,
            "mean_reward": round(mean_r, 4),
            "total_reward": round(sum(ep_rewards.values()), 4),
            "mean_loss": 0.0,
            "entropy": round(agent.alpha, 4),
            "epsilon": 0.0,
            "steps": step_count,
            "time_s": round(elapsed, 1),
            "avg_wait": avg_wait,
            "avg_queue": metrics.get("avg_queue_length", 0),
            "vehicles": vehicles,
            "collisions": 0,
            "best_reward": round(max(best_reward, mean_r), 4),
            "algorithm": "masac",
        }
        log["episodes"].append(ep_log)

        if mean_r > best_reward:
            best_reward = mean_r

        # Save best model by wait_time (only when enough vehicles to be meaningful)
        if vehicles >= 50 and avg_wait < best_wait:
            best_wait = avg_wait
            agent.save(os.path.join(save_dir, "best_model.pt"))
            _status(f"  >> New best model: wait={avg_wait:.2f}s (ep {ep})")

        if on_episode:
            on_episode(ep_log)

        _status(
            f"Ep {ep}/{episodes} | R={mean_r:.3f} | "
            f"wait={metrics.get('avg_wait_time',0):.1f}s | "
            f"alpha={agent.alpha:.3f} | buf={len(agent.buffer)} | "
            f"t={elapsed:.0f}s"
        )

        if ep % save_every == 0:
            agent.save(os.path.join(save_dir, f"model_ep{ep}.pt"))
            with open(log_path, "w") as f:
                json.dump(log, f, indent=2)

    env.close()
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    return agent, log


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
    ap.add_argument("--algorithm", choices=["dqn", "mappo", "masac"], default="dqn",
                    help="RL algorithm: dqn, mappo, or masac (default: dqn)")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel SUMO workers for MAPPO (default: 1, use 2-8 for speed)")
    ap.add_argument("--worker-device", choices=["cpu", "cuda"], default="cpu",
                    help="Device used by parallel rollout workers (default: cpu for stability)")
    ap.add_argument("--entropy-coef", type=float, default=0.0,
                    help="PPO entropy coefficient (default: 0.0 for continuous actions)")
    ap.add_argument("--resume-from", default=None,
                    help="Path to checkpoint to resume training from")
    ap.add_argument("--curriculum", action="store_true",
                    help="Enable staged traffic curriculum for MAPPO parallel training")

    args = ap.parse_args()

    if args.eval_only:
        import torch as _torch
        ckpt = _torch.load(args.eval_only, map_location="cpu", weights_only=True)
        algo = ckpt.get("algorithm", "dqn")
        if algo in ("mappo", "mappo_parallel"):
            agent = MAPPOAgent(OBS_DIM, ACT_DIM, args.hidden)
            agent.load(args.eval_only)
            results = evaluate_mappo(
                agent,
                args.net,
                args.route,
                args.cfg,
                episodes=args.episodes,
                delta_time=args.delta_time,
                sim_length=args.sim_length,
                gui=args.gui,
                seed=args.seed,
            )
        else:
            agent = TrafficDQNAgent(OBS_DIM, ACT_DIM, args.hidden)
            agent.load(args.eval_only)
            results = evaluate(
                agent,
                args.net,
                args.route,
                args.cfg,
                episodes=args.episodes,
                delta_time=args.delta_time,
                sim_length=args.sim_length,
                gui=args.gui,
                seed=args.seed,
            )
        print(f"\n  Results: {json.dumps(results, indent=2)}")
    elif args.algorithm == "mappo":
        if args.workers > 1:
            train_mappo_parallel(
                net_file=args.net,
                route_file=args.route,
                sumo_cfg=args.cfg,
                episodes=args.episodes,
                delta_time=args.delta_time,
                sim_length=args.sim_length,
                hidden=args.hidden,
                lr=args.lr,
                gamma=args.gamma,
                entropy_coef=args.entropy_coef,
                save_dir=args.save_dir,
                save_every=args.save_every,
                seed=args.seed,
                num_workers=args.workers,
                worker_device=args.worker_device,
                curriculum=args.curriculum,
                resume_from=args.resume_from,
            )
        else:
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
                entropy_coef=args.entropy_coef,
                save_dir=args.save_dir,
                save_every=args.save_every,
                seed=args.seed,
                gui=args.gui,
            )
    elif args.algorithm == "masac":
        train_masac(
            net_file=args.net,
            route_file=args.route,
            sumo_cfg=args.cfg,
            episodes=args.episodes,
            delta_time=args.delta_time,
            sim_length=args.sim_length,
            hidden=args.hidden,
            lr=args.lr,
            gamma=args.gamma,
            buffer_capacity=args.buffer,
            batch_size=args.batch_size,
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
