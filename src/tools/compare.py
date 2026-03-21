"""
FlowMind AI - Compare baseline (default timing) vs trained model.

Runs two simulations side by side:
  1. Baseline: SUMO default traffic light timing (no AI)
  2. AI Model: trained best_model.pt controlling all TLS

Outputs a comparison table with wait time, queue, throughput, collisions.

Usage:
    python -m src.tools.compare
    python -m src.tools.compare --model checkpoints/best_model.pt --episodes 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.ai.traffic_env import (
    SumoTrafficEnv, OBS_DIM, OLD_OBS_DIM, ACT_DIM,
    remap_obs_for_old_model,
)
from src.ai.dqn_agent import TrafficDQNAgent
from src.ai.mappo_agent import MAPPOAgent


def run_baseline(net_file, route_file, sumo_cfg, sim_length=1800,
                 delta_time=30, seed=1000, episodes=3):
    """Run the SAME SumoTrafficEnv as the AI but with a 'do nothing' policy.

    Every step, the agent just keeps the current phase (action 0 = first
    green phase for each TLS). This means the env does the same warm-up,
    same roundabout handling, same metric collection — only difference is
    no intelligent phase switching.
    """
    env = SumoTrafficEnv(
        net_file=net_file, route_file=route_file, sumo_cfg=sumo_cfg,
        delta_time=delta_time, sim_length=sim_length, gui=False, seed=seed,
    )

    results = []
    for ep in range(1, episodes + 1):
        t0 = time.time()
        obs, _ = env.reset(seed=seed + ep)
        terminated = truncated = False

        while not (terminated or truncated):
            # "Do nothing" — keep current phase for every TLS
            actions = {}
            for tid in env.tls_ids:
                # Action 0 = keep first green phase (no switching)
                green_phases = env._green_phases.get(tid, [0])
                current = env._current_phases.get(tid, green_phases[0])
                # Find action index that maps to current phase
                action = 0
                for ai, gp in enumerate(green_phases):
                    if gp == current:
                        action = ai
                        break
                actions[tid] = action
            obs, rewards, terminated, truncated, _ = env.step(actions)

        elapsed = time.time() - t0
        metrics = env.get_metrics()
        results.append({
            "episode": ep,
            "avg_wait": metrics.get("avg_wait_time", 0),
            "avg_queue": metrics.get("avg_queue_length", 0),
            "throughput": metrics.get("throughput", 0),
            "vehicles_end": metrics.get("total_vehicles", 0),
            "time_s": round(elapsed, 1),
        })

    env.close()
    return results


def run_model(model_path, net_file, route_file, sumo_cfg, hidden=256,
              sim_length=3600, delta_time=30, seed=1000, episodes=3):
    """Run trained model greedily, collect metrics per episode."""
    import torch
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    ckpt_obs_dim = ckpt.get("obs_dim", OBS_DIM)
    algorithm = ckpt.get("algorithm", "dqn")

    if algorithm == "mappo":
        agent = MAPPOAgent(ckpt_obs_dim, ACT_DIM, hidden)
    else:
        agent = TrafficDQNAgent(ckpt_obs_dim, ACT_DIM, hidden)
    agent.load(model_path)

    env = SumoTrafficEnv(
        net_file=net_file, route_file=route_file, sumo_cfg=sumo_cfg,
        delta_time=delta_time, sim_length=sim_length, gui=False, seed=seed,
    )

    needs_remap = (ckpt_obs_dim == OLD_OBS_DIM and OBS_DIM != OLD_OBS_DIM)
    results = []

    for ep in range(1, episodes + 1):
        t0 = time.time()
        obs, _ = env.reset(seed=seed + ep)
        terminated = truncated = False
        total_reward = 0.0
        total_throughput = 0

        while not (terminated or truncated):
            actions = {}
            if algorithm == "mappo":
                global_obs = np.mean(
                    [obs[tid] for tid in env.tls_ids], axis=0
                ).astype(np.float32)
            for tid in env.tls_ids:
                valid = env.get_valid_actions(tid)
                o = remap_obs_for_old_model(obs[tid]) if needs_remap else obs[tid]
                if algorithm == "mappo":
                    a, _, _ = agent.select_action(o, global_obs, valid, greedy=True)
                else:
                    a = agent.select_action(o, valid, greedy=True)
                actions[tid] = a
            obs, rewards, terminated, truncated, _ = env.step(actions)
            total_reward += sum(rewards.values())

        elapsed = time.time() - t0
        metrics = env.get_metrics()
        results.append({
            "episode": ep,
            "avg_wait": metrics.get("avg_wait_time", 0),
            "avg_queue": metrics.get("avg_queue_length", 0),
            "throughput": metrics.get("throughput", 0),
            "vehicles_end": metrics.get("total_vehicles", 0),
            "total_reward": round(total_reward, 2),
            "time_s": round(elapsed, 1),
        })

    env.close()
    return results


def compare(baseline_results, model_results):
    """Print comparison table."""
    def avg(lst, key):
        return sum(r[key] for r in lst) / max(len(lst), 1)

    b_wait = avg(baseline_results, "avg_wait")
    m_wait = avg(model_results, "avg_wait")
    b_queue = avg(baseline_results, "avg_queue")
    m_queue = avg(model_results, "avg_queue")
    b_tp = avg(baseline_results, "throughput")
    m_tp = avg(model_results, "throughput")

    print()
    print("=" * 65)
    print("  FlowMind AI - Baseline vs Trained Model Comparison")
    print("=" * 65)
    print()
    print(f"  {'Metric':<25s} {'Baseline':>15s} {'AI Model':>15s} {'Change':>12s}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*12}")

    def row(label, base, model, fmt=".1f", lower_better=True):
        diff = model - base
        if base > 0:
            pct = diff / base * 100
        else:
            pct = 0
        arrow = "v" if (diff < 0 and lower_better) or (diff > 0 and not lower_better) else "^"
        color_sign = "+" if diff > 0 else ""
        change = f"{color_sign}{pct:.0f}% {arrow}"
        print(f"  {label:<25s} {base:>15{fmt}} {model:>15{fmt}} {change:>12s}")

    row("Avg Wait Time (s)", b_wait, m_wait)
    row("Avg Queue Length", b_queue, m_queue)
    if b_tp > 0:
        row("Throughput (arrived)", b_tp, m_tp, fmt=".0f", lower_better=False)

    print()
    wait_imp = (b_wait - m_wait) / max(b_wait, 0.1) * 100
    queue_imp = (b_queue - m_queue) / max(b_queue, 0.1) * 100
    tp_imp = (m_tp - b_tp) / max(b_tp, 1) * 100

    print(f"  SUMMARY:")
    print(f"    Wait time  : {wait_imp:+.1f}% {'(improved)' if wait_imp > 0 else '(worse)'}")
    print(f"    Queue      : {queue_imp:+.1f}% {'(improved)' if queue_imp > 0 else '(worse)'}")
    print(f"    Throughput : {tp_imp:+.1f}% {'(improved)' if tp_imp > 0 else '(worse)'}")
    print("=" * 65)

    return {
        "baseline": {"avg_wait": b_wait, "avg_queue": b_queue,
                     "throughput": b_tp},
        "model": {"avg_wait": m_wait, "avg_queue": m_queue,
                  "throughput": m_tp},
        "improvement": {"wait_pct": round(wait_imp, 1),
                        "queue_pct": round(queue_imp, 1),
                        "throughput_pct": round(tp_imp, 1)},
    }


def main():
    ap = argparse.ArgumentParser(
        description="FlowMind AI - Compare baseline vs trained model")
    ap.add_argument("--model", default=os.path.join(
        _PROJECT_ROOT, "checkpoints", "best_model.pt"))
    ap.add_argument("--net", default=os.path.join(
        _PROJECT_ROOT, "sumo", "danang", "danang.net.xml"))
    ap.add_argument("--route", default=os.path.join(
        _PROJECT_ROOT, "sumo", "danang", "danang.rou.xml"))
    ap.add_argument("--cfg", default=os.path.join(
        _PROJECT_ROOT, "sumo", "danang", "danang.sumocfg"))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--sim-length", type=int, default=1800,
                        help="Sim steps (1800 = 900 real seconds at step_length=0.5)")
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        print(f"Model not found: {args.model}")
        sys.exit(1)

    print("=" * 65)
    print("  FlowMind AI - Running Comparison")
    print("=" * 65)

    print(f"\n  [1/2] Running BASELINE (default timing, {args.episodes} episodes)...")
    t0 = time.time()
    baseline = run_baseline(
        args.net, args.route, args.cfg,
        sim_length=args.sim_length, seed=args.seed, episodes=args.episodes)
    for r in baseline:
        print(f"    Ep {r['episode']}: wait={r['avg_wait']:.1f}s "
              f"queue={r['avg_queue']:.1f} tp={r['throughput']} "
              f"({r['time_s']:.0f}s)")
    print(f"    Total: {time.time()-t0:.0f}s")

    print(f"\n  [2/2] Running AI MODEL ({args.episodes} episodes)...")
    t0 = time.time()
    model = run_model(
        args.model, args.net, args.route, args.cfg,
        hidden=args.hidden, sim_length=args.sim_length,
        seed=args.seed, episodes=args.episodes)
    for r in model:
        print(f"    Ep {r['episode']}: wait={r['avg_wait']:.1f}s "
              f"queue={r['avg_queue']:.1f} tp={r['throughput']} "
              f"reward={r.get('total_reward', 0):.1f} ({r['time_s']:.0f}s)")
    print(f"    Total: {time.time()-t0:.0f}s")

    result = compare(baseline, model)

    # Save results
    import json
    out_path = os.path.join(_PROJECT_ROOT, "checkpoints", "comparison.json")
    with open(out_path, "w") as f:
        json.dump({"baseline": baseline, "model": model, **result}, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
