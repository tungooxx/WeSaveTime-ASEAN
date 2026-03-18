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

import traci

from src.ai.traffic_env import (
    SumoTrafficEnv, OBS_DIM, OLD_OBS_DIM, ACT_DIM,
    remap_obs_for_old_model,
)
from src.ai.dqn_agent import TrafficDQNAgent


def run_baseline(net_file, route_file, sumo_cfg, sim_length=3600,
                 seed=1000, episodes=3):
    """Run SUMO with default timing (no AI), collect metrics per episode."""
    import shutil
    binary = shutil.which("sumo")
    if not binary:
        sumo_home = os.environ.get("SUMO_HOME", "")
        binary = os.path.join(sumo_home, "bin", "sumo") if sumo_home else "sumo"

    results = []
    for ep in range(1, episodes + 1):
        label = f"baseline_{ep}"
        try:
            traci.getConnection(label).close()
        except Exception:
            pass

        cmd = [binary, "-c", sumo_cfg,
               "--seed", str(seed + ep),
               "--no-step-log", "true",
               "--no-warnings", "true",
               "--collision.action", "warn",
               "--collision.check-junctions", "true",
               "--time-to-teleport", "300",
               "--step-length", "1"]
        traci.start(cmd, label=label)
        conn = traci.getConnection(label)

        import traci.constants as tc

        # Use edge-level calls in batch (no per-vehicle loops)
        edge_ids = [e for e in conn.edge.getIDList() if not e.startswith(":")]

        wait_samples = []
        queue_samples = []
        total_collisions = 0
        total_throughput = 0
        t0 = time.time()

        # Step through simulation, sampling every 200 steps (fast)
        for step in range(sim_length):
            conn.simulationStep()

            if step % 200 == 0 and step > 0:
                try:
                    total_collisions += conn.simulation.getCollidingVehiclesNumber()
                    total_throughput += conn.simulation.getArrivedNumber()
                except Exception:
                    pass

                step_wait = 0.0
                step_queue = 0
                count = 0
                for eid in edge_ids[:500]:  # sample up to 500 edges
                    try:
                        step_queue += conn.edge.getLastStepHaltingNumber(eid)
                        step_wait += conn.edge.getWaitingTime(eid)
                        count += 1
                    except Exception:
                        pass
                n = max(count, 1)
                wait_samples.append(step_wait / n)
                queue_samples.append(step_queue / n)

        elapsed = time.time() - t0
        try:
            final_veh = conn.vehicle.getIDCount()
        except Exception:
            final_veh = 0
        conn.close()

        results.append({
            "episode": ep,
            "avg_wait": round(np.mean(wait_samples) if wait_samples else 0, 1),
            "avg_queue": round(np.mean(queue_samples) if queue_samples else 0, 1),
            "collisions": total_collisions,
            "throughput": total_throughput,
            "vehicles_end": final_veh,
            "time_s": round(elapsed, 1),
        })

    return results


def run_model(model_path, net_file, route_file, sumo_cfg, hidden=256,
              sim_length=3600, delta_time=10, seed=1000, episodes=3):
    """Run trained model greedily, collect metrics per episode."""
    import torch
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    ckpt_obs_dim = ckpt.get("obs_dim", OBS_DIM)

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

        while not (terminated or truncated):
            actions = {}
            for tid in env.tls_ids:
                valid = env.get_valid_actions(tid)
                o = remap_obs_for_old_model(obs[tid]) if needs_remap else obs[tid]
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
            "collisions": metrics.get("collisions", 0),
            "throughput": 0,
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
    b_col = sum(r["collisions"] for r in baseline_results)
    m_col = sum(r["collisions"] for r in model_results)
    b_tp = sum(r["throughput"] for r in baseline_results)
    m_tp = sum(r.get("throughput", 0) for r in model_results)

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
    row("Total Collisions", b_col, m_col, fmt="d")
    if b_tp > 0:
        row("Throughput (arrived)", b_tp, m_tp, fmt="d", lower_better=False)

    print()
    wait_imp = (b_wait - m_wait) / max(b_wait, 0.1) * 100
    queue_imp = (b_queue - m_queue) / max(b_queue, 0.1) * 100
    col_imp = (b_col - m_col) / max(b_col, 1) * 100

    print(f"  SUMMARY:")
    print(f"    Wait time  : {wait_imp:+.1f}% {'(improved)' if wait_imp > 0 else '(worse)'}")
    print(f"    Queue      : {queue_imp:+.1f}% {'(improved)' if queue_imp > 0 else '(worse)'}")
    print(f"    Collisions : {col_imp:+.1f}% {'(safer)' if col_imp > 0 else '(more dangerous)'}")
    print("=" * 65)

    return {
        "baseline": {"avg_wait": b_wait, "avg_queue": b_queue,
                     "collisions": b_col, "throughput": b_tp},
        "model": {"avg_wait": m_wait, "avg_queue": m_queue,
                  "collisions": m_col, "throughput": m_tp},
        "improvement": {"wait_pct": round(wait_imp, 1),
                        "queue_pct": round(queue_imp, 1),
                        "collision_pct": round(col_imp, 1)},
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
    ap.add_argument("--sim-length", type=int, default=3600)
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
              f"queue={r['avg_queue']:.1f} col={r['collisions']} "
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
              f"queue={r['avg_queue']:.1f} col={r['collisions']} "
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
