"""
FlowMind AI - Analyze which intersections need traffic lights.

Two analyses:
  1. Uncontrolled junctions — find busy intersections without any TLS
     and recommend where to add new traffic lights.
  2. Existing TLS — after training, identify which TLS the AI learned
     to turn off (action=7), meaning they work better without signals.

Usage:
    python -m src.ai.analyze_junctions \
        --net sumo/danang/danang.net.xml \
        --route sumo/danang/danang.rou.xml \
        --cfg sumo/danang/danang.sumocfg \
        --sample-steps 500

    With trained model (also shows which existing TLS should be removed):
    python -m src.ai.analyze_junctions \
        --net sumo/danang/danang.net.xml \
        --route sumo/danang/danang.rou.xml \
        --cfg sumo/danang/danang.sumocfg \
        --model checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import traci
from src.simulation.tls_metadata import (
    TLSMetadata,
    discover_uncontrolled_junctions,
    analyze_junctions_with_traci,
)


def _find_sumo() -> str:
    found = shutil.which("sumo")
    if found:
        return found
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        return os.path.join(sumo_home, "bin", "sumo")
    raise FileNotFoundError("Cannot find 'sumo'")


def analyze_need_for_tls(
    net_file: str,
    route_file: str,
    sumo_cfg: str | None = None,
    sample_steps: int = 500,
    min_incoming: int = 3,
    seed: int = 42,
    top_n: int = 30,
) -> dict:
    """Analyze which uncontrolled junctions need traffic lights."""

    print("=" * 60)
    print("  FlowMind AI - Junction Analysis")
    print("=" * 60)

    # 1. Discover uncontrolled junctions from the network file
    print(f"\n  Scanning network: {net_file}")
    all_junctions = discover_uncontrolled_junctions(net_file, min_incoming)
    print(f"  Found {len(all_junctions)} uncontrolled junctions with >= {min_incoming} incoming roads")

    # Only analyze top candidates (sorted by incoming edges/lanes already)
    max_analyze = min(200, len(all_junctions))
    junctions = all_junctions[:max_analyze]
    print(f"  Analyzing top {max_analyze} busiest junctions...")

    # 2. Also show existing TLS stats
    meta = TLSMetadata(net_file)
    print(f"  Existing TLS: {len(meta)} total, {len(meta.get_non_trivial())} non-trivial")

    if not junctions:
        print("\n  No uncontrolled junctions found to analyze.")
        return {"candidates": [], "existing_tls": len(meta)}

    # 3. Start SUMO and measure congestion at uncontrolled junctions
    print(f"\n  Starting SUMO simulation ({sample_steps} steps)...")

    # Close any stale connection
    try:
        traci.getConnection("analysis").close()
    except (traci.TraCIException, KeyError):
        pass

    binary = _find_sumo()
    if sumo_cfg:
        cmd = [binary, "-c", os.path.abspath(sumo_cfg)]
    else:
        cmd = [binary, "-n", os.path.abspath(net_file), "-r", os.path.abspath(route_file)]
    cmd += [
        "--seed", str(seed),
        "--no-step-log", "true",
        "--no-warnings", "true",
        "--error-log", os.devnull,
    ]

    traci.start(cmd, label="analysis")
    conn = traci.getConnection("analysis")

    # Warm up — need enough time for vehicles to spawn and reach junctions
    # Da Nang cfg uses step-length=0.5, so 2000 steps = 1000s of sim time
    warmup = 2000
    print(f"  Warming up ({warmup} steps)...")
    for _ in range(warmup):
        conn.simulationStep()
    veh_count = conn.vehicle.getIDCount()
    print(f"  Vehicles in network after warmup: {veh_count}")

    # Analyze
    results = analyze_junctions_with_traci(conn, junctions, sample_steps)
    conn.close()

    # 4. Print results
    print(f"\n  Top {min(top_n, len(results))} junctions that may need traffic lights:")
    print(f"  {'Rank':>4}  {'Junction ID':>15}  {'Type':>18}  {'Roads':>5}  "
          f"{'Lanes':>5}  {'AvgWait':>8}  {'AvgQueue':>8}  {'Score':>6}  Recommendation")
    print("  " + "-" * 110)

    for i, r in enumerate(results[:top_n]):
        print(
            f"  {i+1:4d}  {r['junction_id']:>15}  {r['junction_type']:>18}  "
            f"{r['incoming_edges']:5d}  {r['incoming_lanes']:5d}  "
            f"{r['avg_wait']:8.1f}  {r['avg_queue']:8.1f}  "
            f"{r['congestion_score']:6.3f}  {r['recommendation']}"
        )

    # Count by recommendation
    strong = sum(1 for r in results if "STRONGLY" in r["recommendation"])
    consider = sum(1 for r in results if "Consider" in r["recommendation"])
    monitor = sum(1 for r in results if "Monitor" in r["recommendation"])
    print(f"\n  Summary:")
    print(f"    STRONGLY recommended: {strong}")
    print(f"    Consider adding TLS:  {consider}")
    print(f"    Monitor:              {monitor}")
    print(f"    No TLS needed:        {len(results) - strong - consider - monitor}")

    return {
        "candidates": results[:top_n],
        "total_uncontrolled": len(junctions),
        "strong_recommend": strong,
        "consider": consider,
        "existing_tls": len(meta),
    }


def analyze_tls_removal(
    net_file: str,
    route_file: str,
    model_path: str,
    sumo_cfg: str | None = None,
    sim_length: int = 3600,
    delta_time: int = 10,
    seed: int = 42,
) -> list[dict]:
    """Run a trained model and identify which existing TLS it consistently turns off.

    If the AI learned that action=7 (TLS off) is best for certain intersections,
    those intersections might be better off without traffic signals.
    """
    from src.ai.traffic_env import SumoTrafficEnv, OBS_DIM, ACT_DIM, ACT_OFF
    from src.ai.dqn_agent import TrafficDQNAgent

    print("\n  Analyzing which existing TLS the AI recommends removing...")

    agent = TrafficDQNAgent(OBS_DIM, ACT_DIM, hidden=256)
    agent.load(model_path)
    print(f"  Loaded model: {model_path}")

    env = SumoTrafficEnv(
        net_file=net_file,
        route_file=route_file,
        sumo_cfg=sumo_cfg,
        delta_time=delta_time,
        sim_length=sim_length,
        seed=seed,
    )

    obs, _ = env.reset(seed=seed)

    # Count how often each TLS gets the "off" action
    off_counts: dict[str, int] = {tid: 0 for tid in env.tls_ids}
    total_steps = 0
    terminated = truncated = False

    while not (terminated or truncated):
        actions: dict[str, int] = {}
        for tid in env.tls_ids:
            valid = env.get_valid_actions(tid)
            action = agent.select_action(obs[tid], valid, greedy=True)
            actions[tid] = action
            if action == ACT_OFF:
                off_counts[tid] += 1

        obs, _, terminated, truncated, _ = env.step(actions)
        total_steps += 1

    env.close()

    # Report TLS that the AI turns off > 50% of the time
    results: list[dict] = []
    for tid, count in off_counts.items():
        off_ratio = count / max(total_steps, 1)
        if off_ratio > 0.3:  # turned off > 30% of the time
            if off_ratio > 0.7:
                rec = "REMOVE TLS — AI keeps it off most of the time"
            elif off_ratio > 0.5:
                rec = "Consider removing — off more than half the time"
            else:
                rec = "Partially unnecessary — off during some periods"

            results.append({
                "tls_id": tid,
                "off_ratio": round(off_ratio, 3),
                "off_steps": count,
                "total_steps": total_steps,
                "recommendation": rec,
            })

    results.sort(key=lambda r: r["off_ratio"], reverse=True)

    if results:
        print(f"\n  TLS the AI recommends removing or simplifying:")
        print(f"  {'TLS ID':>15}  {'Off%':>6}  Recommendation")
        print("  " + "-" * 65)
        for r in results:
            print(f"  {r['tls_id']:>15}  {r['off_ratio']*100:5.1f}%  {r['recommendation']}")
    else:
        print(f"\n  No TLS consistently turned off by the AI (all are useful).")

    return results


def main() -> None:
    ap = argparse.ArgumentParser(
        description="FlowMind AI - Analyze which intersections need/don't need traffic lights"
    )
    ap.add_argument("--net", required=True, help="SUMO .net.xml")
    ap.add_argument("--route", required=True, help="SUMO .rou.xml")
    ap.add_argument("--cfg", default=None, help="SUMO .sumocfg")
    ap.add_argument("--sample-steps", type=int, default=2000)
    ap.add_argument("--min-incoming", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--model", default=None, help="Trained model for TLS removal analysis")
    ap.add_argument("--output", default=None, help="Save results to JSON file")

    args = ap.parse_args()

    # 1. Analyze uncontrolled junctions
    results = analyze_need_for_tls(
        net_file=args.net,
        route_file=args.route,
        sumo_cfg=args.cfg,
        sample_steps=args.sample_steps,
        min_incoming=args.min_incoming,
        seed=args.seed,
        top_n=args.top,
    )

    # 2. If a trained model is provided, also analyze TLS removal
    removal_results = []
    if args.model:
        removal_results = analyze_tls_removal(
            net_file=args.net,
            route_file=args.route,
            model_path=args.model,
            sumo_cfg=args.cfg,
            seed=args.seed,
        )
        results["tls_to_remove"] = removal_results

    # 3. Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {args.output}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
