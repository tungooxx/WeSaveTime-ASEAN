"""
FlowMind AI - Distill trained DQN into deployable fixed timing plans.

Runs the trained agent greedily for several episodes, records every action
per TLS, and converts the learned policy into:

  1. Fixed-cycle timing plans per TLS (Phase A for Xs -> Phase B for Ys ...)
  2. SUMO additional file (.add.xml) with optimised programs
  3. Human-readable report with recommendations (add / remove / timing)
  4. Clipboard-ready summary for sharing with AI or stakeholders

Usage:
    python -m src.tools.distill
    python -m src.tools.distill --model checkpoints/best_model.pt --episodes 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.ai.traffic_env import (
    SumoTrafficEnv, OBS_DIM, OLD_OBS_DIM, ACT_DIM, ACT_OFF,
    remap_obs_for_old_model,
)
from src.ai.dqn_agent import TrafficDQNAgent


# ── Distillation ───────────────────────────────────────────────────────

def distill(
    model_path: str,
    net_file: str,
    route_file: str,
    sumo_cfg: str | None = None,
    episodes: int = 5,
    hidden: int = 256,
    delta_time: int = 10,
    sim_length: int = 3600,
    seed: int = 1000,
    output_dir: str = "checkpoints",
) -> dict:
    """Run the trained agent greedily and distill into fixed timing.

    Returns a dict with the full distillation report.
    """
    print("=" * 60)
    print("  FlowMind AI — Policy Distillation")
    print("=" * 60)

    # ── Load agent (auto-detect obs_dim from checkpoint) ───────────
    import torch
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    ckpt_obs_dim = ckpt.get("obs_dim", OBS_DIM)
    agent = TrafficDQNAgent(ckpt_obs_dim, ACT_DIM, hidden)
    agent.load(model_path)
    print(f"  Model loaded: {model_path} (obs_dim={ckpt_obs_dim})")

    # ── Create env ─────────────────────────────────────────────────
    env = SumoTrafficEnv(
        net_file=net_file, route_file=route_file, sumo_cfg=sumo_cfg,
        delta_time=delta_time, sim_length=sim_length, gui=False, seed=seed,
    )
    print(f"  TLS agents: {env.num_agents}")
    print(f"  Existing: {len(env.existing_tls_ids)}, "
          f"Candidates: {len(env.candidate_tls_ids)}")
    print()

    # ── Record greedy actions across episodes ──────────────────────
    # Per TLS: list of (step, action) tuples
    tls_actions: dict[str, list[int]] = defaultdict(list)
    # Per TLS per step: which phase was chosen
    tls_step_actions: dict[str, list[list[int]]] = {
        tid: [] for tid in env.tls_ids
    }
    ep_metrics: list[dict] = []

    for ep in range(1, episodes + 1):
        t0 = time.time()
        obs, _ = env.reset(seed=seed + ep)
        terminated = truncated = False
        step = 0
        ep_reward = 0.0

        # Per-episode action tracker
        ep_actions: dict[str, list[int]] = {tid: [] for tid in env.tls_ids}

        # Check if model needs obs remapping (old 26-dim vs new 39-dim)
        needs_remap = (ckpt_obs_dim == OLD_OBS_DIM and OBS_DIM != OLD_OBS_DIM)

        while not (terminated or truncated):
            actions = {}
            for tid in env.tls_ids:
                valid = env.get_valid_actions(tid)
                o = remap_obs_for_old_model(obs[tid]) if needs_remap else obs[tid]
                a = agent.select_action(o, valid, greedy=True)
                actions[tid] = a
                tls_actions[tid].append(a)
                ep_actions[tid].append(a)

            obs, rewards, terminated, truncated, _ = env.step(actions)
            ep_reward += sum(rewards.values())
            step += 1

        metrics = env.get_metrics()
        elapsed = time.time() - t0
        ep_metrics.append({
            "episode": ep,
            "reward": round(ep_reward, 3),
            "wait": round(metrics.get("avg_wait_time", 0), 1),
            "queue": round(metrics.get("avg_queue_length", 0), 1),
            "vehicles": metrics.get("total_vehicles", 0),
            "steps": step,
        })
        print(f"  Distill ep {ep}/{episodes}: "
              f"R={ep_reward:+.2f} Wait={metrics.get('avg_wait_time', 0):.1f}s "
              f"Queue={metrics.get('avg_queue_length', 0):.1f} "
              f"Veh={metrics.get('total_vehicles', 0)} ({elapsed:.0f}s)")

        # Store per-episode step actions
        for tid in env.tls_ids:
            tls_step_actions[tid].append(ep_actions[tid])

    env.close()

    # ── Analyze action distributions ───────────────────────────────
    print()
    print("  Analyzing learned policy...")

    tls_meta = env._tls_meta
    report: dict = {
        "model": model_path,
        "episodes": episodes,
        "metrics": ep_metrics,
        "tls_plans": {},
        "add_tls": [],
        "remove_tls": [],
        "keep_existing": [],
        "keep_off": [],
    }

    # Load candidate info if available
    candidate_file = os.path.join(
        os.path.dirname(net_file), "candidate_tls.json")
    candidate_info = {}
    if os.path.isfile(candidate_file):
        with open(candidate_file) as f:
            cdata = json.load(f)
        for c in cdata.get("candidates", []):
            candidate_info[c["id"]] = c

    green_phases = env._green_phases

    for tid in env.tls_ids:
        actions = tls_actions[tid]
        total = len(actions)
        if total == 0:
            continue

        counter = Counter(actions)
        off_count = counter.get(ACT_OFF, 0)
        off_pct = off_count / total * 100

        # Get TLS phase info
        tls_info = tls_meta.get(tid)
        gp = green_phases.get(tid, [0])

        # Determine category
        is_candidate = tid in env.candidate_tls_ids
        is_existing = tid in env.existing_tls_ids

        # Build phase distribution (excluding OFF)
        phase_dist: list[dict] = []
        active_total = total - off_count
        for action_idx in range(ACT_DIM - 1):  # 0..6
            count = counter.get(action_idx, 0)
            if count == 0:
                continue
            phase_idx = gp[action_idx] if action_idx < len(gp) else gp[0]
            state_str = ""
            if tls_info and phase_idx < len(tls_info.phases):
                state_str = tls_info.phases[phase_idx].state
            pct = count / total * 100
            # Convert proportion to time allocation within delta_time cycle
            time_alloc = round(delta_time * count / total, 1)
            phase_dist.append({
                "action": action_idx,
                "phase_idx": phase_idx,
                "state": state_str,
                "count": count,
                "pct": round(pct, 1),
                "time_s": time_alloc,
            })
        phase_dist.sort(key=lambda p: p["count"], reverse=True)

        # Build fixed cycle plan (only active phases with > 5% usage)
        cycle_phases = [p for p in phase_dist if p["pct"] > 5]
        if cycle_phases and active_total > 0:
            # Normalize to a full cycle
            cycle_total = sum(p["count"] for p in cycle_phases)
            cycle_length = delta_time * (total / max(active_total, 1))
            cycle_length = max(cycle_length, 30)  # min 30s cycle
            cycle_length = min(cycle_length, 120)  # max 120s cycle
            for p in cycle_phases:
                p["cycle_time_s"] = round(
                    cycle_length * p["count"] / cycle_total, 0)

        plan = {
            "tls_id": tid,
            "is_candidate": is_candidate,
            "total_actions": total,
            "off_pct": round(off_pct, 1),
            "phase_distribution": phase_dist,
            "cycle_plan": cycle_phases,
        }

        # Add candidate congestion info
        if is_candidate and tid in candidate_info:
            ci = candidate_info[tid]
            plan["congestion_score"] = ci.get("congestion_score", 0)
            plan["baseline_wait"] = ci.get("avg_wait", 0)

        report["tls_plans"][tid] = plan

        # Categorize
        if is_candidate:
            if off_pct < 40:
                report["add_tls"].append({
                    "id": tid,
                    "off_pct": round(off_pct, 1),
                    "active_pct": round(100 - off_pct, 1),
                    "top_phase": cycle_phases[0] if cycle_phases else None,
                    "congestion_score": candidate_info.get(
                        tid, {}).get("congestion_score", 0),
                })
            else:
                report["keep_off"].append({
                    "id": tid, "off_pct": round(off_pct, 1)})
        elif is_existing:
            if off_pct > 60:
                report["remove_tls"].append({
                    "id": tid, "off_pct": round(off_pct, 1)})
            else:
                report["keep_existing"].append({
                    "id": tid, "off_pct": round(off_pct, 1),
                    "n_phases": len(cycle_phases)})

    # ── Generate SUMO additional file ──────────────────────────────
    add_xml = _generate_sumo_additional(report, env, delta_time)
    add_path = os.path.join(output_dir, "distilled_timing.add.xml")
    os.makedirs(output_dir, exist_ok=True)
    with open(add_path, "w") as f:
        f.write(add_xml)
    report["sumo_additional_file"] = add_path

    # ── Generate human-readable report ────────────────────────────
    text = _generate_text_report(report, env)
    report_path = os.path.join(output_dir, "distillation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)
    report["report_file"] = report_path

    # ── Save full JSON ────────────────────────────────────────────
    json_path = os.path.join(output_dir, "distillation.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ── Print summary ─────────────────────────────────────────────
    print(text)
    print(f"\n  Files saved:")
    print(f"    Timing plans : {add_path}")
    print(f"    Full report  : {report_path}")
    print(f"    JSON data    : {json_path}")

    return report


def _generate_sumo_additional(
    report: dict, env: SumoTrafficEnv, delta_time: int
) -> str:
    """Generate a SUMO .add.xml with optimised TLS programs."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<!-- FlowMind AI - Distilled traffic signal timing -->',
             '<!-- Generated from trained DQN agent -->', '<additional>']

    tls_meta = env._tls_meta

    for tid, plan in report["tls_plans"].items():
        cycle = plan.get("cycle_plan", [])
        if not cycle or plan["off_pct"] > 60:
            continue  # Skip TLS the agent wants off

        # Build SUMO phase elements
        lines.append(f'  <tlLogic id="{tid}" type="static" '
                     f'programID="flowmind_ai" offset="0">')

        yellow_time = env.yellow_time
        tls_info = tls_meta.get(tid)

        for p in cycle:
            state = p.get("state", "")
            dur = max(int(p.get("cycle_time_s", delta_time)), 5)
            if state:
                lines.append(f'    <phase duration="{dur}" state="{state}"/>')
                # Add yellow transition
                yellow = state.replace("G", "y").replace("g", "y")
                lines.append(
                    f'    <phase duration="{yellow_time}" state="{yellow}"/>')

        lines.append('  </tlLogic>')

    lines.append('</additional>')
    return "\n".join(lines) + "\n"


def _generate_text_report(report: dict, env: SumoTrafficEnv) -> str:
    """Generate a human-readable distillation report."""
    lines = []
    lines.append("=" * 60)
    lines.append("  FlowMind AI - Distillation Report")
    lines.append("=" * 60)
    lines.append("")

    # Metrics summary
    eps = report["metrics"]
    avg_r = np.mean([e["reward"] for e in eps])
    avg_w = np.mean([e["wait"] for e in eps])
    avg_q = np.mean([e["queue"] for e in eps])
    lines.append(f"  Evaluation ({len(eps)} episodes, greedy policy):")
    lines.append(f"    Avg reward : {avg_r:+.2f}")
    lines.append(f"    Avg wait   : {avg_w:.1f}s")
    lines.append(f"    Avg queue  : {avg_q:.1f}")
    lines.append("")

    # TLS summary
    n_existing = len(env.existing_tls_ids)
    n_candidates = len(env.candidate_tls_ids)
    n_add = len(report["add_tls"])
    n_remove = len(report["remove_tls"])
    n_keep_off = len(report["keep_off"])

    lines.append(f"  TLS Summary:")
    lines.append(f"    Existing signals  : {n_existing}")
    lines.append(f"    Candidate tested  : {n_candidates}")
    lines.append(f"    --> Add new TLS   : +{n_add}")
    lines.append(f"    --> Remove TLS    : -{n_remove}")
    lines.append(f"    --> No TLS needed : {n_keep_off}")
    lines.append("")

    # ADD recommendations
    if report["add_tls"]:
        lines.append("-" * 60)
        lines.append("  RECOMMEND: ADD NEW TRAFFIC LIGHTS")
        lines.append("-" * 60)
        for r in sorted(report["add_tls"],
                        key=lambda x: x["active_pct"], reverse=True):
            lines.append(f"    + {r['id']}")
            lines.append(f"      Agent active {r['active_pct']:.0f}% | "
                         f"Congestion score: {r.get('congestion_score', '?')}")
        lines.append("")

    # REMOVE recommendations
    if report["remove_tls"]:
        lines.append("-" * 60)
        lines.append("  RECOMMEND: REMOVE TRAFFIC LIGHTS")
        lines.append("-" * 60)
        for r in report["remove_tls"]:
            lines.append(f"    - {r['id']}  (OFF {r['off_pct']:.0f}%)")
        lines.append("")

    # KEEP OFF (candidates not needed)
    if report["keep_off"]:
        lines.append("-" * 60)
        lines.append("  CANDIDATES: NO SIGNAL NEEDED")
        lines.append("-" * 60)
        for r in report["keep_off"]:
            lines.append(f"    . {r['id']}  (OFF {r['off_pct']:.0f}%)")
        lines.append("")

    # Phase timing for top TLS
    lines.append("-" * 60)
    lines.append("  OPTIMISED PHASE TIMING (top active TLS)")
    lines.append("-" * 60)
    shown = 0
    for tid, plan in sorted(
        report["tls_plans"].items(),
        key=lambda x: x[1]["off_pct"]
    ):
        if plan["off_pct"] > 60 or not plan.get("cycle_plan"):
            continue
        if shown >= 20:  # Show top 20
            lines.append(f"    ... and {len(report['tls_plans']) - shown} more")
            break

        lines.append(f"    {tid}:")
        for p in plan["cycle_plan"]:
            dur = int(p.get("cycle_time_s", 10))
            lines.append(f"      Phase {p['phase_idx']:>2d}: "
                         f"{dur:>3d}s  ({p['pct']:.0f}%)  {p['state']}")
        shown += 1
    lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="FlowMind AI - Distill DQN into fixed timing plans")
    ap.add_argument("--model", default=os.path.join(
        _PROJECT_ROOT, "checkpoints", "best_model.pt"))
    ap.add_argument("--net", default=os.path.join(
        _PROJECT_ROOT, "sumo", "danang", "danang.net.xml"))
    ap.add_argument("--route", default=os.path.join(
        _PROJECT_ROOT, "sumo", "danang", "danang.rou.xml"))
    ap.add_argument("--cfg", default=os.path.join(
        _PROJECT_ROOT, "sumo", "danang", "danang.sumocfg"))
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--delta-time", type=int, default=10)
    ap.add_argument("--sim-length", type=int, default=3600)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--output-dir", default=os.path.join(
        _PROJECT_ROOT, "checkpoints"))
    args = ap.parse_args()

    distill(
        model_path=args.model,
        net_file=args.net,
        route_file=args.route,
        sumo_cfg=args.cfg,
        episodes=args.episodes,
        hidden=args.hidden,
        delta_time=args.delta_time,
        sim_length=args.sim_length,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
