"""
FlowMind AI - Reward computation for traffic signal RL.

Design philosophy:
  - Efficiency is the primary objective (wait time, queue, throughput, pressure)
  - Safety is a HARD CONSTRAINT, not a weighted trade-off
  - When no collision: reward is purely about efficiency (~-1 to +0.5)
  - When collision: massive penalty (-5 per collision) overwhelms everything
  - This prevents the agent from being "too conservative" or "too reckless"

Per-TLS reward structure:
  efficiency = wait_improvement + queue_penalty + fairness + throughput + pressure
  safety     = -5.0 per collision, -0.5 per emergency brake (only when they happen)
  reward     = efficiency + safety
"""

from __future__ import annotations

import numpy as np


def compute_tls_reward(
    old_waiting: float,
    new_waiting: float,
    queue_lengths: list[float],
    old_throughput: int = 0,
    new_throughput: int = 0,
    pressure: float = 0.0,
    collisions: int = 0,
    emergency_brakes: int = 0,
    max_queue_cap: float = 50.0,
    # Efficiency weights (sum to 1.0, only affect the efficiency part)
    w_wait: float = 0.40,
    w_queue: float = 0.25,
    w_fairness: float = 0.10,
    w_throughput: float = 0.10,
    w_pressure: float = 0.15,
    # Safety penalty magnitudes (not weights — these are absolute penalties)
    collision_penalty: float = 5.0,
    ebrake_penalty: float = 0.5,
) -> float:
    """Compute scalar reward for one TLS (higher = better).

    The reward has two independent parts:
      1. Efficiency score (continuous, -1 to +1 range)
      2. Safety penalty (sparse, only when collisions/ebrakes happen)

    This design means:
      - 99% of steps: agent optimizes purely for traffic flow
      - Rare collision: agent gets hit with -5, learns "never do this again"
      - Agent won't become overly conservative (safety doesn't compete with flow)
    """
    # ── Efficiency (continuous, every step) ──────────────────────────

    # 1. Waiting-time improvement (primary signal)
    # Negative delta = wait decreased = good
    delta_wait = new_waiting - old_waiting
    wait_term = -w_wait * float(np.clip(delta_wait / 100.0, -2.0, 2.0))

    # 2. Average queue penalty
    avg_q = float(np.mean(queue_lengths)) if queue_lengths else 0.0
    queue_term = -w_queue * (avg_q / max_queue_cap)

    # 3. Max-queue fairness (don't starve any road)
    max_q = float(max(queue_lengths)) if queue_lengths else 0.0
    fairness_term = -w_fairness * (max_q / max_queue_cap)

    # 4. Throughput bonus (vehicles flowing through)
    tp_change = new_throughput - old_throughput
    throughput_term = w_throughput * float(np.clip(tp_change / 10.0, -1.0, 1.0))

    # 5. Pressure (outgoing > incoming = good flow, from sumo-rl)
    pressure_term = w_pressure * float(np.clip(pressure / 20.0, -1.0, 1.0))

    efficiency = (wait_term + queue_term + fairness_term
                  + throughput_term + pressure_term)

    # ── Safety (sparse, only when bad things happen) ─────────────────
    # These are ABSOLUTE penalties, not weighted against efficiency.
    # A single collision = -5.0, which wipes out ~10 good steps of reward.
    # This teaches the agent "collisions are unacceptable" without making
    # it afraid to change phases during normal operation.
    safety = 0.0
    if collisions > 0:
        safety -= collision_penalty * collisions
    if emergency_brakes > 0:
        safety -= ebrake_penalty * emergency_brakes

    return float(efficiency + safety)


def compute_global_reward(
    avg_wait_before: float,
    avg_wait_after: float,
    congestion_before: float,
    congestion_after: float,
    throughput_before: int,
    throughput_after: int,
) -> float:
    """Network-wide reward (for logging / comparison)."""
    wait_imp = (avg_wait_before - avg_wait_after) / max(avg_wait_before, 1.0)
    cong_imp = (congestion_before - congestion_after) / max(congestion_before, 1.0)
    tp_imp = (throughput_after - throughput_before) / max(throughput_before, 1)
    return float(
        0.4 * np.clip(wait_imp, -1, 1)
        + 0.4 * np.clip(cong_imp, -1, 1)
        + 0.2 * np.clip(tp_imp, -1, 1)
    )
