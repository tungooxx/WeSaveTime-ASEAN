"""
FlowMind AI - Reward computation for traffic signal RL.

Level 1: Pure timing optimization reward.

Per-TLS reward structure:
  reward = wait_improvement + queue_penalty + fairness + throughput + pressure + switch_penalty
  Range: approximately -1 to +0.5 per step
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
    phase_changed: bool = False,
    transition_cost: float = 1.0,
    max_queue_cap: float = 50.0,
    # Baseline targets (from comparison runs)
    baseline_wait: float = 25.0,    # baseline avg wait per TLS
    baseline_queue: float = 0.9,    # baseline avg queue per TLS
    w_wait: float = 0.20,
    w_queue: float = 0.10,
    w_fairness: float = 0.05,
    w_throughput: float = 0.10,
    w_pressure: float = 0.25,
    w_switch: float = 0.10,
    w_baseline: float = 0.20,       # baseline bonus weight
) -> float:
    """Compute scalar reward for one TLS (higher = better).

    Combines penalty terms with a baseline bonus:
    - Penalty terms: fine-grained signal for what to optimize
    - Baseline bonus: positive when AI beats SUMO default, negative when worse
      This gives the agent a TARGET — it knows when it's doing well.

    Args:
        baseline_wait: Expected wait time under SUMO default timing.
        baseline_queue: Expected queue length under SUMO default timing.
        transition_cost: Per-TLS scaling for switch penalty.
    """
    # 1. Waiting-time improvement (step-to-step delta)
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

    # 5. Pressure (outgoing > incoming = good flow)
    pressure_term = w_pressure * float(np.clip(pressure / 20.0, -1.0, 1.0))

    # 6. Phase-switch penalty scaled by intersection transition cost
    switch_term = -w_switch * transition_cost if phase_changed else 0.0

    # 7. Baseline bonus: POSITIVE when beating baseline, NEGATIVE when worse
    #    wait bonus: +1 when wait=0, 0 when wait=baseline, -1 when wait=2*baseline
    #    queue bonus: same logic
    bl_wait = max(baseline_wait, 1.0)
    bl_queue = max(baseline_queue, 0.1)
    wait_bonus = float(np.clip((bl_wait - new_waiting) / bl_wait, -1.0, 1.0))
    queue_bonus = float(np.clip((bl_queue - avg_q) / bl_queue, -1.0, 1.0))
    baseline_term = w_baseline * (0.7 * wait_bonus + 0.3 * queue_bonus)

    return float(wait_term + queue_term + fairness_term
                 + throughput_term + pressure_term + switch_term
                 + baseline_term)


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
