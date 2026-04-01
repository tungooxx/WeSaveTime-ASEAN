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
    w_wait: float = 0.20,
    w_queue: float = 0.25,
    w_throughput: float = 0.10,
    w_pressure: float = 0.40,
    w_switch: float = 0.05,
) -> float:
    """Compute scalar reward for one TLS (higher = better).

    Pressure-primary reward (PressLight-style):
    - Pressure (outgoing - incoming vehicles) is the main signal
    - Queue and wait penalties discourage congestion buildup
    - Small switch penalty discourages unnecessary phase changes

    Range: approximately -0.7 to +0.5 per step.
    """
    # 1. Pressure: outgoing > incoming = good flow (main signal)
    pressure_term = w_pressure * float(np.clip(pressure / 10.0, -1.0, 1.0))

    # 2. Queue penalty: absolute level, not delta
    avg_q = float(np.mean(queue_lengths)) if queue_lengths else 0.0
    queue_term = -w_queue * (avg_q / max_queue_cap)

    # 3. Wait delta: penalise increasing wait, reward decreasing
    delta_wait = new_waiting - old_waiting
    wait_term = -w_wait * float(np.clip(delta_wait / 50.0, -1.0, 1.0))

    # 4. Throughput bonus: vehicles clearing intersection (fewer = better flow)
    tp_change = old_throughput - new_throughput
    throughput_term = w_throughput * float(np.clip(tp_change / 10.0, -1.0, 1.0))

    # 5. Small switch penalty to discourage unnecessary phase changes
    switch_term = -w_switch * transition_cost if phase_changed else 0.0

    return float(pressure_term + queue_term + wait_term
                 + throughput_term + switch_term)


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
