"""
FlowMind AI - Signal Optimization via Q-Learning

Uses a tabular Q-learning approach to learn optimal traffic signal timings.
State space is discretized from real-time intersection telemetry;
actions map to concrete green-phase durations for each direction.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Timing action presets
# Each action defines green_ns and green_ew durations in seconds.
# Total cycle = green_ns + 3 (yellow) + green_ew + 3 (yellow)
# ---------------------------------------------------------------------------
TIMING_ACTIONS: list[dict[str, int]] = [
    {"green_ns": 15, "green_ew": 15, "label": "short_equal"},
    {"green_ns": 30, "green_ew": 15, "label": "medium_ns"},
    {"green_ns": 15, "green_ew": 30, "label": "medium_ew"},
    {"green_ns": 30, "green_ew": 30, "label": "medium_equal"},
    {"green_ns": 45, "green_ew": 15, "label": "long_ns"},
    {"green_ns": 15, "green_ew": 45, "label": "long_ew"},
    {"green_ns": 45, "green_ew": 30, "label": "long_ns_med_ew"},
    {"green_ns": 30, "green_ew": 45, "label": "long_ew_med_ns"},
    {"green_ns": 45, "green_ew": 45, "label": "long_equal"},
    {"green_ns": 60, "green_ew": 30, "label": "priority_ns"},
    {"green_ns": 30, "green_ew": 60, "label": "priority_ew"},
    {"green_ns": 60, "green_ew": 60, "label": "priority_equal"},
]

# Discretization helpers -------------------------------------------------------

def _bin_queue(length: float) -> int:
    """Bin a queue length into discrete buckets: 0-4 → 0, 5-14 → 1, 15-29 → 2, 30+ → 3."""
    if length < 5:
        return 0
    if length < 15:
        return 1
    if length < 30:
        return 2
    return 3


def _time_bucket(hour: float) -> int:
    """Map hour-of-day (0-23.99) to 6 buckets."""
    if hour < 6:
        return 0   # night
    if hour < 9:
        return 1   # morning rush
    if hour < 12:
        return 2   # midday
    if hour < 15:
        return 3   # afternoon
    if hour < 19:
        return 4   # evening rush
    return 5       # evening


def _weather_code(weather: str | None) -> int:
    """Map a weather description to an integer code."""
    mapping = {
        "clear": 0, "sunny": 0,
        "cloudy": 1, "overcast": 1,
        "rain": 2, "drizzle": 2, "shower": 2,
        "storm": 3, "thunderstorm": 3, "heavy_rain": 3,
        "fog": 4, "mist": 4,
        "snow": 5, "ice": 5,
    }
    if weather is None:
        return 0
    return mapping.get(weather.lower().strip(), 0)


# ---------------------------------------------------------------------------
# SignalOptimizer
# ---------------------------------------------------------------------------

class SignalOptimizer:
    """Tabular Q-learning optimizer for traffic signal timing."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount: float = 0.95,
        epsilon: float = 0.15,
    ) -> None:
        self.lr = learning_rate
        self.discount = discount
        self.epsilon = epsilon

        # Q-table: state_key -> np array of Q-values (one per action)
        self.q_table: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(len(TIMING_ACTIONS), dtype=np.float64)
        )

        # Bookkeeping
        self.episode_count = 0
        self.total_updates = 0

    # -- state representation ------------------------------------------------

    def get_state_key(
        self,
        intersection_state: dict[str, Any],
        scenario: dict[str, Any] | None = None,
    ) -> tuple:
        """Build a hashable state tuple from intersection telemetry.

        Expected keys in *intersection_state*:
            queue_lengths : dict[str, float]  – per-lane queue lengths
            hour          : float             – hour of day (0-23.99)

        Optional keys in *scenario*:
            weather : str
        """
        queues = intersection_state.get("queue_lengths", {})
        # Aggregate per-direction (N/S vs E/W) then bin
        ns_q = sum(v for k, v in queues.items() if k.lower().startswith(("n", "s")))
        ew_q = sum(v for k, v in queues.items() if k.lower().startswith(("e", "w")))
        ns_bin = _bin_queue(ns_q)
        ew_bin = _bin_queue(ew_q)

        hour = intersection_state.get("hour", 12.0)
        t_bucket = _time_bucket(hour)

        weather = (scenario or {}).get("weather", "clear")
        w_code = _weather_code(weather)

        return (ns_bin, ew_bin, t_bucket, w_code)

    # -- action selection ----------------------------------------------------

    def choose_action(self, state_key: tuple) -> dict[str, Any]:
        """Epsilon-greedy action selection.  Returns a timing dict."""
        if random.random() < self.epsilon:
            idx = random.randrange(len(TIMING_ACTIONS))
        else:
            idx = int(np.argmax(self.q_table[state_key]))
        action = dict(TIMING_ACTIONS[idx])
        action["action_index"] = idx
        action["cycle_time"] = action["green_ns"] + 3 + action["green_ew"] + 3
        return action

    # -- learning ------------------------------------------------------------

    def update(
        self,
        state_key: tuple,
        action: dict[str, Any],
        reward: float,
        next_state_key: tuple,
    ) -> None:
        """Standard Q-learning update rule."""
        idx = action.get("action_index", 0)
        current_q = self.q_table[state_key][idx]
        best_next = float(np.max(self.q_table[next_state_key]))

        new_q = current_q + self.lr * (reward + self.discount * best_next - current_q)
        self.q_table[state_key][idx] = new_q

        self.total_updates += 1

    # -- reward computation --------------------------------------------------

    @staticmethod
    def compute_reward(
        avg_wait: float,
        congestion_score: float,
        emission_factor: float = 0.0,
    ) -> float:
        """Reward = negative cost.  Lower wait / congestion / emissions → higher reward."""
        congestion_penalty = congestion_score * 0.5
        return -(avg_wait + congestion_penalty + emission_factor * 0.3)

    # -- high-level optimization ---------------------------------------------

    def optimize_intersection(
        self,
        intersection_state: dict[str, Any],
        scenario: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Recommend timing for a single intersection.

        Returns dict with recommended_timing, predicted_improvement, reasoning.
        """
        state_key = self.get_state_key(intersection_state, scenario)
        action = self.choose_action(state_key)

        # Estimate improvement from Q-values
        q_vals = self.q_table[state_key]
        best_q = float(np.max(q_vals))
        mean_q = float(np.mean(q_vals))
        predicted_improvement = max(0.0, (best_q - mean_q) / (abs(mean_q) + 1e-6) * 100)
        predicted_improvement = round(min(predicted_improvement, 40.0), 1)

        # Build human-readable reasoning
        queues = intersection_state.get("queue_lengths", {})
        ns_total = sum(v for k, v in queues.items() if k.lower().startswith(("n", "s")))
        ew_total = sum(v for k, v in queues.items() if k.lower().startswith(("e", "w")))

        if ns_total > ew_total * 1.5:
            direction_note = "N/S corridor is significantly more congested"
        elif ew_total > ns_total * 1.5:
            direction_note = "E/W corridor is significantly more congested"
        else:
            direction_note = "Both corridors have comparable load"

        reasoning = (
            f"{direction_note}. "
            f"Recommending {action['label']} timing "
            f"(NS {action['green_ns']}s / EW {action['green_ew']}s, "
            f"cycle {action['cycle_time']}s). "
            f"Q-value confidence based on {self.total_updates} updates."
        )

        return {
            "recommended_timing": {
                "green_ns": action["green_ns"],
                "green_ew": action["green_ew"],
                "yellow": 3,
                "cycle_time": action["cycle_time"],
                "label": action["label"],
                "action_index": action["action_index"],
            },
            "predicted_improvement": predicted_improvement,
            "reasoning": reasoning,
            "state_key": state_key,
        }

    def optimize_all(
        self,
        simulation_state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Optimize every intersection in *simulation_state*.

        *simulation_state* should have:
            intersections : list[dict]  – each with 'id', 'queue_lengths', 'hour', …
            scenario      : dict | None
        """
        intersections = simulation_state.get("intersections", [])
        scenario = simulation_state.get("scenario")
        recommendations: list[dict[str, Any]] = []

        for isct in intersections:
            rec = self.optimize_intersection(isct, scenario)
            rec["intersection_id"] = isct.get("id", "unknown")
            recommendations.append(rec)

        return recommendations

    # -- green corridor (emergency) ------------------------------------------

    def create_green_corridor(
        self,
        from_id: str,
        to_id: str,
        intersections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Create emergency green-wave timing from *from_id* to *to_id*.

        Intersections are assumed to be ordered along the corridor route.
        Each gets a staggered priority green in the travel direction with
        progressive offsets so the emergency vehicle sees continuous green.
        """
        corridor: list[dict[str, Any]] = []
        in_corridor = False
        offset = 0
        travel_speed_mps = 15  # ~54 km/h assumed emergency speed
        avg_block_distance_m = 200  # assumed average block length

        for isct in intersections:
            isct_id = isct.get("id", "")

            if isct_id == from_id:
                in_corridor = True
            if not in_corridor:
                continue

            corridor.append({
                "intersection_id": isct_id,
                "green_ns": 60,
                "green_ew": 10,
                "yellow": 3,
                "cycle_time": 76,
                "priority": True,
                "offset_seconds": offset,
                "label": "emergency_corridor",
            })

            offset += int(avg_block_distance_m / travel_speed_mps)

            if isct_id == to_id:
                break

        return corridor

    # -- introspection -------------------------------------------------------

    def get_policy_summary(self) -> dict[str, Any]:
        """Return a human-readable summary of the learned policy."""
        num_states = len(self.q_table)
        if num_states == 0:
            return {
                "states_visited": 0,
                "total_updates": self.total_updates,
                "message": "No policy learned yet – the optimizer has not received any updates.",
            }

        best_actions: dict[str, int] = defaultdict(int)
        for state_key, q_vals in self.q_table.items():
            best_idx = int(np.argmax(q_vals))
            label = TIMING_ACTIONS[best_idx]["label"]
            best_actions[label] += 1

        sorted_actions = sorted(best_actions.items(), key=lambda x: -x[1])

        return {
            "states_visited": num_states,
            "total_updates": self.total_updates,
            "learning_rate": self.lr,
            "discount_factor": self.discount,
            "epsilon": self.epsilon,
            "preferred_timings": [
                {"timing": label, "states_count": count}
                for label, count in sorted_actions
            ],
            "action_space_size": len(TIMING_ACTIONS),
        }
