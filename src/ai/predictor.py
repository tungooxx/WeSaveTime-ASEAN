"""
FlowMind AI - Congestion Prediction

Uses exponential moving average, trend detection, and simple linear regression
to forecast congestion levels and per-lane queue lengths.
No external ML libraries required beyond numpy.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


class CongestionPredictor:
    """Predict future congestion from a rolling window of traffic metrics."""

    def __init__(self, window_size: int = 30) -> None:
        self.window_size = window_size

        # Global history: list of (tick, metrics_dict)
        self._history: list[tuple[float, dict[str, Any]]] = []

        # Per-intersection queue history: intersection_id -> list of (tick, {lane: length})
        self._queue_history: dict[str, list[tuple[float, dict[str, float]]]] = defaultdict(list)

        # EMA state
        self._ema_congestion: float | None = None
        self._ema_alpha: float = 2.0 / (window_size + 1)

    # -- data ingestion ------------------------------------------------------

    def record(self, tick: float, metrics: dict[str, Any]) -> None:
        """Store a metrics snapshot at simulation *tick*.

        Expected keys in *metrics*:
            congestion_score : float (0-100)
            avg_wait_time    : float (seconds)
            throughput       : float (vehicles / min)
            queue_lengths    : dict[intersection_id, dict[lane, float]]  (optional)
        """
        self._history.append((tick, dict(metrics)))

        # Trim to window
        if len(self._history) > self.window_size * 5:
            self._history = self._history[-self.window_size * 3 :]

        # Update EMA
        score = metrics.get("congestion_score", 0.0)
        if self._ema_congestion is None:
            self._ema_congestion = score
        else:
            self._ema_congestion = (
                self._ema_alpha * score + (1 - self._ema_alpha) * self._ema_congestion
            )

        # Store per-intersection queues
        queues_by_intersection = metrics.get("queue_lengths", {})
        for isct_id, lane_queues in queues_by_intersection.items():
            self._queue_history[isct_id].append((tick, dict(lane_queues)))
            if len(self._queue_history[isct_id]) > self.window_size * 5:
                self._queue_history[isct_id] = self._queue_history[isct_id][
                    -self.window_size * 3 :
                ]

    # -- linear regression helper --------------------------------------------

    @staticmethod
    def _linear_regression(
        x: np.ndarray, y: np.ndarray
    ) -> tuple[float, float]:
        """Simple OLS: returns (slope, intercept)."""
        n = len(x)
        if n < 2:
            return 0.0, float(y[-1]) if n else 0.0
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        ss_xx = np.sum((x - x_mean) ** 2)
        if ss_xx < 1e-12:
            return 0.0, float(y_mean)
        slope = float(ss_xy / ss_xx)
        intercept = float(y_mean - slope * x_mean)
        return slope, intercept

    # -- congestion prediction -----------------------------------------------

    def predict_congestion(self, horizon_minutes: float = 5.0) -> dict[str, Any]:
        """Predict congestion *horizon_minutes* into the future.

        Returns:
            predicted_score   : float (0-100)
            trend             : str   (increasing / decreasing / stable)
            peak_time_estimate: float (minutes until predicted peak, or 0)
            confidence        : float (0-1)
        """
        if len(self._history) < 2:
            current = self._ema_congestion if self._ema_congestion is not None else 50.0
            return {
                "predicted_score": round(current, 1),
                "trend": "stable",
                "peak_time_estimate": 0.0,
                "confidence": 0.1,
            }

        # Use recent window
        recent = self._history[-self.window_size :]
        ticks = np.array([t for t, _ in recent], dtype=np.float64)
        scores = np.array(
            [m.get("congestion_score", 0.0) for _, m in recent], dtype=np.float64
        )

        slope, intercept = self._linear_regression(ticks, scores)

        # Project forward – assume 1 tick ≈ 1 second
        horizon_ticks = horizon_minutes * 60.0
        future_tick = ticks[-1] + horizon_ticks
        predicted_raw = slope * future_tick + intercept

        # Blend linear prediction with EMA for robustness
        ema = self._ema_congestion if self._ema_congestion is not None else scores[-1]
        blend_weight = min(len(recent) / self.window_size, 1.0)
        predicted = blend_weight * predicted_raw + (1 - blend_weight) * ema
        predicted = float(np.clip(predicted, 0.0, 100.0))

        # Trend classification
        if slope > 0.02:
            trend = "increasing"
        elif slope < -0.02:
            trend = "decreasing"
        else:
            trend = "stable"

        # Peak time estimate: if increasing, estimate when score hits 90
        peak_time = 0.0
        if trend == "increasing" and predicted < 95 and slope > 0:
            ticks_to_90 = (90.0 - scores[-1]) / slope
            if ticks_to_90 > 0:
                peak_time = round(ticks_to_90 / 60.0, 1)

        # Confidence based on data quantity and residual variance
        residuals = scores - (slope * ticks + intercept)
        residual_std = float(np.std(residuals)) if len(residuals) > 1 else 50.0
        data_factor = min(len(recent) / self.window_size, 1.0)
        noise_factor = max(0.0, 1.0 - residual_std / 50.0)
        confidence = round(data_factor * 0.6 + noise_factor * 0.4, 2)
        confidence = float(np.clip(confidence, 0.05, 0.99))

        return {
            "predicted_score": round(predicted, 1),
            "trend": trend,
            "peak_time_estimate": peak_time,
            "confidence": confidence,
        }

    # -- per-intersection queue prediction -----------------------------------

    def predict_queue_lengths(
        self, intersection_id: str, horizon_minutes: float = 5.0
    ) -> dict[str, float]:
        """Predict per-lane queue lengths for *intersection_id*."""
        history = self._queue_history.get(intersection_id, [])
        if len(history) < 2:
            # Return latest known queues or empty
            if history:
                return {k: round(v, 1) for k, v in history[-1][1].items()}
            return {}

        recent = history[-self.window_size :]
        ticks = np.array([t for t, _ in recent], dtype=np.float64)
        horizon_ticks = horizon_minutes * 60.0
        future_tick = ticks[-1] + horizon_ticks

        # Collect all lane names
        all_lanes: set[str] = set()
        for _, lanes in recent:
            all_lanes.update(lanes.keys())

        predictions: dict[str, float] = {}
        for lane in sorted(all_lanes):
            values = np.array(
                [entry.get(lane, 0.0) for _, entry in recent], dtype=np.float64
            )
            slope, intercept = self._linear_regression(ticks, values)
            pred = slope * future_tick + intercept
            predictions[lane] = round(float(np.clip(pred, 0.0, 200.0)), 1)

        return predictions

    # -- trend ---------------------------------------------------------------

    def get_trend(self) -> str:
        """Return overall congestion trend: improving / worsening / stable."""
        if len(self._history) < 3:
            return "stable"

        recent = self._history[-self.window_size :]
        scores = np.array(
            [m.get("congestion_score", 0.0) for _, m in recent], dtype=np.float64
        )
        ticks = np.array([t for t, _ in recent], dtype=np.float64)

        slope, _ = self._linear_regression(ticks, scores)

        if slope < -0.02:
            return "improving"
        if slope > 0.02:
            return "worsening"
        return "stable"

    # -- history access ------------------------------------------------------

    def get_history(self, last_n: int = 60) -> list[dict[str, Any]]:
        """Return the last *last_n* recorded metrics snapshots."""
        entries = self._history[-last_n:]
        return [
            {"tick": tick, **metrics}
            for tick, metrics in entries
        ]
