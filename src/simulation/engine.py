"""
FlowMind AI - Main traffic simulation engine.

Creates a grid of intersections in central Hanoi (Hoan Kiem district) and runs
a discrete-time simulation with configurable scenarios, vehicle spawning,
congestion detection, and AI signal-timing overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .intersection import (
    Intersection,
    Vehicle,
    VEHICLE_TYPES,
    create_default_intersection,
)
from .scenarios import Scenario, get_scenario, list_scenarios, SCENARIOS


# ---------------------------------------------------------------------------
# Hanoi Hoan Kiem intersection definitions (real coordinates)
# ---------------------------------------------------------------------------

_HANOI_INTERSECTIONS = [
    # id, name, lat, lon, NS green, EW green
    # Compact grid in central Hoan Kiem — ~200-400m between each pair
    ("HK01", "Hang Bai - Trang Thi",              21.02500, 105.84920, 35, 25),
    ("HK02", "Hang Khay - Dinh Tien Hoang",       21.02500, 105.85280, 30, 30),
    ("HK03", "Ba Trieu - Hai Ba Trung",           21.02260, 105.84920, 30, 25),
    ("HK04", "Le Thai To - Hang Trong",            21.02740, 105.85280, 25, 30),
    ("HK05", "Trang Tien - Ngo Quyen",             21.02740, 105.84920, 30, 25),
    ("HK06", "Ly Thuong Kiet - Quang Trung",       21.02260, 105.85280, 35, 20),
]


# ---------------------------------------------------------------------------
# Fuel consumption estimation constants
# ---------------------------------------------------------------------------

# litres per second while idling / moving (rough averages)
_FUEL_IDLE: dict[str, float] = {
    "motorbike": 0.0003, "car": 0.0008, "bus": 0.002, "truck": 0.003,
}
_FUEL_MOVE: dict[str, float] = {
    "motorbike": 0.0006, "car": 0.0015, "bus": 0.004, "truck": 0.005,
}


@dataclass
class _EmergencyVehicle:
    """Tracks an active emergency vehicle traversing the network."""

    vehicle: Vehicle
    path: list[int]            # indices into engine.intersections
    current_path_idx: int = 0
    active: bool = True


class SimulationEngine:
    """Discrete-time traffic simulation over a network of intersections."""

    def __init__(self, seed: Optional[int] = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.tick: int = 0
        self.dt: float = 1.0                  # seconds per step

        # Active scenario (default = sunny morning)
        self._scenario: Scenario = get_scenario("sunny_morning")

        # Build intersection network
        self.intersections: list[Intersection] = []
        for id_, name, lat, lon, ns, ew in _HANOI_INTERSECTIONS:
            ix = create_default_intersection(id_, name, lat, lon, ns, ew)
            self.intersections.append(ix)

        # Aggregate counters
        self.total_vehicles_spawned: int = 0
        self.total_vehicles_passed: int = 0
        self.total_fuel_l: float = 0.0
        self.total_co2_g: float = 0.0

        # Per-tick tracking for speed averaging
        self._speed_samples: list[float] = []

        # Emergency vehicles
        self._emergencies: list[_EmergencyVehicle] = []

        # Snapshot storage for before/after comparison
        self._before_metrics: Optional[dict] = None

        # User density override (None = use scenario default)
        self._density_override: Optional[float] = None

    # ------------------------------------------------------------------
    # Core simulation loop
    # ------------------------------------------------------------------

    def step(self, dt: Optional[float] = None) -> dict:
        """Advance simulation by one step. Returns current metrics dict."""
        if dt is not None:
            self.dt = dt

        self.tick += 1
        self._speed_samples.clear()

        # 1. spawn new vehicles
        self.spawn_vehicles()

        # 2. tick each intersection (move vehicles, update signals)
        tick_passed = 0
        for ix in self.intersections:
            passed = ix.tick(self.dt)
            tick_passed += passed
            # collect speed + fuel data from live vehicles
            for lane in ix.lanes:
                for v in lane.vehicle_queue:
                    self._speed_samples.append(v.speed)
                    if v.speed < 5.0:
                        self.total_fuel_l += _FUEL_IDLE.get(v.type, 0.001) * self.dt
                    else:
                        self.total_fuel_l += _FUEL_MOVE.get(v.type, 0.002) * self.dt

        self.total_vehicles_passed += tick_passed

        # 3. advance emergency vehicles
        self._advance_emergencies()

        # 4. accumulate CO2
        self.total_co2_g = sum(ix.total_co2 for ix in self.intersections)

        return self.get_metrics()

    def run(self, steps: int = 100, dt: float = 1.0) -> dict:
        """Run *steps* simulation ticks and return final metrics."""
        for _ in range(steps):
            self.step(dt)
        return self.get_metrics()

    # ------------------------------------------------------------------
    # Vehicle spawning
    # ------------------------------------------------------------------

    def set_base_density(self, density: float) -> None:
        """Override vehicle spawn density (vehicles per minute)."""
        self._density_override = max(1.0, min(density, 100.0))

    def get_base_density(self) -> float:
        """Return current effective base density."""
        if self._density_override is not None:
            return self._density_override
        return self._scenario.base_density

    def spawn_vehicles(self) -> int:
        """Generate vehicles across all intersections for one tick."""
        sc = self._scenario
        # density is per minute; convert to per-tick probability
        effective_density = self._density_override if self._density_override is not None else sc.base_density
        per_tick = effective_density * (self.dt / 60.0)
        count_to_spawn = int(self.rng.poisson(per_tick))

        # check for directional surge
        surge_dir: Optional[str] = None
        surge_mult: float = 1.0
        for evt in sc.special_events:
            if evt.get("type") == "directional_surge":
                surge_dir = evt["direction"]
                surge_mult = evt.get("multiplier", 2.0)

        total = 0
        for ix in self.intersections:
            n = max(1, count_to_spawn // len(self.intersections))
            # slight random variation per intersection
            n = int(self.rng.poisson(n))
            injected = ix.inject_vehicles(
                self.rng,
                sc.vehicle_mix,
                n,
                speed_factor=sc.road_speed_factor,
                surge_direction=surge_dir,
                surge_multiplier=surge_mult,
            )
            total += injected
        self.total_vehicles_spawned += total
        return total

    # ------------------------------------------------------------------
    # Scenario management
    # ------------------------------------------------------------------

    def apply_scenario(self, scenario_id: str) -> Scenario:
        """Switch to a new scenario, applying its special events."""
        # store before-metrics for comparison
        self._before_metrics = self.get_metrics()

        sc = get_scenario(scenario_id)
        self._scenario = sc

        # reset lane blocks
        for ix in self.intersections:
            ix.unblock_all_lanes()

        # apply special events
        for evt in sc.special_events:
            etype = evt.get("type")
            if etype == "blocked_lane":
                idx = evt.get("intersection_idx", 0)
                if 0 <= idx < len(self.intersections):
                    self.intersections[idx].block_lane(evt.get("lane_direction", "S"))

            elif etype == "emergency_vehicle":
                from_idx = evt.get("from_intersection_idx", 0)
                to_idx = evt.get("to_intersection_idx", len(self.intersections) - 1)
                self.inject_emergency_vehicle(from_idx, to_idx)

        return sc

    # ------------------------------------------------------------------
    # State and metrics
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Return full simulation state as a plain dict."""
        emergencies = []
        for em in self._emergencies:
            if em.active and em.current_path_idx < len(em.path):
                ix_idx = em.path[em.current_path_idx]
                if 0 <= ix_idx < len(self.intersections):
                    ix = self.intersections[ix_idx]
                    emergencies.append({
                        "lat": ix.lat,
                        "lon": ix.lon,
                        "intersection_id": ix.id,
                        "position": round(em.vehicle.position, 1),
                        "speed": round(em.vehicle.speed, 1),
                    })
        return {
            "tick": self.tick,
            "time_of_day": self._scenario.time_of_day,
            "weather": self._scenario.weather,
            "rain_intensity": self._scenario.rain_intensity,
            "intersections": [ix.to_dict() for ix in self.intersections],
            "total_vehicles": self._count_live_vehicles(),
            "passed_vehicles": self.total_vehicles_passed,
            "metrics": self.get_metrics(),
            "emergencies": emergencies,
            "base_density": self.get_base_density(),
        }

    def get_metrics(self) -> dict:
        """Return aggregate performance metrics."""
        all_waits: list[float] = []
        all_queues: list[int] = []

        for ix in self.intersections:
            all_waits.append(ix.get_avg_wait_time())
            for lane in ix.lanes:
                all_queues.append(lane.queue_length)

        avg_wait = sum(all_waits) / len(all_waits) if all_waits else 0.0
        avg_queue = sum(all_queues) / len(all_queues) if all_queues else 0.0
        avg_speed = (
            sum(self._speed_samples) / len(self._speed_samples)
            if self._speed_samples
            else 0.0
        )

        # congestion score: 0 (free-flow) to 100 (gridlock)
        # based on average queue occupancy and wait times
        max_possible_queue = 20
        queue_ratio = min(avg_queue / max(max_possible_queue, 1), 1.0)
        wait_ratio = min(avg_wait / 120.0, 1.0)  # 120s = very congested
        congestion_score = min(100.0, (queue_ratio * 60 + wait_ratio * 40))

        metrics = {
            "avg_wait_time": round(avg_wait, 2),
            "avg_queue_length": round(avg_queue, 2),
            "throughput": self.total_vehicles_passed,
            "congestion_score": round(congestion_score, 1),
            "emission_estimate": round(self.total_co2_g, 1),
            "avg_speed": round(avg_speed, 1),
            "fuel_consumption_l": round(self.total_fuel_l, 4),
        }

        if self._before_metrics is not None:
            metrics["before_metrics"] = self._before_metrics

        return metrics

    # ------------------------------------------------------------------
    # Congestion detection
    # ------------------------------------------------------------------

    def detect_congestion(self, threshold: float = 0.6) -> list[dict]:
        """Return intersections where average occupancy exceeds *threshold*."""
        congested: list[dict] = []
        for ix in self.intersections:
            occupancies = [lane.occupancy for lane in ix.lanes]
            avg_occ = sum(occupancies) / len(occupancies) if occupancies else 0.0
            if avg_occ >= threshold:
                congested.append({
                    "intersection_id": ix.id,
                    "name": ix.name,
                    "avg_occupancy": round(avg_occ, 3),
                    "queue_lengths": ix.get_queue_lengths(),
                    "avg_wait_time": round(ix.get_avg_wait_time(), 2),
                })
        return congested

    # ------------------------------------------------------------------
    # AI signal timing override
    # ------------------------------------------------------------------

    def set_signal_timing(
        self, intersection_id: str, timing: dict[str, float]
    ) -> bool:
        """Override signal timing at the given intersection.

        *timing* maps direction (N/S/E/W) to green-phase duration in seconds.
        Returns True if the intersection was found.
        """
        for ix in self.intersections:
            if ix.id == intersection_id:
                ix.set_signal_timing(timing)
                return True
        return False

    # ------------------------------------------------------------------
    # Lane blocking
    # ------------------------------------------------------------------

    def block_lane(self, intersection_id: str, lane_id: str) -> bool:
        """Block a specific lane by intersection and lane id."""
        for ix in self.intersections:
            if ix.id == intersection_id:
                for lane in ix.lanes:
                    if lane.id == lane_id and not lane.blocked:
                        lane.blocked = True
                        return True
        return False

    # ------------------------------------------------------------------
    # Emergency vehicle
    # ------------------------------------------------------------------

    def inject_emergency_vehicle(
        self, from_intersection: int, to_intersection: int
    ) -> None:
        """Inject an ambulance that must traverse from intersection index
        *from_intersection* to *to_intersection* with signal pre-emption.
        """
        path = list(range(from_intersection, to_intersection + 1))
        ev = Vehicle.create("car", position=150.0, speed_factor=1.6)
        ev.type = "ambulance"
        ev.max_speed = 80.0
        ev.co2_g_per_s = 3.5

        emergency = _EmergencyVehicle(vehicle=ev, path=path)
        self._emergencies.append(emergency)

        # pre-empt signal at first intersection on the path
        if path:
            self._preempt_signal(path[0])

    def _advance_emergencies(self) -> None:
        """Move emergency vehicles along their paths each tick."""
        for em in self._emergencies:
            if not em.active:
                continue
            em.vehicle.position -= em.vehicle.max_speed * (1 / 3.6) * self.dt
            if em.vehicle.position <= 0:
                em.current_path_idx += 1
                if em.current_path_idx >= len(em.path):
                    em.active = False
                    continue
                em.vehicle.position = 200.0
                self._preempt_signal(em.path[em.current_path_idx])

    def _preempt_signal(self, intersection_idx: int) -> None:
        """Force green on all directions at an intersection (emergency mode)."""
        if 0 <= intersection_idx < len(self.intersections):
            ix = self.intersections[intersection_idx]
            # give very short cycle to flush traffic, then long green
            for phase in ix.traffic_light.phases:
                phase.duration_seconds = 5.0
            ix.traffic_light.elapsed_time = 0.0
            ix.traffic_light.current_phase_idx = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _count_live_vehicles(self) -> int:
        total = 0
        for ix in self.intersections:
            for lane in ix.lanes:
                total += lane.queue_length
        return total

    def reset(self, seed: Optional[int] = 42) -> None:
        """Reset the engine to initial state."""
        self.__init__(seed=seed)  # type: ignore[misc]

    def snapshot_before(self) -> None:
        """Manually capture current metrics as 'before' for comparison."""
        self._before_metrics = self.get_metrics()

    def list_scenarios(self) -> list[dict]:
        """Return metadata for all available scenarios."""
        return [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "time_of_day": s.time_of_day,
                "weather": s.weather,
                "vehicle_mix": s.vehicle_mix,
            }
            for s in list_scenarios()
        ]
