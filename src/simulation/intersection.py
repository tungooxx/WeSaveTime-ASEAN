"""
FlowMind AI - Intersection, Lane, TrafficLight, Phase, and Vehicle models.

All units:
  - time   : seconds
  - speed  : km/h  (converted to m/s internally where needed)
  - size   : lane-units (1.0 = standard car length ~4.5 m)
  - position: metres from stop-line (0 = stop-line, positive = upstream)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Vehicle type catalogue
# ---------------------------------------------------------------------------

VEHICLE_TYPES: dict[str, dict] = {
    "motorbike": {"speed_kmh": 40.0, "size": 0.3, "accel_ms2": 3.0, "co2_g_per_s": 0.8},
    "car":       {"speed_kmh": 50.0, "size": 1.0, "accel_ms2": 2.5, "co2_g_per_s": 2.3},
    "bus":       {"speed_kmh": 35.0, "size": 2.0, "accel_ms2": 1.5, "co2_g_per_s": 5.0},
    "truck":     {"speed_kmh": 30.0, "size": 2.5, "accel_ms2": 1.2, "co2_g_per_s": 6.5},
}

KMH_TO_MS = 1.0 / 3.6
LANE_UNIT_M = 4.5  # metres per lane-unit


@dataclass
class Vehicle:
    """A single vehicle inside a lane queue."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    type: str = "car"
    speed: float = 0.0          # current speed km/h
    max_speed: float = 50.0     # free-flow speed km/h
    size: float = 1.0           # lane-units
    accel: float = 2.5          # m/s^2
    co2_g_per_s: float = 2.3    # emission rate while idling / moving
    position: float = 0.0       # metres from stop-line (decreasing toward 0)
    wait_time: float = 0.0      # seconds spent at speed ≈ 0
    passed: bool = False        # set True once vehicle crosses stop-line

    @classmethod
    def create(cls, vtype: str, position: float = 50.0,
               speed_factor: float = 1.0) -> Vehicle:
        spec = VEHICLE_TYPES[vtype]
        return cls(
            type=vtype,
            speed=0.0,
            max_speed=spec["speed_kmh"] * speed_factor,
            size=spec["size"],
            accel=spec["accel_ms2"],
            co2_g_per_s=spec["co2_g_per_s"],
            position=position,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "speed": round(self.speed, 1),
            "position": round(self.position, 2),
            "wait_time": round(self.wait_time, 1),
            "size": self.size,
        }


# ---------------------------------------------------------------------------
# Phase / TrafficLight
# ---------------------------------------------------------------------------

@dataclass
class Phase:
    """One signal phase: which directions get green and for how long."""

    green_directions: list[str]         # e.g. ["N", "S"]
    duration_seconds: float = 30.0


@dataclass
class TrafficLight:
    """Cycle-based traffic light controller."""

    phases: list[Phase] = field(default_factory=list)
    current_phase_idx: int = 0
    elapsed_time: float = 0.0
    yellow_duration: float = 3.0
    _in_yellow: bool = False

    # ------ helpers ------

    @property
    def current_phase(self) -> Phase:
        return self.phases[self.current_phase_idx]

    def green_directions(self) -> set[str]:
        if self._in_yellow:
            return set()                    # nobody gets green during yellow
        return set(self.current_phase.green_directions)

    def advance(self, dt: float) -> None:
        """Advance the signal by *dt* seconds."""
        self.elapsed_time += dt

        if self._in_yellow:
            if self.elapsed_time >= self.yellow_duration:
                self.elapsed_time = 0.0
                self._in_yellow = False
                self.current_phase_idx = (
                    (self.current_phase_idx + 1) % len(self.phases)
                )
        else:
            if self.elapsed_time >= self.current_phase.duration_seconds:
                self.elapsed_time = 0.0
                self._in_yellow = True

    def get_signal_state(self) -> dict[str, str]:
        """Return direction -> colour mapping for all four cardinal dirs."""
        greens = self.green_directions()
        state: dict[str, str] = {}
        for d in ("N", "S", "E", "W"):
            if d in greens:
                state[d] = "green"
            elif self._in_yellow and d in set(self.current_phase.green_directions):
                state[d] = "yellow"
            else:
                state[d] = "red"
        return state

    def set_timing(self, timing: dict[str, float]) -> None:
        """Override phase durations.  *timing* maps direction -> seconds.
        Phases whose first green direction appears in *timing* get updated.
        """
        for phase in self.phases:
            for d in phase.green_directions:
                if d in timing:
                    phase.duration_seconds = timing[d]
                    break


# ---------------------------------------------------------------------------
# Lane
# ---------------------------------------------------------------------------

@dataclass
class Lane:
    """One directional lane approaching an intersection."""

    id: str = ""
    direction: str = "N"                # N / S / E / W
    vehicle_queue: list[Vehicle] = field(default_factory=list)
    capacity: int = 20                  # max vehicles
    road_type: str = "two_way"          # one_way / two_way
    width_m: float = 3.5
    blocked: bool = False

    @property
    def queue_length(self) -> int:
        return len(self.vehicle_queue)

    @property
    def occupancy(self) -> float:
        """Sum of vehicle sizes vs capacity (in lane-units)."""
        total_size = sum(v.size for v in self.vehicle_queue)
        return total_size / max(self.capacity, 1)

    def avg_wait_time(self) -> float:
        if not self.vehicle_queue:
            return 0.0
        return sum(v.wait_time for v in self.vehicle_queue) / len(self.vehicle_queue)

    def inject_vehicle(self, vehicle: Vehicle) -> bool:
        """Add a vehicle to the back of the queue. Returns False if at capacity."""
        total_size = sum(v.size for v in self.vehicle_queue)
        if total_size + vehicle.size > self.capacity or self.blocked:
            return False
        # place behind the last vehicle
        if self.vehicle_queue:
            last = self.vehicle_queue[-1]
            vehicle.position = last.position + last.size * LANE_UNIT_M + 2.0
        else:
            vehicle.position = 30.0  # default arrival distance
        self.vehicle_queue.append(vehicle)
        return True

    def remove_passed_vehicles(self) -> list[Vehicle]:
        """Remove and return vehicles that have crossed the stop-line."""
        passed = [v for v in self.vehicle_queue if v.passed]
        self.vehicle_queue = [v for v in self.vehicle_queue if not v.passed]
        return passed

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "direction": self.direction,
            "queue_length": self.queue_length,
            "capacity": self.capacity,
            "road_type": self.road_type,
            "width_m": self.width_m,
            "blocked": self.blocked,
            "vehicles": [v.to_dict() for v in self.vehicle_queue],
        }


# ---------------------------------------------------------------------------
# Intersection
# ---------------------------------------------------------------------------

@dataclass
class Intersection:
    """A single signalised intersection with four approach lanes."""

    id: str = ""
    name: str = ""
    lat: float = 0.0
    lon: float = 0.0
    lanes: list[Lane] = field(default_factory=list)
    traffic_light: TrafficLight = field(default_factory=TrafficLight)

    # cumulative counters
    vehicles_passed: int = 0
    total_wait_accumulated: float = 0.0
    total_co2: float = 0.0

    # ---- convenience properties ----

    @property
    def signal_state(self) -> dict[str, str]:
        return self.traffic_light.get_signal_state()

    @property
    def signal_timing(self) -> dict[str, float]:
        timing: dict[str, float] = {}
        for phase in self.traffic_light.phases:
            for d in phase.green_directions:
                timing[d] = phase.duration_seconds
        return timing

    # ---- simulation methods ----

    def tick(self, dt: float) -> int:
        """Advance the intersection by *dt* seconds.

        Returns the number of vehicles that passed through this tick.
        """
        self.traffic_light.advance(dt)
        green_dirs = self.traffic_light.green_directions()

        passed_count = 0

        for lane in self.lanes:
            if lane.blocked:
                # blocked lanes: vehicles just wait
                for v in lane.vehicle_queue:
                    v.wait_time += dt
                    self.total_co2 += v.co2_g_per_s * dt * 0.5  # idle
                continue

            is_green = lane.direction in green_dirs

            for v in lane.vehicle_queue:
                if v.passed:
                    continue

                if is_green:
                    # accelerate toward max_speed
                    target_speed = v.max_speed
                    speed_ms = v.speed * KMH_TO_MS
                    target_ms = target_speed * KMH_TO_MS
                    new_speed_ms = min(speed_ms + v.accel * dt, target_ms)
                    v.speed = new_speed_ms / KMH_TO_MS

                    # move toward stop-line
                    v.position -= new_speed_ms * dt
                    if v.position <= 0:
                        v.position = 0.0
                        v.passed = True
                        passed_count += 1
                        self.vehicles_passed += 1
                        self.total_wait_accumulated += v.wait_time
                else:
                    # red / yellow: decelerate and wait
                    if v.position <= LANE_UNIT_M * 2:
                        # close to stop-line -> stop
                        v.speed = 0.0
                        v.wait_time += dt
                    else:
                        # still approaching, slow down
                        speed_ms = v.speed * KMH_TO_MS
                        new_speed_ms = max(speed_ms - v.accel * 2 * dt, 0.0)
                        v.speed = new_speed_ms / KMH_TO_MS
                        v.position -= new_speed_ms * dt

                # emissions while in queue
                if v.speed < 5.0:
                    self.total_co2 += v.co2_g_per_s * dt * 0.5
                else:
                    self.total_co2 += v.co2_g_per_s * dt

            # remove passed vehicles
            lane.remove_passed_vehicles()

        return passed_count

    def get_queue_lengths(self) -> dict[str, int]:
        return {lane.direction: lane.queue_length for lane in self.lanes}

    def get_avg_wait_time(self) -> float:
        waits = [lane.avg_wait_time() for lane in self.lanes if lane.vehicle_queue]
        return sum(waits) / len(waits) if waits else 0.0

    def inject_vehicles(
        self,
        rng: np.random.Generator,
        vehicle_mix: dict[str, float],
        count: int,
        speed_factor: float = 1.0,
        surge_direction: Optional[str] = None,
        surge_multiplier: float = 1.0,
    ) -> int:
        """Inject *count* random vehicles across lanes. Returns actual injected count."""
        types = list(vehicle_mix.keys())
        probs = np.array([vehicle_mix[t] for t in types], dtype=float)
        probs /= probs.sum()

        injected = 0
        for _ in range(count):
            vtype = rng.choice(types, p=probs)
            # pick a lane (weighted toward surge direction if set)
            weights = []
            for lane in self.lanes:
                w = 1.0
                if surge_direction and lane.direction == surge_direction:
                    w = surge_multiplier
                if lane.blocked:
                    w = 0.0
                weights.append(w)
            warr = np.array(weights, dtype=float)
            if warr.sum() == 0:
                break
            warr /= warr.sum()
            lane_idx = int(rng.choice(len(self.lanes), p=warr))
            lane = self.lanes[lane_idx]

            v = Vehicle.create(vtype, position=30.0 + rng.uniform(0, 40),
                               speed_factor=speed_factor)
            if lane.inject_vehicle(v):
                injected += 1
        return injected

    def remove_passed_vehicles(self) -> list[Vehicle]:
        """Remove all passed vehicles from every lane."""
        removed: list[Vehicle] = []
        for lane in self.lanes:
            removed.extend(lane.remove_passed_vehicles())
        return removed

    def set_signal_timing(self, timing: dict[str, float]) -> None:
        self.traffic_light.set_timing(timing)

    def block_lane(self, lane_direction: str) -> bool:
        """Block the first unblocked lane matching *lane_direction*."""
        for lane in self.lanes:
            if lane.direction == lane_direction and not lane.blocked:
                lane.blocked = True
                return True
        return False

    def unblock_all_lanes(self) -> None:
        for lane in self.lanes:
            lane.blocked = False

    def to_dict(self) -> dict:
        tl = self.traffic_light
        if tl._in_yellow:
            remaining = round(tl.yellow_duration - tl.elapsed_time, 1)
        else:
            remaining = round(tl.current_phase.duration_seconds - tl.elapsed_time, 1)
        return {
            "id": self.id,
            "name": self.name,
            "lat": self.lat,
            "lon": self.lon,
            "lanes": [l.to_dict() for l in self.lanes],
            "signal_state": self.signal_state,
            "signal_remaining": max(0.0, remaining),
            "queue_lengths": self.get_queue_lengths(),
            "avg_wait_time": round(self.get_avg_wait_time(), 2),
            "vehicles_passed": self.vehicles_passed,
            "total_co2_g": round(self.total_co2, 1),
        }


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def create_default_intersection(
    id: str, name: str, lat: float, lon: float,
    ns_green: float = 30.0, ew_green: float = 25.0,
) -> Intersection:
    """Build a standard 4-approach intersection with default timing."""
    lanes = [
        Lane(id=f"{id}_N", direction="N", capacity=20, road_type="two_way", width_m=3.5),
        Lane(id=f"{id}_S", direction="S", capacity=20, road_type="two_way", width_m=3.5),
        Lane(id=f"{id}_E", direction="E", capacity=18, road_type="two_way", width_m=3.0),
        Lane(id=f"{id}_W", direction="W", capacity=18, road_type="two_way", width_m=3.0),
    ]
    light = TrafficLight(phases=[
        Phase(green_directions=["N", "S"], duration_seconds=ns_green),
        Phase(green_directions=["E", "W"], duration_seconds=ew_green),
    ])
    return Intersection(id=id, name=name, lat=lat, lon=lon,
                        lanes=lanes, traffic_light=light)
