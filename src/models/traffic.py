"""
FlowMind AI - Pydantic models for traffic simulation state and API responses.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class VehicleState(BaseModel):
    """Snapshot of a single vehicle in the simulation."""

    type: str = Field(..., description="Vehicle type: motorbike, car, bus, truck")
    speed: float = Field(..., ge=0, description="Current speed in km/h")
    position: float = Field(
        ..., ge=0, description="Position along the lane (0 = queue tail)"
    )
    lane_id: str = Field(..., description="Lane the vehicle currently occupies")
    wait_time: float = Field(0.0, ge=0, description="Seconds spent waiting at red")


class LaneState(BaseModel):
    """Snapshot of a single lane."""

    id: str
    direction: str
    queue_length: int = 0
    capacity: int = 20
    road_type: str = "two_way"
    width_m: float = 3.5
    blocked: bool = False
    vehicles: list[VehicleState] = Field(default_factory=list)


class IntersectionState(BaseModel):
    """Snapshot of one intersection and all its lanes."""

    id: str
    name: str
    lat: float
    lon: float
    lanes: list[LaneState] = Field(default_factory=list)
    signal_state: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of direction -> current colour (green/red/yellow)",
    )
    queue_lengths: dict[str, int] = Field(default_factory=dict)
    avg_wait_time: float = 0.0


class MetricsSnapshot(BaseModel):
    """Aggregate metrics at a single point in time."""

    avg_wait_time: float = Field(0.0, description="Average vehicle wait (seconds)")
    avg_queue_length: float = Field(0.0, description="Average queue across lanes")
    throughput: int = Field(0, description="Vehicles that passed through")
    congestion_score: float = Field(
        0.0, ge=0, le=100, description="0 = free-flow, 100 = gridlock"
    )
    emission_estimate: float = Field(
        0.0, ge=0, description="Estimated CO2 in grams"
    )
    avg_speed: float = Field(0.0, ge=0, description="Average speed km/h")
    fuel_consumption_l: float = Field(
        0.0, ge=0, description="Estimated fuel litres consumed"
    )


class SimulationState(BaseModel):
    """Full state returned by SimulationEngine.get_state()."""

    tick: int = 0
    time_of_day: float = Field(
        8.0, ge=0, le=24, description="Hour of day (fractional)"
    )
    weather: str = "sunny"
    rain_intensity: float = 0.0
    intersections: list[IntersectionState] = Field(default_factory=list)
    total_vehicles: int = 0
    passed_vehicles: int = 0
    metrics: MetricsSnapshot = Field(default_factory=MetricsSnapshot)


class MetricsResponse(BaseModel):
    """Comparison response: before vs after AI optimisation."""

    avg_wait_time: float = 0.0
    avg_queue_length: float = 0.0
    throughput: int = 0
    congestion_score: float = 0.0
    emission_estimate: float = 0.0
    before_metrics: Optional[MetricsSnapshot] = None
    after_metrics: Optional[MetricsSnapshot] = None


class ScenarioInfo(BaseModel):
    """Public-facing scenario metadata."""

    id: str
    name: str
    description: str
    time_of_day: float
    weather: str
    vehicle_mix: dict[str, float]


class SignalTimingUpdate(BaseModel):
    """Payload for overriding signal timing at one intersection."""

    intersection_id: str
    timing: dict[str, float] = Field(
        ...,
        description="Mapping of direction -> green-phase duration in seconds",
    )
