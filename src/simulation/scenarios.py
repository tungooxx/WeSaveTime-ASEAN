"""
FlowMind AI - Six preset traffic scenarios for Vietnamese smart-city simulation.

Each scenario captures time-of-day, weather, vehicle composition, traffic density,
and optional special events (lane blockages, emergency vehicles, school surges).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Scenario:
    """Full description of a simulation scenario."""

    id: str
    name: str
    description: str
    time_of_day: float                          # 0-24 fractional hour
    weather: str                                # sunny / rainy / cloudy / clear
    rain_intensity: float = 0.0                 # 0-1
    vehicle_mix: dict[str, float] = field(default_factory=dict)  # type -> ratio (sums to ~1)
    base_density: float = 20.0                  # vehicles spawned per simulation-minute
    road_speed_factor: float = 1.0              # multiplier on free-flow speed
    visibility: float = 1.0                     # 0-1
    special_events: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# The six preset scenarios
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, Scenario] = {}


def _register(s: Scenario) -> Scenario:
    SCENARIOS[s.id] = s
    return s


_register(Scenario(
    id="sunny_morning",
    name="Sunny Morning Commute",
    description=(
        "Typical 8 AM weekday in Hanoi. Streets dominated by motorbikes with "
        "moderate car traffic heading into the Hoan Kiem business district."
    ),
    time_of_day=8.0,
    weather="sunny",
    rain_intensity=0.0,
    vehicle_mix={"motorbike": 0.60, "car": 0.25, "bus": 0.10, "truck": 0.05},
    base_density=30.0,
    road_speed_factor=1.0,
    visibility=1.0,
))

_register(Scenario(
    id="heavy_rain",
    name="Heavy Afternoon Rain",
    description=(
        "Sudden tropical downpour at 3 PM. Visibility drops, road surfaces "
        "become slippery, and many motorbike riders switch to taxis/cars."
    ),
    time_of_day=15.0,
    weather="rainy",
    rain_intensity=0.8,
    vehicle_mix={"motorbike": 0.35, "car": 0.40, "bus": 0.15, "truck": 0.10},
    base_density=25.0,
    road_speed_factor=0.6,
    visibility=0.4,
))

_register(Scenario(
    id="nighttime",
    name="Late-Night Low Traffic",
    description=(
        "11 PM on a weeknight. Very few vehicles on the road, higher speeds "
        "possible. Mostly cars and a handful of trucks making deliveries."
    ),
    time_of_day=23.0,
    weather="clear",
    rain_intensity=0.0,
    vehicle_mix={"motorbike": 0.20, "car": 0.45, "bus": 0.05, "truck": 0.30},
    base_density=8.0,
    road_speed_factor=1.1,
    visibility=0.7,
))

_register(Scenario(
    id="lane_disruption",
    name="Lane Disruption (Construction / Accident)",
    description=(
        "5 PM rush hour with a lane blocked on a major approach. Traffic must "
        "merge, causing cascading congestion upstream."
    ),
    time_of_day=17.0,
    weather="cloudy",
    rain_intensity=0.0,
    vehicle_mix={"motorbike": 0.50, "car": 0.30, "bus": 0.12, "truck": 0.08},
    base_density=32.0,
    road_speed_factor=0.85,
    visibility=0.9,
    special_events=[
        {"type": "blocked_lane", "intersection_idx": 2, "lane_direction": "S"},
    ],
))

_register(Scenario(
    id="rush_hour_school",
    name="School Rush Hour",
    description=(
        "7:30 AM near a school zone. A directional surge of motorbikes and "
        "cars from the west, plus slow-moving buses picking up students."
    ),
    time_of_day=7.5,
    weather="sunny",
    rain_intensity=0.0,
    vehicle_mix={"motorbike": 0.55, "car": 0.25, "bus": 0.15, "truck": 0.05},
    base_density=40.0,
    road_speed_factor=0.9,
    visibility=1.0,
    special_events=[
        {"type": "directional_surge", "direction": "W", "multiplier": 2.5},
    ],
))

_register(Scenario(
    id="emergency_vehicle",
    name="Emergency Vehicle Priority",
    description=(
        "Normal 2 PM traffic with an ambulance that must cross the network "
        "from intersection 0 to intersection 5 with signal pre-emption."
    ),
    time_of_day=14.0,
    weather="sunny",
    rain_intensity=0.0,
    vehicle_mix={"motorbike": 0.50, "car": 0.30, "bus": 0.12, "truck": 0.08},
    base_density=22.0,
    road_speed_factor=1.0,
    visibility=1.0,
    special_events=[
        {
            "type": "emergency_vehicle",
            "from_intersection_idx": 0,
            "to_intersection_idx": 5,
            "vehicle_type": "ambulance",
            "priority": "absolute",
        },
    ],
))


def get_scenario(scenario_id: str) -> Scenario:
    """Return a scenario by id, raising KeyError if unknown."""
    if scenario_id not in SCENARIOS:
        raise KeyError(
            f"Unknown scenario '{scenario_id}'. "
            f"Available: {list(SCENARIOS.keys())}"
        )
    return SCENARIOS[scenario_id]


def list_scenarios() -> list[Scenario]:
    """Return all scenarios in definition order."""
    return list(SCENARIOS.values())
