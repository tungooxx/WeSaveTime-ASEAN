"""
FlowMind AI - SUMO-backed traffic simulation engine.

Drop-in replacement for SimulationEngine that uses SUMO via TraCI for
microsimulation of vehicle dynamics on the Hanoi Hoan Kiem grid.
"""

from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path
from typing import Optional

import traci
import sumolib


# ---------------------------------------------------------------------------
# Intersection metadata (must match SUMO .nod.xml and old engine)
# ---------------------------------------------------------------------------

_INTERSECTIONS = [
    ("HK01", "Hang Bai - Trang Thi",         21.02500, 105.84920),
    ("HK02", "Hang Khay - Dinh Tien Hoang",  21.02500, 105.85280),
    ("HK03", "Ba Trieu - Hai Ba Trung",       21.02260, 105.84920),
    ("HK04", "Le Thai To - Hang Trong",       21.02740, 105.85280),
    ("HK05", "Trang Tien - Ngo Quyen",        21.02740, 105.84920),
    ("HK06", "Ly Thuong Kiet - Quang Trung",  21.02260, 105.85280),
]

# Edges approaching each intersection, grouped by cardinal direction
# These map TLS id -> {direction: [incoming_edge_ids]}
_APPROACH_EDGES: dict[str, dict[str, list[str]]] = {
    "HK01": {"N": ["HK05_HK01"], "S": ["HK03_HK01"], "E": ["HK02_HK01"], "W": ["W_HK01_in"]},
    "HK02": {"N": ["HK04_HK02"], "S": ["HK06_HK02"], "E": ["E_HK02_in"],  "W": ["HK01_HK02"]},
    "HK03": {"N": ["HK01_HK03"], "S": ["S_HK03_in"],  "E": ["HK06_HK03"], "W": ["W_HK03_in"]},
    "HK04": {"N": ["N_HK04_in"],  "S": ["HK02_HK04"], "E": ["E_HK04_in"],  "W": ["HK05_HK04"]},
    "HK05": {"N": ["N_HK05_in"],  "S": ["HK01_HK05"], "E": ["HK04_HK05"], "W": ["W_HK05_in"]},
    "HK06": {"N": ["HK02_HK06"], "S": ["S_HK06_in"],  "E": ["E_HK06_in"],  "W": ["HK03_HK06"]},
}

# Default SUMO config location (relative to project root)
_DEFAULT_CFG = os.path.join(os.path.dirname(__file__), "..", "..", "sumo", "hanoi_hk.sumocfg")


def _find_sumo_binary(gui: bool = False) -> str:
    """Locate the sumo or sumo-gui binary."""
    name = "sumo-gui" if gui else "sumo"
    # pip-installed eclipse-sumo
    found = shutil.which(name)
    if found:
        return found
    # SUMO_HOME
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        p = os.path.join(sumo_home, "bin", name)
        if os.path.isfile(p) or os.path.isfile(p + ".exe"):
            return p
    raise FileNotFoundError(f"Cannot find '{name}'. Install SUMO or set SUMO_HOME.")


class SumoEngine:
    """SUMO-backed simulation engine with the same public API as SimulationEngine."""

    def __init__(
        self,
        sumo_cfg: Optional[str] = None,
        gui: bool = False,
        seed: int = 42,
    ) -> None:
        self._cfg = os.path.abspath(sumo_cfg or _DEFAULT_CFG)
        self._gui = gui
        self._seed = seed
        self._lock = threading.Lock()
        self._conn_label = "flowmind"

        # State
        self.tick: int = 0
        self.total_vehicles_passed: int = 0
        self.total_fuel_l: float = 0.0
        self.total_co2_g: float = 0.0

        # Scenario state (for API compatibility — time_of_day is a float: 8.0 = 08:00)
        self._time_of_day: float = 8.0
        self._weather: str = "sunny"
        self._rain_intensity: float = 0.0
        self._scenario_id: str = "sunny_morning"
        self._base_density: float = 30.0

        # Snapshot for before/after comparison
        self._before_metrics: Optional[dict] = None

        # Emergency vehicle tracking
        self._emergencies: list[dict] = []

        # Blocked lanes
        self._blocked_lanes: set[str] = set()

        # Intersection lookup
        self._ix_meta = {ix[0]: ix for ix in _INTERSECTIONS}

        # Start SUMO
        self._start_sumo()

    # ------------------------------------------------------------------
    # SUMO lifecycle
    # ------------------------------------------------------------------

    def _start_sumo(self) -> None:
        # Close any stale connection with the same label
        try:
            old = traci.getConnection(self._conn_label)
            old.close()
        except (traci.TraCIException, KeyError):
            pass

        binary = _find_sumo_binary(self._gui)
        cmd = [
            binary,
            "-c", self._cfg,
            "--seed", str(self._seed),
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--time-to-teleport", "-1",  # disable teleporting
        ]
        traci.start(cmd, label=self._conn_label)
        self._conn = traci.getConnection(self._conn_label)

    def _restart_sumo(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
        self._start_sumo()

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def step(self, dt: Optional[float] = None) -> dict:
        """Advance SUMO by one simulation step. Returns metrics."""
        with self._lock:
            self._conn.simulationStep()
            self.tick += 1

            # Advance simulated clock (SUMO step = 0.5s)
            self._time_of_day += 0.5 / 3600.0  # seconds → hours
            if self._time_of_day >= 24.0:
                self._time_of_day -= 24.0

            # Count vehicles that left the network this step
            arrived = self._conn.simulation.getArrivedNumber()
            self.total_vehicles_passed += arrived

            # Accumulate emissions
            for veh_id in self._conn.vehicle.getIDList():
                self.total_co2_g += self._conn.vehicle.getCO2Emission(veh_id) / 1000.0
                self.total_fuel_l += self._conn.vehicle.getFuelConsumption(veh_id) / 1000.0

        return self.get_metrics()

    def run(self, steps: int = 100, dt: float = 1.0) -> dict:
        for _ in range(steps):
            self.step(dt)
        return self.get_metrics()

    # ------------------------------------------------------------------
    # State and metrics (same schema as SimulationEngine)
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        with self._lock:
            intersections = []
            for ix_id, name, lat, lon in _INTERSECTIONS:
                ix_data = self._get_intersection_state(ix_id, name, lat, lon)
                intersections.append(ix_data)

            total_vehicles = self._conn.vehicle.getIDCount()

        return {
            "tick": self.tick,
            "time_of_day": self._time_of_day,
            "weather": self._weather,
            "rain_intensity": self._rain_intensity,
            "intersections": intersections,
            "total_vehicles": total_vehicles,
            "passed_vehicles": self.total_vehicles_passed,
            "metrics": self.get_metrics(),
            "emergencies": list(self._emergencies),
            "base_density": self._base_density,
        }

    def _get_intersection_state(self, ix_id: str, name: str, lat: float, lon: float) -> dict:
        """Build per-intersection state dict matching old engine format."""
        # Traffic light state
        try:
            tls_state = self._conn.trafficlight.getRedYellowGreenState(ix_id)
            phase_idx = self._conn.trafficlight.getPhase(ix_id)
        except traci.TraCIException:
            tls_state = ""
            phase_idx = 0

        signal_map = self._parse_tls_state(ix_id, tls_state)

        # Queue lengths per direction
        queues: dict[str, int] = {}
        wait_times: dict[str, float] = {}
        approach = _APPROACH_EDGES.get(ix_id, {})
        for direction, edges in approach.items():
            q = 0
            w = 0.0
            for eid in edges:
                try:
                    q += self._conn.edge.getLastStepHaltingNumber(eid)
                    w += self._conn.edge.getWaitingTime(eid)
                except traci.TraCIException:
                    pass
            queues[direction] = q
            wait_times[direction] = round(w, 2)

        # Lanes
        lanes = []
        for direction in ["N", "S", "E", "W"]:
            lane_id = f"{ix_id}_{direction}"
            lanes.append({
                "id": lane_id,
                "direction": direction,
                "queue_length": queues.get(direction, 0),
                "wait_time": wait_times.get(direction, 0.0),
                "occupancy": min(1.0, queues.get(direction, 0) / 20.0),
                "blocked": lane_id in self._blocked_lanes,
                "vehicle_count": queues.get(direction, 0),
            })

        total_queue = sum(queues.values())
        total_wait = sum(wait_times.values())
        avg_wait = total_wait / max(len(wait_times), 1)

        return {
            "id": ix_id,
            "name": name,
            "lat": lat,
            "lon": lon,
            "signal_state": signal_map,
            "phase_index": phase_idx,
            "lanes": lanes,
            "queue_lengths": queues,
            "avg_wait_time": round(avg_wait, 2),
            "total_queue": total_queue,
        }

    def _parse_tls_state(self, tls_id: str, state_str: str) -> dict[str, str]:
        """Convert SUMO's signal state string to {direction: color}.

        SUMO state chars: G/g = green, y = yellow, r = red.
        With 16 connections per intersection (4 approaches × 2 lanes × 2 turns),
        we check the first link index for each approach direction.
        """
        if not state_str:
            return {"N": "red", "S": "red", "E": "red", "W": "red"}

        n = len(state_str)
        # 4 approach groups, each has n//4 link indices
        group_size = max(n // 4, 1)

        result = {}
        directions = ["N", "S", "E", "W"]
        for i, d in enumerate(directions):
            start = i * group_size
            if start < n:
                ch = state_str[start].lower()
                if ch == 'g':
                    result[d] = "green"
                elif ch == 'y':
                    result[d] = "yellow"
                else:
                    result[d] = "red"
            else:
                result[d] = "red"
        return result

    def get_metrics(self) -> dict:
        with self._lock:
            all_waits = []
            all_queues = []

            for ix_id in [ix[0] for ix in _INTERSECTIONS]:
                approach = _APPROACH_EDGES.get(ix_id, {})
                for direction, edges in approach.items():
                    for eid in edges:
                        try:
                            all_queues.append(self._conn.edge.getLastStepHaltingNumber(eid))
                            all_waits.append(self._conn.edge.getWaitingTime(eid))
                        except traci.TraCIException:
                            pass

            # Average speed across all vehicles
            speeds = []
            for veh_id in self._conn.vehicle.getIDList():
                speeds.append(self._conn.vehicle.getSpeed(veh_id) * 3.6)  # m/s → km/h

        avg_wait = sum(all_waits) / max(len(all_waits), 1)
        avg_queue = sum(all_queues) / max(len(all_queues), 1)
        avg_speed = sum(speeds) / max(len(speeds), 1)

        max_possible_queue = 20
        queue_ratio = min(avg_queue / max(max_possible_queue, 1), 1.0)
        wait_ratio = min(avg_wait / 120.0, 1.0)
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
    # Vehicle positions (NEW — for canvas rendering)
    # ------------------------------------------------------------------

    def get_vehicle_positions(self) -> list[dict]:
        """Return all vehicle positions as [{id, type, lon, lat, speed, angle}]."""
        vehicles = []
        with self._lock:
            for veh_id in self._conn.vehicle.getIDList():
                try:
                    x, y = self._conn.vehicle.getPosition(veh_id)
                    lon, lat = self._conn.simulation.convertGeo(x, y)
                    speed = self._conn.vehicle.getSpeed(veh_id) * 3.6  # km/h
                    angle = self._conn.vehicle.getAngle(veh_id)
                    vtype = self._conn.vehicle.getTypeID(veh_id)
                    length = self._conn.vehicle.getLength(veh_id)
                    vehicles.append({
                        "id": veh_id,
                        "type": vtype,
                        "lon": round(lon, 7),
                        "lat": round(lat, 7),
                        "speed": round(speed, 1),
                        "angle": round(angle, 1),
                        "length": round(length, 1),
                    })
                except traci.TraCIException:
                    pass
        return vehicles

    # ------------------------------------------------------------------
    # Signal timing control
    # ------------------------------------------------------------------

    def set_signal_timing(self, intersection_id: str, timing: dict) -> bool:
        """Override signal timing. timing: {direction: green_seconds} or
        {"green_ns": X, "green_ew": Y} format.
        """
        with self._lock:
            try:
                # Get current program
                logic = self._conn.trafficlight.getAllProgramLogics(intersection_id)
                if not logic:
                    return False

                prog = logic[0]
                phases = list(prog.phases)

                # Determine NS/EW green durations
                ns_green = timing.get("N", timing.get("green_ns", 30))
                ew_green = timing.get("E", timing.get("green_ew", 25))

                # Update phase durations (assuming standard 4-phase: NS-green, NS-yellow, EW-green, EW-yellow)
                if len(phases) >= 4:
                    phases[0] = traci.trafficlight.Phase(int(ns_green), phases[0].state)
                    phases[1] = traci.trafficlight.Phase(3, phases[1].state)  # yellow
                    phases[2] = traci.trafficlight.Phase(int(ew_green), phases[2].state)
                    phases[3] = traci.trafficlight.Phase(3, phases[3].state)  # yellow

                new_logic = traci.trafficlight.Logic(
                    prog.programID, prog.type, prog.currentPhaseIndex, tuple(phases)
                )
                self._conn.trafficlight.setProgramLogic(intersection_id, new_logic)
                return True
            except traci.TraCIException:
                return False

    # ------------------------------------------------------------------
    # Scenario management (compatibility layer)
    # ------------------------------------------------------------------

    def apply_scenario(self, scenario_id: str):
        """Apply a scenario. Returns a scenario-like object for API compat."""
        from .scenarios import get_scenario
        self._before_metrics = self.get_metrics()

        sc = get_scenario(scenario_id)
        self._scenario_id = scenario_id
        self._time_of_day = sc.time_of_day
        self._weather = sc.weather
        self._rain_intensity = sc.rain_intensity
        self._base_density = sc.base_density

        # Handle special events
        for evt in sc.special_events:
            etype = evt.get("type")
            if etype == "blocked_lane":
                idx = evt.get("intersection_idx", 0)
                if 0 <= idx < len(_INTERSECTIONS):
                    ix_id = _INTERSECTIONS[idx][0]
                    direction = evt.get("lane_direction", "S")
                    self._blocked_lanes.add(f"{ix_id}_{direction}")
            elif etype == "emergency_vehicle":
                from_idx = evt.get("from_intersection_idx", 0)
                to_idx = evt.get("to_intersection_idx", len(_INTERSECTIONS) - 1)
                self.inject_emergency_vehicle(from_idx, to_idx)

        return sc

    # ------------------------------------------------------------------
    # Density control
    # ------------------------------------------------------------------

    def set_base_density(self, density: float) -> None:
        self._base_density = max(1.0, min(density, 100.0))
        # Adjust SUMO flow by scaling vehicle insertion probability
        # This is a simplified approach — for more control, modify route files
        # For now we just track the value for API compatibility

    def get_base_density(self) -> float:
        return self._base_density

    # ------------------------------------------------------------------
    # Congestion detection
    # ------------------------------------------------------------------

    def detect_congestion(self, threshold: float = 0.6) -> list[dict]:
        congested = []
        with self._lock:
            for ix_id, name, lat, lon in _INTERSECTIONS:
                approach = _APPROACH_EDGES.get(ix_id, {})
                queues = {}
                total_occ = 0.0
                count = 0
                for direction, edges in approach.items():
                    q = 0
                    for eid in edges:
                        try:
                            q += self._conn.edge.getLastStepHaltingNumber(eid)
                        except traci.TraCIException:
                            pass
                    queues[direction] = q
                    occ = min(q / 20.0, 1.0)
                    total_occ += occ
                    count += 1

                avg_occ = total_occ / max(count, 1)
                if avg_occ >= threshold:
                    wait = 0.0
                    for edges_list in approach.values():
                        for eid in edges_list:
                            try:
                                wait += self._conn.edge.getWaitingTime(eid)
                            except traci.TraCIException:
                                pass
                    congested.append({
                        "intersection_id": ix_id,
                        "name": name,
                        "avg_occupancy": round(avg_occ, 3),
                        "queue_lengths": queues,
                        "avg_wait_time": round(wait / max(count, 1), 2),
                    })
        return congested

    # ------------------------------------------------------------------
    # Lane blocking
    # ------------------------------------------------------------------

    def block_lane(self, intersection_id: str, lane_id: str) -> bool:
        self._blocked_lanes.add(lane_id)
        return True

    # ------------------------------------------------------------------
    # Emergency vehicles
    # ------------------------------------------------------------------

    def inject_emergency_vehicle(self, from_intersection: int, to_intersection: int) -> None:
        if 0 <= from_intersection < len(_INTERSECTIONS) and 0 <= to_intersection < len(_INTERSECTIONS):
            ix = _INTERSECTIONS[from_intersection]
            self._emergencies.append({
                "lat": ix[2],
                "lon": ix[3],
                "intersection_id": ix[0],
                "position": 150.0,
                "speed": 60.0,
            })

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = 42) -> None:
        self._seed = seed if seed is not None else 42
        self.tick = 0
        self.total_vehicles_passed = 0
        self.total_fuel_l = 0.0
        self.total_co2_g = 0.0
        self._before_metrics = None
        self._emergencies.clear()
        self._blocked_lanes.clear()
        self._restart_sumo()

    def snapshot_before(self) -> None:
        self._before_metrics = self.get_metrics()

    def list_scenarios(self) -> list[dict]:
        from .scenarios import list_scenarios as _ls
        return [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "time_of_day": s.time_of_day,
                "weather": s.weather,
                "vehicle_mix": s.vehicle_mix,
            }
            for s in _ls()
        ]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()
