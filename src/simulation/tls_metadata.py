"""
FlowMind AI - Dynamic TLS (Traffic Light System) discovery from SUMO networks.

Reads a SUMO .net.xml file via sumolib and extracts phase programs,
incoming edges, and controlled links for each traffic light.  This lets
the RL environment work with *any* SUMO network (Hanoi, Da Nang, …)
without hard-coding intersection metadata.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional

import sumolib


# ──────────────────────────────────────────────────────────────────────
# Constants — Vietnamese traffic engineering
# ──────────────────────────────────────────────────────────────────────

_DOWNTOWN_SPEED_CAP_MS = 13.89      # 50 km/h — max for urban Hai Chau
_DECEL_RATE = 3.05                   # m/s² comfortable deceleration (ITE)
_PED_WALK_S = 7.0                    # pedestrian walk interval (MUTCD)
_PED_SPEED_MS = 1.0                  # Vietnamese walking speed m/s
_VEH_LENGTH_M = 6.0                  # clearance vehicle length
_MIN_YELLOW_S = 3.0                  # MUTCD minimum yellow
_MIN_ALLRED_S = 1.5                  # minimum all-red clearance
_MAX_CYCLE_STEPS = 240               # 120 real seconds at step_length=0.5
_VN_SATURATION_FLOW = 4092           # PCU/hr/lane (Vietnamese measured)
_SIM_STEP_LENGTH = 0.5               # seconds per simulation step

# Tier-based limits
_TIER_PARAMS = {
    #               veh_min_green_s, max_green_s
    "small":   (10,  25),
    "medium":  (15,  45),
    "large":   (20,  60),
}


# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────

@dataclass
class PhaseInfo:
    """One phase of a traffic light program."""
    index: int
    state: str          # e.g. "GGrrGGrr"
    duration: float     # seconds
    min_dur: float = 0.0
    max_dur: float = 0.0


@dataclass
class TLSGeometry:
    """Per-intersection timing derived from engineering formulas.

    All *_steps values are in simulation steps (2 steps = 1 real second
    at step_length=0.5).
    """
    width_m: float                       # junction bounding-box extent
    approach_speed_ms: float             # max incoming speed (capped)
    total_lanes: int
    tier: str                            # "small" / "medium" / "large"
    # Computed timing in sim steps
    yellow_steps: int
    allred_steps: int
    min_green_steps: int                 # max(vehicle_min, ped_min)
    max_green_steps: int
    # Reference values (real seconds, for logging)
    yellow_s: float
    allred_s: float
    ped_min_green_s: float               # pedestrian binding constraint
    min_green_s: float
    max_green_s: float
    # Webster's default cycle
    default_cycle_s: float = 0.0
    default_green_splits: list[float] = field(default_factory=list)


def _seconds_to_steps(s: float) -> int:
    """Convert real seconds to simulation steps."""
    return max(1, round(s / _SIM_STEP_LENGTH))


def _junction_width(net, tls_id: str) -> float:
    """Estimate junction width from node shape bounding box."""
    try:
        node = net.getNode(tls_id)
        shape = node.getShape()
        if shape and len(shape) >= 2:
            xs = [p[0] for p in shape]
            ys = [p[1] for p in shape]
            return max(max(xs) - min(xs), max(ys) - min(ys))
    except Exception:
        pass
    return 30.0  # fallback median


def _max_approach_speed(net, tls_id: str) -> float:
    """Get max speed of incoming edges, capped at downtown limit."""
    try:
        node = net.getNode(tls_id)
        speeds = [
            e.getSpeed() for e in node.getIncoming()
            if not e.getID().startswith(":")
        ]
        if speeds:
            return min(max(speeds), _DOWNTOWN_SPEED_CAP_MS)
    except Exception:
        pass
    return _DOWNTOWN_SPEED_CAP_MS


def _total_incoming_lanes(net, tls_id: str) -> int:
    try:
        node = net.getNode(tls_id)
        return sum(
            e.getLaneNumber() for e in node.getIncoming()
            if not e.getID().startswith(":")
        )
    except Exception:
        return 4


def compute_tls_geometry(
    net, tls_id: str, num_green_phases: int,
) -> TLSGeometry:
    """Compute per-intersection timing from engineering formulas."""
    width = _junction_width(net, tls_id)
    speed = _max_approach_speed(net, tls_id)
    lanes = _total_incoming_lanes(net, tls_id)

    # Tier classification
    if width < 25:
        tier = "small"
    elif width < 50:
        tier = "medium"
    else:
        tier = "large"

    veh_min_s, max_green_s = _TIER_PARAMS[tier]

    # ── Yellow (ITE / Kell-Fullerton) ──────────────────────────────
    yellow_s = max(_MIN_YELLOW_S, 1.0 + speed / (2 * _DECEL_RATE))

    # ── All-red clearance ──────────────────────────────────────────
    allred_s = max(_MIN_ALLRED_S, (width + _VEH_LENGTH_M) / max(speed, 1.0))

    # ── Pedestrian minimum green ───────────────────────────────────
    ped_min_s = _PED_WALK_S + width / _PED_SPEED_MS

    # ── Actual min green = max(vehicle, ped), capped at max_green ─
    min_green_s = max(veh_min_s, min(ped_min_s, max_green_s))

    # ── Max cycle constraint ───────────────────────────────────────
    # Ensure total cycle ≤ 120 real seconds
    n = max(num_green_phases, 1)
    total_cycle_s = n * (min_green_s + yellow_s + allred_s)
    if total_cycle_s > 120:
        available = 120 - n * (yellow_s + allred_s)
        min_green_s = max(available / n, 7.0)  # floor 7s

    # ── Webster's optimal cycle ────────────────────────────────────
    lost_per_phase = 1.5 + yellow_s + allred_s  # startup + clearance
    L = n * lost_per_phase
    Y = min(n * 0.30, 0.90)  # default: 30% capacity per phase
    denom = max(1 - Y, 0.10)
    c_opt = (1.5 * L + 5) / denom
    c_opt = max(60, min(c_opt, 120))
    available_green = c_opt - L
    splits = [available_green / n] * n

    # Convert to sim steps
    return TLSGeometry(
        width_m=round(width, 1),
        approach_speed_ms=round(speed, 2),
        total_lanes=lanes,
        tier=tier,
        yellow_steps=_seconds_to_steps(yellow_s),
        allred_steps=_seconds_to_steps(allred_s),
        min_green_steps=_seconds_to_steps(min_green_s),
        max_green_steps=_seconds_to_steps(max_green_s),
        yellow_s=round(yellow_s, 1),
        allred_s=round(allred_s, 1),
        ped_min_green_s=round(ped_min_s, 1),
        min_green_s=round(min_green_s, 1),
        max_green_s=round(max_green_s, 1),
        default_cycle_s=round(c_opt, 1),
        default_green_splits=[round(s, 1) for s in splits],
    )


@dataclass
class TLSInfo:
    """Metadata for a single traffic light system."""
    id: str
    incoming_edges: list[str] = field(default_factory=list)
    incoming_lanes: list[str] = field(default_factory=list)
    phases: list[PhaseInfo] = field(default_factory=list)
    num_connections: int = 0
    geometry: Optional[TLSGeometry] = None

    @property
    def num_phases(self) -> int:
        return len(self.phases)

    @property
    def num_green_phases(self) -> int:
        """Count phases that contain at least one green (G/g) and are not
        purely yellow/all-red transitions."""
        count = 0
        for p in self.phases:
            has_green = any(c in ("G", "g") for c in p.state)
            all_yr = all(c in ("y", "Y", "r", "R") for c in p.state)
            if has_green and not all_yr:
                count += 1
        return count

    def green_phase_indices(self) -> list[int]:
        """Return the *phase indices* that are usable green phases."""
        indices: list[int] = []
        for p in self.phases:
            has_green = any(c in ("G", "g") for c in p.state)
            all_yr = all(c in ("y", "Y", "r", "R") for c in p.state)
            if has_green and not all_yr:
                indices.append(p.index)
        return indices


# ──────────────────────────────────────────────────────────────────────
# TLS discovery
# ──────────────────────────────────────────────────────────────────────

class TLSMetadata:
    """Discover and cache TLS metadata from a SUMO .net.xml file."""

    def __init__(self, net_file: str) -> None:
        self.net_file = os.path.abspath(net_file)
        self._net = sumolib.net.readNet(self.net_file, withPrograms=True)
        self._tls_map: dict[str, TLSInfo] = {}
        self._discover()

    # ── internal ──────────────────────────────────────────────────────

    def _discover(self) -> None:
        for tls in self._net.getTrafficLights():
            tls_id = tls.getID()
            info = TLSInfo(id=tls_id)

            # Phases from the first program
            programs = tls.getPrograms()
            if programs:
                prog = list(programs.values())[0]
                for i, phase in enumerate(prog.getPhases()):
                    info.phases.append(PhaseInfo(
                        index=i,
                        state=phase.state,
                        duration=phase.duration,
                        min_dur=getattr(phase, "minDur", phase.duration),
                        max_dur=getattr(phase, "maxDur", phase.duration),
                    ))

            # Incoming edges / lanes via controlled connections
            connections = tls.getConnections()
            info.num_connections = len(connections)

            seen_edges: set[str] = set()
            seen_lanes: set[str] = set()
            for conn in connections:
                in_lane = conn[0]
                edge_id = in_lane.getEdge().getID()
                lane_id = in_lane.getID()
                if edge_id not in seen_edges:
                    info.incoming_edges.append(edge_id)
                    seen_edges.add(edge_id)
                if lane_id not in seen_lanes:
                    info.incoming_lanes.append(lane_id)
                    seen_lanes.add(lane_id)

            # Compute per-intersection geometry & timing
            info.geometry = compute_tls_geometry(
                self._net, tls_id, info.num_green_phases,
            )

            self._tls_map[tls_id] = info

    # ── public API ────────────────────────────────────────────────────

    @property
    def all_tls(self) -> list[TLSInfo]:
        return list(self._tls_map.values())

    def get(self, tls_id: str) -> Optional[TLSInfo]:
        return self._tls_map.get(tls_id)

    def get_non_trivial(
        self,
        min_green_phases: int = 2,
        min_incoming: int = 2,
        exclude_roundabout_radius: float = 200.0,
    ) -> list[TLSInfo]:
        """Return only TLS worth optimizing.

        Skips:
          - Pedestrian crossings / trivial signals
          - TLS near roundabouts (within exclude_roundabout_radius meters)
            because roundabouts work better with yield-based flow.
        """
        # Find roundabout centers
        ra_centers: list[tuple[float, float]] = []
        for ra in self._net.getRoundabouts():
            xs, ys = [], []
            for nid in ra.getNodes():
                try:
                    n = self._net.getNode(nid)
                    x, y = n.getCoord()
                    xs.append(x); ys.append(y)
                except Exception:
                    pass
            if xs:
                ra_centers.append((sum(xs) / len(xs), sum(ys) / len(ys)))

        import math

        def _near_roundabout(tls_id: str) -> bool:
            if not ra_centers:
                return False
            try:
                node = self._net.getNode(tls_id)
                tx, ty = node.getCoord()
            except Exception:
                return False
            for cx, cy in ra_centers:
                if math.sqrt((tx - cx) ** 2 + (ty - cy) ** 2) < exclude_roundabout_radius:
                    return True
            return False

        return [
            info
            for info in self._tls_map.values()
            if info.num_green_phases >= min_green_phases
            and len(info.incoming_edges) >= min_incoming
            and not _near_roundabout(info.id)
        ]

    def get_tls_ids(self) -> list[str]:
        return list(self._tls_map.keys())

    def __len__(self) -> int:
        return len(self._tls_map)

    def summary(self) -> dict:
        all_tls = self.all_tls
        non_trivial = self.get_non_trivial()
        n = max(len(all_tls), 1)
        return {
            "total_tls": len(all_tls),
            "non_trivial_tls": len(non_trivial),
            "avg_phases": round(sum(t.num_phases for t in all_tls) / n, 1),
            "avg_incoming_edges": round(
                sum(len(t.incoming_edges) for t in all_tls) / n, 1
            ),
            "max_phases": max((t.num_phases for t in all_tls), default=0),
            "max_incoming_edges": max(
                (len(t.incoming_edges) for t in all_tls), default=0
            ),
        }


# ──────────────────────────────────────────────────────────────────────
# Junction analysis — find uncontrolled intersections that may need a TLS
# ──────────────────────────────────────────────────────────────────────

@dataclass
class JunctionInfo:
    """Metadata for an uncontrolled junction."""
    id: str
    x: float
    y: float
    junction_type: str          # "priority", "right_before_left", etc.
    incoming_edges: list[str] = field(default_factory=list)
    num_connections: int = 0
    num_incoming_lanes: int = 0


def discover_uncontrolled_junctions(
    net_file: str,
    min_incoming_edges: int = 3,
) -> list[JunctionInfo]:
    """Find junctions WITHOUT traffic lights that have enough incoming roads
    to potentially benefit from signal control.

    Args:
        net_file: path to SUMO .net.xml
        min_incoming_edges: only return junctions with at least this many
            distinct incoming edges (skip simple T-intersections and merges)

    Returns:
        List of JunctionInfo for candidate locations, sorted by number
        of incoming edges (busiest first).
    """
    net = sumolib.net.readNet(os.path.abspath(net_file))

    tls_node_ids: set[str] = set()
    for tls in net.getTrafficLights():
        tls_node_ids.add(tls.getID())

    results: list[JunctionInfo] = []

    for node in net.getNodes():
        nid = node.getID()
        ntype = node.getType()

        # Skip nodes that already have TLS
        if nid in tls_node_ids:
            continue

        # Skip internal / dead-end / rail nodes
        if ntype in ("internal", "dead_end", "rail_signal", "rail_crossing"):
            continue

        incoming = node.getIncoming()
        if len(incoming) < min_incoming_edges:
            continue

        x, y = node.getCoord()

        edge_ids = [e.getID() for e in incoming if not e.getID().startswith(":")]
        total_lanes = sum(e.getLaneNumber() for e in incoming if not e.getID().startswith(":"))

        conns = 0
        for e in incoming:
            conns += len(e.getOutgoing())

        results.append(JunctionInfo(
            id=nid,
            x=x,
            y=y,
            junction_type=ntype,
            incoming_edges=edge_ids,
            num_connections=conns,
            num_incoming_lanes=total_lanes,
        ))

    results.sort(key=lambda j: (len(j.incoming_edges), j.num_incoming_lanes), reverse=True)
    return results


def analyze_junctions_with_traci(
    conn,
    junctions: list[JunctionInfo],
    sample_steps: int = 100,
) -> list[dict]:
    """Run SUMO for *sample_steps* and measure congestion at uncontrolled junctions.

    Must be called while a TraCI connection is active and simulation is running.

    Returns a list of dicts sorted by congestion_score (worst first):
        {junction_id, incoming_edges, avg_wait, avg_queue, congestion_score,
         num_incoming, recommendation}
    """
    # Accumulate metrics, sampling every 10th step to reduce TraCI overhead
    accum: dict[str, dict] = {
        j.id: {"waits": [], "queues": [], "info": j}
        for j in junctions
    }
    sample_interval = 10

    for step in range(sample_steps):
        conn.simulationStep()
        if step % sample_interval != 0:
            continue
        for j in junctions:
            total_wait = 0.0
            total_queue = 0
            for eid in j.incoming_edges:
                try:
                    total_wait += conn.edge.getWaitingTime(eid)
                    total_queue += conn.edge.getLastStepHaltingNumber(eid)
                except Exception:
                    pass
            accum[j.id]["waits"].append(total_wait)
            accum[j.id]["queues"].append(total_queue)

    # Score each junction
    results: list[dict] = []
    for jid, data in accum.items():
        j = data["info"]
        avg_wait = sum(data["waits"]) / max(len(data["waits"]), 1)
        avg_queue = sum(data["queues"]) / max(len(data["queues"]), 1)
        # Score: weighted combination of wait + queue + road complexity
        score = (
            0.4 * min(avg_wait / 100.0, 1.0)
            + 0.3 * min(avg_queue / 20.0, 1.0)
            + 0.3 * min(len(j.incoming_edges) / 6.0, 1.0)
        )

        if score > 0.5:
            rec = "STRONGLY recommended for TLS"
        elif score > 0.3:
            rec = "Consider adding TLS"
        elif score > 0.15:
            rec = "Monitor — may need TLS during peak hours"
        else:
            rec = "No TLS needed"

        results.append({
            "junction_id": jid,
            "junction_type": j.junction_type,
            "incoming_edges": len(j.incoming_edges),
            "incoming_lanes": j.num_incoming_lanes,
            "avg_wait": round(avg_wait, 2),
            "avg_queue": round(avg_queue, 2),
            "congestion_score": round(score, 3),
            "recommendation": rec,
            "x": j.x,
            "y": j.y,
        })

    results.sort(key=lambda r: r["congestion_score"], reverse=True)
    return results
