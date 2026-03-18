"""
FlowMind AI - Dynamic traffic event system for SUMO simulations.

Supports: accident, concert, flood, heavy_rain, construction, vip_motorcade.
Each event applies TraCI changes (speed reduction, lane closure, demand surge)
and draws coloured polygons / POI markers on the SUMO-gui map.
Events auto-expire after their duration and all changes are reverted.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Optional

import sumolib
import traci


# ── Event data ──────────────────────────────────────────────────────────

EVENT_TYPES = [
    "accident",
    "concert",
    "flood",
    "heavy_rain",
    "construction",
    "vip",
]

EVENT_COLORS = {
    "accident":     (255, 50, 50, 180),      # red
    "concert":      (255, 165, 0, 160),       # orange
    "flood":        (50, 120, 220, 160),      # blue
    "heavy_rain":   (120, 120, 140, 100),     # dark gray
    "construction": (255, 220, 50, 160),      # yellow
    "vip":          (160, 80, 220, 160),      # purple
}

EVENT_LABELS = {
    "accident":     "Accident",
    "concert":      "Concert / Festival",
    "flood":        "Flood",
    "heavy_rain":   "Heavy Rain",
    "construction": "Road Construction",
    "vip":          "VIP Motorcade",
}


@dataclass
class TrafficEvent:
    id: str
    event_type: str
    affected_edges: list[str]
    intensity: float          # 0.0 – 1.0
    start_time: float         # sim seconds when event begins
    duration: float           # sim seconds
    description: str = ""
    _applied: bool = field(default=False, repr=False)
    _expired: bool = field(default=False, repr=False)

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    @property
    def label(self) -> str:
        return EVENT_LABELS.get(self.event_type, self.event_type)

    def time_remaining(self, sim_time: float) -> float:
        return max(0.0, self.end_time - sim_time)


# ── Saved state (for reverting) ─────────────────────────────────────────

@dataclass
class _SavedEdgeState:
    edge_id: str
    max_speed: float
    lane_allowed: dict[str, list[str]]  # lane_id -> allowed classes


# ── Event Manager ───────────────────────────────────────────────────────

class EventManager:
    """Manages dynamic traffic events during a SUMO simulation."""

    def __init__(self, conn: traci.Connection, net_file: str) -> None:
        self._conn = conn
        self._net = sumolib.net.readNet(net_file, withPrograms=False)
        self._events: dict[str, TrafficEvent] = {}
        self._saved: dict[str, list[_SavedEdgeState]] = {}  # event_id -> saved states
        self._poly_ids: dict[str, list[str]] = {}            # event_id -> polygon IDs
        self._poi_ids: dict[str, list[str]] = {}             # event_id -> POI IDs

    # ── Public API ──────────────────────────────────────────────────

    def add_event(self, event: TrafficEvent) -> str:
        self._events[event.id] = event
        return event.id

    def remove_event(self, event_id: str) -> None:
        event = self._events.get(event_id)
        if event and event._applied:
            self._revert(event)
        if event_id in self._events:
            del self._events[event_id]

    def get_active(self) -> list[TrafficEvent]:
        return [e for e in self._events.values() if e._applied and not e._expired]

    def get_all(self) -> list[TrafficEvent]:
        return list(self._events.values())

    def tick(self, sim_time: float) -> list[str]:
        """Call every sim step. Returns list of log messages for events that changed."""
        messages: list[str] = []

        for eid, event in list(self._events.items()):
            # Apply event when start_time is reached
            if not event._applied and sim_time >= event.start_time:
                self._apply(event)
                event._applied = True
                messages.append(
                    f"{event.label} triggered on "
                    f"{self._edge_name(event.affected_edges[0]) if event.affected_edges else 'city-wide'} "
                    f"({event.duration / 60:.0f} min)"
                )

            # Expire event when duration ends
            if event._applied and not event._expired and sim_time >= event.end_time:
                self._revert(event)
                event._expired = True
                messages.append(f"{event.label} ended — conditions restored")

        # Cleanup expired events
        expired = [eid for eid, e in self._events.items() if e._expired]
        for eid in expired:
            del self._events[eid]

        return messages

    # ── Apply events ────────────────────────────────────────────────

    def _apply(self, event: TrafficEvent) -> None:
        # Save original state
        self._save_state(event)

        handler = {
            "accident": self._apply_accident,
            "concert": self._apply_concert,
            "flood": self._apply_flood,
            "heavy_rain": self._apply_heavy_rain,
            "construction": self._apply_construction,
            "vip": self._apply_vip,
        }.get(event.event_type)

        if handler:
            handler(event)

        # Draw visuals
        self._draw_polygon(event)
        self._draw_poi(event)

    def _apply_accident(self, event: TrafficEvent) -> None:
        """Close lanes and drastically reduce speed at crash site."""
        for eid in event.affected_edges:
            try:
                lanes = self._get_lane_ids(eid)
                # Close half the lanes (at least 1)
                close_count = max(1, int(len(lanes) * event.intensity))
                for lane_id in lanes[:close_count]:
                    self._conn.lane.setAllowed(lane_id, [])
                # Reduce speed on remaining open lanes
                for lane_id in lanes[close_count:]:
                    self._conn.lane.setMaxSpeed(lane_id, 1.39)  # ~5 km/h rubbernecking
            except traci.TraCIException:
                pass

    def _apply_concert(self, event: TrafficEvent) -> None:
        """Reduce speed around venue (crowd spilling onto roads)."""
        for eid in event.affected_edges:
            try:
                speed_factor = 1.0 - (0.6 * event.intensity)  # up to 60% slower
                edge = self._net.getEdge(eid)
                new_speed = max(edge.getSpeed() * speed_factor, 2.78)  # min 10 km/h
                self._conn.edge.setMaxSpeed(eid, new_speed)
            except (traci.TraCIException, KeyError):
                pass

    def _apply_flood(self, event: TrafficEvent) -> None:
        """Close flooded roads or severely reduce speed."""
        for eid in event.affected_edges:
            try:
                if event.intensity > 0.7:
                    # Severe flood — close road entirely
                    for lane_id in self._get_lane_ids(eid):
                        self._conn.lane.setAllowed(lane_id, [])
                else:
                    # Moderate flood — reduce speed to 30%
                    edge = self._net.getEdge(eid)
                    new_speed = max(edge.getSpeed() * 0.3, 1.39)
                    self._conn.edge.setMaxSpeed(eid, new_speed)
            except (traci.TraCIException, KeyError):
                pass

    def _apply_heavy_rain(self, event: TrafficEvent) -> None:
        """Reduce speed on ALL edges in the network."""
        speed_factor = 1.0 - (0.4 * event.intensity)  # up to 40% slower
        for edge in self._net.getEdges():
            eid = edge.getID()
            if eid.startswith(":"):
                continue
            try:
                new_speed = max(edge.getSpeed() * speed_factor, 2.78)
                self._conn.edge.setMaxSpeed(eid, new_speed)
            except traci.TraCIException:
                pass

    def _apply_construction(self, event: TrafficEvent) -> None:
        """Close one lane and reduce speed on remaining."""
        for eid in event.affected_edges:
            try:
                lanes = self._get_lane_ids(eid)
                if lanes:
                    self._conn.lane.setAllowed(lanes[0], [])  # close rightmost lane
                for lane_id in lanes[1:]:
                    self._conn.lane.setMaxSpeed(lane_id, 5.56)  # 20 km/h
            except traci.TraCIException:
                pass

    def _apply_vip(self, event: TrafficEvent) -> None:
        """Temporarily close route edges for motorcade."""
        for eid in event.affected_edges:
            try:
                for lane_id in self._get_lane_ids(eid):
                    self._conn.lane.setAllowed(lane_id, ["emergency"])
            except traci.TraCIException:
                pass

    # ── Revert events ───────────────────────────────────────────────

    def _revert(self, event: TrafficEvent) -> None:
        saved = self._saved.pop(event.id, [])
        for state in saved:
            try:
                self._conn.edge.setMaxSpeed(state.edge_id,state.max_speed)
            except traci.TraCIException:
                pass
            for lane_id, allowed in state.lane_allowed.items():
                try:
                    if allowed:
                        self._conn.lane.setAllowed(lane_id, allowed)
                    else:
                        # Reset to default (allow all)
                        self._conn.lane.setAllowed(lane_id, ["all"])
                except traci.TraCIException:
                    pass

        # Remove visuals
        for poly_id in self._poly_ids.pop(event.id, []):
            try:
                self._conn.polygon.remove(poly_id)
            except traci.TraCIException:
                pass
        for poi_id in self._poi_ids.pop(event.id, []):
            try:
                self._conn.poi.remove(poi_id)
            except traci.TraCIException:
                pass

    # ── Save/restore helpers ────────────────────────────────────────

    def _save_state(self, event: TrafficEvent) -> None:
        states: list[_SavedEdgeState] = []
        edges_to_save = event.affected_edges

        if event.event_type == "heavy_rain":
            # Save ALL edges for city-wide events
            edges_to_save = [
                e.getID() for e in self._net.getEdges()
                if not e.getID().startswith(":")
            ]

        for eid in edges_to_save:
            try:
                edge = self._net.getEdge(eid)
                lane_allowed: dict[str, list[str]] = {}
                for lane in edge.getLanes():
                    lid = lane.getID()
                    try:
                        allowed = list(self._conn.lane.getAllowed(lid))
                    except traci.TraCIException:
                        allowed = []
                    lane_allowed[lid] = allowed

                states.append(_SavedEdgeState(
                    edge_id=eid,
                    max_speed=edge.getSpeed(),
                    lane_allowed=lane_allowed,
                ))
            except (KeyError, traci.TraCIException):
                pass

        self._saved[event.id] = states

    def _get_lane_ids(self, edge_id: str) -> list[str]:
        try:
            edge = self._net.getEdge(edge_id)
            return [lane.getID() for lane in edge.getLanes()]
        except KeyError:
            return []

    def _edge_name(self, edge_id: str) -> str:
        try:
            edge = self._net.getEdge(edge_id)
            return edge.getName() or edge_id
        except KeyError:
            return edge_id

    # ── Visual overlays ─────────────────────────────────────────────

    def _draw_polygon(self, event: TrafficEvent) -> None:
        color = EVENT_COLORS.get(event.event_type, (200, 200, 200, 150))
        polys: list[str] = []

        for eid in event.affected_edges[:10]:  # limit polygons
            try:
                edge = self._net.getEdge(eid)
                shape = edge.getShape()
                if len(shape) < 2:
                    continue
                # Create a buffer polygon around the edge shape
                poly_shape = self._buffer_shape(shape, width=15.0)
                poly_id = f"evt_{event.id}_{eid}"
                self._conn.polygon.add(
                    poly_id, poly_shape, color, fill=True,
                    layer=10, polygonType=event.event_type,
                )
                polys.append(poly_id)
            except (traci.TraCIException, KeyError):
                pass

        self._poly_ids[event.id] = polys

    def _draw_poi(self, event: TrafficEvent) -> None:
        pois: list[str] = []
        if not event.affected_edges:
            return
        try:
            edge = self._net.getEdge(event.affected_edges[0])
            shape = edge.getShape()
            mid = shape[len(shape) // 2]
            poi_id = f"poi_{event.id}"
            color = EVENT_COLORS.get(event.event_type, (200, 200, 200, 255))
            self._conn.poi.add(
                poi_id, mid[0], mid[1],
                color, event.label, layer=20,
            )
            pois.append(poi_id)
        except (traci.TraCIException, KeyError):
            pass
        self._poi_ids[event.id] = pois

    @staticmethod
    def _buffer_shape(shape: list[tuple], width: float = 15.0) -> list[tuple]:
        """Create a simple polygon buffer around a polyline."""
        left: list[tuple] = []
        right: list[tuple] = []

        for i in range(len(shape) - 1):
            x1, y1 = shape[i]
            x2, y2 = shape[i + 1]
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length < 0.01:
                continue
            nx, ny = -dy / length * width, dx / length * width
            left.append((x1 + nx, y1 + ny))
            left.append((x2 + nx, y2 + ny))
            right.append((x1 - nx, y1 - ny))
            right.append((x2 - nx, y2 - ny))

        right.reverse()
        return left + right


# ── Helper to create events easily ──────────────────────────────────────

def make_event(
    event_type: str,
    affected_edges: list[str],
    intensity: float = 0.5,
    start_time: float = 0.0,
    duration: float = 600.0,
    description: str = "",
) -> TrafficEvent:
    return TrafficEvent(
        id=str(uuid.uuid4())[:8],
        event_type=event_type,
        affected_edges=affected_edges,
        intensity=intensity,
        start_time=start_time,
        duration=duration,
        description=description,
    )
