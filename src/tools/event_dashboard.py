"""
FlowMind AI - Event Dashboard for SUMO-gui.

Starts SUMO-gui, then opens a Tkinter panel where you can manually trigger
traffic events (accident, concert, flood, heavy rain, construction, VIP).
Events appear as coloured polygons on the SUMO-gui map and affect traffic
in real time.

Usage:
    python -m src.tools.event_dashboard --net sumo/danang/danang.net.xml --cfg sumo/danang/danang.sumocfg
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import threading
import time
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import sumolib
import traci
import traci.constants as tc

import tkinter as tk
from tkinter import ttk

from src.simulation.events import (
    EventManager, TrafficEvent, make_event,
    EVENT_TYPES, EVENT_LABELS, EVENT_COLORS,
)


# ── Edge data (for road picker) ─────────────────────────────────────────

def _load_roads(net_file: str, min_lanes: int = 2) -> list[tuple[str, str, float, float]]:
    """Return [(edge_id, display_name, center_x, center_y)] for significant roads."""
    net = sumolib.net.readNet(net_file, withPrograms=False)
    roads: list[tuple[str, str, float, float]] = []
    for edge in net.getEdges():
        eid = edge.getID()
        if eid.startswith(":"):
            continue
        if edge.getLaneNumber() < min_lanes:
            continue
        name = edge.getName() or eid
        display = f"{name} ({edge.getLaneNumber()}L, {edge.getLength():.0f}m)"
        shape = edge.getShape()
        mid = shape[len(shape) // 2] if shape else (0.0, 0.0)
        roads.append((eid, display, mid[0], mid[1]))
    roads.sort(key=lambda r: r[1])
    return roads


# ── SUMO launcher ───────────────────────────────────────────────────────

def _find_sumo_gui() -> str:
    found = shutil.which("sumo-gui")
    if found:
        return found
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        p = os.path.join(sumo_home, "bin", "sumo-gui")
        if os.path.isfile(p) or os.path.isfile(p + ".exe"):
            return p
    raise FileNotFoundError("Cannot find 'sumo-gui'.")


# ── Simulation runner ───────────────────────────────────────────────────

class SimRunner:
    """Runs SUMO in a background thread, ticks EventManager each step."""

    def __init__(
        self,
        net_file: str,
        sumo_cfg: str | None,
        route_file: str | None,
        seed: int = 42,
        steps_per_tick: int = 10,
    ):
        self.net_file = os.path.abspath(net_file)
        self.sumo_cfg = os.path.abspath(sumo_cfg) if sumo_cfg else None
        self.route_file = os.path.abspath(route_file) if route_file else None
        self.seed = seed
        self.steps_per_tick = steps_per_tick

        self._conn: traci.Connection | None = None
        self._label = "event_dashboard"
        self.running = False
        self.sim_time = 0.0
        self.vehicle_count = 0
        self.event_mgr: EventManager | None = None

        self._lock = threading.Lock()
        self._log_messages: list[str] = []

    def start(self) -> None:
        binary = _find_sumo_gui()
        try:
            traci.getConnection(self._label).close()
        except (traci.TraCIException, KeyError):
            pass

        if self.sumo_cfg:
            cmd = [binary, "-c", self.sumo_cfg]
        else:
            cmd = [binary, "-n", self.net_file]
            if self.route_file:
                cmd += ["-r", self.route_file]
        cmd += [
            "--seed", str(self.seed),
            "--start", "true",
            "--quit-on-end", "false",
            "--delay", "0",
            "--no-warnings", "true",
        ]

        traci.start(cmd, label=self._label)
        self._conn = traci.getConnection(self._label)
        self.running = True
        self.event_mgr = EventManager(self._conn, self.net_file)

    def tick(self) -> None:
        if not self._conn or not self.running:
            return
        for _ in range(self.steps_per_tick):
            try:
                self._conn.simulationStep()
            except traci.TraCIException:
                self.running = False
                return

        try:
            self.sim_time = self._conn.simulation.getTime()
            self.vehicle_count = self._conn.vehicle.getIDCount()
        except traci.TraCIException:
            pass

        if self.event_mgr:
            msgs = self.event_mgr.tick(self.sim_time)
            if msgs:
                with self._lock:
                    self._log_messages.extend(msgs)

    def drain_log(self) -> list[str]:
        with self._lock:
            msgs = list(self._log_messages)
            self._log_messages.clear()
        return msgs

    def close(self) -> None:
        self.running = False
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


def _sim_loop(runner: SimRunner):
    while runner.running:
        runner.tick()


# ── Event Dialog ────────────────────────────────────────────────────────

class EventDialog(tk.Toplevel):
    """Pop-up dialog to configure and trigger an event."""

    def __init__(self, parent, event_type: str,
                 roads: list[tuple[str, str, float, float]], on_trigger):
        super().__init__(parent)
        self.event_type = event_type
        self.roads = roads
        self.on_trigger = on_trigger
        self.result = None

        self.title(f"Trigger: {EVENT_LABELS[event_type]}")
        self.geometry("420x340")
        self.configure(bg="#1e1e2e")
        self.resizable(False, False)
        self.grab_set()

        lbl_style = {"fg": "#cdd6f4", "bg": "#1e1e2e", "font": ("Segoe UI", 10)}

        row = 0

        # Road picker (skip for heavy_rain — city-wide)
        self._road_var = tk.StringVar()
        if event_type != "heavy_rain":
            tk.Label(self, text="Road:", **lbl_style).grid(
                row=row, column=0, padx=10, pady=8, sticky=tk.W)
            road_names = [r[1] for r in roads]
            self._road_cb = ttk.Combobox(
                self, textvariable=self._road_var, values=road_names,
                width=40, state="readonly")
            self._road_cb.grid(row=row, column=1, padx=10, pady=8)
            if road_names:
                self._road_cb.current(0)
            row += 1

        # Duration slider (minutes)
        tk.Label(self, text="Duration (min):", **lbl_style).grid(
            row=row, column=0, padx=10, pady=8, sticky=tk.W)
        dur_range = {
            "accident": (5, 60, 20),
            "concert": (30, 240, 120),
            "flood": (30, 360, 120),
            "heavy_rain": (10, 120, 30),
            "construction": (30, 480, 120),
            "vip": (2, 15, 5),
        }
        lo, hi, default = dur_range.get(event_type, (5, 120, 30))
        self._dur_var = tk.IntVar(value=default)
        self._dur_scale = tk.Scale(
            self, from_=lo, to=hi, orient=tk.HORIZONTAL,
            variable=self._dur_var, length=250,
            bg="#313244", fg="#cdd6f4", highlightbackground="#1e1e2e",
            troughcolor="#45475a",
        )
        self._dur_scale.grid(row=row, column=1, padx=10, pady=8)
        row += 1

        # Intensity slider
        tk.Label(self, text="Intensity:", **lbl_style).grid(
            row=row, column=0, padx=10, pady=8, sticky=tk.W)
        self._int_var = tk.DoubleVar(value=0.5)
        self._int_scale = tk.Scale(
            self, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL,
            variable=self._int_var, length=250,
            bg="#313244", fg="#cdd6f4", highlightbackground="#1e1e2e",
            troughcolor="#45475a",
        )
        self._int_scale.grid(row=row, column=1, padx=10, pady=8)
        row += 1

        # Trigger button
        btn = tk.Button(
            self, text="Trigger Event", font=("Segoe UI", 12, "bold"),
            bg="#f38ba8", fg="white", activebackground="#eb6f92",
            command=self._trigger, padx=20, pady=8,
        )
        btn.grid(row=row, column=0, columnspan=2, pady=20)

    def _trigger(self):
        # Find selected edge
        edges: list[str] = []
        if self.event_type != "heavy_rain":
            idx = 0
            for i, road in enumerate(self.roads):
                if road[1] == self._road_var.get():
                    idx = i
                    break
            edges = [self.roads[idx][0]]

        self.on_trigger(
            self.event_type,
            edges,
            self._int_var.get(),
            self._dur_var.get() * 60,  # minutes -> seconds
        )
        self.destroy()


# ── Main Dashboard ──────────────────────────────────────────────────────

class EventDashboard:

    def __init__(self, runner: SimRunner, roads: list[tuple[str, str]]):
        self.runner = runner
        self.roads = roads

        self.root = tk.Tk()
        self.root.title("FlowMind AI - Event Dashboard")
        self.root.geometry("780x700")
        self.root.configure(bg="#1e1e2e")

        lbl = {"fg": "#cdd6f4", "bg": "#1e1e2e"}

        # ── Header ──────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg="#1e1e2e")
        hdr.pack(fill=tk.X, padx=10, pady=(10, 5))
        tk.Label(hdr, text="Event Dashboard", font=("Segoe UI", 16, "bold"),
                 **lbl).pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="Starting...")
        tk.Label(hdr, textvariable=self.status_var,
                 font=("Segoe UI", 11), fg="#a6adc8", bg="#1e1e2e").pack(side=tk.RIGHT)

        # ── Event Trigger Buttons ───────────────────────────────────
        btn_frame = tk.LabelFrame(
            self.root, text=" Trigger Events ", font=("Segoe UI", 10, "bold"),
            fg="#cdd6f4", bg="#1e1e2e", padx=10, pady=10,
        )
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        btn_colors = {
            "accident": "#f38ba8",
            "concert": "#fab387",
            "flood": "#74c7ec",
            "heavy_rain": "#9399b2",
            "construction": "#f9e2af",
            "vip": "#cba6f7",
        }

        for i, etype in enumerate(EVENT_TYPES):
            fg = "white" if etype not in ("construction", "heavy_rain") else "#1e1e2e"
            btn = tk.Button(
                btn_frame, text=EVENT_LABELS[etype],
                font=("Segoe UI", 10, "bold"),
                bg=btn_colors.get(etype, "#585b70"), fg=fg,
                activebackground="#585b70",
                width=16, pady=6,
                command=lambda et=etype: self._open_dialog(et),
            )
            btn.grid(row=i // 3, column=i % 3, padx=5, pady=5)

        # ── Active Events Table ─────────────────────────────────────
        evt_frame = tk.LabelFrame(
            self.root, text=" Active Events ", font=("Segoe UI", 10, "bold"),
            fg="#cdd6f4", bg="#1e1e2e", padx=5, pady=5,
        )
        evt_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        cols = ("type", "location", "intensity", "remaining")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("E.Treeview", background="#313244", foreground="#cdd6f4",
                        fieldbackground="#313244", rowheight=28,
                        font=("Consolas", 10))
        style.configure("E.Treeview.Heading", background="#45475a",
                        foreground="#cdd6f4", font=("Segoe UI", 10, "bold"))

        self.tree = ttk.Treeview(evt_frame, columns=cols, show="headings",
                                 style="E.Treeview", height=8)
        heads = {"type": ("Event", 150), "location": ("Location", 220),
                 "intensity": ("Intensity", 80), "remaining": ("Time Left", 100)}
        for c, (label, w) in heads.items():
            self.tree.heading(c, text=label)
            self.tree.column(c, width=w, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Double-click to fly to event location in SUMO-gui
        self.tree.bind("<Double-1>", self._on_double_click)

        # Remove button
        rm_btn = tk.Button(
            evt_frame, text="Remove Selected Event", font=("Segoe UI", 10),
            bg="#45475a", fg="#cdd6f4", command=self._remove_selected, pady=4,
        )
        rm_btn.pack(pady=5)

        # ── Event Log ───────────────────────────────────────────────
        log_frame = tk.LabelFrame(
            self.root, text=" Event Log ", font=("Segoe UI", 10, "bold"),
            fg="#cdd6f4", bg="#1e1e2e", padx=5, pady=5,
        )
        log_frame.pack(fill=tk.X, padx=10, pady=(5, 10))

        self.log_text = tk.Text(
            log_frame, height=6, bg="#313244", fg="#a6e3a1",
            font=("Consolas", 9), state=tk.DISABLED, wrap=tk.WORD,
        )
        self.log_text.pack(fill=tk.X)

        # Track event IDs and edge IDs for the tree
        self._tree_event_ids: dict[str, str] = {}   # tree_iid -> event_id
        self._tree_edge_ids: dict[str, str] = {}    # tree_iid -> first affected edge_id

        # Build edge -> (cx, cy) lookup from roads data
        self._edge_coords: dict[str, tuple[float, float]] = {
            r[0]: (r[2], r[3]) for r in self.roads
        }

    def _get_road_name(self, edge_id: str) -> str:
        """Look up display name for an edge from the roads list."""
        for eid, display, _, _ in self.roads:
            if eid == edge_id:
                return display[:35]
        return edge_id[:20]

    def _on_double_click(self, _event):
        """Double-click an active event row to fly SUMO-gui camera there."""
        item = self.tree.focus()
        edge_id = self._tree_edge_ids.get(item)
        if not edge_id:
            return
        coords = self._edge_coords.get(edge_id)
        if coords and self.runner._conn:
            try:
                self.runner._conn.gui.setOffset("View #0", coords[0], coords[1])
                self.runner._conn.gui.setZoom("View #0", 800)
            except traci.TraCIException:
                pass

    def _open_dialog(self, event_type: str):
        EventDialog(self.root, event_type, self.roads, self._on_trigger)

    def _on_trigger(self, event_type: str, edges: list[str], intensity: float,
                    duration: float):
        if not self.runner.event_mgr:
            return
        event = make_event(
            event_type=event_type,
            affected_edges=edges,
            intensity=intensity,
            start_time=self.runner.sim_time,  # trigger immediately
            duration=duration,
        )
        self.runner.event_mgr.add_event(event)
        self._add_log(f"{event.label} triggered (intensity={intensity:.1f}, "
                      f"duration={duration/60:.0f} min)")

    def _remove_selected(self):
        sel = self.tree.focus()
        eid = self._tree_event_ids.get(sel)
        if eid and self.runner.event_mgr:
            self.runner.event_mgr.remove_event(eid)
            self._add_log(f"Event {eid} manually removed")

    def _add_log(self, msg: str):
        minutes = self.runner.sim_time / 60
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{minutes:6.1f}m] {msg}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _refresh(self):
        # Drain log messages from sim thread
        for msg in self.runner.drain_log():
            self._add_log(msg)

        # Update status
        self.status_var.set(
            f"Sim: {self.runner.sim_time/60:.1f} min | "
            f"Vehicles: {self.runner.vehicle_count}"
        )

        # Update active events table
        self.tree.delete(*self.tree.get_children())
        self._tree_event_ids.clear()
        self._tree_edge_ids.clear()

        if self.runner.event_mgr:
            for evt in self.runner.event_mgr.get_active():
                remaining = evt.time_remaining(self.runner.sim_time)
                # Show road name if available
                edge_id = evt.affected_edges[0] if evt.affected_edges else ""
                loc = self._get_road_name(edge_id) if edge_id else "City-wide"
                iid = self.tree.insert("", tk.END, values=(
                    evt.label,
                    loc,
                    f"{evt.intensity:.1f}",
                    f"{remaining/60:.1f} min",
                ))
                self._tree_event_ids[iid] = evt.id
                self._tree_edge_ids[iid] = edge_id

        if self.runner.running:
            self.root.after(2000, self._refresh)

    def run(self):
        self._refresh()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.mainloop()

    def _close(self):
        self.runner.close()
        self.root.destroy()


# ── Main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="FlowMind AI - Event Dashboard (SUMO-gui + Tkinter)")
    ap.add_argument("--net", required=True, help="SUMO .net.xml")
    ap.add_argument("--route", default=None, help="SUMO .rou.xml")
    ap.add_argument("--cfg", default=None, help="SUMO .sumocfg")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=10,
                    help="Sim steps between ticks")
    args = ap.parse_args()

    print("=" * 55)
    print("  FlowMind AI - Event Dashboard")
    print("=" * 55)

    roads = _load_roads(args.net)
    print(f"  Network: {args.net}")
    print(f"  Roads:   {len(roads)} (2+ lanes)")

    runner = SimRunner(
        net_file=args.net,
        sumo_cfg=args.cfg,
        route_file=args.route,
        seed=args.seed,
        steps_per_tick=args.steps,
    )
    runner.start()
    print(f"  SUMO-gui started.")

    sim_thread = threading.Thread(target=_sim_loop, args=(runner,), daemon=True)
    sim_thread.start()

    dashboard = EventDashboard(runner, roads)
    dashboard.run()


if __name__ == "__main__":
    main()
