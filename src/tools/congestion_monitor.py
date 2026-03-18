"""
FlowMind AI - Live Congestion Monitor for SUMO-gui.

Starts SUMO-gui with TraCI, then opens a Tkinter panel showing the most
congested roads in real time.  Double-click any road to fly the SUMO-gui
camera there.

Key optimisation: uses TraCI **subscriptions** so all edge data arrives in
one TCP round-trip per step (instead of thousands of individual get calls).

Usage:
    python -m src.tools.congestion_monitor --net sumo/danang/danang.net.xml --cfg sumo/danang/danang.sumocfg
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import sumolib
import traci
import traci.constants as tc

import tkinter as tk
from tkinter import ttk


# ── Data ────────────────────────────────────────────────────────────────

@dataclass
class EdgeInfo:
    id: str
    name: str
    lanes: int
    length: float
    center_x: float
    center_y: float
    max_speed: float  # m/s


@dataclass
class CongestionSnapshot:
    edge_id: str
    name: str
    waiting_time: float
    halting: int
    mean_speed: float
    occupancy: float
    vehicle_count: int
    score: float = 0.0

    @property
    def speed_kmh(self) -> float:
        return self.mean_speed * 3.6

    @property
    def level(self) -> str:
        if self.score >= 0.7:
            return "SEVERE"
        if self.score >= 0.4:
            return "MODERATE"
        if self.score >= 0.15:
            return "LIGHT"
        return "FREE"


# ── Helpers ─────────────────────────────────────────────────────────────

def _find_sumo_gui() -> str:
    found = shutil.which("sumo-gui")
    if found:
        return found
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        p = os.path.join(sumo_home, "bin", "sumo-gui")
        if os.path.isfile(p) or os.path.isfile(p + ".exe"):
            return p
    raise FileNotFoundError("Cannot find 'sumo-gui'. Install SUMO or set SUMO_HOME.")


def _load_edges(net_file: str, min_lanes: int = 2) -> dict[str, EdgeInfo]:
    """Load edges with >= min_lanes from the .net.xml (skip alleys)."""
    net = sumolib.net.readNet(net_file, withPrograms=False)
    edges: dict[str, EdgeInfo] = {}
    for edge in net.getEdges():
        eid = edge.getID()
        if eid.startswith(":"):
            continue
        if edge.getLaneNumber() < min_lanes:
            continue
        name = edge.getName() or eid
        shape = edge.getShape()
        mid = len(shape) // 2
        cx, cy = shape[mid] if shape else (0.0, 0.0)
        edges[eid] = EdgeInfo(
            id=eid,
            name=name,
            lanes=edge.getLaneNumber(),
            length=edge.getLength(),
            center_x=cx,
            center_y=cy,
            max_speed=edge.getSpeed(),
        )
    return edges


def _score(wait: float, halt: int, speed: float, max_speed: float, occ: float) -> float:
    speed_ratio = 1.0 - min(speed / max(max_speed, 0.1), 1.0)
    return (
        0.35 * speed_ratio
        + 0.25 * min(wait / 200.0, 1.0)
        + 0.25 * min(halt / 15.0, 1.0)
        + 0.15 * min(occ, 1.0)
    )


# ── Subscription variables we need per edge ─────────────────────────────
_SUB_VARS = [
    tc.LAST_STEP_VEHICLE_NUMBER,
    tc.LAST_STEP_MEAN_SPEED,
    tc.LAST_STEP_OCCUPANCY,
    tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
    tc.VAR_WAITING_TIME,
]


# ── Monitor ─────────────────────────────────────────────────────────────

class CongestionMonitor:

    def __init__(
        self,
        net_file: str,
        route_file: str | None = None,
        sumo_cfg: str | None = None,
        seed: int = 42,
        top_n: int = 50,
        min_lanes: int = 2,
        steps_between_polls: int = 20,
    ):
        self.net_file = os.path.abspath(net_file)
        self.route_file = os.path.abspath(route_file) if route_file else None
        self.sumo_cfg = os.path.abspath(sumo_cfg) if sumo_cfg else None
        self.seed = seed
        self.top_n = top_n
        self.steps_between_polls = steps_between_polls

        self._edges = _load_edges(self.net_file, min_lanes)
        self._conn: traci.Connection | None = None
        self._label = "congestion_monitor"
        self.running = False
        self._sim_time = 0.0
        self._vehicle_count = 0

        self._lock = threading.Lock()
        self._latest: list[CongestionSnapshot] = []

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def start_sumo(self) -> None:
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
            "--delay", "0",           # max speed rendering
            "--no-warnings", "true",
        ]

        traci.start(cmd, label=self._label)
        self._conn = traci.getConnection(self._label)
        self.running = True

        # Subscribe to all monitored edges (one-time setup)
        for eid in self._edges:
            self._conn.edge.subscribe(eid, _SUB_VARS)

        print(f"  Subscribed to {len(self._edges)} edges (2+ lanes).")

    def poll(self) -> None:
        """Advance N sim steps, then read all subscription results at once."""
        if not self._conn:
            return

        # Advance simulation in bulk (no per-step data collection)
        for _ in range(self.steps_between_polls):
            try:
                self._conn.simulationStep()
            except traci.TraCIException:
                self.running = False
                return

        try:
            self._sim_time = self._conn.simulation.getTime()
            self._vehicle_count = self._conn.vehicle.getIDCount()
        except traci.TraCIException:
            pass

        # Read ALL subscription results in one batch
        try:
            all_results = self._conn.edge.getAllSubscriptionResults()
        except traci.TraCIException:
            return

        snapshots: list[CongestionSnapshot] = []
        for eid, data in all_results.items():
            info = self._edges.get(eid)
            if not info:
                continue

            veh = data.get(tc.LAST_STEP_VEHICLE_NUMBER, 0)
            halt = data.get(tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0)
            wait = data.get(tc.VAR_WAITING_TIME, 0.0)

            if veh == 0 and halt == 0 and wait == 0:
                continue

            speed = data.get(tc.LAST_STEP_MEAN_SPEED, 0.0)
            occ = data.get(tc.LAST_STEP_OCCUPANCY, 0.0)
            sc = _score(wait, halt, speed, info.max_speed, occ)

            snapshots.append(CongestionSnapshot(
                edge_id=eid,
                name=info.name,
                waiting_time=round(wait, 1),
                halting=halt,
                mean_speed=round(speed, 2),
                occupancy=round(occ, 3),
                vehicle_count=veh,
                score=round(sc, 3),
            ))

        snapshots.sort(key=lambda c: c.score, reverse=True)
        with self._lock:
            self._latest = snapshots[:self.top_n]

    def get_latest(self) -> tuple[list[CongestionSnapshot], float, int]:
        with self._lock:
            return list(self._latest), self._sim_time, self._vehicle_count

    def fly_to(self, edge_id: str) -> None:
        info = self._edges.get(edge_id)
        if not info or not self._conn:
            return
        try:
            self._conn.gui.setOffset("View #0", info.center_x, info.center_y)
            self._conn.gui.setZoom("View #0", 800)
        except traci.TraCIException:
            pass

    def close(self) -> None:
        self.running = False
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


# ── Simulation thread ───────────────────────────────────────────────────

def _sim_loop(monitor: CongestionMonitor):
    while monitor.running:
        monitor.poll()


# ── Tkinter UI ──────────────────────────────────────────────────────────

LEVEL_COLORS = {
    "SEVERE": "#ff3333",
    "MODERATE": "#ffaa00",
    "LIGHT": "#88cc44",
    "FREE": "#44aa88",
}


class CongestionPanel:

    def __init__(self, monitor: CongestionMonitor):
        self.monitor = monitor
        self.root = tk.Tk()
        self.root.title("FlowMind AI - Live Congestion Monitor")
        self.root.geometry("920x720")
        self.root.configure(bg="#1e1e2e")

        # ── Header ──────────────────────────────────────────────────
        header = tk.Frame(self.root, bg="#1e1e2e")
        header.pack(fill=tk.X, padx=10, pady=(10, 5))

        tk.Label(
            header, text="Live Congestion Monitor",
            font=("Segoe UI", 16, "bold"), fg="#cdd6f4", bg="#1e1e2e",
        ).pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Warming up...")
        tk.Label(
            header, textvariable=self.status_var,
            font=("Segoe UI", 11), fg="#a6adc8", bg="#1e1e2e",
        ).pack(side=tk.RIGHT)

        # ── Legend ──────────────────────────────────────────────────
        leg = tk.Frame(self.root, bg="#1e1e2e")
        leg.pack(fill=tk.X, padx=10, pady=(0, 5))
        for lvl, col in LEVEL_COLORS.items():
            tk.Label(leg, text=f" {lvl} ", font=("Segoe UI", 9, "bold"),
                     fg="white", bg=col).pack(side=tk.LEFT, padx=2)
        tk.Label(
            leg, text="   Double-click a road to fly there",
            font=("Segoe UI", 9, "italic"), fg="#a6adc8", bg="#1e1e2e",
        ).pack(side=tk.LEFT, padx=10)

        # ── Table ───────────────────────────────────────────────────
        cols = ("rank", "level", "road", "score", "wait", "stopped",
                "speed", "vehicles", "occupancy")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("C.Treeview", background="#313244",
                        foreground="#cdd6f4", fieldbackground="#313244",
                        rowheight=26, font=("Consolas", 10))
        style.configure("C.Treeview.Heading", background="#45475a",
                        foreground="#cdd6f4", font=("Segoe UI", 10, "bold"))
        style.map("C.Treeview", background=[("selected", "#585b70")])

        tf = tk.Frame(self.root, bg="#1e1e2e")
        tf.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.tree = ttk.Treeview(tf, columns=cols, show="headings",
                                 style="C.Treeview", selectmode="browse")
        heads = {
            "rank": ("#", 40), "level": ("Level", 80),
            "road": ("Road Name", 250), "score": ("Score", 65),
            "wait": ("Wait(s)", 70), "stopped": ("Stopped", 65),
            "speed": ("km/h", 60), "vehicles": ("Veh", 50),
            "occupancy": ("Occ%", 60),
        }
        for c, (label, w) in heads.items():
            self.tree.heading(c, text=label)
            self.tree.column(c, width=w, minwidth=w,
                             anchor=tk.W if c == "road" else tk.CENTER)

        sb = ttk.Scrollbar(tf, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.tree.bind("<Double-1>", self._on_dbl)

        for lvl, col in LEVEL_COLORS.items():
            self.tree.tag_configure(lvl, foreground=col)

        # ── Summary ─────────────────────────────────────────────────
        self.summary_var = tk.StringVar(value="")
        tk.Label(
            self.root, textvariable=self.summary_var,
            font=("Segoe UI", 10), fg="#a6adc8", bg="#1e1e2e", anchor=tk.W,
        ).pack(fill=tk.X, padx=10, pady=(0, 10))

        self._row_edges: dict[str, str] = {}

    # ── events ──────────────────────────────────────────────────────

    def _on_dbl(self, _event):
        item = self.tree.focus()
        eid = self._row_edges.get(item)
        if eid:
            self.monitor.fly_to(eid)

    def _refresh(self):
        snaps, sim_t, vehs = self.monitor.get_latest()

        self.tree.delete(*self.tree.get_children())
        self._row_edges.clear()

        for i, s in enumerate(snaps):
            iid = self.tree.insert("", tk.END, values=(
                i + 1, s.level, s.name[:45],
                f"{s.score:.2f}", f"{s.waiting_time:.0f}", s.halting,
                f"{s.speed_kmh:.1f}", s.vehicle_count,
                f"{s.occupancy * 100:.1f}",
            ), tags=(s.level,))
            self._row_edges[iid] = s.edge_id

        sev = sum(1 for s in snaps if s.level == "SEVERE")
        mod = sum(1 for s in snaps if s.level == "MODERATE")
        self.status_var.set(f"Sim: {sim_t / 60:.1f} min | Vehicles: {vehs}")
        self.summary_var.set(
            f"Top {len(snaps)} congested roads | "
            f"SEVERE: {sev} | MODERATE: {mod}"
        )

        if self.monitor.running:
            self.root.after(3000, self._refresh)

    def run(self):
        self._refresh()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.mainloop()

    def _close(self):
        self.monitor.close()
        self.root.destroy()


# ── Main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="FlowMind AI - Live Congestion Monitor")
    ap.add_argument("--net", required=True, help="SUMO .net.xml")
    ap.add_argument("--route", default=None, help="SUMO .rou.xml")
    ap.add_argument("--cfg", default=None, help="SUMO .sumocfg")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--min-lanes", type=int, default=2,
                    help="Only monitor edges with >= N lanes (skip alleys)")
    ap.add_argument("--steps", type=int, default=20,
                    help="Sim steps between each congestion poll")
    args = ap.parse_args()

    monitor = CongestionMonitor(
        net_file=args.net,
        route_file=args.route,
        sumo_cfg=args.cfg,
        seed=args.seed,
        top_n=args.top,
        min_lanes=args.min_lanes,
        steps_between_polls=args.steps,
    )

    print("=" * 55)
    print("  FlowMind AI - Live Congestion Monitor")
    print("=" * 55)
    print(f"  Network:    {args.net}")
    print(f"  Monitoring: {monitor.edge_count} edges (>= {args.min_lanes} lanes)")
    print(f"  Poll every: {args.steps} sim steps")
    print()

    monitor.start_sumo()

    sim_thread = threading.Thread(target=_sim_loop, args=(monitor,), daemon=True)
    sim_thread.start()

    panel = CongestionPanel(monitor)
    panel.run()


if __name__ == "__main__":
    main()
