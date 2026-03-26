"""
FlowMind AI - TLS Inspector Tool

Lists all TLS in the network with their category (AI / Forced Green /
Default / Roundabout / Clustered). Click a row to fly the SUMO-GUI
camera to that TLS.

Usage:
    1. Start sumo-gui first: sumo-gui -c sumo/danang/danang.sumocfg
    2. Run this tool:         python -m src.tools.tls_inspector
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import tkinter as tk
from tkinter import ttk

import sumolib
import traci

from src.simulation.tls_metadata import TLSMetadata

# ── Colors ──────────────────────────────────────────────────────────
BG = "#1e1e2e"
FG = "#cdd6f4"
FG2 = "#a6adc8"
GREEN = "#a6e3a1"
RED = "#f38ba8"
YELLOW = "#f9e2af"
BLUE = "#89b4fa"
ORANGE = "#fab387"


def main():
    net_file = os.path.join(_PROJECT_ROOT, "sumo", "danang", "danang.net.xml")
    cfg_file = os.path.join(_PROJECT_ROOT, "sumo", "danang", "danang.sumocfg")
    net = sumolib.net.readNet(net_file)
    meta = TLSMetadata(net_file)

    # Start sumo-gui
    import shutil
    binary = shutil.which("sumo-gui")
    if not binary:
        sumo_home = os.environ.get("SUMO_HOME", "")
        binary = os.path.join(sumo_home, "bin", "sumo-gui") if sumo_home else "sumo-gui"

    try:
        traci.getConnection("inspector").close()
    except Exception:
        pass

    cmd = [binary, "-c", cfg_file,
           "--no-step-log", "true",
           "--no-warnings", "true",
           "--start", "true",
           "--quit-on-end", "true"]
    traci.start(cmd, label="inspector")
    conn = traci.getConnection("inspector")
    nt_ids = {t.id for t in meta.get_non_trivial()}

    # Find roundabout centers
    ra_centers = []
    for ra in net.getRoundabouts():
        xs, ys = [], []
        for nid in ra.getNodes():
            try:
                n = net.getNode(nid)
                x, y = n.getCoord()
                xs.append(x); ys.append(y)
            except Exception:
                pass
        if xs:
            ra_centers.append((sum(xs) / len(xs), sum(ys) / len(ys)))

    # Get coords for clustering detection
    all_coords = {}
    for t in meta.all_tls:
        try:
            node = net.getNode(t.id)
            all_coords[t.id] = node.getCoord()
        except Exception:
            pass

    # Build TLS list with categories
    tls_rows = []
    for t in meta.all_tls:
        gp = t.num_green_phases
        ie = len(t.incoming_edges)
        nc = t.num_connections

        # Get road names
        roads = set()
        for eid in t.incoming_edges[:4]:
            try:
                e = net.getEdge(eid)
                name = e.getName() or eid[:20]
                roads.add(name.split("#")[0][:25])
            except Exception:
                roads.add(eid[:20])
        road_str = " / ".join(sorted(roads)[:3])

        # Get coords
        x, y = all_coords.get(t.id, (0, 0))

        # Near roundabout?
        near_ra = False
        for cx, cy in ra_centers:
            if math.sqrt((x - cx) ** 2 + (y - cy) ** 2) < 300:
                near_ra = True
                break

        # Clustered?
        nearby = 0
        if t.id in all_coords:
            x1, y1 = all_coords[t.id]
            for oid, (x2, y2) in all_coords.items():
                if oid != t.id and math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < 30:
                    nearby += 1
        clustered = nearby >= 2

        # Has conflicting phases?
        has_conflict = False
        for p in t.phases:
            has_g = any(c in ("G", "g") for c in p.state)
            has_r = any(c in ("r", "R") for c in p.state)
            if has_g and has_r:
                has_conflict = True
                break

        # Category:
        # Multi-phase (GP>=2) = AI controlled
        # Single-phase = keep default SUMO timing (pedestrian safety)
        if gp >= 2:
            cat = "AI"
            color = GREEN
        elif near_ra:
            cat = "DEFAULT (roundabout)"
            color = ORANGE
        elif clustered:
            cat = "DEFAULT (clustered)"
            color = YELLOW
        else:
            cat = "DEFAULT (ped crossing)"
            color = BLUE

        # Phase states
        phase_str = " | ".join(p.state for p in t.phases[:4])

        tls_rows.append({
            "id": t.id, "cat": cat, "color": color,
            "gp": gp, "ie": ie, "conns": nc,
            "roads": road_str, "phases": phase_str,
            "x": x, "y": y,
        })

    # ── Build GUI ────────────────────────────────────────────────────
    root = tk.Tk()
    root.title("FlowMind AI - TLS Inspector")
    root.geometry("1100x700")
    root.configure(bg=BG)

    tk.Label(root, text="TLS Inspector — Click to fly to TLS in SUMO-GUI",
             font=("Segoe UI", 14, "bold"), fg=FG, bg=BG).pack(pady=10)

    # Summary
    cats = {}
    for r in tls_rows:
        cats[r["cat"]] = cats.get(r["cat"], 0) + 1
    summary = " | ".join(f"{k}: {v}" for k, v in sorted(cats.items()))
    tk.Label(root, text=summary, font=("Segoe UI", 10), fg=FG2, bg=BG).pack()

    # Table
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TLS.Treeview", background="#313244", foreground=FG,
                    fieldbackground="#313244", rowheight=22,
                    font=("Consolas", 9))
    style.configure("TLS.Treeview.Heading", background="#45475a",
                    foreground=FG, font=("Segoe UI", 9, "bold"))
    style.map("TLS.Treeview", background=[("selected", "#585b70")])

    cols = ("cat", "id", "gp", "ie", "roads", "phases")
    tree = ttk.Treeview(root, columns=cols, show="headings",
                        style="TLS.Treeview", selectmode="browse")

    heads = {
        "cat": ("Category", 100), "id": ("TLS ID", 220),
        "gp": ("GP", 40), "ie": ("IE", 40),
        "roads": ("Roads", 300), "phases": ("Phases", 350),
    }
    for c, (label, w) in heads.items():
        tree.heading(c, text=label)
        tree.column(c, width=w, minwidth=w)

    sb = ttk.Scrollbar(root, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=sb.set)

    tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5, side=tk.LEFT)
    sb.pack(fill=tk.Y, side=tk.RIGHT, pady=5, padx=(0, 10))

    # Tag colors
    for cat_name, color in [("AI", GREEN),
                             ("DEFAULT (ped crossing)", BLUE),
                             ("DEFAULT (roundabout)", ORANGE),
                             ("DEFAULT (clustered)", YELLOW)]:
        tree.tag_configure(cat_name, foreground=color)

    # Populate
    row_data = {}
    for r in tls_rows:
        iid = tree.insert("", tk.END, values=(
            r["cat"], r["id"][:40], r["gp"], r["ie"],
            r["roads"][:50], r["phases"][:60],
        ), tags=(r["cat"],))
        row_data[iid] = r

    # Click to fly
    def on_click(_event):
        item = tree.focus()
        r = row_data.get(item)
        if not r:
            return
        try:
            conn.gui.setOffset("View #0", r["x"], r["y"])
            conn.gui.setZoom("View #0", 1500)
        except Exception:
            pass

    tree.bind("<Double-1>", on_click)

    def on_close():
        try:
            conn.close()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
