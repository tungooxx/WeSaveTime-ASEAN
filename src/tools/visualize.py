"""
FlowMind AI - Visualize trained model in SUMO-gui.

Loads best_model.pt, runs SUMO-gui with the trained agent making greedy
decisions.  Adds visual overlays on the map:

  - Green circles  : existing TLS the agent KEEPS
  - Red circles    : existing TLS the agent wants to REMOVE (OFF > 60%)
  - Orange diamonds: candidate TLS the agent wants to ADD
  - Gray diamonds  : candidate TLS the agent keeps OFF (not needed)

A small Tkinter stats panel runs alongside showing live metrics.

Usage:
    python -m src.tools.visualize
    python -m src.tools.visualize --model checkpoints/best_model.pt
    python -m src.tools.visualize --speed 0.1          # slower playback
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import tkinter as tk
from tkinter import ttk

import sumolib
import traci
import traci.constants as tc

from src.ai.traffic_env import (
    SumoTrafficEnv, OBS_DIM, OLD_OBS_DIM, ACT_DIM, ACT_OFF,
    remap_obs_for_old_model,
)
from src.ai.dqn_agent import TrafficDQNAgent
from src.ai.mappo_agent import MAPPOAgent

# ── Colors (Catppuccin Mocha) ────────────────────────────────────────
BG = "#1e1e2e"
BG2 = "#313244"
FG = "#cdd6f4"
FG2 = "#a6adc8"
GREEN = "#a6e3a1"
RED = "#f38ba8"
BLUE = "#89b4fa"
YELLOW = "#f9e2af"
ORANGE = "#fab387"

# SUMO-gui overlay colors (R,G,B,A in 0-255)
CLR_KEEP = (100, 200, 100, 200)      # green — keep existing TLS
CLR_REMOVE = (240, 100, 100, 200)    # red — agent wants to remove
CLR_ADD = (240, 180, 80, 200)        # orange — agent wants to add
CLR_OFF = (150, 150, 150, 120)       # gray — candidate not needed


def _rgb_str(rgba):
    return f"{rgba[0]},{rgba[1]},{rgba[2]},{rgba[3]}"


# ── Congestion monitor helpers ─────────────────────────────────────────
_CONG_SUB_VARS = [
    tc.LAST_STEP_VEHICLE_NUMBER,
    tc.LAST_STEP_MEAN_SPEED,
    tc.LAST_STEP_OCCUPANCY,
    tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
    tc.VAR_WAITING_TIME,
]

LEVEL_COLORS = {
    "SEVERE": "#ff3333",
    "MODERATE": "#ffaa00",
    "LIGHT": "#88cc44",
    "FREE": "#44aa88",
}


def _congestion_score(wait, halt, speed, max_speed, occ):
    speed_ratio = 1.0 - min(speed / max(max_speed, 0.1), 1.0)
    return (
        0.35 * speed_ratio
        + 0.25 * min(wait / 200.0, 1.0)
        + 0.25 * min(halt / 15.0, 1.0)
        + 0.15 * min(occ, 1.0)
    )


def _congestion_level(score):
    if score >= 0.7:
        return "SEVERE"
    if score >= 0.4:
        return "MODERATE"
    if score >= 0.15:
        return "LIGHT"
    return "FREE"


class Visualizer:
    """Run trained model in SUMO-gui with overlays and stats panel."""

    def __init__(self, model_path: str, net_file: str, route_file: str,
                 sumo_cfg: str, hidden: int = 256, delta_time: int = 10,
                 sim_length: int = 3600, seed: int = 1000, speed: float = 0.05):
        self.model_path = model_path
        self.net_file = net_file
        self.route_file = route_file
        self.sumo_cfg = sumo_cfg
        self.hidden = hidden
        self.delta_time = delta_time
        self.sim_length = sim_length
        self.seed = seed
        self.speed = speed  # delay between steps (seconds)

        self._stop = False

    def run(self):
        """Start SUMO-gui + stats panel."""
        # Build Tkinter panel in main thread, sim in background
        self._build_panel()
        self._sim_thread = threading.Thread(target=self._run_sim, daemon=True)
        self._sim_thread.start()
        self.root.mainloop()

    def _build_panel(self):
        self.root = tk.Tk()
        self.root.title("FlowMind AI - Live Visualization")
        self.root.geometry("520x780")
        self.root.configure(bg=BG)
        self.root.attributes("-topmost", True)

        # Header
        tk.Label(self.root, text="Live Agent Visualization",
                 font=("Segoe UI", 14, "bold"), fg=FG, bg=BG
                 ).pack(padx=10, pady=(10, 5))

        self._status_var = tk.StringVar(value="Loading model...")
        tk.Label(self.root, textvariable=self._status_var,
                 font=("Segoe UI", 10), fg=FG2, bg=BG).pack()

        # Metrics frame
        mf = tk.LabelFrame(self.root, text=" Live Metrics ",
                           font=("Segoe UI", 10, "bold"), fg=FG, bg=BG,
                           padx=10, pady=8)
        mf.pack(fill=tk.X, padx=10, pady=10)

        self._metrics = {}
        labels = [
            ("sim_time", "Sim Time"),
            ("vehicles", "Vehicles"),
            ("avg_wait", "Avg Wait (s)"),
            ("avg_queue", "Avg Queue"),
            ("reward", "Step Reward"),
            ("total_reward", "Total Reward"),
        ]
        for i, (key, text) in enumerate(labels):
            tk.Label(mf, text=text + ":", font=("Segoe UI", 9),
                     fg=FG2, bg=BG, anchor=tk.W).grid(row=i, column=0,
                                                        sticky=tk.W, pady=2)
            var = tk.StringVar(value="--")
            tk.Label(mf, textvariable=var, font=("Consolas", 10, "bold"),
                     fg=FG, bg=BG, anchor=tk.E, width=14).grid(
                row=i, column=1, sticky=tk.E, pady=2)
            self._metrics[key] = var

        # TLS decisions frame
        tf = tk.LabelFrame(self.root, text=" Agent TLS Decisions ",
                           font=("Segoe UI", 10, "bold"), fg=FG, bg=BG,
                           padx=10, pady=8)
        tf.pack(fill=tk.X, padx=10, pady=5)

        self._tls_vars = {}
        tls_labels = [
            ("existing_keep", "Existing: Keep", GREEN),
            ("existing_remove", "Existing: Remove", RED),
            ("candidate_add", "Candidate: Add TLS", ORANGE),
            ("candidate_off", "Candidate: No TLS", FG2),
        ]
        for i, (key, text, color) in enumerate(tls_labels):
            tk.Label(tf, text=text + ":", font=("Segoe UI", 9),
                     fg=color, bg=BG, anchor=tk.W).grid(
                row=i, column=0, sticky=tk.W, pady=2)
            var = tk.StringVar(value="0")
            tk.Label(tf, textvariable=var, font=("Consolas", 11, "bold"),
                     fg=color, bg=BG, anchor=tk.E, width=6).grid(
                row=i, column=1, sticky=tk.E, pady=2)
            self._tls_vars[key] = var

        # [Level 2 REMOVED] Active events display removed

        # ── Tabbed notebook: TLS Log + Congestion ─────────────────
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TNotebook", background=BG)
        style.configure("Dark.TNotebook.Tab", background=BG2,
                        foreground=FG, padding=[8, 4],
                        font=("Segoe UI", 9, "bold"))
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", "#585b70")],
                  foreground=[("selected", "#cdd6f4")])

        notebook = ttk.Notebook(self.root, style="Dark.TNotebook")
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # ── Tab 1: TLS Action Log ────────────────────────────────
        log_frame = tk.Frame(notebook, bg=BG)
        notebook.add(log_frame, text=" TLS Action Log ")

        self._log_text = tk.Text(log_frame, font=("Consolas", 8),
                                  bg=BG2, fg=FG, height=10, width=48,
                                  wrap=tk.WORD, state=tk.DISABLED,
                                  borderwidth=0, highlightthickness=0)
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL,
                                       command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=log_scrollbar.set)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._log_text.pack(fill=tk.BOTH, expand=True)

        self._log_text.tag_configure("header", foreground=BLUE)
        self._log_text.tag_configure("change", foreground=YELLOW)
        self._log_text.tag_configure("keep", foreground=FG2)

        # ── Tab 2: Live Congestion Monitor ────────────────────────
        cong_frame = tk.Frame(notebook, bg=BG)
        notebook.add(cong_frame, text=" Congestion Monitor ")

        # Legend bar
        leg = tk.Frame(cong_frame, bg=BG)
        leg.pack(fill=tk.X, padx=5, pady=(5, 2))
        for lvl, col in LEVEL_COLORS.items():
            tk.Label(leg, text=f" {lvl} ", font=("Segoe UI", 8, "bold"),
                     fg="white", bg=col).pack(side=tk.LEFT, padx=2)
        tk.Label(leg, text="  Dbl-click to fly",
                 font=("Segoe UI", 8, "italic"), fg=FG2, bg=BG
                 ).pack(side=tk.LEFT, padx=6)

        self._cong_summary_var = tk.StringVar(value="Waiting for data...")
        tk.Label(cong_frame, textvariable=self._cong_summary_var,
                 font=("Segoe UI", 9), fg=FG2, bg=BG).pack(anchor=tk.W, padx=5)

        # Congestion table
        cols = ("rank", "level", "road", "score", "wait", "stopped",
                "speed", "veh", "occ")
        style.configure("Cong.Treeview", background=BG2,
                        foreground=FG, fieldbackground=BG2,
                        rowheight=22, font=("Consolas", 9))
        style.configure("Cong.Treeview.Heading", background="#45475a",
                        foreground=FG, font=("Segoe UI", 9, "bold"))
        style.map("Cong.Treeview", background=[("selected", "#585b70")])

        ctf = tk.Frame(cong_frame, bg=BG)
        ctf.pack(fill=tk.BOTH, expand=True, padx=5, pady=3)

        self._cong_tree = ttk.Treeview(ctf, columns=cols, show="headings",
                                        style="Cong.Treeview", selectmode="browse")
        heads = {
            "rank": ("#", 30), "level": ("Level", 70),
            "road": ("Road", 180), "score": ("Score", 50),
            "wait": ("Wait", 50), "stopped": ("Stop", 45),
            "speed": ("km/h", 50), "veh": ("Veh", 40),
            "occ": ("Occ%", 45),
        }
        for c, (label, w) in heads.items():
            self._cong_tree.heading(c, text=label)
            self._cong_tree.column(c, width=w, minwidth=w,
                                    anchor=tk.W if c == "road" else tk.CENTER)

        csb = ttk.Scrollbar(ctf, orient=tk.VERTICAL, command=self._cong_tree.yview)
        self._cong_tree.configure(yscrollcommand=csb.set)
        csb.pack(side=tk.RIGHT, fill=tk.Y)
        self._cong_tree.pack(fill=tk.BOTH, expand=True)

        for lvl, col in LEVEL_COLORS.items():
            self._cong_tree.tag_configure(lvl, foreground=col)

        self._cong_tree.bind("<Double-1>", self._on_cong_dbl_click)
        self._cong_row_edges: dict[str, tuple[float, float]] = {}  # iid -> (x, y)

        # Stop button
        tk.Button(self.root, text="Stop & Close", font=("Segoe UI", 10, "bold"),
                  bg=RED, fg="white", width=18, pady=4,
                  command=self._on_close).pack(pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_cong_dbl_click(self, _event):
        """Fly SUMO-gui camera to double-clicked congested road."""
        item = self._cong_tree.focus()
        coords = self._cong_row_edges.get(item)
        if coords and hasattr(self, '_env') and self._env and self._env._conn:
            try:
                self._env._conn.gui.setOffset("View #0", coords[0], coords[1])
                self._env._conn.gui.setZoom("View #0", 800)
            except Exception:
                pass

    def _on_close(self):
        self._stop = True
        time.sleep(0.3)
        self.root.destroy()

    def _update_var(self, key: str, value: str):
        """Thread-safe update of a Tkinter variable."""
        try:
            self.root.after(0, lambda: self._metrics[key].set(value))
        except Exception:
            pass

    def _update_tls_var(self, key: str, value: str):
        try:
            self.root.after(0, lambda: self._tls_vars[key].set(value))
        except Exception:
            pass

    def _set_status(self, msg: str):
        try:
            self.root.after(0, lambda: self._status_var.set(msg))
        except Exception:
            pass

    def _append_log(self, text: str, tag: str = ""):
        """Thread-safe append to the TLS action log."""
        def _do():
            self._log_text.configure(state=tk.NORMAL)
            if tag:
                self._log_text.insert(tk.END, text + "\n", tag)
            else:
                self._log_text.insert(tk.END, text + "\n")
            self._log_text.see(tk.END)
            self._log_text.configure(state=tk.DISABLED)
        try:
            self.root.after(0, _do)
        except Exception:
            pass

    def _run_sim(self):
        """Background thread: load model, start SUMO-gui, run agent."""
        try:
            self._set_status("Loading model...")

            # Detect obs_dim from checkpoint (handles old 26-dim models)
            import torch
            ckpt = torch.load(self.model_path, map_location="cpu",
                              weights_only=True)
            ckpt_obs_dim = ckpt.get("obs_dim", OBS_DIM)
            self._model_obs_dim = ckpt_obs_dim
            algorithm = ckpt.get("algorithm", "dqn")
            self._algorithm = algorithm

            if algorithm == "mappo":
                agent = MAPPOAgent(ckpt_obs_dim, ACT_DIM, self.hidden)
                agent.load(self.model_path)
            else:
                agent = TrafficDQNAgent(ckpt_obs_dim, ACT_DIM, self.hidden)
                agent.load(self.model_path)

            self._set_status("Starting SUMO-gui...")

            env = SumoTrafficEnv(
                net_file=self.net_file,
                route_file=self.route_file,
                sumo_cfg=self.sumo_cfg,
                delta_time=self.delta_time,
                sim_length=self.sim_length,
                gui=True,
                seed=self.seed,
            )

            obs, _ = env.reset(seed=self.seed)

            # Load candidate info for overlay classification
            candidate_file = os.path.join(
                os.path.dirname(self.net_file), "candidate_tls.json")
            candidate_info = {}
            if os.path.isfile(candidate_file):
                with open(candidate_file) as f:
                    cdata = json.load(f)
                for c in cdata.get("candidates", []):
                    candidate_info[c["id"]] = c

            self._set_status(f"Running: {env.num_agents} TLS agents")

            # Build TLS road name lookup from sumolib network
            tls_road_names: dict[str, str] = {}
            for tid in env.tls_ids:
                try:
                    node = env._net.getNode(tid)
                    # Get unique road names from incoming edges
                    road_names = set()
                    for edge in node.getIncoming():
                        name = edge.getName() or edge.getID()
                        # Trim long IDs, prefer human-readable names
                        if name:
                            road_names.add(name.split("#")[0])
                    tls_road_names[tid] = " / ".join(sorted(road_names)[:3]) or tid
                except Exception:
                    tls_road_names[tid] = tid

            # Track previous phases to detect changes
            prev_phases: dict[str, int] = {}

            # Per-TLS action tracking for live overlay updates
            action_counts: dict[str, Counter] = {
                tid: Counter() for tid in env.tls_ids
            }
            total_reward = 0.0
            step = 0
            terminated = truncated = False

            # Set initial SUMO-gui view (zoom, delay)
            self._env = env  # store ref for fly-to
            try:
                conn = env._conn
                conn.gui.setSchema("View #0", "real world")
                conn.gui.setDelay(50)  # 50ms between steps for visibility
            except Exception:
                conn = env._conn

            # Setup congestion monitoring subscriptions on same connection
            self._setup_congestion_subscriptions(conn, env._net)

            # Check if model needs obs remapping (old 26-dim vs new 39-dim)
            needs_remap = (ckpt_obs_dim == OLD_OBS_DIM and OBS_DIM != OLD_OBS_DIM)

            while not (terminated or truncated) and not self._stop:
                actions = {}
                if self._algorithm == "mappo":
                    global_obs = np.mean(
                        [obs[tid] for tid in env.tls_ids], axis=0
                    ).astype(np.float32)
                for tid in env.tls_ids:
                    valid = env.get_valid_actions(tid)
                    o = remap_obs_for_old_model(obs[tid]) if needs_remap else obs[tid]
                    if self._algorithm == "mappo":
                        a, _, _ = agent.select_action(o, global_obs, valid, greedy=True)
                    else:
                        a = agent.select_action(o, valid, greedy=True)
                    actions[tid] = a
                    action_counts[tid][a] += 1

                # Step environment
                obs, rewards, terminated, truncated, info = env.step(actions)
                step_reward = sum(rewards.values())
                total_reward += step_reward
                step += 1

                # Get metrics
                metrics = env.get_metrics()

                # Update panel
                self._update_var("sim_time", f"{metrics.get('sim_time', 0)}s")
                self._update_var("vehicles", str(metrics.get('total_vehicles', 0)))
                self._update_var("avg_wait", f"{metrics.get('avg_wait_time', 0):.1f}")
                self._update_var("avg_queue", f"{metrics.get('avg_queue_length', 0):.1f}")
                self._update_var("reward", f"{step_reward:+.3f}")
                self._update_var("total_reward", f"{total_reward:+.1f}")

                # Log TLS phase changes
                changes_this_step = []
                for tid in env.tls_ids:
                    cur_phase = env._current_phases.get(tid, 0)
                    old_phase = prev_phases.get(tid, -1)
                    if old_phase != cur_phase:
                        road = tls_road_names.get(tid, tid)
                        # Get phase state string from SUMO
                        try:
                            state_str = env._conn.trafficlight.getRedYellowGreenState(tid)
                        except Exception:
                            state_str = "?"
                        dur = env.delta_time
                        changes_this_step.append(
                            f"  {road[:28]:<28s} P{old_phase}->P{cur_phase} "
                            f"d={dur}s [{state_str[:16]}]"
                        )
                    prev_phases[tid] = cur_phase

                if changes_this_step:
                    sim_t = metrics.get('sim_time', 0)
                    self._append_log(f"--- Step {step} | t={sim_t}s ---", "header")
                    for line in changes_this_step:
                        self._append_log(line, "change")

                # Update TLS overlays every 10 steps
                if step % 10 == 1:
                    self._update_overlays(env, action_counts, candidate_info)

                # Poll congestion data every 3 steps
                if step % 3 == 0:
                    self._poll_congestion(env._conn)

                # [Level 2 REMOVED] Active events display removed

                self._set_status(
                    f"Step {step} | {metrics.get('sim_time', 0)}s | "
                    f"Veh={metrics.get('total_vehicles', 0)}")

                if self.speed > 0:
                    time.sleep(self.speed)

            # Final summary
            self._set_status(
                f"Done! {step} steps, total reward: {total_reward:+.1f}")
            env.close()

        except Exception as e:
            import traceback
            self._set_status(f"Error: {e}")
            print(traceback.format_exc())

    def _setup_congestion_subscriptions(self, conn, net):
        """Subscribe to edge data for congestion monitoring."""
        self._cong_edges: dict[str, dict] = {}
        for edge in net.getEdges():
            eid = edge.getID()
            if eid.startswith(":"):
                continue
            if edge.getLaneNumber() < 2:
                continue
            shape = edge.getShape()
            mid = len(shape) // 2
            cx, cy = shape[mid] if shape else (0.0, 0.0)
            self._cong_edges[eid] = {
                "name": edge.getName() or eid,
                "max_speed": edge.getSpeed(),
                "cx": cx, "cy": cy,
            }
            conn.edge.subscribe(eid, _CONG_SUB_VARS)

    def _poll_congestion(self, conn):
        """Read subscription results and update the congestion table."""
        try:
            all_results = conn.edge.getAllSubscriptionResults()
        except Exception:
            return

        rows = []
        for eid, data in all_results.items():
            info = self._cong_edges.get(eid)
            if not info:
                continue
            veh = data.get(tc.LAST_STEP_VEHICLE_NUMBER, 0)
            halt = data.get(tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0)
            wait = data.get(tc.VAR_WAITING_TIME, 0.0)
            if veh == 0 and halt == 0 and wait == 0:
                continue
            speed = data.get(tc.LAST_STEP_MEAN_SPEED, 0.0)
            occ = data.get(tc.LAST_STEP_OCCUPANCY, 0.0)
            sc = _congestion_score(wait, halt, speed, info["max_speed"], occ)
            lvl = _congestion_level(sc)
            rows.append((sc, lvl, info["name"], wait, halt, speed, veh, occ,
                         info["cx"], info["cy"], eid))

        rows.sort(key=lambda r: r[0], reverse=True)
        top = rows[:40]

        def _update_tree():
            self._cong_tree.delete(*self._cong_tree.get_children())
            self._cong_row_edges.clear()
            for i, (sc, lvl, name, wait, halt, speed, veh, occ, cx, cy, eid) in enumerate(top):
                iid = self._cong_tree.insert("", tk.END, values=(
                    i + 1, lvl, name[:35],
                    f"{sc:.2f}", f"{wait:.0f}", halt,
                    f"{speed * 3.6:.1f}", veh,
                    f"{occ * 100:.1f}",
                ), tags=(lvl,))
                self._cong_row_edges[iid] = (cx, cy)
            sev = sum(1 for r in top if r[1] == "SEVERE")
            mod = sum(1 for r in top if r[1] == "MODERATE")
            self._cong_summary_var.set(
                f"Top {len(top)} roads | SEVERE: {sev} | MODERATE: {mod}")

        try:
            self.root.after(0, _update_tree)
        except Exception:
            pass

    def _update_overlays(self, env, action_counts, candidate_info):
        """Add colored POIs on SUMO-gui map for each TLS."""
        conn = env._conn
        if conn is None:
            return

        n_keep = n_remove = n_add = n_off = 0

        for tid in env.tls_ids:
            counts = action_counts[tid]
            total = sum(counts.values())
            if total == 0:
                continue

            # [Level 2 COMMENTED OUT] TLS candidate/location optimization
            # All TLS are treated as existing-keep in Level 1
            color = CLR_KEEP
            n_keep += 1
            label = "KEEP"

            # Get position
            try:
                node = env._net.getNode(tid)
                x, y = node.getCoord()
            except Exception:
                continue

            poi_id = f"ai_{tid}"
            try:
                # Only remove if it already exists
                if poi_id in conn.poi.getIDList():
                    conn.poi.remove(poi_id)
                conn.poi.add(
                    poi_id, x, y,
                    color=color,
                    poiType=f"flowmind_{label.lower()}",
                    layer=10,
                    width=15,
                    height=15,
                )
            except Exception:
                pass

        # Update panel counts
        self._update_tls_var("existing_keep", str(n_keep))
        self._update_tls_var("existing_remove", str(n_remove))
        self._update_tls_var("candidate_add", str(n_add))
        self._update_tls_var("candidate_off", str(n_off))


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="FlowMind AI - Visualize trained model in SUMO-gui")
    ap.add_argument("--model", default=os.path.join(
        _PROJECT_ROOT, "checkpoints", "best_model.pt"))
    ap.add_argument("--net", default=os.path.join(
        _PROJECT_ROOT, "sumo", "danang", "danang.net.xml"))
    ap.add_argument("--route", default=os.path.join(
        _PROJECT_ROOT, "sumo", "danang", "danang.rou.xml"))
    ap.add_argument("--cfg", default=os.path.join(
        _PROJECT_ROOT, "sumo", "danang", "danang.sumocfg"))
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--delta-time", type=int, default=10)
    ap.add_argument("--sim-length", type=int, default=3600)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--speed", type=float, default=0.05,
                    help="Delay between steps in seconds (0=max speed)")
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        print(f"Model not found: {args.model}")
        print("Train first: python -m src.tools.rl_dashboard")
        sys.exit(1)

    viz = Visualizer(
        model_path=args.model,
        net_file=args.net,
        route_file=args.route,
        sumo_cfg=args.cfg,
        hidden=args.hidden,
        delta_time=args.delta_time,
        sim_length=args.sim_length,
        seed=args.seed,
        speed=args.speed,
    )
    viz.run()


if __name__ == "__main__":
    main()
