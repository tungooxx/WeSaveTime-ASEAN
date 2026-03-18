"""
FlowMind AI - Signal Timing Map.

Visualizes traffic signal locations on the road network and compares
default SUMO timing vs AI-optimized timing after training.

Usage:
    python -m src.tools.timing_map
"""

from __future__ import annotations

import os
import queue
import shutil
import sys
import threading
import traceback
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.colors as mcolors

import sumolib

from src.simulation.tls_metadata import TLSMetadata

# ── Defaults ────────────────────────────────────────────────────────────

_DANANG = {
    "net": os.path.join(_PROJECT_ROOT, "sumo", "danang", "danang.net.xml"),
    "route": os.path.join(_PROJECT_ROOT, "sumo", "danang", "danang.rou.xml"),
    "cfg": os.path.join(_PROJECT_ROOT, "sumo", "danang", "danang.sumocfg"),
}

# ── Colors (Catppuccin Mocha) ───────────────────────────────────────────

BG = "#1e1e2e"
BG2 = "#313244"
BG3 = "#45475a"
FG = "#cdd6f4"
FG2 = "#a6adc8"
GREEN = "#a6e3a1"
RED = "#f38ba8"
BLUE = "#89b4fa"
YELLOW = "#f9e2af"
MAUVE = "#cba6f7"


# ── Dashboard ───────────────────────────────────────────────────────────

class TimingMapDashboard:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("FlowMind AI - Signal Timing Map")
        self.root.geometry("1250x820")
        self.root.configure(bg=BG)
        self.root.minsize(1000, 650)

        self._msg_q: queue.Queue = queue.Queue()
        self._net = None
        self._tls_info: dict[str, dict] = {}
        self._tls_ids: list[str] = []
        self._edge_shapes: list[tuple] = []

        self._baseline: dict[str, dict] = {}
        self._trained: dict[str, dict] = {}
        self._recommendations: dict = {}
        self._candidate_ids: set[str] = set()
        self._existing_ids: set[str] = set()
        self._selected_tls: str | None = None
        self._scatter = None
        self._cbar = None

        self._build_ui()
        self._load_network()
        self._draw_map()

    # ── UI construction ─────────────────────────────────────────────

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self.root, bg=BG)
        hdr.pack(fill=tk.X, padx=10, pady=(10, 5))
        tk.Label(hdr, text="Signal Timing Map",
                 font=("Segoe UI", 16, "bold"), fg=FG, bg=BG).pack(side=tk.LEFT)
        self._status_var = tk.StringVar(value="Loading...")
        tk.Label(hdr, textvariable=self._status_var,
                 font=("Segoe UI", 11), fg=FG2, bg=BG).pack(side=tk.RIGHT)

        # Main: left map + right details
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # ── Left: Map ──────────────────────────────────────────────
        left = tk.Frame(main, bg=BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._fig = Figure(figsize=(8, 6), dpi=90, facecolor=BG)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor(BG2)
        self._fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)

        self._canvas = FigureCanvasTkAgg(self._fig, master=left)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._canvas.mpl_connect("pick_event", self._on_pick)

        # ── Right: Details panel ───────────────────────────────────
        right = tk.LabelFrame(main, text=" Signal Details ",
                              font=("Segoe UI", 10, "bold"), fg=FG, bg=BG,
                              width=330, padx=10, pady=8)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right.pack_propagate(False)

        self._detail_text = tk.Text(
            right, bg=BG2, fg=FG, font=("Consolas", 9),
            wrap=tk.WORD, relief=tk.FLAT, padx=8, pady=8,
            state=tk.DISABLED, insertbackground=FG,
        )
        self._detail_text.pack(fill=tk.BOTH, expand=True)

        for tag, cfg in {
            "header":    {"font": ("Segoe UI", 12, "bold"), "foreground": FG},
            "sub":       {"font": ("Segoe UI", 10, "bold"), "foreground": BLUE},
            "green":     {"foreground": GREEN},
            "red":       {"foreground": RED},
            "yellow":    {"foreground": YELLOW},
            "mauve":     {"foreground": MAUVE},
            "dim":       {"foreground": FG2},
        }.items():
            self._detail_text.tag_configure(tag, **cfg)

        # ── Bottom: Controls ───────────────────────────────────────
        bottom = tk.Frame(self.root, bg=BG)
        bottom.pack(fill=tk.X, padx=10, pady=(5, 10))

        self._base_btn = tk.Button(
            bottom, text="Run Baseline", font=("Segoe UI", 10, "bold"),
            bg=BLUE, fg="#1e1e2e", width=14, pady=4,
            command=self._run_baseline)
        self._base_btn.pack(side=tk.LEFT, padx=5)

        self._train_btn = tk.Button(
            bottom, text="Run Trained", font=("Segoe UI", 10, "bold"),
            bg=GREEN, fg="#1e1e2e", width=14, pady=4,
            command=self._run_trained)
        self._train_btn.pack(side=tk.LEFT, padx=5)

        self._compare_btn = tk.Button(
            bottom, text="Compare", font=("Segoe UI", 10, "bold"),
            bg=YELLOW, fg="#1e1e2e", width=14, pady=4,
            state=tk.DISABLED, command=self._compare)
        self._compare_btn.pack(side=tk.LEFT, padx=5)

        tk.Label(bottom, text="Model:", font=("Segoe UI", 9),
                 fg=FG2, bg=BG).pack(side=tk.LEFT, padx=(20, 5))
        self._model_var = tk.StringVar(
            value=os.path.join(_PROJECT_ROOT, "checkpoints", "best_model.pt"))
        tk.Entry(bottom, textvariable=self._model_var, width=35,
                 bg=BG2, fg=FG, font=("Consolas", 8),
                 insertbackground=FG).pack(side=tk.LEFT, padx=5)
        tk.Button(bottom, text="...", command=self._browse_model,
                  bg=BG3, fg=FG, width=3).pack(side=tk.LEFT)

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Select trained model",
            filetypes=[("PyTorch", "*.pt")],
            initialdir=os.path.join(_PROJECT_ROOT, "checkpoints"))
        if path:
            self._model_var.set(path)

    # ── Network loading ─────────────────────────────────────────────

    def _load_network(self):
        net_file = _DANANG["net"]
        self._net = sumolib.net.readNet(net_file, withPrograms=True)

        # Edge shapes for road background
        for edge in self._net.getEdges():
            eid = edge.getID()
            if eid.startswith(":"):
                continue
            shape = edge.getShape()
            if len(shape) >= 2:
                xs = [p[0] for p in shape]
                ys = [p[1] for p in shape]
                self._edge_shapes.append((xs, ys, edge.getLaneNumber()))

        # TLS info
        meta = TLSMetadata(net_file)
        non_trivial = meta.get_non_trivial(min_green_phases=2, min_incoming=2)

        for tls in non_trivial:
            try:
                node = self._net.getNode(tls.id)
                x, y = node.getCoord()
            except KeyError:
                coords = []
                for eid in tls.incoming_edges:
                    try:
                        shape = self._net.getEdge(eid).getShape()
                        coords.append(shape[-1])
                    except KeyError:
                        pass
                if not coords:
                    continue
                x = np.mean([c[0] for c in coords])
                y = np.mean([c[1] for c in coords])

            # Street name from first named incoming edge
            name = tls.id
            for eid in tls.incoming_edges:
                try:
                    ename = self._net.getEdge(eid).getName()
                    if ename:
                        name = ename
                        break
                except KeyError:
                    pass

            self._tls_info[tls.id] = {
                "x": x, "y": y, "name": name,
                "phases": tls.phases,
                "green_indices": tls.green_phase_indices(),
                "incoming_edges": tls.incoming_edges,
                "num_connections": tls.num_connections,
            }

        self._tls_ids = sorted(self._tls_info.keys())

        # Load candidate/existing info
        import json
        candidate_file = os.path.join(os.path.dirname(net_file), "candidate_tls.json")
        if os.path.isfile(candidate_file):
            with open(candidate_file) as f:
                cdata = json.load(f)
            self._existing_ids = set(cdata.get("existing_tls", []))
            self._candidate_ids = {c["id"] for c in cdata.get("candidates", [])}
        else:
            self._existing_ids = set(self._tls_ids)

        # Load recommendations from training log
        log_file = os.path.join(_PROJECT_ROOT, "checkpoints", "training_log.json")
        if os.path.isfile(log_file):
            with open(log_file) as f:
                log_data = json.load(f)
            self._recommendations = log_data.get("recommendations", {})

        n_existing = sum(1 for t in self._tls_ids if t in self._existing_ids)
        n_candidate = sum(1 for t in self._tls_ids if t in self._candidate_ids)
        self._status_var.set(
            f"Loaded: {n_existing} existing + {n_candidate} candidate TLS")

    # ── Map drawing ─────────────────────────────────────────────────

    def _draw_map(self):
        ax = self._ax
        ax.clear()
        ax.set_facecolor(BG2)

        # Remove old colorbar
        if self._cbar is not None:
            self._cbar.remove()
            self._cbar = None

        # Road network
        for xs, ys, lanes in self._edge_shapes:
            lw = max(0.3, min(lanes * 0.4, 2.0))
            ax.plot(xs, ys, color="#4a4a5a", linewidth=lw, alpha=0.5, zorder=1)

        if not self._tls_info:
            self._canvas.draw_idle()
            return

        # Separate existing vs candidate TLS
        rec_add = {r["id"] for r in self._recommendations.get("add", [])}
        rec_remove = {r["id"] for r in self._recommendations.get("remove", [])}

        xs = [self._tls_info[t]["x"] for t in self._tls_ids]
        ys = [self._tls_info[t]["y"] for t in self._tls_ids]
        sizes = [
            max(40, min(self._tls_info[t]["num_connections"] * 5, 140))
            for t in self._tls_ids
        ]

        # Determine marker colors and shapes
        if self._baseline and self._trained:
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "rg", [(0.95, 0.3, 0.35), (1, 1, 0.6), (0.4, 0.9, 0.4)])
            improvements = []
            for tid in self._tls_ids:
                b_w = self._baseline.get(tid, {}).get("avg_wait", 0)
                t_w = self._trained.get(tid, {}).get("avg_wait", 0)
                imp = (b_w - t_w) / b_w if b_w > 0.1 else 0.0
                improvements.append(max(-1.0, min(1.0, imp)))
            norm = mcolors.Normalize(vmin=-1, vmax=1)
            colors = [cmap(norm(v)) for v in improvements]

            self._scatter = ax.scatter(
                xs, ys, c=colors, s=sizes,
                edgecolors="white", linewidths=0.8, zorder=10,
                picker=True, pickradius=10)

            sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            self._cbar = self._fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
            self._cbar.set_label("Wait Time Improvement", color=FG2, fontsize=8)
            self._cbar.ax.tick_params(colors=FG2, labelsize=7)
        else:
            # Color by type: blue=existing, orange=candidate
            colors = []
            for tid in self._tls_ids:
                if tid in self._candidate_ids:
                    colors.append(YELLOW)  # candidate
                else:
                    colors.append(BLUE)    # existing
            self._scatter = ax.scatter(
                xs, ys, c=colors, s=sizes,
                edgecolors="white", linewidths=0.8, zorder=10,
                picker=True, pickradius=10)

        # Overlay recommendation markers
        for tid in self._tls_ids:
            info = self._tls_info[tid]
            if tid in rec_add:
                ax.scatter(info["x"], info["y"], marker="^", s=200,
                           c=GREEN, edgecolors="white", linewidths=1.5,
                           zorder=15, alpha=0.9)
            elif tid in rec_remove:
                ax.scatter(info["x"], info["y"], marker="X", s=180,
                           c=RED, edgecolors="white", linewidths=1.5,
                           zorder=15, alpha=0.9)

        ax.set_aspect("equal")
        ax.tick_params(colors=FG2, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(BG3)

        # Title with recommendation summary
        n_exist = sum(1 for t in self._tls_ids if t in self._existing_ids)
        n_cand = sum(1 for t in self._tls_ids if t in self._candidate_ids)
        mode = "Comparison" if (self._baseline and self._trained) else "Overview"
        title = f"{n_exist} Existing + {n_cand} Candidate TLS \u2014 {mode}"
        if rec_add or rec_remove:
            title += f" | \u25b2+{len(rec_add)} \u2716-{len(rec_remove)}"
        ax.set_title(title, fontsize=10, color=FG, pad=8)

        self._canvas.draw_idle()

    # ── Pick handler ────────────────────────────────────────────────

    def _on_pick(self, event):
        if event.artist != self._scatter:
            return
        ind = event.ind[0]
        tls_id = self._tls_ids[ind]
        self._selected_tls = tls_id
        self._show_details(tls_id)

    # ── Detail panel ────────────────────────────────────────────────

    def _show_details(self, tls_id: str):
        info = self._tls_info.get(tls_id)
        if not info:
            return

        txt = self._detail_text
        txt.config(state=tk.NORMAL)
        txt.delete("1.0", tk.END)

        txt.insert(tk.END, f"{info['name']}\n", "header")
        txt.insert(tk.END, f"ID: {tls_id}\n", "dim")

        # Type badge
        if tls_id in self._candidate_ids:
            txt.insert(tk.END, "CANDIDATE", "yellow")
            txt.insert(tk.END, " (no TLS currently)\n", "dim")
        else:
            txt.insert(tk.END, "EXISTING", "green")
            txt.insert(tk.END, " traffic light\n", "dim")

        # Recommendation badge
        rec_add_ids = {r["id"] for r in self._recommendations.get("add", [])}
        rec_remove_ids = {r["id"] for r in self._recommendations.get("remove", [])}
        if tls_id in rec_add_ids:
            txt.insert(tk.END, "\u25b2 RECOMMEND: ADD TRAFFIC LIGHT\n", "green")
        elif tls_id in rec_remove_ids:
            txt.insert(tk.END, "\u2716 RECOMMEND: REMOVE TRAFFIC LIGHT\n", "red")

        txt.insert(tk.END, f"Connections: {info['num_connections']}  |  "
                           f"Incoming: {len(info['incoming_edges'])}\n\n", "dim")

        # Default phase program
        txt.insert(tk.END, "Default Phase Program\n", "sub")
        txt.insert(tk.END, "\u2500" * 32 + "\n", "dim")
        total_cycle = sum(p.duration for p in info["phases"])
        for phase in info["phases"]:
            is_green = phase.index in info["green_indices"]
            tag = "green" if is_green else "yellow"
            marker = "\u25cf" if is_green else "\u25cb"
            pct = phase.duration / total_cycle * 100 if total_cycle > 0 else 0
            txt.insert(tk.END,
                       f"  {marker} Phase {phase.index}: {phase.duration:.0f}s "
                       f"({pct:.0f}%)\n", tag)
            txt.insert(tk.END, f"    {self._format_state(phase.state)}\n", "dim")
        txt.insert(tk.END, f"  Cycle: {total_cycle:.0f}s\n\n", "dim")

        # Baseline metrics
        if tls_id in self._baseline:
            b = self._baseline[tls_id]
            txt.insert(tk.END, "Baseline Metrics\n", "sub")
            txt.insert(tk.END, "\u2500" * 32 + "\n", "dim")
            txt.insert(tk.END, f"  Avg Wait:  {b['avg_wait']:.1f}s\n")
            txt.insert(tk.END, f"  Avg Queue: {b['avg_queue']:.1f}\n\n")

        # Trained metrics + action distribution
        if tls_id in self._trained:
            t = self._trained[tls_id]
            txt.insert(tk.END, "AI Optimized\n", "sub")
            txt.insert(tk.END, "\u2500" * 32 + "\n", "dim")
            txt.insert(tk.END, f"  Avg Wait:  {t['avg_wait']:.1f}s\n")
            txt.insert(tk.END, f"  Avg Queue: {t['avg_queue']:.1f}\n\n")

            if "action_dist" in t and t["action_dist"]:
                txt.insert(tk.END, "AI Phase Selection\n", "sub")
                txt.insert(tk.END, "\u2500" * 32 + "\n", "dim")
                dist = t["action_dist"]
                total = sum(dist.values())
                gp = info["green_indices"]
                for act in sorted(dist.keys()):
                    count = dist[act]
                    pct = count / total * 100 if total > 0 else 0
                    if act == 7:
                        label = "OFF"
                    elif act < len(gp):
                        label = f"Phase {gp[act]}"
                    else:
                        label = f"Act {act}"
                    n_bars = int(pct / 5)
                    bar = "\u2588" * n_bars + "\u2591" * (20 - n_bars)
                    txt.insert(tk.END, f"  {label:10s} {bar} {pct:.0f}%\n", "green")
                txt.insert(tk.END, "\n")

        # Comparison
        if tls_id in self._baseline and tls_id in self._trained:
            b = self._baseline[tls_id]
            t = self._trained[tls_id]
            txt.insert(tk.END, "Improvement\n", "sub")
            txt.insert(tk.END, "\u2500" * 32 + "\n", "dim")

            for metric, label in [("avg_wait", "Wait"), ("avg_queue", "Queue")]:
                bv, tv = b[metric], t[metric]
                if bv > 0.01:
                    chg = (bv - tv) / bv * 100
                    tag = "green" if chg > 0 else "red"
                    arrow = "\u2193" if chg > 0 else "\u2191"
                    unit = "s" if metric == "avg_wait" else ""
                    txt.insert(tk.END,
                               f"  {label}: {bv:.1f}{unit} \u2192 {tv:.1f}{unit} "
                               f"({arrow}{abs(chg):.0f}%)\n", tag)

        # Training-log recommendations (if no live eval but training log exists)
        timing = self._recommendations.get("timing", {})
        if tls_id in timing and tls_id not in self._trained:
            td = timing[tls_id]
            txt.insert(tk.END, "Training Result\n", "sub")
            txt.insert(tk.END, "\u2500" * 32 + "\n", "dim")
            txt.insert(tk.END, f"  OFF: {td['off_pct']:.0f}% of actions\n",
                       "red" if td["off_pct"] > 50 else "green")
            dist = td.get("action_dist", {})
            total = td.get("total_actions", 1)
            gp = info["green_indices"]
            for act_str in sorted(dist.keys(), key=int):
                act = int(act_str)
                pct = dist[act_str] / total * 100
                if act == 7:
                    continue  # already shown
                if act < len(gp):
                    label = f"Phase {gp[act]}"
                else:
                    label = f"Act {act}"
                n_bars = int(pct / 5)
                bar = "\u2588" * n_bars + "\u2591" * (20 - n_bars)
                txt.insert(tk.END, f"  {label:10s} {bar} {pct:.0f}%\n", "mauve")

        txt.config(state=tk.DISABLED)

    @staticmethod
    def _format_state(state: str) -> str:
        greens = sum(1 for c in state if c in "Gg")
        reds = sum(1 for c in state if c in "Rr")
        yellows = sum(1 for c in state if c in "Yy")
        return f"[G:{greens}  Y:{yellows}  R:{reds}]"

    # ── Run Baseline ────────────────────────────────────────────────

    def _run_baseline(self):
        self._base_btn.config(state=tk.DISABLED)
        self._status_var.set("Running baseline (default timing)...")
        threading.Thread(target=self._baseline_thread, daemon=True).start()
        self._poll("baseline")

    def _baseline_thread(self):
        try:
            import traci
            for label in ["timing_baseline"]:
                try:
                    traci.getConnection(label).close()
                except Exception:
                    pass

            binary = shutil.which("sumo")
            if not binary:
                sumo_home = os.environ.get("SUMO_HOME", "")
                if sumo_home:
                    bp = os.path.join(sumo_home, "bin", "sumo")
                    if os.path.isfile(bp) or os.path.isfile(bp + ".exe"):
                        binary = bp
            if not binary:
                raise FileNotFoundError("Cannot find 'sumo'. Set SUMO_HOME.")

            cmd = [binary, "-c", _DANANG["cfg"],
                   "--seed", "42",
                   "--no-step-log", "true",
                   "--no-warnings", "true",
                   "--time-to-teleport", "300"]

            traci.start(cmd, label="timing_baseline")
            conn = traci.getConnection("timing_baseline")

            # Controlled lanes per TLS
            tls_lanes: dict[str, list[str]] = {}
            for tid in self._tls_ids:
                try:
                    tls_lanes[tid] = list(set(
                        conn.trafficlight.getControlledLanes(tid)))
                except Exception:
                    tls_lanes[tid] = []

            accum = {tid: {"wait": 0.0, "queue": 0.0, "n": 0}
                     for tid in self._tls_ids}

            step = 0
            sample_every = 20  # every 10 sim-seconds (step_length=0.5)
            max_steps = 3600 * 2  # 3600s at step_length=0.5

            while step < max_steps:
                try:
                    conn.simulationStep()
                except Exception:
                    break
                step += 1

                if step % sample_every == 0:
                    for tid in self._tls_ids:
                        w = q = 0.0
                        for lane in tls_lanes[tid]:
                            try:
                                w += conn.lane.getWaitingTime(lane)
                                q += conn.lane.getLastStepHaltingNumber(lane)
                            except Exception:
                                pass
                        nl = max(len(tls_lanes[tid]), 1)
                        accum[tid]["wait"] += w / nl
                        accum[tid]["queue"] += q / nl
                        accum[tid]["n"] += 1

            conn.close()

            result = {}
            for tid in self._tls_ids:
                a = accum[tid]
                n = max(a["n"], 1)
                result[tid] = {
                    "avg_wait": a["wait"] / n,
                    "avg_queue": a["queue"] / n,
                }
            self._msg_q.put(("baseline_done", result))

        except Exception as e:
            self._msg_q.put(("error", f"Baseline error:\n{e}\n{traceback.format_exc()}"))

    # ── Run Trained ─────────────────────────────────────────────────

    def _run_trained(self):
        model_path = self._model_var.get()
        if not os.path.isfile(model_path):
            model_path = filedialog.askopenfilename(
                title="Select trained model",
                filetypes=[("PyTorch", "*.pt")],
                initialdir=os.path.join(_PROJECT_ROOT, "checkpoints"))
            if not model_path:
                return
            self._model_var.set(model_path)

        self._train_btn.config(state=tk.DISABLED)
        self._status_var.set("Running trained model evaluation...")
        threading.Thread(target=self._trained_thread,
                         args=(model_path,), daemon=True).start()
        self._poll("trained")

    def _trained_thread(self, model_path: str):
        try:
            import traci
            for label in ["rl_training"]:
                try:
                    traci.getConnection(label).close()
                except Exception:
                    pass

            from src.ai.traffic_env import SumoTrafficEnv, OBS_DIM, ACT_DIM
            from src.ai.dqn_agent import TrafficDQNAgent

            env = SumoTrafficEnv(
                net_file=_DANANG["net"],
                route_file=_DANANG["route"],
                sumo_cfg=_DANANG["cfg"],
                delta_time=10, sim_length=3600,
                gui=False, seed=42,
            )

            agent = TrafficDQNAgent(OBS_DIM, ACT_DIM, hidden=256)
            agent.load(model_path)

            accum = {tid: {"wait": 0.0, "queue": 0.0, "n": 0}
                     for tid in env.tls_ids}
            action_counts: dict[str, dict[int, int]] = {
                tid: {} for tid in env.tls_ids}

            # Cache controlled lanes
            obs, _ = env.reset(seed=42)
            tls_lanes: dict[str, list[str]] = {}
            for tid in env.tls_ids:
                try:
                    tls_lanes[tid] = list(set(
                        env._conn.trafficlight.getControlledLanes(tid)))
                except Exception:
                    tls_lanes[tid] = []

            terminated = truncated = False
            while not (terminated or truncated):
                actions = {
                    tid: agent.select_action(
                        obs[tid], env.get_valid_actions(tid), greedy=True)
                    for tid in env.tls_ids
                }
                obs, rewards, terminated, truncated, info = env.step(actions)

                for tid, act in actions.items():
                    action_counts[tid][act] = action_counts[tid].get(act, 0) + 1

                for tid in env.tls_ids:
                    w = q = 0.0
                    for lane in tls_lanes.get(tid, []):
                        try:
                            w += env._conn.lane.getWaitingTime(lane)
                            q += env._conn.lane.getLastStepHaltingNumber(lane)
                        except Exception:
                            pass
                    nl = max(len(tls_lanes.get(tid, [])), 1)
                    accum[tid]["wait"] += w / nl
                    accum[tid]["queue"] += q / nl
                    accum[tid]["n"] += 1

            env.close()

            result = {}
            for tid in env.tls_ids:
                a = accum[tid]
                n = max(a["n"], 1)
                result[tid] = {
                    "avg_wait": a["wait"] / n,
                    "avg_queue": a["queue"] / n,
                    "action_dist": action_counts.get(tid, {}),
                }
            self._msg_q.put(("trained_done", result))

        except Exception as e:
            self._msg_q.put(("error", f"Trained eval error:\n{e}\n{traceback.format_exc()}"))

    # ── Poll results ────────────────────────────────────────────────

    def _poll(self, mode: str):
        try:
            msg_type, data = self._msg_q.get_nowait()

            if msg_type == "baseline_done":
                self._baseline = data
                self._base_btn.config(state=tk.NORMAL)
                self._status_var.set(f"Baseline done — {len(data)} TLS")
                if self._baseline and self._trained:
                    self._compare_btn.config(state=tk.NORMAL)
                    self._draw_map()
                if self._selected_tls:
                    self._show_details(self._selected_tls)
                return

            elif msg_type == "trained_done":
                self._trained = data
                self._train_btn.config(state=tk.NORMAL)
                self._status_var.set(f"Trained eval done — {len(data)} TLS")
                if self._baseline and self._trained:
                    self._compare_btn.config(state=tk.NORMAL)
                    self._draw_map()
                if self._selected_tls:
                    self._show_details(self._selected_tls)
                return

            elif msg_type == "error":
                self._base_btn.config(state=tk.NORMAL)
                self._train_btn.config(state=tk.NORMAL)
                self._status_var.set("Error")
                messagebox.showerror("Error", data)
                return

        except queue.Empty:
            pass

        self.root.after(1000, lambda: self._poll(mode))

    def _compare(self):
        if self._baseline and self._trained:
            self._draw_map()

            # Show summary
            improvements = []
            for tid in self._tls_ids:
                b_w = self._baseline.get(tid, {}).get("avg_wait", 0)
                t_w = self._trained.get(tid, {}).get("avg_wait", 0)
                if b_w > 0.1:
                    improvements.append((b_w - t_w) / b_w * 100)

            if improvements:
                avg_imp = np.mean(improvements)
                improved = sum(1 for x in improvements if x > 0)
                self._status_var.set(
                    f"Avg improvement: {avg_imp:+.1f}% | "
                    f"{improved}/{len(improvements)} signals improved")

    # ── Run ─────────────────────────────────────────────────────────

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self.root.destroy()


# ── Main ────────────────────────────────────────────────────────────────

def main():
    dashboard = TimingMapDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
