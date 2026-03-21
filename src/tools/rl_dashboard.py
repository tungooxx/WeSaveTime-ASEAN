"""
FlowMind AI - RL Training Dashboard.

Tkinter GUI to configure, run, and monitor DQN training for traffic signal
optimization.  Live charts show reward, wait time, and epsilon per episode.

Usage:
    python -m src.tools.rl_dashboard
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


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
ORANGE = "#fab387"


# ── Training thread ─────────────────────────────────────────────────────

class TrainingThread(threading.Thread):
    """Runs DQN training in background, posting metrics to a queue."""

    def __init__(self, params: dict, metric_q: queue.Queue,
                 stop_event: threading.Event):
        super().__init__(daemon=True)
        self.params = params
        self.metric_q = metric_q
        self.stop_event = stop_event
        self.error: str | None = None

    def run(self):
        try:
            # Kill any stale SUMO/TraCI connections
            import traci
            for label in ["rl_training", "congestion_monitor"]:  # [Level 2 REMOVED] event_dashboard
                try:
                    traci.getConnection(label).close()
                except Exception:
                    pass

            common_kwargs = dict(
                net_file=self.params["net"],
                route_file=self.params["route"],
                sumo_cfg=self.params["cfg"],
                episodes=self.params["episodes"],
                delta_time=self.params["delta_time"],
                sim_length=self.params["sim_length"],
                hidden=self.params["hidden"],
                lr=self.params["lr"],
                gamma=self.params["gamma"],
                batch_size=self.params["batch_size"],
                buffer_capacity=self.params["buffer_capacity"],
                epsilon_start=self.params["epsilon_start"],
                epsilon_end=self.params["epsilon_end"],
                epsilon_decay=self.params["epsilon_decay"],
                target_update=self.params["target_update"],
                save_dir=self.params["save_dir"],
                save_every=self.params["save_every"],
                seed=self.params["seed"],
                gui=self.params.get("gui", False),
                on_episode=lambda ep: self.metric_q.put(("episode", ep)),
                on_status=lambda msg: self.metric_q.put(("status", msg)),
                stop_check=lambda: self.stop_event.is_set(),
            )

            algorithm = self.params.get("algorithm", "dqn")
            if algorithm == "mappo":
                from src.ai.train import train_mappo_with_callbacks
                # MAPPO uses different params than DQN
                mappo_kwargs = dict(
                    net_file=self.params["net"],
                    route_file=self.params["route"],
                    sumo_cfg=self.params["cfg"],
                    episodes=self.params["episodes"],
                    delta_time=self.params["delta_time"],
                    sim_length=self.params["sim_length"],
                    hidden=self.params["hidden"],
                    lr=self.params["lr"],
                    gamma=self.params["gamma"],
                    save_dir=self.params["save_dir"],
                    save_every=self.params["save_every"],
                    seed=self.params["seed"],
                    gui=self.params.get("gui", False),
                    on_episode=lambda ep: self.metric_q.put(("episode", ep)),
                    on_status=lambda msg: self.metric_q.put(("status", msg)),
                    stop_check=lambda: self.stop_event.is_set(),
                )
                train_mappo_with_callbacks(**mappo_kwargs)
            elif self.params.get("dyna", False):
                from src.ai.train import train_dyna_with_callbacks
                train_dyna_with_callbacks(**common_kwargs)
            else:
                from src.ai.train import train_with_callbacks
                train_with_callbacks(**common_kwargs)
            self.metric_q.put(("done", None))
        except Exception as e:
            import traceback
            err_msg = f"{e}\n\n{traceback.format_exc()}"
            self.error = err_msg
            self.metric_q.put(("error", err_msg))


# ── Dashboard ───────────────────────────────────────────────────────────

class RLDashboard:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("FlowMind AI - RL Training Dashboard")
        self.root.geometry("1200x850")
        self.root.configure(bg=BG)
        self.root.minsize(900, 600)

        self._metric_q: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._training_thread: TrainingThread | None = None
        self._is_training = False

        # Collected metrics for charts
        self._ep_nums: list[int] = []
        self._rewards: list[float] = []
        self._waits: list[float] = []
        self._epsilons: list[float] = []
        self._best_reward = -float("inf")

        self._build_ui()

    # ── UI construction ─────────────────────────────────────────────

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self.root, bg=BG)
        hdr.pack(fill=tk.X, padx=10, pady=(10, 5))
        tk.Label(hdr, text="RL Training Dashboard",
                 font=("Segoe UI", 16, "bold"), fg=FG, bg=BG).pack(side=tk.LEFT)
        self._status_var = tk.StringVar(value="Ready")
        tk.Label(hdr, textvariable=self._status_var,
                 font=("Segoe UI", 11), fg=FG2, bg=BG).pack(side=tk.RIGHT)

        # Main area: left params + right charts
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # ── Left: Hyperparameters ───────────────────────────────────
        left = tk.LabelFrame(main, text=" Hyperparameters ",
                             font=("Segoe UI", 10, "bold"), fg=FG, bg=BG,
                             padx=10, pady=8)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        self._params: dict[str, tk.Variable] = {}
        param_defs = [
            ("episodes",       "Episodes",        100,    int),
            ("lr",             "Learning Rate",    0.001,  float),
            ("hidden",         "Hidden Size",      256,    int),
            ("gamma",          "Gamma",            0.99,   float),
            ("batch_size",     "Batch Size",       64,     int),
            ("buffer_capacity","Buffer Size",      200000, int),
            ("epsilon_decay",  "Epsilon Decay",    500000, int),
            ("delta_time",     "Delta Time (s)",   30,     int),
            ("sim_length",     "Sim Steps",         1800,   int),
            ("seed",           "Seed",             42,     int),
        ]

        self._param_entries: list[tk.Entry] = []
        for i, (key, label, default, _) in enumerate(param_defs):
            tk.Label(left, text=label + ":", font=("Segoe UI", 9),
                     fg=FG2, bg=BG, anchor=tk.W).grid(
                row=i, column=0, sticky=tk.W, pady=2)
            var = tk.StringVar(value=str(default))
            self._params[key] = var
            entry = tk.Entry(left, textvariable=var, width=12,
                             bg=BG2, fg=FG, insertbackground=FG,
                             font=("Consolas", 10))
            entry.grid(row=i, column=1, padx=(10, 0), pady=2)
            self._param_entries.append(entry)
        self._param_types = {k: t for k, _, _, t in param_defs}

        # File paths
        row = len(param_defs)
        tk.Label(left, text="Network:", font=("Segoe UI", 9),
                 fg=FG2, bg=BG).grid(row=row, column=0, sticky=tk.W, pady=(10, 2))
        self._net_var = tk.StringVar(value=_DANANG["net"])
        tk.Entry(left, textvariable=self._net_var, width=22,
                 bg=BG2, fg=FG, font=("Consolas", 8),
                 insertbackground=FG).grid(row=row, column=1, pady=(10, 2))

        row += 1
        tk.Label(left, text="Routes:", font=("Segoe UI", 9),
                 fg=FG2, bg=BG).grid(row=row, column=0, sticky=tk.W, pady=2)
        self._route_var = tk.StringVar(value=_DANANG["route"])
        tk.Entry(left, textvariable=self._route_var, width=22,
                 bg=BG2, fg=FG, font=("Consolas", 8),
                 insertbackground=FG).grid(row=row, column=1, pady=2)

        row += 1
        tk.Label(left, text="Config:", font=("Segoe UI", 9),
                 fg=FG2, bg=BG).grid(row=row, column=0, sticky=tk.W, pady=2)
        self._cfg_var = tk.StringVar(value=_DANANG["cfg"])
        tk.Entry(left, textvariable=self._cfg_var, width=22,
                 bg=BG2, fg=FG, font=("Consolas", 8),
                 insertbackground=FG).grid(row=row, column=1, pady=2)

        # Buttons
        row += 1
        btn_frame = tk.Frame(left, bg=BG)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=(15, 5))

        self._start_btn = tk.Button(
            btn_frame, text="Start Training", font=("Segoe UI", 10, "bold"),
            bg=GREEN, fg="#1e1e2e", activebackground="#81c784",
            width=14, pady=4, command=self._start_training)
        self._start_btn.grid(row=0, column=0, padx=3)

        self._stop_btn = tk.Button(
            btn_frame, text="Stop", font=("Segoe UI", 10, "bold"),
            bg=RED, fg="white", activebackground="#e57373",
            width=8, pady=4, state=tk.DISABLED, command=self._stop_training)
        self._stop_btn.grid(row=0, column=1, padx=3)

        row += 1
        self._gui_var = tk.BooleanVar(value=False)
        tk.Checkbutton(left, text="Train with SUMO-gui",
                       variable=self._gui_var, font=("Segoe UI", 9),
                       fg=ORANGE, bg=BG, selectcolor=BG2,
                       activebackground=BG, activeforeground=ORANGE,
                       ).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)

        row += 1
        tk.Label(left, text="Algorithm:", font=("Segoe UI", 9),
                 fg=FG2, bg=BG).grid(row=row, column=0, sticky=tk.W, pady=2)
        self._algo_var = tk.StringVar(value="mappo")
        algo_combo = ttk.Combobox(left, textvariable=self._algo_var,
                                   values=["dqn", "mappo"], width=10,
                                   state="readonly")
        algo_combo.grid(row=row, column=1, sticky=tk.W, pady=2)

        row += 1
        self._dyna_var = tk.BooleanVar(value=False)
        tk.Checkbutton(left, text="Dyna mode (DQN only, AI surrogate)",
                       variable=self._dyna_var, font=("Segoe UI", 9),
                       fg=BLUE, bg=BG, selectcolor=BG2,
                       activebackground=BG, activeforeground=BLUE,
                       ).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)

        row += 1
        self._eval_btn = tk.Button(
            left, text="Evaluate Best Model", font=("Segoe UI", 10),
            bg=BLUE, fg="#1e1e2e", width=22, pady=4, command=self._evaluate)
        self._eval_btn.grid(row=row, column=0, columnspan=2, pady=5)

        # ── Right: Charts ───────────────────────────────────────────
        right = tk.Frame(main, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._fig = Figure(figsize=(6, 5), dpi=90, facecolor=BG)
        self._fig.subplots_adjust(hspace=0.45, left=0.12, right=0.95,
                                  top=0.95, bottom=0.08)

        self._ax_reward = self._fig.add_subplot(311)
        self._ax_wait = self._fig.add_subplot(312)
        self._ax_eps = self._fig.add_subplot(313)

        for ax, title, color in [
            (self._ax_reward, "Mean Reward", GREEN),
            (self._ax_wait, "Avg Wait Time (s)", RED),
            (self._ax_eps, "Epsilon", BLUE),
        ]:
            ax.set_facecolor(BG2)
            ax.set_title(title, fontsize=10, color=FG, pad=6)
            ax.tick_params(colors=FG2, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(BG3)
            ax.grid(True, alpha=0.2, color=FG2)

        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── Bottom: Episode log table ───────────────────────────────
        bottom = tk.LabelFrame(self.root, text=" Episode Log ",
                               font=("Segoe UI", 10, "bold"), fg=FG, bg=BG,
                               padx=5, pady=5)
        bottom.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 5))

        # Toolbar ABOVE the table (always visible)
        toolbar = tk.Frame(bottom, bg=BG)
        toolbar.pack(fill=tk.X, pady=(0, 4))
        tk.Label(toolbar, text="Double-click row for TLS details",
                 font=("Segoe UI", 8), fg=FG2, bg=BG).pack(side=tk.LEFT)
        self._copy_btn = tk.Button(
            toolbar, text="Copy Full Log", font=("Segoe UI", 9),
            bg=YELLOW, fg="#1e1e2e", width=14, command=self._copy_log)
        self._copy_btn.pack(side=tk.RIGHT)

        cols = ("src", "ep", "reward", "loss", "eps", "wait", "queue",
                "veh", "add", "rm", "time")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("RL.Treeview", background=BG2, foreground=FG,
                        fieldbackground=BG2, rowheight=24,
                        font=("Consolas", 9))
        style.configure("RL.Treeview.Heading", background=BG3,
                        foreground=FG, font=("Segoe UI", 9, "bold"))
        style.map("RL.Treeview", background=[("selected", BG3)])

        tree_frame = tk.Frame(bottom, bg=BG)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        self._tree = ttk.Treeview(tree_frame, columns=cols, show="headings",
                                  style="RL.Treeview", height=6)
        heads = {
            "src": ("Src", 35), "ep": ("Ep", 40), "reward": ("Reward", 70),
            "loss": ("Loss", 65), "eps": ("Eps", 55),
            "wait": ("Wait", 55), "queue": ("Queue", 50),
            "veh": ("Veh", 45), "add": ("+Add", 40),
            "rm": ("-Rm", 40), "time": ("Time", 45),
        }
        for c, (label, w) in heads.items():
            self._tree.heading(c, text=label)
            self._tree.column(c, width=w, anchor=tk.CENTER, minwidth=w)

        sb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.pack(fill=tk.BOTH, expand=True)

        # Click to inspect episode details
        self._tree.bind("<Double-1>", self._on_episode_click)

        # Full log accumulator
        self._full_log_lines: list[str] = []
        # Per-episode TLS details cache
        self._ep_details: dict[int, list[dict]] = {}

    # ── Training control ────────────────────────────────────────────

    def _get_params(self) -> dict:
        p = {}
        for key, var in self._params.items():
            cast = self._param_types[key]
            try:
                p[key] = cast(var.get())
            except ValueError:
                messagebox.showerror("Invalid Parameter",
                                     f"Bad value for {key}: {var.get()}")
                return {}
        p["net"] = self._net_var.get()
        p["route"] = self._route_var.get()
        p["cfg"] = self._cfg_var.get()
        p["save_dir"] = os.path.join(_PROJECT_ROOT, "checkpoints")
        p["save_every"] = 10
        p["epsilon_start"] = 1.0
        p["epsilon_end"] = 0.05
        p["target_update"] = 1000
        p["gui"] = self._gui_var.get()
        p["dyna"] = self._dyna_var.get()
        p["algorithm"] = self._algo_var.get()
        return p

    def _start_training(self):
        params = self._get_params()
        if not params:
            return

        # Verify files exist
        for key in ("net", "route", "cfg"):
            path = params.get(key, "")
            if path and not os.path.isfile(path):
                messagebox.showerror("File Not Found", f"{key}: {path}")
                return

        # Reset state
        self._ep_nums.clear()
        self._rewards.clear()
        self._waits.clear()
        self._epsilons.clear()
        self._best_reward = -float("inf")
        self._tree.delete(*self._tree.get_children())
        self._full_log_lines.clear()
        self._stop_event.clear()

        # Disable controls
        self._is_training = True
        self._start_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)
        self._eval_btn.config(state=tk.DISABLED)
        for entry in self._param_entries:
            entry.config(state=tk.DISABLED)

        # Launch thread
        self._training_thread = TrainingThread(
            params, self._metric_q, self._stop_event)
        self._training_thread.start()
        self._status_var.set("Training...")
        self._poll_metrics()

    def _stop_training(self):
        self._stop_event.set()
        self._status_var.set("Stopping after current episode...")
        self._stop_btn.config(state=tk.DISABLED)

    def _training_finished(self, error: str | None = None):
        self._is_training = False
        self._start_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        self._eval_btn.config(state=tk.NORMAL)
        for entry in self._param_entries:
            entry.config(state=tk.NORMAL)

        if error:
            self._status_var.set(f"Error: {error[:60]}")
            self._full_log_lines.append(f"\nERROR: {error}")
            messagebox.showerror("Training Error", error)
        else:
            self._status_var.set(
                f"Done! Best reward: {self._best_reward:.4f}")
            # Append recommendations from training log
            self._append_recommendations_to_log()

    def _append_recommendations_to_log(self):
        """Read training_log.json and append recommendations summary."""
        log_path = os.path.join(_PROJECT_ROOT, "checkpoints", "training_log.json")
        try:
            with open(log_path) as f:
                log = json.load(f)
        except Exception:
            return

        recs = log.get("recommendations", {})
        cfg = log.get("config", {})

        self._full_log_lines.append("")
        self._full_log_lines.append("=" * 60)
        self._full_log_lines.append("  TRAINING COMPLETE")
        self._full_log_lines.append("=" * 60)
        self._full_log_lines.append(
            f"  Best reward: {self._best_reward:.4f}")
        self._full_log_lines.append(
            f"  TLS agents: {cfg.get('num_agents', '?')}")
        self._full_log_lines.append("")

        if recs.get("add"):
            self._full_log_lines.append("  ADD NEW TRAFFIC LIGHTS:")
            for r in recs["add"]:
                self._full_log_lines.append(
                    f"    + {r['id']} (active "
                    f"{100 - r['off_pct']:.0f}% of time)")

        if recs.get("remove"):
            self._full_log_lines.append("  REMOVE TRAFFIC LIGHTS:")
            for r in recs["remove"]:
                self._full_log_lines.append(
                    f"    - {r['id']} (OFF {r['off_pct']:.0f}% of time)")

        if recs.get("keep_off"):
            self._full_log_lines.append("  CANDIDATES - NO TLS NEEDED:")
            for r in recs["keep_off"]:
                self._full_log_lines.append(
                    f"    . {r['id']} (OFF {r['off_pct']:.0f}%)")

        n_add = len(recs.get("add", []))
        n_rm = len(recs.get("remove", []))
        n_keep = len(recs.get("keep_off", []))
        self._full_log_lines.append("")
        self._full_log_lines.append(
            f"  Summary: +{n_add} new TLS, -{n_rm} removals, "
            f"{n_keep} candidates not needed")
        self._full_log_lines.append("=" * 60)

    def _copy_log(self):
        """Copy the full training log to clipboard."""
        # Build header with hyperparameters
        header = ["FlowMind AI - RL Training Log", "=" * 40]
        for key, var in self._params.items():
            header.append(f"  {key}: {var.get()}")
        header.append(f"  network: {self._net_var.get()}")
        header.append(f"  routes: {self._route_var.get()}")
        header.append(f"  config: {self._cfg_var.get()}")
        header.append("=" * 40)
        header.append("")

        full_text = "\n".join(header + self._full_log_lines)
        self.root.clipboard_clear()
        self.root.clipboard_append(full_text)
        self._status_var.set("Log copied to clipboard!")
        self.root.after(3000, lambda: self._status_var.set(
            f"Done! Best reward: {self._best_reward:.4f}"
            if not self._is_training else "Ready"))

    # ── Metric polling ──────────────────────────────────────────────

    def _poll_metrics(self):
        try:
            while True:
                msg_type, data = self._metric_q.get_nowait()
                if msg_type == "episode":
                    self._on_episode(data)
                elif msg_type == "status":
                    self._status_var.set(data)
                elif msg_type == "done":
                    self._training_finished()
                    return
                elif msg_type == "error":
                    self._training_finished(error=data)
                    return
        except queue.Empty:
            pass

        if self._is_training:
            self.root.after(1000, self._poll_metrics)

    def _on_episode(self, ep: dict):
        ep_num = ep["episode"]
        total = ep.get("total_episodes", "?")
        reward = ep["mean_reward"]
        self._best_reward = max(self._best_reward, reward)
        # [TLS CANDIDATE COMMENTED OUT]
        n_add = 0  # ep.get("tls_add", 0)
        n_rm = 0   # ep.get("tls_remove", 0)

        # Cache TLS details for click-to-inspect
        self._ep_details[ep_num] = {
            "tls": ep.get("tls_details", []),
            # [Level 2 REMOVED] "events": ep.get("event_log", []),
        }

        # Store for charts
        self._ep_nums.append(ep_num)
        self._rewards.append(reward)
        self._waits.append(ep["avg_wait"])
        self._epsilons.append(ep["epsilon"])

        # Update table
        source = ep.get("source", "sumo")
        src_label = "S" if source == "surrogate" else "R"
        veh_display = ep["vehicles"] if ep["vehicles"] > 0 else "~"
        self._tree.insert("", tk.END, values=(
            src_label,
            ep_num,
            f"{reward:+.3f}",
            f"{ep['mean_loss']:.4f}",
            f"{ep['epsilon']:.3f}",
            f"{ep['avg_wait']:.1f}",
            f"{ep['avg_queue']:.1f}",
            veh_display,
            f"+{n_add}" if n_add else "0",
            f"-{n_rm}" if n_rm else "0",
            f"{ep['time_s']:.0f}s",
        ))
        self._tree.yview_moveto(1.0)  # auto-scroll

        # Accumulate full log line
        n_col = ep.get("collisions", 0)
        source = ep.get("source", "sumo")
        src_tag = "S" if source == "surrogate" else "R"  # S=surrogate, R=real
        line = (
            f"[{src_tag}] Ep {ep_num:>4}/{total} | "
            f"R={reward:+.4f} | Loss={ep['mean_loss']:.4f} | "
            f"eps={ep['epsilon']:.3f} | "
            f"Wait={ep['avg_wait']:.1f}s | Queue={ep['avg_queue']:.1f} | "
            f"Veh={ep['vehicles']} | Col={n_col} | "
            f"+Add={n_add} -Rm={n_rm} | {ep['time_s']:.0f}s"
        )
        self._full_log_lines.append(line)
        # [Level 2 REMOVED] Event log messages removed

        # Update status
        self._status_var.set(
            f"Ep {ep_num}/{total} | Best: {self._best_reward:+.3f} | "
            f"+Add={n_add} -Rm={n_rm} | eps={ep['epsilon']:.3f}")

        # Update charts
        self._update_charts()

    def _update_charts(self):
        for ax, data, color in [
            (self._ax_reward, self._rewards, GREEN),
            (self._ax_wait, self._waits, RED),
            (self._ax_eps, self._epsilons, BLUE),
        ]:
            ax.clear()
            ax.set_facecolor(BG2)
            ax.plot(self._ep_nums, data, color=color, linewidth=1.5)
            if data:
                ax.fill_between(self._ep_nums, data,
                                alpha=0.15, color=color)
            ax.tick_params(colors=FG2, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(BG3)
            ax.grid(True, alpha=0.2, color=FG2)

        titles = [
            (self._ax_reward, "Mean Reward"),
            (self._ax_wait, "Avg Wait Time (s)"),
            (self._ax_eps, "Epsilon"),
        ]
        for ax, title in titles:
            ax.set_title(title, fontsize=10, color=FG, pad=6)

        self._canvas.draw_idle()

    # ── Episode detail popup ─────────────────────────────────────

    def _on_episode_click(self, event):
        """Double-click on episode row to show per-TLS detail popup."""
        try:
            sel = self._tree.selection()
            if not sel:
                # Try focus item instead
                sel = [self._tree.focus()]
            if not sel or not sel[0]:
                return
            item = self._tree.item(sel[0])
            vals = item.get("values", [])
            if len(vals) < 2:
                return

            # values[0] = Src ("R"/"S"), values[1] = Ep number
            # Tkinter may return int or str depending on theme
            ep_num = int(str(vals[1]).strip())

            details = self._ep_details.get(ep_num, {})
            tls_details = details.get("tls", []) if isinstance(details, dict) else details
            # [Level 2 REMOVED] event_log removed

            # Fallback: try loading from training_log.json on disk
            if not tls_details:
                try:
                    log_path = os.path.join(_PROJECT_ROOT, "checkpoints",
                                            "training_log.json")
                    with open(log_path) as f:
                        log_data = json.load(f)
                    for ep_entry in log_data.get("episodes", []):
                        if ep_entry.get("episode") == ep_num:
                            tls_details = ep_entry.get("tls_details", [])
                            break
                except Exception:
                    pass

            if not tls_details:
                messagebox.showinfo(
                    f"Episode {ep_num}",
                    f"Source: {vals[0]}\n"
                    f"Reward: {vals[2]}\n"
                    f"Wait: {vals[5]}s | Queue: {vals[6]}\n"
                    f"+Add: {vals[8]} | -Rm: {vals[9]}\n\n"
                    f"(No per-TLS details for this episode)")
                return

            self._show_detail_popup(ep_num, tls_details)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load details:\n{e}")

    def _show_detail_popup(self, ep_num: int, details: list[dict]):
        """Show a popup window with per-TLS breakdown for one episode."""

        win = tk.Toplevel(self.root)
        win.title(f"Episode {ep_num} - TLS Details")
        win.geometry("900x550")
        win.configure(bg=BG)
        win.attributes("-topmost", True)

        # Header
        tk.Label(win, text=f"Episode {ep_num} - Per-TLS Details",
                 font=("Segoe UI", 13, "bold"), fg=FG, bg=BG
                 ).pack(padx=10, pady=(10, 5))

        # Summary counts
        n_keep = sum(1 for d in details if d["decision"] == "KEEP")
        n_remove = sum(1 for d in details if d["decision"] == "REMOVE")
        n_add = sum(1 for d in details if d["decision"] == "ADD")
        n_off = sum(1 for d in details if d["decision"] == "NO TLS")

        summary = (f"KEEP: {n_keep}  |  REMOVE: {n_remove}  |  "
                   f"ADD: {n_add}  |  NO TLS: {n_off}")
        tk.Label(win, text=summary, font=("Consolas", 10),
                 fg=YELLOW, bg=BG).pack()

        # Table — use a plain tk.Text widget for reliable dark-theme rendering
        frame = tk.Frame(win, bg=BG)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        text = tk.Text(frame, bg=BG2, fg=FG, font=("Consolas", 9),
                       wrap=tk.NONE, borderwidth=0, highlightthickness=0,
                       padx=8, pady=6)

        # Tags for coloring rows
        text.tag_configure("header", foreground=FG2,
                           font=("Consolas", 9, "bold"))
        text.tag_configure("keep", foreground=GREEN)
        text.tag_configure("remove", foreground=RED)
        text.tag_configure("add", foreground=ORANGE)
        text.tag_configure("no_tls", foreground=FG2)
        text.tag_configure("sep", foreground=BG3)

        # Header
        hdr = (f"{'Decision':<10} {'Type':<11} {'Road Name':<26} "
               f"{'OFF%':>5} {'Wait':>7} {'Queue':>6} {'Roads':>5}  "
               f"{'TLS ID'}")
        text.insert(tk.END, hdr + "\n", "header")
        text.insert(tk.END, "-" * 100 + "\n", "sep")

        # Rows
        for d in details:
            dec = d["decision"]
            tag = {"KEEP": "keep", "REMOVE": "remove",
                   "ADD": "add", "NO TLS": "no_tls"}.get(dec, "")
            road = d["road_name"]
            if len(road) > 24:
                road = road[:22] + ".."
            tls_id = d["tls_id"]
            if len(tls_id) > 35:
                tls_id = tls_id[:33] + ".."

            line = (f"{dec:<10} {d['type']:<11} {road:<26} "
                    f"{d['off_pct']:>4.0f}% {d['wait']:>6.0f}s "
                    f"{d['queue']:>5} {d['n_incoming']:>5}  "
                    f"{tls_id}")
            text.insert(tk.END, line + "\n", tag)

        # [Level 2 REMOVED] Event log section removed

        text.config(state=tk.DISABLED)

        sb_y = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text.yview)
        sb_x = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=text.xview)
        text.configure(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)
        sb_y.pack(side=tk.RIGHT, fill=tk.Y)
        sb_x.pack(side=tk.BOTTOM, fill=tk.X)
        text.pack(fill=tk.BOTH, expand=True)

        # Copy button
        def _copy_details():
            lines = [f"Episode {ep_num} - TLS Details", summary, ""]
            lines.append(f"{'Decision':<10} {'Type':<10} {'Road':<25} "
                         f"{'OFF%':>5} {'Wait':>7} {'Queue':>6} {'Roads':>5}")
            lines.append("-" * 75)
            for d in details:
                road = d["road_name"][:23]
                lines.append(
                    f"{d['decision']:<10} {d['type']:<10} {road:<25} "
                    f"{d['off_pct']:>4.0f}% {d['wait']:>6.0f}s "
                    f"{d['queue']:>5} {d['n_incoming']:>5}")
            # [Level 2 REMOVED] Event log section removed
            win.clipboard_clear()
            win.clipboard_append("\n".join(lines))
            self._status_var.set(f"Ep {ep_num} details copied!")

        tk.Button(win, text="Copy Details", font=("Segoe UI", 9),
                  bg=YELLOW, fg="#1e1e2e", width=14,
                  command=_copy_details).pack(pady=(0, 10))

    # ── Evaluation ──────────────────────────────────────────────────

    def _evaluate(self):
        model_path = os.path.join(_PROJECT_ROOT, "checkpoints", "best_model.pt")
        if not os.path.isfile(model_path):
            model_path = filedialog.askopenfilename(
                title="Select model checkpoint",
                filetypes=[("PyTorch", "*.pt")],
                initialdir=os.path.join(_PROJECT_ROOT, "checkpoints"),
            )
            if not model_path:
                return

        self._status_var.set("Evaluating...")
        self._eval_btn.config(state=tk.DISABLED)

        def _run_eval():
            try:
                from src.ai.train import evaluate
                from src.ai.dqn_agent import TrafficDQNAgent
                from src.ai.traffic_env import OBS_DIM, ACT_DIM

                agent = TrafficDQNAgent(OBS_DIM, ACT_DIM,
                                        int(self._params["hidden"].get()))
                agent.load(model_path)
                results = evaluate(
                    agent,
                    net_file=self._net_var.get(),
                    route_file=self._route_var.get(),
                    sumo_cfg=self._cfg_var.get(),
                    episodes=3,
                    seed=int(self._params["seed"].get()),
                )
                self._metric_q.put(("eval_done", results))
            except Exception as e:
                self._metric_q.put(("eval_error", str(e)))

        threading.Thread(target=_run_eval, daemon=True).start()
        self._poll_eval()

    def _poll_eval(self):
        try:
            msg_type, data = self._metric_q.get_nowait()
            self._eval_btn.config(state=tk.NORMAL)
            if msg_type == "eval_done":
                self._status_var.set("Eval complete")
                messagebox.showinfo(
                    "Evaluation Results",
                    f"Mean Reward: {data['mean_reward']:.3f}\n"
                    f"Mean Wait: {data['mean_wait']:.1f}s\n"
                    f"Mean Queue: {data['mean_queue']:.1f}\n"
                    f"Episodes: {len(data['episodes'])}"
                )
            elif msg_type == "eval_error":
                self._status_var.set("Eval failed")
                messagebox.showerror("Evaluation Error", data)
            return
        except queue.Empty:
            pass
        self.root.after(1000, self._poll_eval)

    # ── Run ─────────────────────────────────────────────────────────

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        if self._is_training:
            self._stop_event.set()
            time.sleep(0.5)
        self.root.destroy()


# ── Main ────────────────────────────────────────────────────────────────

def main():
    dashboard = RLDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
