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

import traci

from src.ai.traffic_env import (
    SumoTrafficEnv, OBS_DIM, OLD_OBS_DIM, ACT_DIM, ACT_OFF,
    remap_obs_for_old_model,
)
from src.ai.dqn_agent import TrafficDQNAgent

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
        self.root.geometry("420x520")
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

        # Active events display
        ef = tk.LabelFrame(self.root, text=" Active Events ",
                           font=("Segoe UI", 10, "bold"), fg=ORANGE, bg=BG,
                           padx=10, pady=5)
        ef.pack(fill=tk.X, padx=10, pady=5)
        self._events_var = tk.StringVar(value="None")
        tk.Label(ef, textvariable=self._events_var,
                 font=("Consolas", 9), fg=YELLOW, bg=BG,
                 anchor=tk.W, justify=tk.LEFT).pack(fill=tk.X)

        # Legend
        lf = tk.LabelFrame(self.root, text=" Map Legend ",
                           font=("Segoe UI", 10, "bold"), fg=FG, bg=BG,
                           padx=10, pady=5)
        lf.pack(fill=tk.X, padx=10, pady=5)

        legends = [
            ("Green circle", "Agent keeps this signal", GREEN),
            ("Red circle", "Agent wants to REMOVE", RED),
            ("Orange diamond", "Agent wants to ADD here", ORANGE),
            ("Gray diamond", "Candidate - not needed", FG2),
        ]
        for i, (sym, desc, color) in enumerate(legends):
            tk.Label(lf, text=f"{sym}: {desc}",
                     font=("Segoe UI", 8), fg=color, bg=BG,
                     anchor=tk.W).grid(row=i, column=0, sticky=tk.W)

        # Stop button
        tk.Button(self.root, text="Stop & Close", font=("Segoe UI", 10, "bold"),
                  bg=RED, fg="white", width=18, pady=4,
                  command=self._on_close).pack(pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

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

            agent = TrafficDQNAgent(ckpt_obs_dim, ACT_DIM, self.hidden)
            agent.load(self.model_path)
            # greedy=True is passed to select_action, no need to set epsilon

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

            # Per-TLS action tracking for live overlay updates
            action_counts: dict[str, Counter] = {
                tid: Counter() for tid in env.tls_ids
            }
            total_reward = 0.0
            step = 0
            terminated = truncated = False

            # Set initial SUMO-gui view (zoom, delay)
            try:
                conn = env._conn
                conn.gui.setSchema("View #0", "real world")
                conn.gui.setDelay(50)  # 50ms between steps for visibility
            except Exception:
                pass

            # Check if model needs obs remapping (old 26-dim vs new 39-dim)
            needs_remap = (ckpt_obs_dim == OLD_OBS_DIM and OBS_DIM != OLD_OBS_DIM)

            while not (terminated or truncated) and not self._stop:
                actions = {}
                for tid in env.tls_ids:
                    valid = env.get_valid_actions(tid)
                    o = remap_obs_for_old_model(obs[tid]) if needs_remap else obs[tid]
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

                # Update TLS overlays every 10 steps
                if step % 10 == 1:
                    self._update_overlays(env, action_counts, candidate_info)

                # Update active events display
                active_events = env.get_active_events()
                if active_events:
                    evt_lines = []
                    for ae in active_events:
                        evt_lines.append(
                            f"{ae['type'].upper()} "
                            f"(int={ae['intensity']:.0%}, "
                            f"{ae['remaining']:.0f}s left)")
                    try:
                        self.root.after(0, lambda l=evt_lines:
                            self._events_var.set("\n".join(l)))
                    except Exception:
                        pass
                else:
                    try:
                        self.root.after(0, lambda:
                            self._events_var.set("None"))
                    except Exception:
                        pass

                self._set_status(
                    f"Step {step} | {metrics.get('sim_time', 0)}s | "
                    f"Veh={metrics.get('total_vehicles', 0)} | "
                    f"Events={len(active_events)}")

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

            off_pct = counts.get(ACT_OFF, 0) / total
            is_candidate = tid in env.candidate_tls_ids
            is_existing = tid in env.existing_tls_ids

            # Determine color and shape
            if is_candidate:
                if off_pct < 0.4:
                    color = CLR_ADD
                    n_add += 1
                    label = "ADD"
                else:
                    color = CLR_OFF
                    n_off += 1
                    label = "OFF"
            elif is_existing:
                if off_pct > 0.6:
                    color = CLR_REMOVE
                    n_remove += 1
                    label = "REMOVE"
                else:
                    color = CLR_KEEP
                    n_keep += 1
                    label = "KEEP"
            else:
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
                    width=20 if is_candidate else 15,
                    height=20 if is_candidate else 15,
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
