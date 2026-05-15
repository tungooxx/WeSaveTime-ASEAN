"""
FlowMind AI - SUMO-first cinematic story demo.

Separate from the normal visualizer. This tool stages a 5-scene presentation
and uses real GAT attention to explain one hotspot directly on the SUMO map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tkinter as tk

try:
    import pygetwindow as gw
except ImportError:  # pragma: no cover
    gw = None

from PIL import Image, ImageDraw, ImageFont

import traci
import traci.constants as tc

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.ai.mappo_agent import MAPPOAgent
from src.ai.traffic_env import ACT_DIM, OBS_DIM, SumoTrafficEnv, remap_obs_for_old_model


VIEW_ID = "View #0"
HEAT_SUB_VARS = [
    tc.LAST_STEP_VEHICLE_NUMBER,
    tc.LAST_STEP_MEAN_SPEED,
    tc.LAST_STEP_OCCUPANCY,
    tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
    tc.VAR_WAITING_TIME,
]

HUD_BG = "#111827"
HUD_BG2 = "#1f2937"
HUD_FG = "#f8fafc"
HUD_MUTED = "#cbd5e1"
HUD_ACCENT = "#38bdf8"
HUD_WARN = "#f97316"

CLR_FLOWING = (66, 184, 131, 170)
CLR_BUSY = (250, 178, 68, 190)
CLR_BACKING = (239, 68, 68, 210)
CLR_FOCUS = (255, 215, 64, 235)
CLR_NEIGHBOR = (56, 189, 248, 215)
CLR_NEIGHBOR_SOFT = (125, 211, 252, 180)
CLR_CARD = (15, 23, 42, 225)
CLR_CARD_ACCENT = (56, 189, 248, 240)
CLR_BASELINE = (248, 113, 113, 220)
CLR_AI = (34, 197, 94, 220)

LABEL_DIR = Path(_PROJECT_ROOT) / "tmp" / "demo_story_labels"


def _trim_name(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "Unnamed road"
    return text.split("#")[0].strip() or text


def _unique_label(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _scene_control_steps(seconds: float, delta_time: int, gui_delay_ms: int) -> int:
    control_wall_ms = max(delta_time * gui_delay_ms, 1)
    return max(1, int(round((seconds * 1000.0) / control_wall_ms)))


def _baseline_action(env: SumoTrafficEnv) -> np.ndarray:
    return np.full(env.act_dim, 0.5, dtype=np.float32)


def _congestion_score(wait: float, halt: float, speed: float, max_speed: float, occ: float) -> float:
    speed_ratio = 1.0 - min(speed / max(max_speed, 0.1), 1.0)
    return (
        0.35 * speed_ratio
        + 0.25 * min(wait / 200.0, 1.0)
        + 0.25 * min(halt / 15.0, 1.0)
        + 0.15 * min(occ, 1.0)
    )


@dataclass
class CheckpointSpec:
    obs_dim: int
    act_dim: int
    hidden: int
    algorithm: str
    gat_out: int
    neighbor_feat_dim: int

    @property
    def uses_gat(self) -> bool:
        return self.algorithm == "mappo" and self.gat_out > 0

    @property
    def needs_remap(self) -> bool:
        return self.obs_dim < OBS_DIM


@dataclass
class HotspotInfo:
    tls_id: str
    name: str
    x: float
    y: float
    queue: float
    wait: float
    density: float
    score: float


class LabelSpriteFactory:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        self._font = self._load_font(16)
        self._font_small = self._load_font(14)
        self._font_title = self._load_font(20)

    def _load_font(self, size: int):
        for name in ("segoeui.ttf", "arial.ttf", "calibri.ttf"):
            try:
                return ImageFont.truetype(name, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    def render(self, *, key: str, title: str, lines: list[str], accent) -> str:
        payload = json.dumps(
            {"key": key, "title": title, "lines": lines, "accent": accent},
            sort_keys=True,
        ).encode("utf-8")
        cache_key = hashlib.md5(payload).hexdigest()
        cached = self._cache.get(cache_key)
        if cached and os.path.isfile(cached):
            return cached

        probe = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
        draw = ImageDraw.Draw(probe)
        width = 180
        height = 44
        title_bbox = draw.textbbox((0, 0), title, font=self._font_title)
        width = max(width, title_bbox[2] - title_bbox[0] + 32)
        height += title_bbox[3] - title_bbox[1]
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=self._font_small)
            width = max(width, bbox[2] - bbox[0] + 32)
            height += bbox[3] - bbox[1] + 8
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=18, fill=CLR_CARD)
        draw.rounded_rectangle((0, 0, 12, height - 1), radius=12, fill=accent)
        y = 14
        draw.text((22, y), title, font=self._font_title, fill=(248, 250, 252, 255))
        y += title_bbox[3] - title_bbox[1] + 10
        for line in lines:
            draw.text((22, y), line, font=self._font_small, fill=(226, 232, 240, 255))
            bbox = draw.textbbox((22, y), line, font=self._font_small)
            y = bbox[3] + 6
        path = self.root / f"{cache_key}.png"
        img.save(path)
        self._cache[cache_key] = str(path)
        return str(path)


class SumoOverlayRenderer:
    def __init__(self, env: SumoTrafficEnv, prefix: str, labels: LabelSpriteFactory):
        self.env = env
        self.conn = env._conn
        self.net = env._net
        self.prefix = prefix
        self.labels = labels
        self._scene_polygons: set[str] = set()
        self._scene_pois: set[str] = set()
        self._heat_polygons: set[str] = set()
        self._edge_meta: dict[str, dict] = {}

    def _pid(self, *parts: str) -> str:
        return f"{self.prefix}_{'_'.join(parts)}"

    def clear_scene(self) -> None:
        for pid in list(self._scene_pois):
            try:
                self.conn.poi.remove(pid)
            except Exception:
                pass
        for pid in list(self._scene_polygons):
            try:
                self.conn.polygon.remove(pid)
            except Exception:
                pass
        self._scene_pois.clear()
        self._scene_polygons.clear()

    def clear_heatmap(self) -> None:
        for pid in list(self._heat_polygons):
            try:
                self.conn.polygon.remove(pid)
            except Exception:
                pass
        self._heat_polygons.clear()
        self._edge_meta.clear()

    def clear_all(self) -> None:
        self.clear_scene()
        self.clear_heatmap()

    def setup_heatmap(self) -> None:
        self._edge_meta.clear()
        for edge in self.net.getEdges():
            eid = edge.getID()
            if eid.startswith(":") or edge.getLaneNumber() < 2:
                continue
            shape = [(float(x), float(y)) for x, y in edge.getShape()]
            if len(shape) < 2:
                continue
            self._edge_meta[eid] = {
                "shape": shape,
                "name": _trim_name(edge.getName() or eid),
                "max_speed": edge.getSpeed(),
            }
            self.conn.edge.subscribe(eid, HEAT_SUB_VARS)

    def update_heatmap(self) -> None:
        if not self._edge_meta:
            self.setup_heatmap()
        try:
            results = self.conn.edge.getAllSubscriptionResults()
        except Exception:
            return
        for eid, meta in self._edge_meta.items():
            data = results.get(eid, {})
            score = _congestion_score(
                data.get(tc.VAR_WAITING_TIME, 0.0),
                data.get(tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0),
                data.get(tc.LAST_STEP_MEAN_SPEED, 0.0),
                meta["max_speed"],
                data.get(tc.LAST_STEP_OCCUPANCY, 0.0),
            )
            if score >= 0.66:
                color, width = CLR_BACKING, 9
            elif score >= 0.33:
                color, width = CLR_BUSY, 6
            else:
                color, width = CLR_FLOWING, 4
            self._upsert_polygon(
                self._pid("heat", eid),
                meta["shape"],
                color,
                fill=False,
                line_width=width,
                layer=3,
                group="heat",
            )

    def set_macro_view(self) -> None:
        (xmin, ymin), (xmax, ymax) = self.net.getBBoxXY()
        pad_x = (xmax - xmin) * 0.06
        pad_y = (ymax - ymin) * 0.06
        self.conn.gui.setBoundary(VIEW_ID, xmin - pad_x, ymin - pad_y, xmax + pad_x, ymax + pad_y)

    def set_focus_view(self, x: float, y: float, radius: float) -> None:
        self.conn.gui.setBoundary(VIEW_ID, x - radius, y - radius, x + radius, y + radius)

    def current_boundary(self) -> tuple[float, float, float, float]:
        boundary = self.conn.gui.getBoundary(VIEW_ID)
        try:
            if len(boundary) == 2 and hasattr(boundary[0], "__len__"):
                (xmin, ymin), (xmax, ymax) = boundary
            elif len(boundary) == 4:
                xmin, ymin, xmax, ymax = boundary
            else:
                raise ValueError(f"Unexpected GUI boundary format: {boundary!r}")
            return float(xmin), float(ymin), float(xmax), float(ymax)
        except Exception:
            (xmin, ymin), (xmax, ymax) = self.net.getBBoxXY()
            return float(xmin), float(ymin), float(xmax), float(ymax)

    def draw_banner(self, title: str, subtitle: str, accent) -> None:
        xmin, ymin, xmax, ymax = self.current_boundary()
        span_x = xmax - xmin
        span_y = ymax - ymin
        self._upsert_label(
            self._pid("banner"),
            xmin + span_x * 0.22,
            ymax - span_y * 0.08,
            title,
            [subtitle] if subtitle else [],
            max(span_x * 0.28, 180),
            max(span_y * 0.10, 80),
            accent,
        )

    def draw_focus(self, hotspot: HotspotInfo, radius: float = 70.0) -> None:
        self._upsert_polygon(
            self._pid("focus", "ring"),
            self._circle(hotspot.x, hotspot.y, radius),
            CLR_FOCUS,
            fill=False,
            line_width=6,
            layer=15,
            group="scene",
        )
        self._upsert_label(
            self._pid("focus", "label"),
            hotspot.x,
            hotspot.y + radius + 28,
            f"FOCUS · {hotspot.name}",
            [f"Queue {hotspot.queue:.2f} | Wait {hotspot.wait:.2f} | Density {hotspot.density:.2f}"],
            210,
            72,
            CLR_FOCUS,
        )

    def draw_incoming_edge(self, edge_id: str, score: float, label: str) -> None:
        try:
            edge = self.net.getEdge(edge_id)
        except Exception:
            return
        shape = [(float(x), float(y)) for x, y in edge.getShape()]
        if len(shape) < 2:
            return
        mid = shape[len(shape) // 2]
        color = CLR_BACKING if score >= 0.66 else CLR_BUSY if score >= 0.33 else CLR_FLOWING
        self._upsert_polygon(
            self._pid("incoming", edge_id),
            shape,
            color,
            fill=False,
            line_width=8,
            layer=14,
            group="scene",
        )
        self._upsert_label(
            self._pid("incoming", edge_id, "label"),
            mid[0],
            mid[1],
            _trim_name(edge.getName() or edge_id),
            [label],
            160,
            60,
            color,
        )

    def draw_neighbors(self, hotspot: HotspotInfo, neighbors: list[dict]) -> None:
        for idx, row in enumerate(neighbors[:3]):
            color = CLR_NEIGHBOR if idx == 0 else CLR_NEIGHBOR_SOFT
            width = max(4, int(3 + row["weight"] * 10))
            self._upsert_polygon(
                self._pid("neighbor", str(idx), "line"),
                [(hotspot.x, hotspot.y), (row["x"], row["y"])],
                color,
                fill=False,
                line_width=width,
                layer=14,
                group="scene",
            )
            self._upsert_polygon(
                self._pid("neighbor", str(idx), "arrow"),
                self._arrowhead(hotspot.x, hotspot.y, row["x"], row["y"], 20 + 18 * row["weight"]),
                color,
                fill=True,
                line_width=1,
                layer=14,
                group="scene",
            )
            self._upsert_label(
                self._pid("neighbor", str(idx), "label"),
                row["x"],
                row["y"] + 34,
                row["name"],
                [
                    f"Attention {row['weight'] * 100:.0f}% · {row['tag']}",
                    f"Vehicles waiting {row['queue']:.2f}",
                    f"Delay building up {row['wait']:.2f}",
                    f"Road is crowded {row['density']:.2f}",
                ],
                190,
                96,
                color,
            )

    def draw_card(self, title: str, lines: list[str], accent) -> None:
        xmin, ymin, xmax, ymax = self.current_boundary()
        span_x = xmax - xmin
        span_y = ymax - ymin
        self._upsert_label(
            self._pid("card"),
            xmin + max(span_x * 0.14, 120),
            ymin + max(span_y * 0.14, 120),
            title,
            lines,
            max(span_x * 0.30, 220),
            max(span_y * 0.24, 140),
            accent,
        )

    def _upsert_polygon(self, pid: str, shape: list[tuple[float, float]], color, *,
                        fill: bool, line_width: float, layer: int, group: str) -> None:
        try:
            self.conn.polygon.add(pid, shape, color, fill, "demo_story", layer, line_width)
        except Exception:
            try:
                self.conn.polygon.setShape(pid, shape)
                self.conn.polygon.setColor(pid, color)
                self.conn.polygon.setFilled(pid, fill)
                self.conn.polygon.setLineWidth(pid, line_width)
            except Exception:
                return
        if group == "heat":
            self._heat_polygons.add(pid)
        else:
            self._scene_polygons.add(pid)

    def _upsert_label(self, pid: str, x: float, y: float, title: str,
                      lines: list[str], width: float, height: float, accent) -> None:
        img_path = self.labels.render(key=pid, title=title, lines=lines, accent=accent)
        try:
            self.conn.poi.add(pid, x, y, (255, 255, 255, 0), "demo_story", 20, img_path, width, height, 0, "")
        except Exception:
            try:
                self.conn.poi.setPosition(pid, x, y)
                self.conn.poi.setImageFile(pid, img_path)
                self.conn.poi.setWidth(pid, width)
                self.conn.poi.setHeight(pid, height)
            except Exception:
                return
        self._scene_pois.add(pid)

    @staticmethod
    def _circle(cx: float, cy: float, radius: float, points: int = 28) -> list[tuple[float, float]]:
        return [
            (cx + radius * math.cos(2 * math.pi * i / points), cy + radius * math.sin(2 * math.pi * i / points))
            for i in range(points)
        ]

    @staticmethod
    def _arrowhead(x1: float, y1: float, x2: float, y2: float, size: float) -> list[tuple[float, float]]:
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy) or 1.0
        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        tip = (x2, y2)
        base = (x2 - ux * size, y2 - uy * size)
        left = (base[0] + px * size * 0.45, base[1] + py * size * 0.45)
        right = (base[0] - px * size * 0.45, base[1] - py * size * 0.45)
        return [tip, left, right]


class StoryDirector:
    def __init__(self, args):
        self.args = args
        self.scene_var = None
        self.status_var = None
        self.detail_var = None
        self._scene_text = "Preparing demo..."
        self._status_text = "Loading..."
        self._detail_text = "Hotkeys: Space pause/resume | Right next scene | R restart scene | Esc quit"
        self._paused = False
        self._stop = False
        self._scene_command: Optional[str] = None
        self._command_lock = threading.Lock()
        self._story_thread: Optional[threading.Thread] = None
        self._root: Optional[tk.Tk] = None
        self._active_envs: list[SumoTrafficEnv] = []
        self._comparison = self._load_comparison(args.comparison)
        self._agent: Optional[MAPPOAgent] = None
        self._spec: Optional[CheckpointSpec] = None
        self._labels = LabelSpriteFactory(LABEL_DIR)

    def run(self) -> None:
        self._build_hud()
        self._story_thread = threading.Thread(target=self._run_story, daemon=True)
        self._story_thread.start()
        assert self._root is not None
        self._root.mainloop()

    def _build_hud(self) -> None:
        root = tk.Tk()
        root.title("FlowMind Story Director")
        root.geometry("450x190+30+30")
        root.configure(bg=HUD_BG)
        root.attributes("-topmost", True)
        root.bind("<Escape>", lambda _e: self._request_stop())
        root.bind("<space>", lambda _e: self._toggle_pause())
        root.bind("<Right>", lambda _e: self._set_scene_command("next"))
        root.bind("<Key-r>", lambda _e: self._set_scene_command("restart"))
        root.bind("<Key-R>", lambda _e: self._set_scene_command("restart"))

        self.scene_var = tk.StringVar(master=root, value=self._scene_text)
        self.status_var = tk.StringVar(master=root, value=self._status_text)
        self.detail_var = tk.StringVar(master=root, value=self._detail_text)

        tk.Label(root, text="SUMO Story Demo", bg=HUD_BG, fg=HUD_FG,
                 font=("Segoe UI", 16, "bold")).pack(anchor="w", padx=14, pady=(12, 2))
        tk.Label(root, textvariable=self.scene_var, bg=HUD_BG, fg=HUD_ACCENT,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=14)
        tk.Label(root, textvariable=self.status_var, bg=HUD_BG, fg=HUD_FG,
                 font=("Segoe UI", 10)).pack(anchor="w", padx=14, pady=(8, 0))
        tk.Label(root, textvariable=self.detail_var, bg=HUD_BG, fg=HUD_MUTED,
                 font=("Segoe UI", 9), wraplength=420, justify=tk.LEFT).pack(anchor="w", padx=14, pady=(8, 0))

        controls = tk.Frame(root, bg=HUD_BG)
        controls.pack(anchor="w", padx=14, pady=(12, 10))
        tk.Button(controls, text="Pause / Resume", width=14, bg=HUD_BG2, fg=HUD_FG,
                  command=self._toggle_pause).pack(side=tk.LEFT)
        tk.Button(controls, text="Next Scene", width=12, bg=HUD_BG2, fg=HUD_FG,
                  command=lambda: self._set_scene_command("next")).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(controls, text="Restart Scene", width=12, bg=HUD_BG2, fg=HUD_FG,
                  command=lambda: self._set_scene_command("restart")).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(controls, text="Quit", width=10, bg=HUD_WARN, fg="white",
                  command=self._request_stop).pack(side=tk.LEFT, padx=(8, 0))

        root.protocol("WM_DELETE_WINDOW", self._request_stop)
        self._root = root

    def _safe_ui(self, fn) -> None:
        try:
            if self._root is not None:
                self._root.after(0, fn)
        except Exception:
            pass

    def _set_scene(self, text: str) -> None:
        self._scene_text = text
        if self.scene_var is not None:
            self._safe_ui(lambda: self.scene_var.set(text))

    def _set_status(self, text: str) -> None:
        self._status_text = text
        if self.status_var is not None:
            self._safe_ui(lambda: self.status_var.set(text))

    def _set_detail(self, text: str) -> None:
        self._detail_text = text
        if self.detail_var is not None:
            self._safe_ui(lambda: self.detail_var.set(text))

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        self._set_status("Paused" if self._paused else "Running")

    def _request_stop(self) -> None:
        self._stop = True
        self._close_all_envs()
        try:
            if self._root is not None:
                self._root.destroy()
        except Exception:
            pass

    def _set_scene_command(self, command: str) -> None:
        with self._command_lock:
            self._scene_command = command

    def _consume_scene_command(self) -> Optional[str]:
        with self._command_lock:
            cmd = self._scene_command
            self._scene_command = None
            return cmd

    def _wait_if_paused(self) -> Optional[str]:
        while self._paused and not self._stop:
            cmd = self._consume_scene_command()
            if cmd:
                return cmd
            time.sleep(0.1)
        return self._consume_scene_command()

    def _hold_scene(self, seconds: float) -> str:
        end = time.time() + seconds
        while time.time() < end and not self._stop:
            cmd = self._wait_if_paused()
            if cmd in {"restart", "next"}:
                return cmd
            time.sleep(0.08)
        return "quit" if self._stop else "done"

    def _register_env(self, env: SumoTrafficEnv) -> None:
        if env not in self._active_envs:
            self._active_envs.append(env)

    def _close_env(self, env: Optional[SumoTrafficEnv]) -> None:
        if env is None:
            return
        try:
            env.close()
        finally:
            if env in self._active_envs:
                self._active_envs.remove(env)

    def _close_all_envs(self) -> None:
        for env in list(self._active_envs):
            self._close_env(env)

    def _load_comparison(self, path: str) -> dict:
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def _load_agent(self) -> tuple[MAPPOAgent, CheckpointSpec]:
        import torch

        ckpt = torch.load(self.args.model, map_location="cpu", weights_only=True)
        obs_dim = ckpt.get("obs_dim", OBS_DIM)
        act_dim = ckpt.get("act_dim", ACT_DIM)
        algorithm = ckpt.get("algorithm", "dqn")
        model_state = ckpt.get("model", {})

        if "actor_backbone.0.weight" in model_state:
            hidden = model_state["actor_backbone.0.weight"].shape[0]
            gat_out = model_state["actor_backbone.0.weight"].shape[1] - obs_dim
        else:
            hidden = model_state.get("critic.0.weight", np.zeros((256, 1))).shape[0] if model_state else 256
            gat_out = 0

        neighbor_feat_dim = ckpt.get("neighbor_feat_dim")
        if neighbor_feat_dim is None and "gat.attn.weight" in model_state:
            neighbor_feat_dim = model_state["gat.attn.weight"].shape[1] // 2
        if neighbor_feat_dim is None:
            neighbor_feat_dim = 5

        spec = CheckpointSpec(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden=hidden,
            algorithm=algorithm,
            gat_out=gat_out,
            neighbor_feat_dim=neighbor_feat_dim,
        )
        if not spec.uses_gat:
            raise RuntimeError(
                f"GAT checkpoint required for demo story. '{self.args.model}' is {algorithm} with gat_out={gat_out}."
            )

        agent = MAPPOAgent(
            obs_dim,
            act_dim,
            hidden=hidden,
            gat_out=gat_out,
            neighbor_feat_dim=neighbor_feat_dim,
            device="cpu",
        )
        agent.load(self.args.model)
        self._agent = agent
        self._spec = spec
        return agent, spec

    def _make_env(self, tag: str) -> tuple[SumoTrafficEnv, dict[str, np.ndarray]]:
        env = SumoTrafficEnv(
            net_file=self.args.net,
            route_file=self.args.route,
            sumo_cfg=self.args.cfg,
            delta_time=self.args.delta_time,
            sim_length=self.args.sim_length,
            gui=True,
            seed=self.args.seed,
        )
        env._conn_label = _unique_label(f"story_{tag}")
        obs, _ = env.reset(seed=self.args.seed)
        env._last_neighbor_feats, env._last_neighbor_masks = env._get_neighbor_obs()
        try:
            env._conn.gui.setSchema(VIEW_ID, "real world")
            env._conn.gui.setDelay(self.args.gui_delay_ms)
        except Exception:
            pass
        self._register_env(env)
        return env, obs

    def _tls_name(self, env: SumoTrafficEnv, tls_id: str) -> str:
        try:
            node = env._net.getNode(tls_id)
            names = sorted({
                _trim_name(edge.getName() or edge.getID())
                for edge in node.getIncoming()
                if not edge.getID().startswith(":")
            })
            return " / ".join(names[:3]) or tls_id
        except Exception:
            return tls_id

    def _find_hotspot(self, env: SumoTrafficEnv) -> HotspotInfo:
        best: Optional[HotspotInfo] = None
        for tid in env.tls_ids:
            obs = env._obs_for(tid)
            queue = float(np.mean(obs[:12]))
            wait = float(np.mean(obs[12:24]))
            density = float(np.mean(obs[24:36]))
            score = 0.4 * queue + 0.4 * wait + 0.2 * density
            try:
                x, y = env._net.getNode(tid).getCoord()
            except Exception:
                x, y = 0.0, 0.0
            item = HotspotInfo(
                tls_id=tid,
                name=self._tls_name(env, tid),
                x=float(x),
                y=float(y),
                queue=queue,
                wait=wait,
                density=density,
                score=score,
            )
            if best is None or item.score > best.score:
                best = item
        if best is None:
            raise RuntimeError("Could not determine hotspot from the current SUMO state.")
        return best

    def _incoming_edges(self, env: SumoTrafficEnv, tls_id: str) -> list[dict]:
        tls = env._tls_map.get(tls_id)
        rows = []
        if tls is None:
            return rows
        for eid in tls.incoming_edges:
            if eid.startswith(":"):
                continue
            try:
                edge = env._net.getEdge(eid)
                veh = env._conn.edge.getLastStepVehicleNumber(eid)
                queue = env._conn.edge.getLastStepHaltingNumber(eid) / 50.0
                wait = env._conn.edge.getWaitingTime(eid) / 300.0
                density = veh / max(env._edge_capacity.get(eid, 1.0), 1.0)
            except Exception:
                queue = wait = density = 0.0
                edge = env._net.getEdge(eid)
            score = 0.4 * min(queue, 1.0) + 0.4 * min(wait, 1.0) + 0.2 * min(density, 1.0)
            rows.append({
                "edge_id": eid,
                "name": _trim_name(edge.getName() or eid),
                "queue": min(queue, 1.0),
                "wait": min(wait, 1.0),
                "density": min(density, 1.0),
                "score": score,
            })
        rows.sort(key=lambda item: item["score"], reverse=True)
        return rows

    def _story_tag(self, feat: np.ndarray) -> str:
        queue, wait, density, phase, green = [float(v) for v in feat[:5]]
        if density >= 0.75 and queue >= 0.45:
            return "spillback risk"
        if wait >= 0.55 or queue >= 0.55:
            return "queue building"
        if phase >= 0.45 and green <= 0.45:
            return "still draining"
        if density <= 0.35 and queue <= 0.30:
            return "safe to release"
        return "watching flow"

    def _story_reason(self, name: str, feat: np.ndarray) -> str:
        queue, wait, density, phase, green = [float(v) for v in feat[:5]]
        if queue >= 0.55:
            return f"Upstream queue on {name} is rising."
        if wait >= 0.55:
            return f"Delay is building up on {name}."
        if density >= 0.70:
            return f"The road toward {name} is already crowded."
        if phase >= 0.45 and green <= 0.45:
            return f"The neighbor signal on {name} is still draining traffic."
        if green >= 0.65:
            return f"The green window on {name} has already been used heavily."
        return f"Traffic around {name} still looks manageable."

    def _decode_duration_seconds(self, env: SumoTrafficEnv, tls_id: str, action) -> float:
        green_phases = env._green_phases.get(tls_id, [0])
        ci = env._cycle_index.get(tls_id, 0)
        phase_ci = ci if env._countdown.get(tls_id, 0) > 0 else (ci + 1) % len(green_phases)
        return env.decode_duration_steps(tls_id, action, phase_ci=phase_ci) * 0.5

    def _ai_actions(self, env: SumoTrafficEnv, obs: dict[str, np.ndarray],
                    explain_tid: Optional[str] = None) -> tuple[dict[str, np.ndarray], Optional[dict]]:
        assert self._agent is not None and self._spec is not None
        raw_global = np.mean([obs[tid] for tid in env.tls_ids], axis=0).astype(np.float32)
        global_obs = (
            remap_obs_for_old_model(raw_global, target_dim=self._spec.obs_dim)
            if self._spec.needs_remap else raw_global
        )
        neighbor_feats_dict, neighbor_masks_dict = env.get_neighbor_obs()
        actions: dict[str, np.ndarray] = {}
        explanation = None
        for tid in env.tls_ids:
            local_obs = remap_obs_for_old_model(obs[tid], target_dim=self._spec.obs_dim) if self._spec.needs_remap else obs[tid]
            nf = neighbor_feats_dict.get(tid)
            nm = neighbor_masks_dict.get(tid)
            if tid == explain_tid:
                explanation = self._agent.explain_action(
                    local_obs, global_obs, env.get_valid_actions(tid), greedy=True,
                    neighbor_feats=nf, neighbor_mask=nm,
                )
                actions[tid] = explanation["action"]
            else:
                action, _, _ = self._agent.select_action(
                    local_obs, global_obs, env.get_valid_actions(tid), greedy=True,
                    neighbor_feats=nf, neighbor_mask=nm,
                )
                actions[tid] = action
        return actions, explanation

    def _advance_baseline(self, env: SumoTrafficEnv, obs: dict[str, np.ndarray], steps: int,
                          renderer: Optional[SumoOverlayRenderer] = None) -> dict[str, np.ndarray]:
        terminated = truncated = False
        for idx in range(steps):
            if terminated or truncated or self._stop:
                break
            cmd = self._wait_if_paused()
            if cmd == "restart":
                raise RuntimeError("restart")
            if cmd == "next":
                break
            if renderer is not None:
                renderer.update_heatmap()
            actions = {tid: _baseline_action(env) for tid in env.tls_ids}
            obs, _, terminated, truncated, _ = env.step(actions)
            metrics = env.get_metrics()
            self._set_status(
                f"Baseline replay · step {idx + 1}/{steps} · wait={metrics.get('avg_wait_time', 0):.1f}s · veh={metrics.get('total_vehicles', 0)}"
            )
        return obs

    def _advance_ai(self, env: SumoTrafficEnv, obs: dict[str, np.ndarray], steps: int) -> dict[str, np.ndarray]:
        terminated = truncated = False
        for idx in range(steps):
            if terminated or truncated or self._stop:
                break
            cmd = self._wait_if_paused()
            if cmd == "restart":
                raise RuntimeError("restart")
            if cmd == "next":
                break
            actions, _ = self._ai_actions(env, obs)
            obs, _, terminated, truncated, _ = env.step(actions)
            metrics = env.get_metrics()
            self._set_status(
                f"AI replay align · step {idx + 1}/{steps} · wait={metrics.get('avg_wait_time', 0):.1f}s · veh={metrics.get('total_vehicles', 0)}"
            )
        return obs

    def _advance_until_hotspot_decision(self, env: SumoTrafficEnv, obs: dict[str, np.ndarray], hotspot_tid: str) -> dict[str, np.ndarray]:
        attempts = 0
        while env._countdown.get(hotspot_tid, 0) > 0 and attempts < 6 and not self._stop:
            cmd = self._wait_if_paused()
            if cmd == "restart":
                raise RuntimeError("restart")
            if cmd == "next":
                break
            actions, _ = self._ai_actions(env, obs)
            obs, _, terminated, truncated, _ = env.step(actions)
            attempts += 1
            if terminated or truncated:
                break
        return obs

    def _build_neighbor_story(self, env: SumoTrafficEnv, hotspot: HotspotInfo,
                              explanation: Optional[dict]) -> tuple[list[dict], list[str], str]:
        if explanation is None:
            return [], [f"No attention explanation available for {hotspot.name}."], "Why this light chose a stable cycle"
        neighbor_feats, neighbor_masks = env.get_neighbor_obs()
        feats = neighbor_feats.get(hotspot.tls_id)
        mask = neighbor_masks.get(hotspot.tls_id)
        weights = explanation.get("attention_weights")
        if feats is None or mask is None or weights is None:
            return [], [f"No valid neighbor attention available for {hotspot.name}."], "Why this light chose a stable cycle"

        rows = []
        for idx, ((neighbor_tid, edge_id), enabled) in enumerate(zip(env._neighbor_info.get(hotspot.tls_id, []), mask)):
            if not enabled or idx >= len(weights):
                continue
            feat = feats[idx]
            try:
                nx, ny = env._net.getNode(neighbor_tid).getCoord()
            except Exception:
                nx, ny = hotspot.x, hotspot.y
            edge_name = _trim_name(env._net.getEdge(edge_id).getName() or edge_id)
            rows.append({
                "tls_id": neighbor_tid,
                "name": self._tls_name(env, neighbor_tid),
                "road": edge_name,
                "x": float(nx),
                "y": float(ny),
                "weight": float(weights[idx]),
                "queue": float(feat[0]),
                "wait": float(feat[1]),
                "density": float(feat[2]),
                "tag": self._story_tag(feat),
                "reason": self._story_reason(edge_name, feat),
            })
        rows = [row for row in rows if row["weight"] > 0.0]
        rows.sort(key=lambda item: item["weight"], reverse=True)
        top = rows[:3]

        duration_sec = self._decode_duration_seconds(env, hotspot.tls_id, explanation["action"])
        midpoint = (env._min_green_steps.get(hotspot.tls_id, 30) + env._max_green_steps.get(hotspot.tls_id, 90)) * 0.25
        if duration_sec >= midpoint + 4:
            title = f"Why this light stayed green {duration_sec:.0f}s"
        elif duration_sec <= midpoint - 4:
            title = f"Why this light switched sooner ({duration_sec:.0f}s)"
        else:
            title = f"Why this light chose {duration_sec:.0f}s"

        lines = []
        if hotspot.score >= 0.45:
            lines.append(f"{hotspot.name} is carrying visible congestion right now.")
        elif hotspot.queue >= 0.35:
            lines.append(f"{hotspot.name} still has vehicles waiting on key approaches.")
        else:
            lines.append(f"{hotspot.name} is relatively stable, so timing can stay controlled.")
        for row in top[:2]:
            lines.append(row["reason"])
        if len(lines) < 3 and top:
            lines.append(f"{top[0]['road']} still has room, so releasing more vehicles is safe.")
        return top, lines[:3], title

    def _auto_tile_sumo_windows(self) -> None:
        if gw is None or os.name != "nt":
            print("SUMO auto-tiling unavailable. Position the windows manually if needed.")
            return
        time.sleep(1.5)
        wins = [
            win for win in gw.getAllWindows()
            if win.title and "SUMO" in win.title and "Story Director" not in win.title
        ]
        if len(wins) < 2:
            print("Could not find two SUMO windows to tile automatically.")
            return
        wins = wins[-2:]
        screen_w = self._root.winfo_screenwidth() if self._root is not None else 1920
        screen_h = self._root.winfo_screenheight() if self._root is not None else 1080
        width = screen_w // 2
        height = max(screen_h - 80, 700)
        for win, (x, y) in zip(wins, [(0, 0), (width, 0)]):
            try:
                if win.isMinimized:
                    win.restore()
                win.moveTo(x, y)
                win.resizeTo(width, height)
            except Exception:
                pass

    def _scene_baseline_pain(self) -> tuple[str, Optional[dict]]:
        self._set_scene("Scene 1 · Baseline pain")
        self._set_detail("Macro view. Let the city breathe for a moment, then show where default timing starts to back traffic up.")
        env, obs = self._make_env("baseline_pain")
        renderer = SumoOverlayRenderer(env, "scene1", self._labels)
        renderer.setup_heatmap()
        renderer.set_macro_view()
        renderer.draw_banner("Scene 1 · Baseline pain", "Default timing lets pressure spread across the district.", CLR_BASELINE)
        steps = _scene_control_steps(self.args.scene1_seconds, self.args.delta_time, self.args.gui_delay_ms)
        try:
            obs = self._advance_baseline(env, obs, steps, renderer=renderer)
        except RuntimeError as exc:
            self._close_env(env)
            if str(exc) == "restart":
                return "restart", None
            raise
        hotspot = self._find_hotspot(env)
        return "done", {"env": env, "obs": obs, "renderer": renderer, "hotspot": hotspot, "control_steps": steps}

    def _scene_freeze_explain(self, state: dict) -> str:
        self._set_scene("Scene 2 · Freeze and explain")
        self._set_detail("Pause on one hotspot and label the incoming roads that are creating the pain.")
        env: SumoTrafficEnv = state["env"]
        renderer: SumoOverlayRenderer = state["renderer"]
        hotspot: HotspotInfo = state["hotspot"]

        renderer.clear_scene()
        renderer.clear_heatmap()
        renderer.set_focus_view(hotspot.x, hotspot.y, radius=360)
        renderer.draw_banner("Scene 2 · Freeze and explain", "This is the hotspot the AI will react to next.", CLR_BASELINE)
        renderer.draw_focus(hotspot, radius=85)
        roads = self._incoming_edges(env, hotspot.tls_id)[:3]
        for row in roads:
            renderer.draw_incoming_edge(
                row["edge_id"],
                row["score"],
                f"Queue {row['queue']:.2f} | Wait {row['wait']:.2f} | Density {row['density']:.2f}",
            )
        renderer.draw_card(
            "Why this place gets stuck",
            [f"Hotspot score {hotspot.score:.2f} at {hotspot.name}."] +
            [f"{row['name']} is backing up." for row in roads[:2]],
            CLR_BASELINE,
        )
        return self._hold_scene(self.args.scene2_seconds)

    def _scene_ai_wakes_up(self, hotspot: HotspotInfo, control_steps: int) -> str:
        self._set_scene("Scene 3 · AI wakes up")
        self._set_detail("Fresh AI replay from the same seed. Now the model can check its neighbors before choosing the next duration.")
        env, obs = self._make_env("ai_micro")
        renderer = SumoOverlayRenderer(env, "scene3", self._labels)
        try:
            obs = self._advance_ai(env, obs, control_steps)
            obs = self._advance_until_hotspot_decision(env, obs, hotspot.tls_id)
        except RuntimeError as exc:
            self._close_env(env)
            if str(exc) == "restart":
                return "restart"
            raise

        steps = _scene_control_steps(self.args.scene3_seconds, self.args.delta_time, self.args.gui_delay_ms)
        terminated = truncated = False
        for idx in range(steps):
            if terminated or truncated or self._stop:
                break
            cmd = self._wait_if_paused()
            if cmd in {"restart", "next"}:
                self._close_env(env)
                return cmd
            radius = 260 if idx < max(steps // 2, 1) else 180
            renderer.clear_scene()
            renderer.set_focus_view(hotspot.x, hotspot.y, radius=radius)
            renderer.draw_banner("Scene 3 · AI wakes up", "Level 2 checks nearby roads before choosing the next green duration.", CLR_AI)
            actions, explanation = self._ai_actions(env, obs, explain_tid=hotspot.tls_id)
            neighbors, reason_lines, title = self._build_neighbor_story(env, hotspot, explanation)
            renderer.draw_focus(hotspot, radius=68)
            renderer.draw_neighbors(hotspot, neighbors)
            renderer.draw_card(title, reason_lines, CLR_AI)
            obs, _, terminated, truncated, _ = env.step(actions)
            metrics = env.get_metrics()
            self._set_status(
                f"AI micro scene · step {idx + 1}/{steps} · wait={metrics.get('avg_wait_time', 0):.1f}s · queue={metrics.get('avg_queue_length', 0):.2f}"
            )
        self._close_env(env)
        return "quit" if self._stop else "done"

    def _scene_split_screen(self, hotspot: HotspotInfo, control_steps: int) -> tuple[str, Optional[dict]]:
        self._set_scene("Scene 4 · Split screen")
        self._set_detail("Baseline and AI replay the same moment side-by-side so the difference is visible immediately.")
        baseline_env, baseline_obs = self._make_env("split_baseline")
        ai_env, ai_obs = self._make_env("split_ai")
        try:
            baseline_env._conn.gui.setDelay(1)
            ai_env._conn.gui.setDelay(1)
        except Exception:
            pass
        baseline_renderer = SumoOverlayRenderer(baseline_env, "scene4_base", self._labels)
        ai_renderer = SumoOverlayRenderer(ai_env, "scene4_ai", self._labels)
        try:
            baseline_obs = self._advance_baseline(baseline_env, baseline_obs, control_steps)
            ai_obs = self._advance_ai(ai_env, ai_obs, control_steps)
            ai_obs = self._advance_until_hotspot_decision(ai_env, ai_obs, hotspot.tls_id)
        except RuntimeError as exc:
            self._close_env(baseline_env)
            self._close_env(ai_env)
            if str(exc) == "restart":
                return "restart", None
            raise
        try:
            baseline_env._conn.gui.setDelay(self.args.gui_delay_ms)
            ai_env._conn.gui.setDelay(self.args.gui_delay_ms)
        except Exception:
            pass

        baseline_renderer.set_focus_view(hotspot.x, hotspot.y, radius=220)
        ai_renderer.set_focus_view(hotspot.x, hotspot.y, radius=220)
        self._auto_tile_sumo_windows()

        steps = _scene_control_steps(self.args.scene4_seconds, self.args.delta_time, self.args.gui_delay_ms)
        terminated_base = truncated_base = False
        terminated_ai = truncated_ai = False
        for idx in range(steps):
            if self._stop or terminated_base or truncated_base or terminated_ai or truncated_ai:
                break
            cmd = self._wait_if_paused()
            if cmd in {"restart", "next"}:
                return cmd, {
                    "baseline_env": baseline_env,
                    "ai_env": ai_env,
                    "baseline_renderer": baseline_renderer,
                    "ai_renderer": ai_renderer,
                }
            baseline_renderer.clear_scene()
            ai_renderer.clear_scene()
            baseline_renderer.set_focus_view(hotspot.x, hotspot.y, radius=220)
            ai_renderer.set_focus_view(hotspot.x, hotspot.y, radius=220)
            baseline_renderer.draw_banner("Baseline", "Default fixed timing", CLR_BASELINE)
            baseline_renderer.draw_focus(hotspot, radius=65)
            ai_renderer.draw_banner("Level 2 AI", "Neighbor-aware GAT timing", CLR_AI)
            ai_renderer.draw_focus(hotspot, radius=65)
            ai_actions, explanation = self._ai_actions(ai_env, ai_obs, explain_tid=hotspot.tls_id)
            neighbors, reason_lines, title = self._build_neighbor_story(ai_env, hotspot, explanation)
            ai_renderer.draw_neighbors(hotspot, neighbors)
            ai_renderer.draw_card(title, reason_lines, CLR_AI)

            base_actions = {tid: _baseline_action(baseline_env) for tid in baseline_env.tls_ids}
            base_box: dict = {}
            ai_box: dict = {}

            def _step_env(env, actions, box):
                try:
                    box["result"] = env.step(actions)
                except Exception as exc:
                    box["error"] = exc

            t1 = threading.Thread(target=_step_env, args=(baseline_env, base_actions, base_box))
            t2 = threading.Thread(target=_step_env, args=(ai_env, ai_actions, ai_box))
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            if "error" in base_box:
                raise base_box["error"]
            if "error" in ai_box:
                raise ai_box["error"]

            baseline_obs, _, terminated_base, truncated_base, _ = base_box["result"]
            ai_obs, _, terminated_ai, truncated_ai, _ = ai_box["result"]
            base_metrics = baseline_env.get_metrics()
            ai_metrics = ai_env.get_metrics()
            self._set_status(
                f"Split screen · baseline wait={base_metrics.get('avg_wait_time', 0):.1f}s | AI wait={ai_metrics.get('avg_wait_time', 0):.1f}s"
            )
        return ("quit" if self._stop else "done"), {
            "baseline_env": baseline_env,
            "ai_env": ai_env,
            "baseline_renderer": baseline_renderer,
            "ai_renderer": ai_renderer,
        }

    def _scene_payoff(self, split_state: dict) -> str:
        self._set_scene("Scene 5 · Payoff")
        self._set_detail("Finish with the branch numbers on the same SUMO windows so the story ends with a clear outcome.")
        baseline_renderer: SumoOverlayRenderer = split_state["baseline_renderer"]
        ai_renderer: SumoOverlayRenderer = split_state["ai_renderer"]
        baseline_renderer.clear_scene()
        ai_renderer.clear_scene()
        baseline_renderer.draw_banner("Baseline payoff", "What happens without neighbor-aware timing", CLR_BASELINE)
        ai_renderer.draw_banner("Level 2 payoff", "What coordinated timing changes", CLR_AI)
        baseline = self._comparison.get("baseline", {})
        model = self._comparison.get("model", {})
        payoff_lines = [
            f"Wait {baseline.get('avg_wait', 0):.1f}s -> {model.get('avg_wait', 0):.1f}s",
            f"Queue {baseline.get('avg_queue', 0):.2f} -> {model.get('avg_queue', 0):.2f}",
            f"Throughput {baseline.get('throughput', 0):.0f} -> {model.get('throughput', 0):.0f}",
        ]
        baseline_renderer.draw_card("Payoff", payoff_lines, CLR_BASELINE)
        ai_renderer.draw_card("Payoff", payoff_lines, CLR_AI)
        return self._hold_scene(self.args.scene5_seconds)

    def _run_story(self) -> None:
        baseline_state: Optional[dict] = None
        split_state: Optional[dict] = None
        try:
            self._set_status("Validating checkpoint...")
            self._load_agent()
            self._set_status("Checkpoint ready. Opening SUMO scenes...")

            while not self._stop:
                result, baseline_state = self._scene_baseline_pain()
                if result == "restart":
                    continue
                if result == "quit":
                    return
                break
            assert baseline_state is not None
            hotspot: HotspotInfo = baseline_state["hotspot"]
            control_steps = baseline_state["control_steps"]

            while not self._stop:
                result = self._scene_freeze_explain(baseline_state)
                if result == "restart":
                    continue
                break
            self._close_env(baseline_state["env"])
            if self._stop or result == "quit":
                return

            while not self._stop:
                result = self._scene_ai_wakes_up(hotspot, control_steps)
                if result == "restart":
                    continue
                break
            if self._stop or result == "quit":
                return

            while not self._stop:
                result, split_state = self._scene_split_screen(hotspot, control_steps)
                if result == "restart":
                    if split_state:
                        self._close_env(split_state.get("baseline_env"))
                        self._close_env(split_state.get("ai_env"))
                    continue
                break
            if split_state is None or self._stop or result == "quit":
                return

            while not self._stop:
                result = self._scene_payoff(split_state)
                if result == "restart":
                    continue
                break

            self._set_scene("Story complete")
            self._set_status("Demo finished.")
            self._set_detail("Restart the tool if you want to play the full story again.")
            self._hold_scene(1.5)
        except Exception as exc:
            self._set_status(f"Error: {exc}")
            print(traceback.format_exc())
        finally:
            if split_state:
                self._close_env(split_state.get("baseline_env"))
                self._close_env(split_state.get("ai_env"))
            self._close_all_envs()


def main() -> None:
    ap = argparse.ArgumentParser(description="FlowMind AI - SUMO-first cinematic story demo")
    ap.add_argument("--model", default=os.path.join(_PROJECT_ROOT, "checkpoints", "best_model.pt"))
    ap.add_argument("--comparison", default=os.path.join(_PROJECT_ROOT, "checkpoints", "comparison.json"))
    ap.add_argument("--net", default=os.path.join(_PROJECT_ROOT, "sumo", "danang", "danang.net.xml"))
    ap.add_argument("--route", default=os.path.join(_PROJECT_ROOT, "sumo", "danang", "danang.rou.xml"))
    ap.add_argument("--cfg", default=os.path.join(_PROJECT_ROOT, "sumo", "danang", "danang.sumocfg"))
    ap.add_argument("--delta-time", type=int, default=30)
    ap.add_argument("--sim-length", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--speed", type=float, default=0.05, help="Reserved for future presenter pacing tweaks.")
    ap.add_argument("--gui-delay-ms", type=int, default=50)
    ap.add_argument("--scene1-seconds", type=float, default=12.0)
    ap.add_argument("--scene2-seconds", type=float, default=6.0)
    ap.add_argument("--scene3-seconds", type=float, default=8.0)
    ap.add_argument("--scene4-seconds", type=float, default=12.0)
    ap.add_argument("--scene5-seconds", type=float, default=6.0)
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        print(f"Model not found: {args.model}")
        sys.exit(1)

    StoryDirector(args).run()


if __name__ == "__main__":
    main()
