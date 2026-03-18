"""
FlowMind AI - All API endpoints for the traffic simulation.

Provides REST endpoints for simulation control, AI optimization,
congestion prediction, and real-time state queries.
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional, Union

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..simulation.engine import SimulationEngine
from ..simulation.scenarios import get_scenario, list_scenarios
from ..ai.optimizer import SignalOptimizer
from ..ai.predictor import CongestionPredictor

router = APIRouter()

# ---------------------------------------------------------------------------
# Module-level globals — initialised once at startup
# ---------------------------------------------------------------------------

engine: Optional[Union[SimulationEngine, "SumoEngine"]] = None
optimizer: Optional[SignalOptimizer] = None
predictor: Optional[CongestionPredictor] = None

auto_run: bool = False
auto_run_task: Optional[asyncio.Task] = None
metrics_history: list[dict] = []

MAX_HISTORY = 500


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@router.on_event("startup")
async def startup() -> None:
    global engine, optimizer, predictor

    engine_type = os.getenv("FLOWMIND_ENGINE", "builtin").lower()
    if engine_type == "sumo":
        from ..simulation.sumo_engine import SumoEngine
        sumo_cfg = os.getenv("SUMO_CFG", None)
        engine = SumoEngine(sumo_cfg=sumo_cfg, gui=False)
        print("[FlowMind] Using SUMO engine")
    else:
        engine = SimulationEngine()
        print("[FlowMind] Using built-in engine")

    optimizer = SignalOptimizer()
    predictor = CongestionPredictor()


@router.on_event("shutdown")
async def shutdown() -> None:
    global auto_run, auto_run_task
    auto_run = False
    if auto_run_task is not None and not auto_run_task.done():
        auto_run_task.cancel()
        try:
            await auto_run_task
        except asyncio.CancelledError:
            pass
    # Clean up SUMO connection if applicable
    if engine is not None and hasattr(engine, "close"):
        engine.close()


# ---------------------------------------------------------------------------
# Background auto-step loop
# ---------------------------------------------------------------------------

async def _auto_step_loop() -> None:
    global auto_run
    while auto_run:
        if engine is not None:
            engine.step()
            _record_metrics()
        await asyncio.sleep(0.5)


def _record_metrics() -> None:
    if engine is None:
        return
    state = engine.get_state()
    m = state["metrics"]
    metrics_history.append({
        "tick": state["tick"],
        "time_of_day": state["time_of_day"],
        "avg_wait_time": m["avg_wait_time"],
        "avg_queue_length": m["avg_queue_length"],
        "throughput": m["throughput"],
        "congestion_score": m["congestion_score"],
        "emission_estimate": m["emission_estimate"],
        "avg_speed": m["avg_speed"],
        "fuel_consumption_l": m["fuel_consumption_l"],
    })
    if predictor is not None:
        predictor.record(state["tick"], m)
    if len(metrics_history) > MAX_HISTORY:
        metrics_history.pop(0)


# ---------------------------------------------------------------------------
# Request / response helpers
# ---------------------------------------------------------------------------

class ApplyTimingRequest(BaseModel):
    intersection_id: str
    green_ns: float
    green_ew: float


class EmergencyRequest(BaseModel):
    from_intersection: str
    to_intersection: str


class LaneRequest(BaseModel):
    intersection_id: str
    lane_id: str


class DensityRequest(BaseModel):
    base_density: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/state")
async def get_state():
    if engine is None:
        raise HTTPException(503, "Simulation not initialised")
    return engine.get_state()


@router.get("/metrics")
async def get_metrics():
    if engine is None:
        raise HTTPException(503, "Simulation not initialised")
    return engine.get_metrics()


@router.get("/scenarios")
async def get_scenarios():
    results = []
    for s in list_scenarios():
        results.append({
            "id": s.id,
            "name": s.name,
            "description": s.description,
            "time_of_day": s.time_of_day,
            "weather": s.weather,
            "vehicle_mix": s.vehicle_mix,
        })
    return results


@router.post("/scenario/{scenario_id}")
async def apply_scenario(scenario_id: str):
    if engine is None:
        raise HTTPException(503, "Simulation not initialised")
    try:
        scenario = engine.apply_scenario(scenario_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    _record_metrics()
    return {
        "status": "ok",
        "scenario": scenario_id,
        "description": scenario.description,
    }


@router.post("/step")
async def step(steps: int = Query(1, ge=1, le=100)):
    if engine is None:
        raise HTTPException(503, "Simulation not initialised")
    for _ in range(steps):
        engine.step()
        _record_metrics()
    state = engine.get_state()
    return {
        "status": "ok",
        "steps": steps,
        "tick": state["tick"],
        "metrics": state["metrics"],
    }


@router.post("/optimize")
async def optimize():
    if engine is None or optimizer is None:
        raise HTTPException(503, "Simulation not initialised")

    state = engine.get_state()
    before = dict(state["metrics"])

    for isct in state["intersections"]:
        isct["hour"] = state["time_of_day"]

    recommendations = optimizer.optimize_all(state)

    # Store state keys before applying for Q-learning update
    state_keys = []
    for i, rec in enumerate(recommendations):
        timing = rec.get("recommended_timing", {})
        iid = rec.get("intersection_id", "")
        isct = state["intersections"][i] if i < len(state["intersections"]) else {}
        isct["hour"] = state["time_of_day"]
        state_keys.append({
            "state_key": rec.get("state_key"),
            "action": timing,
        })
        engine.set_signal_timing(iid, {
            "N": timing.get("green_ns", 30),
            "S": timing.get("green_ns", 30),
            "E": timing.get("green_ew", 25),
            "W": timing.get("green_ew", 25),
        })

    for _ in range(20):
        engine.step()
        _record_metrics()

    after_state = engine.get_state()
    after = after_state["metrics"]

    # Close Q-learning loop: update Q-values with actual results
    reward = optimizer.compute_reward(
        after.get("avg_wait_time", 0),
        after.get("congestion_score", 0),
        after.get("emission_estimate", 0) / 1000.0,
    )
    for i, sk in enumerate(state_keys):
        if sk["state_key"] is None:
            continue
        isct = after_state["intersections"][i] if i < len(after_state["intersections"]) else {}
        isct["hour"] = after_state["time_of_day"]
        next_key = optimizer.get_state_key(isct, {"weather": after_state.get("weather", "clear")})
        optimizer.update(sk["state_key"], sk["action"], reward, next_key)

    # Decay exploration over time
    optimizer.epsilon = max(0.05, optimizer.epsilon * 0.995)
    optimizer.episode_count += 1

    return {
        "status": "ok",
        "recommendations": recommendations,
        "before": before,
        "after": after,
    }


@router.post("/apply-timing")
async def apply_timing(body: ApplyTimingRequest):
    if engine is None:
        raise HTTPException(503, "Simulation not initialised")
    engine.set_signal_timing(body.intersection_id, {
        "N": body.green_ns,
        "S": body.green_ns,
        "E": body.green_ew,
        "W": body.green_ew,
    })
    _record_metrics()
    return {
        "status": "ok",
        "intersection_id": body.intersection_id,
        "green_ns": body.green_ns,
        "green_ew": body.green_ew,
    }


@router.get("/predict")
async def predict():
    if engine is None or predictor is None:
        raise HTTPException(503, "Simulation not initialised")
    predictions = {}
    for horizon in [5, 10, 15]:
        predictions[f"{horizon}min"] = predictor.predict_congestion(horizon)
    return {
        "status": "ok",
        "predictions": predictions,
        "trend": predictor.get_trend(),
    }


@router.post("/emergency")
async def emergency(body: EmergencyRequest):
    if engine is None:
        raise HTTPException(503, "Simulation not initialised")
    id_to_idx = {ix.id: i for i, ix in enumerate(engine.intersections)}
    from_idx = id_to_idx.get(body.from_intersection, 0)
    to_idx = id_to_idx.get(body.to_intersection, len(engine.intersections) - 1)
    engine.inject_emergency_vehicle(from_idx, to_idx)
    _record_metrics()
    return {
        "status": "ok",
        "from": body.from_intersection,
        "to": body.to_intersection,
    }


@router.post("/block-lane")
async def block_lane(body: LaneRequest):
    if engine is None:
        raise HTTPException(503, "Simulation not initialised")
    engine.block_lane(body.intersection_id, body.lane_id)
    _record_metrics()
    return {
        "status": "ok",
        "intersection_id": body.intersection_id,
        "lane_id": body.lane_id,
        "blocked": True,
    }


@router.post("/unblock-lane")
async def unblock_lane(body: LaneRequest):
    if engine is None:
        raise HTTPException(503, "Simulation not initialised")
    for ix in engine.intersections:
        if ix.id == body.intersection_id:
            for lane in ix.lanes:
                if lane.id == body.lane_id:
                    lane.blocked = False
                    break
            break
    _record_metrics()
    return {
        "status": "ok",
        "intersection_id": body.intersection_id,
        "lane_id": body.lane_id,
        "blocked": False,
    }


@router.post("/density")
async def set_density(body: DensityRequest):
    if engine is None:
        raise HTTPException(503, "Simulation not initialised")
    engine.set_base_density(body.base_density)
    return {"status": "ok", "base_density": engine.get_base_density()}


@router.post("/reset")
async def reset():
    global auto_run, auto_run_task
    if engine is None:
        raise HTTPException(503, "Simulation not initialised")
    auto_run = False
    if auto_run_task is not None and not auto_run_task.done():
        auto_run_task.cancel()
        try:
            await auto_run_task
        except asyncio.CancelledError:
            pass
        auto_run_task = None
    engine.reset()
    metrics_history.clear()
    return {"status": "ok", "tick": 0}


@router.get("/history")
async def get_history():
    return {"history": metrics_history}


@router.post("/auto-run")
async def toggle_auto_run():
    global auto_run, auto_run_task
    if engine is None:
        raise HTTPException(503, "Simulation not initialised")
    auto_run = not auto_run
    if auto_run:
        auto_run_task = asyncio.create_task(_auto_step_loop())
    else:
        if auto_run_task is not None and not auto_run_task.done():
            auto_run_task.cancel()
            try:
                await auto_run_task
            except asyncio.CancelledError:
                pass
            auto_run_task = None
    return {"status": "ok", "auto_running": auto_run}


@router.get("/vehicles")
async def get_vehicles():
    """Return all vehicle positions (SUMO engine only)."""
    if engine is None:
        raise HTTPException(503, "Simulation not initialised")
    if hasattr(engine, "get_vehicle_positions"):
        vehicles = await asyncio.to_thread(engine.get_vehicle_positions)
        return {"vehicles": vehicles, "count": len(vehicles)}
    return {"vehicles": [], "count": 0}


@router.get("/engine-info")
async def engine_info():
    """Return which engine backend is active."""
    if engine is None:
        raise HTTPException(503, "Simulation not initialised")
    from ..simulation.sumo_engine import SumoEngine
    is_sumo = isinstance(engine, SumoEngine)
    return {"engine": "sumo" if is_sumo else "builtin", "tick": engine.tick}
