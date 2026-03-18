"""
FlowMind AI - FastAPI application entry point.

Serves the main map interface, dashboard, and all REST API endpoints
for the adaptive traffic signal simulation.
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from dotenv import load_dotenv
import os

from .api.routes import router

load_dotenv()

app = FastAPI(
    title="FlowMind AI",
    description="AI adaptive traffic signal system for Vietnamese smart cities",
)

# Static files and templates
BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "web" / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "web" / "templates")

app.include_router(router, prefix="/api")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "vietmap_api_key": os.getenv("VIETMAP_API_KEY", ""),
            "vietmap_services_key": os.getenv("VIETMAP_SERVICES_KEY", ""),
        },
    )


@app.get("/dashboard")
async def dashboard(request: Request):
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "vietmap_api_key": os.getenv("VIETMAP_API_KEY", ""),
        },
    )
