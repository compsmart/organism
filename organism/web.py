from __future__ import annotations

import argparse
import threading
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .env import Action
from .evolution import EvolutionSimulation
from .session import SimulationSession, discover_checkpoints, load_config


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = WORKSPACE_ROOT / "outputs"
WEBUI_ROOT = WORKSPACE_ROOT / "webui"


class ResetRequest(BaseModel):
    seed: int | None = None


class ManualStepRequest(BaseModel):
    action: str


class OptionsRequest(BaseModel):
    deterministic: bool


class LoadCheckpointRequest(BaseModel):
    checkpoint: str | None = None


class WebSessionManager:
    def __init__(self, outputs_root: Path) -> None:
        self.outputs_root = outputs_root
        self.lock = threading.Lock()
        self.session = self._build_default_session()

    def _build_default_session(self) -> SimulationSession:
        checkpoints = discover_checkpoints(self.outputs_root)
        checkpoint = Path(checkpoints[0]["path"]) if checkpoints else None
        config = load_config(None, checkpoint) if checkpoint else None
        return SimulationSession(
            config=config,
            checkpoint=checkpoint,
            seed=11,
            deterministic=False,
        )

    def list_checkpoints(self) -> list[dict[str, str]]:
        return discover_checkpoints(self.outputs_root)

    def get_state(self) -> dict[str, Any]:
        with self.lock:
            return {
                "checkpoints": self.list_checkpoints(),
                "session": self.session.state_dict(),
            }

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        with self.lock:
            self.session.reset(seed=seed)
            return self.session.state_dict()

    def step_policy(self) -> dict[str, Any]:
        with self.lock:
            try:
                self.session.step_policy()
            except RuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return self.session.state_dict()

    def step_manual(self, action_name: str) -> dict[str, Any]:
        try:
            action = Action[action_name.upper()]
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"Unknown action '{action_name}'.") from exc

        with self.lock:
            self.session.step_manual(action)
            return self.session.state_dict()

    def set_options(self, deterministic: bool) -> dict[str, Any]:
        with self.lock:
            self.session.set_deterministic(deterministic)
            return self.session.state_dict()

    def load_checkpoint(self, checkpoint: str | None) -> dict[str, Any]:
        resolved: Path | None
        if checkpoint is None or checkpoint == "":
            resolved = None
        else:
            resolved = Path(checkpoint).resolve()
            if not resolved.exists():
                raise HTTPException(status_code=404, detail="Checkpoint not found.")

        with self.lock:
            config = load_config(None, resolved) if resolved is not None else self.session.config
            self.session.load_checkpoint(resolved, config=config, reset=True)
            return self.session.state_dict()


class EvoResetRequest(BaseModel):
    seed: int | None = None
    population_size: int | None = None


class EvoStepRequest(BaseModel):
    steps: int = 5


class EvoSaveRequest(BaseModel):
    uid: int
    note: str = ""


class EvoSpawnRequest(BaseModel):
    key: str


class EvolutionSessionManager:
    def __init__(self, outputs_root: Path) -> None:
        self.outputs_root = outputs_root
        self.lock = threading.Lock()
        self.simulation = self._build_default()

    def _build_default(self) -> EvolutionSimulation:
        return EvolutionSimulation(seed=42)

    def get_state(self) -> dict[str, Any]:
        with self.lock:
            return self.simulation.state_dict()

    def step(self, steps: int) -> dict[str, Any]:
        with self.lock:
            self.simulation.step(steps)
            return self.simulation.state_dict()

    def reset(self, seed: int | None = None, population_size: int | None = None) -> dict[str, Any]:
        with self.lock:
            self.simulation.reset(seed=seed, population_size=population_size)
            return self.simulation.state_dict()


def create_app() -> FastAPI:
    app = FastAPI(title="Organism Web Viewer")
    manager = WebSessionManager(OUTPUTS_ROOT)
    evo_manager = EvolutionSessionManager(OUTPUTS_ROOT)

    app.mount("/static", StaticFiles(directory=WEBUI_ROOT), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEBUI_ROOT / "index.html")

    @app.get("/evolution", include_in_schema=False)
    def evolution_page() -> FileResponse:
        return FileResponse(WEBUI_ROOT / "evolution.html")

    @app.get("/api/evolution/state")
    def evo_state() -> dict[str, Any]:
        return evo_manager.get_state()

    @app.post("/api/evolution/reset")
    def evo_reset(payload: EvoResetRequest) -> dict[str, Any]:
        return evo_manager.reset(seed=payload.seed, population_size=payload.population_size)

    @app.post("/api/evolution/step")
    def evo_step(payload: EvoStepRequest) -> dict[str, Any]:
        return evo_manager.step(max(1, min(payload.steps, 100)))

    @app.get("/api/evolution/saved")
    def evo_saved() -> list[dict[str, Any]]:
        from .evolution import EvolutionSimulation
        return EvolutionSimulation.list_saved()

    @app.post("/api/evolution/save")
    def evo_save(payload: EvoSaveRequest) -> dict[str, Any]:
        with evo_manager.lock:
            org = next((o for o in evo_manager.simulation.organisms if o.uid == payload.uid), None)
            if org is None:
                raise HTTPException(status_code=404, detail="Organism not found.")
            key = evo_manager.simulation.save_organism(org, note=payload.note)
            return {"key": key}

    @app.post("/api/evolution/spawn")
    def evo_spawn(payload: EvoSpawnRequest) -> dict[str, Any]:
        with evo_manager.lock:
            ok = evo_manager.simulation.spawn_saved(payload.key)
            if not ok:
                raise HTTPException(status_code=404, detail="Saved organism not found.")
            return evo_manager.simulation.state_dict()

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return manager.get_state()

    @app.post("/api/reset")
    def reset(payload: ResetRequest) -> dict[str, Any]:
        return manager.reset(seed=payload.seed)

    @app.post("/api/step/policy")
    def step_policy() -> dict[str, Any]:
        return manager.step_policy()

    @app.post("/api/step/manual")
    def step_manual(payload: ManualStepRequest) -> dict[str, Any]:
        return manager.step_manual(payload.action)

    @app.post("/api/options")
    def set_options(payload: OptionsRequest) -> dict[str, Any]:
        return manager.set_options(payload.deterministic)

    @app.post("/api/load-checkpoint")
    def load_checkpoint(payload: LoadCheckpointRequest) -> dict[str, Any]:
        return manager.load_checkpoint(payload.checkpoint)

    return app


app = create_app()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the organism web viewer.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface.")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port.")
    parser.add_argument(
        "--headless-check",
        action="store_true",
        help="Exercise the API in-process without starting the server.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.headless_check:
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/api/state")
        if response.status_code != 200:
            raise SystemExit("state_failed")
        policy_available = response.json()["session"]["has_policy"]
        if policy_available:
            step_response = client.post("/api/step/policy")
            if step_response.status_code != 200:
                raise SystemExit("step_failed")
        print("web_ok")
        return

    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
