"""WormLab RL REST API v1 (SPEC-05).

Run: ``uvicorn services.api.app:create_app --factory --port 8000``
"""
from __future__ import annotations

import json
import logging
import random
import threading
import time
import uuid
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from worm_core import (ACTIONS, GameError, Level, State, Status, initial_state, level_from_dict, step,
                       valid_action_mask)

log = logging.getLogger("wormlab.api")
logging.basicConfig(level=logging.INFO, format="%(message)s")

SESSION_TTL_S = 30 * 60
INFERENCE_DEADLINE_S = 2.0
MAX_BODY = 64 * 1024
ROOT = Path(__file__).resolve().parents[2]

STATUS_OF = {"UNSUPPORTED_RULESET": 400, "LEVEL_NOT_FOUND": 404, "SESSION_EXPIRED": 404, "MODEL_NOT_FOUND": 404,
             "EPISODE_FINISHED": 409, "MODEL_INCOMPATIBLE": 409, "INVALID_LEVEL": 422, "INVALID_ACTION": 422,
             "INVALID_DIFFICULTY": 422, "INVALID_REQUEST": 422, "LEVEL_POOL_EMPTY": 503,
             "INFERENCE_TIMEOUT": 504, "INTERNAL_VERIFICATION_FAILED": 500, "PAYLOAD_TOO_LARGE": 413}
RETRYABLE = {"LEVEL_POOL_EMPTY", "INFERENCE_TIMEOUT"}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GenerateReq(Strict):
    difficulty: str
    seed: Optional[int] = Field(None, ge=0, le=4294967295)
    exclude_level_ids: List[str] = Field(default_factory=list, max_length=100)


class SessionReq(Strict):
    level_id: str = Field(max_length=64)


class ActionReq(Strict):
    action: str


class SolveReq(Strict):
    level_id: str = Field(max_length=64)
    agent_type: Literal["rl", "bfs"]
    model_id: Optional[str] = Field(None, max_length=128)


PUBLIC_LEVEL_KEYS = ("schema_version", "level_id", "ruleset_id", "grid", "walls", "worm", "apples", "exit", "seed",
                     "generator_version", "verified", "difficulty_label", "difficulty_score", "level_hash",
                     "canonical_hash")


def public_level(rec: dict) -> dict:
    return {k: rec[k] for k in PUBLIC_LEVEL_KEYS}


def state_json(level: Level, st: State) -> dict:
    return {"worm": {"segments": [list(c) for c in st.segments]},
            "remaining_apples": [list(c) for c in sorted(st.apples)],
            "step_count": st.step_count, "max_steps": level.max_steps, "status": st.status.value,
            "action_mask": list(valid_action_mask(level, st))}


class Session:
    __slots__ = ("level_id", "level", "state", "touched")

    def __init__(self, level_id, level):
        self.level_id, self.level = level_id, level
        self.state = initial_state(level)
        self.touched = time.monotonic()


def create_app(db_path: Optional[str] = None, models_dir: Optional[Path] = None,
               training_runs_dir: Optional[Path] = None, serve_web: bool = True) -> FastAPI:
    from ml import registry
    from storage import DEFAULT_DB, LevelStore
    from solver import verify_certificate

    store = LevelStore(db_path or DEFAULT_DB)
    models_dir = models_dir or registry.MODELS_DIR
    sessions: dict[str, Session] = {}
    lock = threading.Lock()
    model_cache: dict = {}

    app = FastAPI(title="WormLab RL API", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["*"], allow_headers=["*"])
    app.state.store = store
    app.state.sessions = sessions

    def err(code: str, message: str, request_id: str):
        return JSONResponse(status_code=STATUS_OF.get(code, 500),
                            content={"error": {"code": code, "message": message, "retryable": code in RETRYABLE,
                                               "request_id": request_id}})

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = str(uuid.uuid4())
        request.state.rid = rid
        if int(request.headers.get("content-length") or 0) > MAX_BODY:
            return err("PAYLOAD_TOO_LARGE", "request body exceeds 64KB", rid)
        t0 = time.perf_counter()
        resp = await call_next(request)
        log.info(json.dumps({"request_id": rid, "method": request.method, "path": request.url.path,
                             "status": resp.status_code, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
                             "error_code": getattr(request.state, "error_code", None)}))
        resp.headers["x-request-id"] = rid
        return resp

    @app.exception_handler(GameError)
    async def game_error(request: Request, e: GameError):
        request.state.error_code = e.code
        return err(e.code, e.message, request.state.rid)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, e: RequestValidationError):
        code = "INVALID_REQUEST"
        locs = {tuple(x["loc"]) for x in e.errors()}
        if ("body", "difficulty") in locs:
            code = "INVALID_DIFFICULTY"
        request.state.error_code = code
        return err(code, "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()),
                   request.state.rid)

    def get_level(level_id: str):
        rec = store.get(level_id)
        if rec is None or not rec.get("verified"):
            raise GameError("LEVEL_NOT_FOUND", f"no published level {level_id!r}")
        return rec

    def get_session(sid: str) -> Session:
        now = time.monotonic()
        with lock:
            for k in [k for k, s in sessions.items() if now - s.touched > SESSION_TTL_S]:
                del sessions[k]
            s = sessions.get(sid)
        if s is None:
            raise GameError("SESSION_EXPIRED", "session not found or expired")
        s.touched = now
        return s

    api = "/api/v1"

    @app.get(api + "/health")
    def health():
        return {"status": "ok", "pool": store.counts("pool")}

    @app.post(api + "/levels/generate", status_code=201)
    def generate(req: GenerateReq):
        if req.difficulty not in ("easy", "medium", "hard"):
            raise GameError("INVALID_DIFFICULTY", f"difficulty must be easy|medium|hard, got {req.difficulty!r}")
        excl = set(req.exclude_level_ids)
        excl_hashes = {r["canonical_hash"] for lid in excl if (r := store.get(lid))}
        pool = [r for r in store.pool(req.difficulty, exclude=excl) if r["canonical_hash"] not in excl_hashes]
        if not pool:
            raise GameError("LEVEL_POOL_EMPTY", f"No certified {req.difficulty} level available.")
        rng = random.Random(req.seed) if req.seed is not None else random.SystemRandom()
        rec = rng.choice(pool)
        cert = store.certificate(rec["level_id"])
        if cert is None or not verify_certificate(level_from_dict(rec), cert):
            raise GameError("INTERNAL_VERIFICATION_FAILED", "stored certificate failed replay")
        return {"level": public_level(rec), "certification": {"verified": True, "optimal_steps": cert["optimal_steps"]}}

    @app.get(api + "/levels/{level_id}")
    def get_level_route(level_id: str):
        return public_level(get_level(level_id))

    @app.post(api + "/sessions", status_code=201)
    def create_session(req: SessionReq):
        level = level_from_dict(get_level(req.level_id))
        sid = str(uuid.uuid4())
        s = Session(req.level_id, level)
        with lock:
            sessions[sid] = s
        return {"session_id": sid, "level_id": req.level_id, "state": state_json(level, s.state)}

    @app.post(api + "/sessions/{sid}/actions")
    def act(sid: str, req: ActionReq):
        from rl_env import reward_of
        s = get_session(sid)
        if req.action not in ACTIONS:
            raise GameError("INVALID_ACTION", f"action must be one of {', '.join(ACTIONS)}")
        with lock:
            r = step(s.level, s.state, req.action)
            s.state = r.state
        ev = r.event
        return {"session_id": sid, "state": state_json(s.level, r.state),
                "event": {"valid_action": ev.valid_action, "ate_apple": ev.ate_apple, "won": ev.won,
                          "deadlock": ev.deadlock},
                "reward": round(reward_of(ev), 4), "terminated": r.terminated, "truncated": r.truncated}

    @app.post(api + "/sessions/{sid}/restart")
    def restart(sid: str):
        s = get_session(sid)
        with lock:
            s.state = initial_state(s.level)
        return {"session_id": sid, "level_id": s.level_id, "state": state_json(s.level, s.state)}

    @app.post(api + "/solve")
    def solve_route(req: SolveReq):
        rec = get_level(req.level_id)
        level = level_from_dict(rec)
        if req.agent_type == "bfs":
            if req.model_id is not None:
                raise GameError("INVALID_REQUEST", "model_id must be omitted for agent_type=bfs")
            t0 = time.perf_counter()
            cert = store.certificate(req.level_id)
            if cert is None or not verify_certificate(level, cert):
                raise GameError("INTERNAL_VERIFICATION_FAILED", "certificate replay failed")
            return {"level_id": req.level_id, "agent_type": "bfs", "model_id": None, "outcome": "SOLVED",
                    "actions": cert["actions"], "steps": cert["optimal_steps"], "optimal": True,
                    "elapsed_ms": int((time.perf_counter() - t0) * 1000)}
        if not req.model_id:
            raise GameError("INVALID_REQUEST", "model_id is required for agent_type=rl")
        from ml.agents import RLAgent, run_episode
        if req.model_id not in model_cache:
            model_cache[req.model_id] = registry.load_model(req.model_id, models_dir)[0]
        r = run_episode(RLAgent(model_cache[req.model_id], req.model_id), level, deadline_s=INFERENCE_DEADLINE_S)
        if r["outcome"] == "INFERENCE_TIMEOUT":
            raise GameError("INFERENCE_TIMEOUT", f"RL inference exceeded {INFERENCE_DEADLINE_S}s")
        outcome = "SOLVED" if r["outcome"] == "SOLVED" else ("TIMEOUT" if r["outcome"] == "TIMEOUT" else "FAILED")
        return {"level_id": req.level_id, "agent_type": "rl", "model_id": req.model_id, "outcome": outcome,
                "failure_reason": None if outcome == "SOLVED" else r["outcome"],
                "actions": r["actions"], "steps": r["steps"], "optimal": False, "elapsed_ms": r["elapsed_ms"]}

    @app.get(api + "/agents")
    def agents():
        keys = ("model_id", "ruleset_id", "observation_shape", "action_mapping", "trained_at", "checkpoint_sha256",
                "seed", "run_id", "evaluation_summary")
        return [{k: m.get(k) for k in keys} for m in registry.list_models(models_dir)]

    @app.get(api + "/training-runs")
    def training_runs():
        out = []
        for m in registry.list_models(models_dir):
            ev = models_dir / m["model_id"] / "evaluation.json"
            if not ev.exists():
                continue
            e = json.loads(ev.read_text(encoding="utf-8"))
            out.append({"run_id": m.get("run_id"), "model_id": m["model_id"], "seed": m.get("seed"),
                        "train_steps": m.get("train_steps"), "trained_at": m.get("trained_at"),
                        "dataset_manifest_hash": e["dataset_hash"], "dataset": e["dataset"],
                        "metrics": e["summary"], "detail": e["metrics"]})
        baselines = []
        bdir = ROOT / "artifacts" / "baselines"
        for f in sorted(bdir.glob("test-*.json")) if bdir.exists() else []:
            b = json.loads(f.read_text(encoding="utf-8"))
            for name, r in b["agents"].items():
                runs = r["runs"] if name == "random" else [r]
                metrics = {t: (runs[0][t] and {"successes": sum(x[t]["successes"] for x in runs),
                                               "total": sum(x[t]["episodes"] for x in runs)})
                           for t in ("easy", "medium", "hard", "overall")}
                baselines.append({"agent": name, "dataset_manifest_hash": b["dataset_hash"], "dataset": b["dataset"],
                                  "eval_runs": len(runs), "metrics": metrics})
        return {"runs": out, "baselines": baselines}

    web = ROOT / "apps" / "web" / "dist"
    if serve_web and web.exists():
        app.mount("/", StaticFiles(directory=web, html=True), name="web")

    return app
