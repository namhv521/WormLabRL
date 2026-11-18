"""
WormLabRL Interactive Web Studio Server (FastAPI).

Launch:
  python studio_server.py --port 8000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import torch
import uvicorn

from train_product import DEFAULT_CKPT, DEFAULT_DESIGNER_CKPT
from worm_physics import WormMap
from worm_ai.model import WormBrainNet, load_brain_checkpoint, save_brain_checkpoint
from worm_ai.oracle import analyze_level
from worm_ai.pcgrl import (
    DesignerPolicyNet,
    ProgressivePCGRLGenerator,
    load_designer_checkpoint,
    prune_unused_objects,
)
from worm_ai.physics import (
    BLOCKED_EVENTS,
    DEFAULT_LEVELS_JSON,
    FATAL_EVENTS,
    STRICT_RULES,
    LevelArena,
    ROOT,
    step_strict_trace,
    validate_initial_level,
)
from worm_ai.trainer import ContinualWormTrainer, frame_dict

STUDIO_HTML = ROOT / "studio" / "index.html"
LIVE_CKPT = ROOT / "checkpoints" / "worm_brain_live.pt"
ARCHIVE_PATH = ROOT / "generated_maps" / "archive.json"


class LevelPayload(BaseModel):
    level: dict[str, Any]
    name: str = "custom"
    ppo_rounds: int = 3
    envs: int = 8


class GeneratePayload(BaseModel):
    tier: int = 2
    escalating: bool = False
    clean_unused: bool = True


class StepPayload(BaseModel):
    level: dict[str, Any]
    body: list[list[int]]
    apples: list[list[int]]
    stones: list[list[int]]
    action: int


class StudioState:
    def __init__(self):
        torch.set_num_threads(4)
        self.lock = threading.Lock()
        raw = json.loads(DEFAULT_LEVELS_JSON.read_text(encoding="utf-8-sig"))
        self.elements = raw["elements"]
        self.levels_a = raw["levelsA"]
        self.levels_b = raw["levelsB"]

        ckpt_to_load = LIVE_CKPT if LIVE_CKPT.exists() else DEFAULT_CKPT
        if ckpt_to_load.exists():
            model, meta = load_brain_checkpoint(ckpt_to_load)
            self.ckpt_source = str(ckpt_to_load.name)
            self.ckpt_meta = meta
        else:
            model = WormBrainNet()
            self.ckpt_source = "fresh_untrained"
            self.ckpt_meta = {}

        self.trainer = ContinualWormTrainer(model=model, seed=73)

        if DEFAULT_DESIGNER_CKPT.exists():
            d_model, _ = load_designer_checkpoint(DEFAULT_DESIGNER_CKPT)
        else:
            d_model = DesignerPolicyNet()

        self.generator = ProgressivePCGRLGenerator(
            model=d_model, archive_path=ARCHIVE_PATH, seed=73
        )
        self.curriculum_tier = 1
        self.difficulty_floor = 14.0
        self.consecutive_wins = 0
        self.coevolution_history: list[dict] = []

    def generated_levels(self) -> list[dict]:
        entries = self.generator.archive.list_by_difficulty()
        out = []
        for idx, item in enumerate(entries, 1):
            lv = dict(item["level"])
            lv["oracle"] = item["oracle"]
            lv.setdefault("name", f"G{lv.get('id', idx)}")
            out.append(lv)
        return out

    def brain_status(self) -> dict:
        return {
            "checkpoint": self.ckpt_source,
            "parameters": self.trainer.model.parameter_count(),
            "total_updates": self.trainer.total_updates,
            "total_env_steps": self.trainer.total_env_steps,
            "mastered_stages": len(self.trainer.replay.mastered_stages),
            "replay_size": len(self.trainer.replay),
            "curriculum_tier": self.curriculum_tier,
            "difficulty_floor": round(self.difficulty_floor, 1),
            "generated_archive_count": len(self.generator.archive.cells),
            "stage_history": self.trainer.stage_history[-20:],
            "coevolution_history": self.coevolution_history[-20:],
        }


app = FastAPI(title="WormLabRL Interactive AI Studio")
STATE: StudioState | None = None


def get_state() -> StudioState:
    global STATE
    if STATE is None:
        STATE = StudioState()
    return STATE


@app.get("/")
def serve_index():
    if not STUDIO_HTML.exists():
        raise HTTPException(status_code=404, detail="Studio UI index.html not found")
    return FileResponse(STUDIO_HTML)


@app.get("/api/status")
def api_status():
    return get_state().brain_status()


@app.get("/api/levels")
def api_levels():
    st = get_state()
    return {
        "levelsA": st.levels_a,
        "levelsB": st.levels_b,
        "levelsG": st.generated_levels(),
        "status": st.brain_status(),
    }


@app.post("/api/init_level")
def api_init_level(payload: LevelPayload):
    valid, reason, board = validate_initial_level(payload.level)
    if not valid or board is None:
        raise HTTPException(status_code=400, detail=f"Invalid map: {reason}")
    return {
        "valid": True,
        "initial": frame_dict(board.initial, step=0, event="ok"),
    }


@app.post("/api/step")
def api_step(payload: StepPayload):
    from worm_physics import State as SimState

    valid, reason, board = validate_initial_level(payload.level)
    if not valid or board is None:
        raise HTTPException(status_code=400, detail=f"Invalid map: {reason}")
    cur = SimState(
        tuple((int(p[0]), int(p[1])) for p in payload.body),
        frozenset((int(p[0]), int(p[1])) for p in payload.apples),
        frozenset((int(p[0]), int(p[1])) for p in payload.stones),
    )
    # First check if current state itself is already dead/unsettled (e.g. legacy state)
    settled_cur, cur_ev = board.settle(cur)
    if cur_ev in FATAL_EVENTS:
        final_f = frame_dict(settled_cur, action=payload.action, event=cur_ev)
        return {
            **final_f,
            "sub_frames": [final_f],
            "is_fatal": True,
            "is_blocked": False,
            "is_win": False,
        }

    sub_states, event = step_strict_trace(board, settled_cur, int(payload.action))
    fall_frames = [
        frame_dict(st, action=payload.action, event="falling") for st in sub_states[:-1]
    ]
    final_frame = frame_dict(sub_states[-1], action=payload.action, event=event)
    return {
        **final_frame,
        "sub_frames": fall_frames + [final_frame],
        "is_fatal": event in FATAL_EVENTS,
        "is_blocked": event in BLOCKED_EVENTS,
        "is_win": event == "win",
    }


@app.post("/api/verify")
def api_verify(payload: LevelPayload):
    res = analyze_level(payload.level, budget=30000, counterfactual_budget=8000, verify_mandatory=True)
    cleaned_lv = payload.level
    if res.solvable and res.unused_apples > 0:
        cleaned_lv, res = prune_unused_objects(payload.level, res)
    return {
        "oracle": res.to_dict(),
        "cleaned_level": cleaned_lv,
    }


@app.post("/api/ai/play")
def api_ai_play(payload: LevelPayload):
    st = get_state()
    valid, reason, board = validate_initial_level(payload.level)
    if not valid or board is None:
        raise HTTPException(status_code=400, detail=f"Invalid map: {reason}")
    arena = LevelArena(board, name=payload.name, level_dict=payload.level)
    with st.lock:
        ep = st.trainer.play_episode(arena, greedy=True, record_frames=True)
    return {
        "stage": ep.stage,
        "won": ep.won,
        "outcome": ep.outcome,
        "steps": ep.steps,
        "total_reward": ep.total_reward,
        "actions": ep.actions,
        "frames": ep.frames,
        "status": st.brain_status(),
    }


@app.post("/api/ai/self_train_stream")
def api_ai_self_train_stream(payload: LevelPayload):
    st = get_state()
    valid, reason, board = validate_initial_level(payload.level)
    if not valid or board is None:
        raise HTTPException(status_code=400, detail=f"Invalid map: {reason}")
    arena = LevelArena(board, name=payload.name, level_dict=payload.level)

    def event_generator():
        events_queue: list[dict] = []

        def on_prog(ev: dict):
            ep_obj = ev.pop("episode", None)
            msg = dict(ev)
            if ep_obj is not None:
                msg["frames"] = ep_obj.frames
            events_queue.append(msg)

        with st.lock:
            summary = st.trainer.play_and_self_train_stage(
                arena,
                ppo_rounds=max(1, min(8, payload.ppo_rounds)),
                envs=max(4, min(16, payload.envs)),
                horizon=36,
                rescue_budget=3000,
                record_frames=True,
                on_progress=on_prog,
            )
            save_brain_checkpoint(LIVE_CKPT, st.trainer.model, st.trainer.optimizer)
            st.ckpt_source = LIVE_CKPT.name

        for item in events_queue:
            yield json.dumps(item) + "\n"

        yield json.dumps({
            "phase": "complete",
            "summary": summary,
            "status": st.brain_status(),
        }) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.post("/api/pcgrl/generate")
def api_pcgrl_generate(payload: GeneratePayload):
    st = get_state()
    with st.lock:
        tier = st.curriculum_tier if payload.escalating else max(1, min(5, payload.tier))
        floor = st.difficulty_floor if payload.escalating else 0.0

        def solver_probe(lv_dict: dict) -> bool:
            ar = LevelArena(WormMap(lv_dict), name="probe", level_dict=lv_dict)
            return st.trainer.play_episode(ar, greedy=True, max_steps=180, record_frames=False).won

        lv, ores = st.generator.generate_verified_map(
            target_tier=tier,
            min_difficulty_floor=floor,
            max_attempts=50,
            solver_eval_fn=solver_probe,
        )
        if payload.escalating:
            st.difficulty_floor = max(st.difficulty_floor + 3.5, ores.difficulty_score + 1.0)
            if st.difficulty_floor >= 30 and st.curriculum_tier < 2:
                st.curriculum_tier = 2
            elif st.difficulty_floor >= 46 and st.curriculum_tier < 3:
                st.curriculum_tier = 3
            elif st.difficulty_floor >= 63 and st.curriculum_tier < 4:
                st.curriculum_tier = 4
            elif st.difficulty_floor >= 76 and st.curriculum_tier < 5:
                st.curriculum_tier = 5

        lv["oracle"] = ores.to_dict()
        return {
            "level": lv,
            "oracle": ores.to_dict(),
            "status": st.brain_status(),
        }


@app.post("/api/coevolution/step")
def api_coevolution_step():
    """Run one complete Co-Evolution cycle: Generate Harder Map -> Zero-Shot -> Self-Train -> Escalate!"""
    st = get_state()
    with st.lock:
        req_tier = st.curriculum_tier
        req_floor = st.difficulty_floor

        def solver_probe(lv_dict: dict) -> bool:
            ar = LevelArena(WormMap(lv_dict), name="probe", level_dict=lv_dict)
            return st.trainer.play_episode(ar, greedy=True, max_steps=180, record_frames=False).won

        lv, ores = st.generator.generate_verified_map(
            target_tier=req_tier,
            min_difficulty_floor=req_floor,
            max_attempts=50,
            solver_eval_fn=solver_probe,
        )
        arena = LevelArena(WormMap(lv), name=lv["name"], level_dict=lv)
        summary = st.trainer.play_and_self_train_stage(
            arena, ppo_rounds=3, envs=8, horizon=36, rescue_budget=3000, record_frames=True
        )
        save_brain_checkpoint(LIVE_CKPT, st.trainer.model, st.trainer.optimizer)
        st.ckpt_source = LIVE_CKPT.name

        if summary["after_train_won"]:
            st.consecutive_wins += 1
            st.difficulty_floor = max(st.difficulty_floor + 4.0, ores.difficulty_score + 1.0)
            if st.consecutive_wins >= 2 and st.curriculum_tier < 5:
                st.curriculum_tier += 1
                st.consecutive_wins = 0

        rec = {
            "round": len(st.coevolution_history) + 1,
            "map_name": lv["name"],
            "tier": ores.difficulty_tier,
            "tier_name": ores.tier_name,
            "difficulty_score": ores.difficulty_score,
            "optimal_steps": ores.optimal_steps,
            "mandatory_apples": ores.mandatory_apples,
            "mandatory_stones": ores.mandatory_stones,
            "zero_shot_won": summary["zero_shot_won"],
            "after_train_won": summary["after_train_won"],
            "after_train_steps": summary["after_train_steps"],
            "next_tier": st.curriculum_tier,
            "next_floor": round(st.difficulty_floor, 1),
        }
        st.coevolution_history.append(rec)
        lv["oracle"] = ores.to_dict()
        return {
            "record": rec,
            "level": lv,
            "oracle": ores.to_dict(),
            "summary": summary,
            "status": st.brain_status(),
        }


@app.post("/api/reset_curriculum")
def api_reset_curriculum():
    st = get_state()
    with st.lock:
        st.curriculum_tier = 1
        st.difficulty_floor = 14.0
        st.consecutive_wins = 0
        return st.brain_status()


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Launch WormLabRL Interactive Web Studio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"[WormLabRL] AI Studio starting at http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
