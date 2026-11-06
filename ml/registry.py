"""Model registry: immutable checkpoints + manifests under ``artifacts/models``.

Loading refuses any checkpoint whose ruleset / observation shape / action
mapping does not match the running env (SPEC-02 §6, SPEC-05 MODEL_INCOMPATIBLE).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional

from worm_core import RULESET_ID, GameError

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "artifacts" / "models"
RUNS_DIR = ROOT / "artifacts" / "runs"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compatible(manifest: dict) -> bool:
    from rl_env import ACTION_MAPPING, OBS_SHAPE
    return (manifest.get("ruleset_id") == RULESET_ID
            and tuple(manifest.get("observation_shape", ())) == OBS_SHAPE
            and manifest.get("action_mapping") == ACTION_MAPPING)


def register(model_path: Path, model_id: str, extra: dict, models_dir: Path = MODELS_DIR) -> dict:
    from rl_env import ACTION_MAPPING, OBS_SHAPE
    d = models_dir / model_id
    d.mkdir(parents=True, exist_ok=True)
    target = d / "model.zip"
    target.write_bytes(Path(model_path).read_bytes())
    manifest = {"model_id": model_id, "ruleset_id": RULESET_ID, "observation_shape": list(OBS_SHAPE),
                "action_mapping": ACTION_MAPPING, "checkpoint_sha256": sha256_file(target), **extra}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def list_models(models_dir: Path = MODELS_DIR, only_compatible: bool = True) -> List[dict]:
    out = []
    for mf in sorted(models_dir.glob("*/manifest.json")):
        m = json.loads(mf.read_text(encoding="utf-8"))
        ev = mf.parent / "evaluation.json"
        if ev.exists():
            m["evaluation_summary"] = json.loads(ev.read_text(encoding="utf-8")).get("summary")
        if not only_compatible or compatible(m):
            out.append(m)
    return out


def get_manifest(model_id: str, models_dir: Path = MODELS_DIR) -> Optional[dict]:
    if not model_id or "/" in model_id or "\\" in model_id or ".." in model_id:
        return None
    mf = models_dir / model_id / "manifest.json"
    return json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else None


def load_model(model_id: str, models_dir: Path = MODELS_DIR):
    from sb3_contrib import MaskablePPO
    m = get_manifest(model_id, models_dir)
    if m is None:
        raise GameError("MODEL_NOT_FOUND", f"unknown model {model_id!r}")
    if not compatible(m):
        raise GameError("MODEL_INCOMPATIBLE", f"model {model_id} does not match {RULESET_ID} env")
    path = models_dir / model_id / "model.zip"
    if sha256_file(path) != m["checkpoint_sha256"]:
        raise GameError("MODEL_INCOMPATIBLE", f"checkpoint hash mismatch for {model_id}")
    model = MaskablePPO.load(str(path), device="cpu")
    if tuple(model.observation_space.shape) != tuple(m["observation_shape"]):
        raise GameError("MODEL_INCOMPATIBLE", "observation shape mismatch")
    return model, m
