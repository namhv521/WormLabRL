"""SPEC-05 §6 contract tests + T-API-01..05 against an isolated DB."""
import json

import pytest
from fastapi.testclient import TestClient

from generator import certified_record, certify, generate_batch
from services.api.app import create_app
from storage import LevelStore
from tests.helpers import demo_level


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("api")
    store = LevelStore(d / "t.db")
    cl, _ = certify(demo_level())
    rec = certified_record(cl, "pool")
    rec["level_id"] = "demo-5x5-001"
    store.publish(rec, cl.certificate)
    for cl in generate_batch("medium", 3, base_seed=9, namespace="api")[0]:
        store.publish(certified_record(cl, "pool"), cl.certificate)
    store.close()
    (d / "models").mkdir()
    app = create_app(db_path=str(d / "t.db"), models_dir=d / "models", serve_web=False)
    return TestClient(app)


API = "/api/v1"


def test_get_level_published_without_certificate(client):
    r = client.get(f"{API}/levels/demo-5x5-001")
    assert r.status_code == 200
    body = r.json()
    assert body["verified"] is True and body["difficulty_score"] == 0.1568 and "actions" not in body
    assert len(body["level_hash"]) == 64 and len(body["canonical_hash"]) == 64
    assert client.get(f"{API}/levels/nope").json()["error"]["code"] == "LEVEL_NOT_FOUND"


def test_api01_generate_tier(client):
    r = client.post(f"{API}/levels/generate", json={"difficulty": "medium", "seed": 1})
    assert r.status_code == 201
    assert r.json()["level"]["difficulty_label"] == "medium" and r.json()["certification"]["verified"]
    r2 = client.post(f"{API}/levels/generate", json={"difficulty": "medium", "seed": 1})
    assert r2.json()["level"]["level_id"] == r.json()["level"]["level_id"]


def test_api04_pool_empty_and_bad_tier(client):
    r = client.post(f"{API}/levels/generate", json={"difficulty": "hard"})
    assert r.status_code == 503 and r.json()["error"] == {**r.json()["error"], "code": "LEVEL_POOL_EMPTY",
                                                          "retryable": True}
    r = client.post(f"{API}/levels/generate", json={"difficulty": "extreme"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_DIFFICULTY"
    r = client.post(f"{API}/levels/generate", json={"difficulty": "easy", "bogus": 1})
    assert r.status_code == 422


def test_generate_excludes(client):
    ids = set()
    for _ in range(3):
        r = client.post(f"{API}/levels/generate", json={"difficulty": "medium", "exclude_level_ids": list(ids)})
        ids.add(r.json()["level"]["level_id"])
    assert len(ids) == 3
    r = client.post(f"{API}/levels/generate", json={"difficulty": "medium", "exclude_level_ids": list(ids)})
    assert r.status_code == 503


def test_api02_session_fixture_win_and_restart(client):
    r = client.post(f"{API}/sessions", json={"level_id": "demo-5x5-001"})
    assert r.status_code == 201
    sid, st = r.json()["session_id"], r.json()["state"]
    assert st["action_mask"] == [True, True, False, True] and st["max_steps"] == 120
    r = client.post(f"{API}/sessions/{sid}/actions", json={"action": "RIGHT"}).json()
    assert r["state"]["worm"]["segments"] == [[2, 2], [1, 2], [1, 3]]
    assert r["state"]["action_mask"] == [True, True, True, False] and r["reward"] == -0.01
    for _ in range(2):
        r = client.post(f"{API}/sessions/{sid}/actions", json={"action": "RIGHT"}).json()
    assert r["state"]["status"] == "WON" and r["state"]["step_count"] == 3 and r["state"]["remaining_apples"] == []
    r = client.post(f"{API}/sessions/{sid}/actions", json={"action": "UP"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "EPISODE_FINISHED"
    r = client.post(f"{API}/sessions/{sid}/restart").json()
    assert r["session_id"] == sid and r["state"]["step_count"] == 0 and r["state"]["status"] == "PLAYING"


def test_blocked_move_and_bad_enum(client):
    sid = client.post(f"{API}/sessions", json={"level_id": "demo-5x5-001"}).json()["session_id"]
    r = client.post(f"{API}/sessions/{sid}/actions", json={"action": "DOWN"})
    assert r.status_code == 200 and not r.json()["event"]["valid_action"]
    assert r.json()["state"]["step_count"] == 1 and r.json()["state"]["worm"]["segments"] == [[1, 2], [1, 3], [1, 4]]
    r = client.post(f"{API}/sessions/{sid}/actions", json={"action": "NORTH"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_ACTION"
    r = client.post(f"{API}/sessions/missing/actions", json={"action": "UP"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "SESSION_EXPIRED"


def test_api03_solve_bfs_and_rl_errors(client):
    r = client.post(f"{API}/solve", json={"level_id": "demo-5x5-001", "agent_type": "bfs"}).json()
    assert r["actions"] == ["RIGHT", "RIGHT", "RIGHT"] and r["optimal"] is True and r["outcome"] == "SOLVED"
    r = client.post(f"{API}/solve", json={"level_id": "demo-5x5-001", "agent_type": "rl", "model_id": "ghost"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "MODEL_NOT_FOUND"
    r = client.post(f"{API}/solve", json={"level_id": "demo-5x5-001", "agent_type": "rl"})
    assert r.status_code == 422


def test_model_incompatible(tmp_path):
    from ml.registry import load_model
    from worm_core import GameError
    d = tmp_path / "models" / "bad"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"model_id": "bad", "ruleset_id": "worm-grid-v1",
                                                 "observation_shape": [5, 8, 8], "action_mapping": []}))
    app = create_app(db_path=":memory:", models_dir=tmp_path / "models", serve_web=False)
    assert TestClient(app).get(f"{API}/agents").json() == []
    with pytest.raises(GameError) as e:
        load_model("bad", tmp_path / "models")
    assert e.value.code == "MODEL_INCOMPATIBLE"


def test_training_runs_shape(client):
    r = client.get(f"{API}/training-runs")
    assert r.status_code == 200 and set(r.json()) == {"runs", "baselines"}
