# SPEC-05 — API & data contracts v1

## 1. Mục tiêu

Cung cấp REST API phục vụ ứng dụng web và UI replay; training chạy **CLI/offline**, không khởi tạo qua public API. API chịu trách nhiệm validate toàn bộ level/action và chỉ phát hành level đã được verify.

- **Base path:** `/api/v1`.
- **Content-Type:** `application/json; charset=utf-8`.
- **Action strings:** `UP | RIGHT | DOWN | LEFT` (tương ứng `0 | 1 | 2 | 3` trong Gym).
- **Các tier:** `easy | medium | hard`; không tự fallback tier.
- **Timestamps:** ISO 8601 UTC; JSON dùng tọa độ `[x,y]` integer.
- **MVP:** demo localhost, same-origin hoặc CORS allowlist; không coi đây là public API bảo mật sẵn.

## 2. Public level schema

```json
{
  "schema_version": "1.0",
  "level_id": "demo-5x5-001",
  "ruleset_id": "worm-grid-v1",
  "grid": {"width": 5, "height": 5},
  "walls": [],
  "worm": {"segments": [[1,2],[1,3],[1,4]]},
  "apples": [[3,2]],
  "exit": [4,2],
  "seed": 42,
  "generator_version": "manual-fixture-v1",
  "verified": true,
  "difficulty_label": "easy",
  "difficulty_score": 0.1568,
  "level_hash": "<sha256>",
  "canonical_hash": "<sha256>"
}
```

**Ghi chú:** `difficulty_score: 0.1568` được tính cho fixture này khi BFS mở rộng 24 state theo thứ tự hành động đã định và kiểm tra goal lúc dequeue; mọi implementation phải tính lại bằng solver runtime, không hard-code. `<sha256>` là placeholder phải được thay bằng hash thật. `verified=true` chỉ hợp lệ sau khi certificate đã được tạo và replay qua engine; fixture chưa ingest vào DB không được tự khai báo published.

- Required LevelDefinition fields: `schema_version`, `ruleset_id`, `grid`, `walls`, `worm`, `apples`, `exit`.
- Required PublishedLevel metadata: `level_id`, `verified`, `difficulty_label`, `difficulty_score`, `seed`, `generator_version`, `level_hash`, `canonical_hash`.
- Không expose certificate/action list mặc định ở `GET /levels`; để hiển thị lời giải dùng `/solve` với phương thức `bfs`.
- Các hash SHA-256 là lowercase hex dài 64 ký tự, `seed` là integer uint32 `[0, 4294967295]` hoặc `null` đối với map import không gắn RNG.
- Schema validation: board 5–8, worm 2–5 ban đầu, apple 1–4, không overlap, contiguous; xem `SPEC-01`.
- Input sai hoặc version không hỗ trợ phải có lỗi; server **không tự sửa map**.

## 3. Endpoints

| Method | Path | Mục đích | Status chính |
|---|---|---|---|
| `POST` | `/levels/generate` | Lấy một level mới đã chứng nhận trong tier | `201`, `503`, `422` |
| `GET` | `/levels/{level_id}` | Lấy một PublishedLevel | `200`, `404` |
| `POST` | `/sessions` | Tạo game session từ level | `201`, `404` |
| `POST` | `/sessions/{id}/actions` | Thực hiện một action/tick | `200`, `404`, `409`, `422` |
| `POST` | `/sessions/{id}/restart` | Reset lại state của session | `200`, `404` |
| `POST` | `/solve` | Tạo replay bằng RL hoặc BFS | `200`, `404`, `409`, `504` |
| `GET` | `/agents` | Liệt kê model compatible | `200` |
| `GET` | `/training-runs` | Xem summary benchmark đã hoàn tất | `200` |
| `GET` | `/health` | Liveness/readiness nội bộ | `200`, `503` |

### 3.1. `POST /levels/generate`

Request:

```json
{"difficulty": "easy", "seed": 42, "exclude_level_ids": []}
```

- `seed` optional; có seed nghĩa là yêu cầu tái lập selection/generation trong cùng pool snapshot/version, **không** cam kết response giống hệt nếu pool đã thay đổi. Tính tái lập tuyệt đối dành cho offline generator cùng config/seed.
- `exclude_level_ids` optional, tối đa 100; server dùng no-repeat theo level/canonical hash tương ứng.
- Success `201`:

```json
{
  "level": {"level_id":"demo-5x5-001", "schema_version":"1.0", "ruleset_id":"worm-grid-v1", "grid":{"width":5,"height":5}, "walls":[], "worm":{"segments":[[1,2],[1,3],[1,4]]}, "apples":[[3,2]], "exit":[4,2], "seed":42, "generator_version":"manual-fixture-v1", "verified":true, "difficulty_label":"easy", "difficulty_score":0.1568, "level_hash":"<sha256>", "canonical_hash":"<sha256>"},
  "certification": {"verified": true, "optimal_steps": 3}
}
```

Các giá trị `<sha256>` là placeholder mô tả, **không phải hash thực**; response production phải có hash thật. Level demo chỉ có thể dùng trong endpoint khi đã certify và nhập DB.

### 3.2. Sessions

Request tạo session:

```json
{"level_id":"demo-5x5-001"}
```

Response tạo session:

```json
{
  "session_id": "<uuid>",
  "level_id": "demo-5x5-001",
  "state": {
    "worm": {"segments": [[1,2],[1,3],[1,4]]},
    "remaining_apples": [[3,2]],
    "step_count": 0,
    "max_steps": 120,
    "status": "PLAYING",
    "action_mask": [true, true, false, true]
  }
}
```

Với fixture, `UP`, `RIGHT`, `LEFT` hợp lệ; `DOWN` đi vào đốt thứ hai. Action mask ban đầu chính xác là `[true, true, false, true]`.

Request action:

```json
{"action":"RIGHT"}
```

Response:

```json
{
  "session_id":"<uuid>",
  "state":{"worm":{"segments":[[2,2],[1,2],[1,3]]},"remaining_apples":[[3,2]],"step_count":1,"max_steps":120,"status":"PLAYING","action_mask":[true,true,true,false]},
  "event":{"valid_action":true,"ate_apple":false,"won":false,"deadlock":false},
  "reward":-0.01,
  "terminated":false,
  "truncated":false
}
```

Sau bước `RIGHT` đầu tiên, `LEFT` là quay 180° vào đốt thứ hai nên invalid. Mask chính xác là `[true, true, true, false]`. API thực luôn tính mask từ core.

- Session hết hạn sau 30 phút không hoạt động, return `404 SESSION_EXPIRED`.
- Action khi `WON | LOST | TIMEOUT`: `409 EPISODE_FINISHED`.
- `restart` reset về state ban đầu, `step_count=0`, giữ session ID.

### 3.3. `POST /solve`

Request:

```json
{"level_id":"demo-5x5-001", "agent_type":"rl", "model_id":"<published-compatible-model-id>"}
```

- `agent_type: "rl"` bắt buộc `model_id`; `agent_type: "bfs"` bỏ `model_id` và trả path từ certificate đã lưu.
- Response:

```json
{
  "level_id": "demo-5x5-001",
  "agent_type": "bfs",
  "model_id": null,
  "outcome": "SOLVED",
  "actions": ["RIGHT","RIGHT","RIGHT"],
  "steps": 3,
  "optimal": true,
  "elapsed_ms": 12
}
```

`elapsed_ms` là **số minh họa**, phải đo khi chạy thực. Với RL, `optimal=false`, `outcome` có thể `SOLVED | FAILED | TIMEOUT`, `actions` là những action thực agent đã chọn. Trường `optimal=true` chỉ cho BFS certificate. Với RL thất bại **không** dùng path BFS để thay thế. Inference vượt 2s thì `504 INFERENCE_TIMEOUT` với error body.

### 3.4. Models & metrics

`GET /agents`: `[{model_id, ruleset_id, observation_shape, action_mapping, trained_at, checkpoint_sha256, evaluation_summary?}]`. Chỉ trả model có artifact/manifest compatible.

`GET /training-runs`: `[{run_id, model_id, seed, train_steps, dataset_manifest_hash, metrics:{easy:{successes:95,total:100},medium:...,hard:...}}]`. Ví dụ success numbers ở đây chỉ minh họa cấu trúc; runtime lấy từ evaluation artifacts.

## 4. Errors & status semantics

```json
{
  "error": {
    "code": "LEVEL_POOL_EMPTY",
    "message": "No certified level available for requested difficulty.",
    "retryable": true,
    "request_id": "<uuid>"
  }
}
```

| HTTP | Code | Nguyên nhân |
|---|---|---|
| 400 | `UNSUPPORTED_RULESET` | Ruleset/schema không hỗ trợ |
| 404 | `LEVEL_NOT_FOUND` | Level ID không tồn tại/published |
| 404 | `SESSION_EXPIRED` | Session ID không còn |
| 409 | `EPISODE_FINISHED` | Step sau terminal state |
| 409 | `MODEL_INCOMPATIBLE` | Checkpoint không khớp env/action mapping |
| 422 | `INVALID_LEVEL` | Board/worm/entities sai |
| 422 | `INVALID_ACTION` | Giá trị action không thuộc enum (khác với move hợp lệ về schema nhưng bị chặn gameplay) |
| 422 | `INVALID_DIFFICULTY` | Tier ngoài easy/medium/hard |
| 503 | `LEVEL_POOL_EMPTY` | Hết map đã chứng nhận theo tier |
| 504 | `INFERENCE_TIMEOUT` | RL inference vượt deadline |
| 500 | `INTERNAL_VERIFICATION_FAILED` | Certificate replay không khớp; alert nội bộ |

Action thuộc enum nhưng đi đâm tường **không** trả 422; engine trả `200` với `event.valid_action=false`, state unchanged, trừ `step_count`. Trường `reward` trong session API được tính bằng hàm reward chung với RL adapter (để debug), không thuộc luật thắng/thua của game core.

## 5. Security & input validation

- Pydantic strict mode: reject extra fields không được định nghĩa; chống dữ liệu vượt max grid/segments.
- Request JSON ≤64KB, body không có code/script/path từ người dùng trong demo.
- Không public endpoint bắt đầu training; giải RL chỉ load model từ allowlisted registry.
- Giới hạn `/solve` theo inference time và `max_steps`; `/generate` theo pool/quota.
- Public deploy cần auth/rate-limit/tls; local demo chỉ trust localhost.

## 6. Contract tests bắt buộc

1. `GET /levels/{id}` chỉ trả published verified; không trả certificate khi chưa gọi solve.
2. `POST /sessions` với fixture rồi `RIGHT` ×3 trả `WON`, 3 bước, không còn táo.
3. `POST /solve` BFS cùng fixture trả path chính xác `RIGHT,RIGHT,RIGHT` và replay thắng.
4. Action bị chặn tăng step_count, không đổi hình học; out-of-enum trả 422.
5. Map không certificate, hết pool, model incompatible có đúng error code.
6. Tất cả response model/level version metadata khớp contract.

## 7. Phụ thuộc

- [`01-game-rules.md`](01-game-rules.md): luật canonical.
- [`03-generator-solver.md`](03-generator-solver.md): certificate và difficulty.
- [`04-system-architecture.md`](04-system-architecture.md): API/storage ownership.
