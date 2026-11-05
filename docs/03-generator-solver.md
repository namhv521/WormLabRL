# SPEC-03 — Exact solver, solvability certificate & level generation

## 1. Outcomes

Tạo level mới theo một trong 3 band `easy | medium | hard`; **không publish nếu chưa chứng nhận có lời giải trong số bước cho phép**. Solver và generator **độc lập với RL agent**, nhờ đó agent có thể thất bại trên một level vẫn được chứng nhận là solvable.

### 1.1. Thuật ngữ

- **Candidate:** level mới sinh, chưa đủ điều kiện sử dụng.
- **SOLVED:** BFS có solution path, đã replay thắng và `path_length ≤ max_steps`.
- **EXHAUSTED:** BFS duyệt hết không gian trạng thái reachable mà không gặp goal; kết luận **không có lời giải** theo ruleset.
- **LIMIT:** chạm giới hạn thời gian/node/bộ nhớ; **chưa biết có lời giải hay không**, không được xem là vô nghiệm.
- **Certificate:** `level_hash`, `ruleset_id`, `solver_version`, `actions`, `optimal_steps`, `expanded_nodes`, `verified_by_replay`, `verifier_version`.
- **Published:** level `SOLVED` + replay verified + khớp band + không vi phạm chính sách dedup.

## 2. Exact BFS solver

### 2.1. Search state

Vì tường và exit là immutable đối với một level, key BFS là:

```text
state_key = (
  tuple(segments_head_to_tail),
  frozenset(remaining_apples)
)
```

- Không đưa `step_count` vào key để BFS tìm đường ngắn nhất theo số move hợp lệ. Từ state đầu, nếu BFS tìm path dài hơn `max_steps` thì không publish.
- Duyệt hành động theo thứ tự `UP, RIGHT, DOWN, LEFT` để kết quả deterministic. `expanded_nodes` tăng mỗi lần một state được lấy khỏi FIFO queue (kể cả goal/deadlock), trước khi xét goal và actions; deadlock không có successor.
- Bỏ action không hợp lệ khỏi graph; action sai không thay đổi state nên không thể giúp tìm path ngắn nhất.
- Thắng đúng quy tắc `SPEC-01`, kể cả đi ngang exit trước khi ăn hết táo.
- `parent`/`action` để reconstruct path; dùng queue FIFO và visited key.
- Giới hạn node đã mở: `max_expanded_nodes = 150_000`; giới hạn thời gian tìm kiếm cho một candidate offline: `max_search_ms = 1_500` trên máy tham chiếu (configurable); khi chạm limit trả `LIMIT`.
- Kết luận `EXHAUSTED` **chỉ** sau khi frontier trống, không có limit bị vượt.
- BFS dùng unit cost và không có action âm, do đó path đầu tiên tới goal là ngắn nhất theo move hợp lệ.

### 2.2. Return contract

```text
solve(level, node_budget, time_budget_ms) -> {
  status: SOLVED | EXHAUSTED | LIMIT | INVALID_LEVEL,
  actions?: [UP|RIGHT|DOWN|LEFT],
  optimal_steps?: int,
  expanded_nodes: int,
  elapsed_ms: int,
  reason?: string
}
```

Khi `SOLVED`, bắt buộc kiểm tra `optimal_steps = len(actions) ≤ max_steps`, rồi chạy **replay từ reset** qua core. Nếu replay không thắng: `INTERNAL_VERIFICATION_FAILED`, không publish và alert.

## 3. Generator pipeline

```text
request(tier, seed)
  -> generate candidate from constrained RNG
  -> schema/ruleset validate
  -> quick structural checks
  -> BFS solve
  -> require SOLVED with valid replay and max_steps
  -> compute algorithmic difficulty score
  -> match requested tier
  -> canonical dedup
  -> persist level + certificate
  -> published level ID
```

### 3.1. Candidate generation — Phase 1 / MVP

1. Chọn board 5×5–8×8, starting body 2–5 segments, số táo 1–4 theo phân bố được version hóa cho tier.
2. Dùng seeded RNG tạo một đường đi thử khả dĩ cho đầu sâu, xây dựng thân liên thông ban đầu rồi **đặt táo/cổng vào các cell phù hợp với witness plan**. Witness chỉ là gợi ý sinh; không phải chứng nhận.
3. Đặt tường vào cell trống, ưu tiên giữ các cell dọc witness path không bị chặn; kiểm tra lại bằng engine vì việc ăn táo làm thay đổi thân.
4. Có thể mutate tường, táo, vị trí exit để tăng số bước/branching; **mọi mutation phải qua BFS và replay**, không dùng witness cũ làm bằng chứng.
5. Khi level không giải được, `LIMIT`, sai tier hoặc trùng lặp thì reject và thử candidate mới.

**Hạn chế bảo đảm:** ở map kích thước/luật hữu hạn số lượng level phân biệt cũng hữu hạn. Nếu `attempts` hoặc ngân sách hết, API phải trả lỗi rõ ràng hoặc dùng pool level đã verify, **không** giảm tier âm thầm.

### 3.2. Limits & retry policy

- Offline generation batch: tối đa `5000 candidate/band/job`, có tổng time budget cấu hình; lưu rejection reason histogram.
- Sinh online không chạy BFS không giới hạn; **ưu tiên lấy từ pool đã certificate** và replenish trong job nền (không phải phản hồi đồng bộ kéo dài).
- MVP prefill pool ≥30 certified level/band cho demo, giữ uniqueness canonical hash trong batch người dùng nhận.
- Nếu tier không còn map sẵn: `503 LEVEL_POOL_EMPTY`, trả `retryable=true`; hoặc bật generation đồng bộ giới hạn 2s trong internal demo, phải log timeout.
- Quota bộ nhớ solver và CPU ngoài request path; backpressure tránh làm treo server.

## 4. Algorithmic difficulty (proxy)

Để tính điểm **tái lập**, dùng số bước tối ưu của BFS và số state đã expand (độc lập latency máy):

```text
s = min(optimal_steps / 40.0, 1.0)
e = min(log10(expanded_nodes + 1) / 5.0, 1.0)
difficulty_score = round(0.60*s + 0.40*e, 4)
```

| Band | Ranh giới score |
|---|---|
| Easy | `0.0000 ≤ score < 0.3500` |
| Medium | `0.3500 ≤ score < 0.6500` |
| Hard | `0.6500 ≤ score ≤ 1.0000` |

- `optimal_steps` và `expanded_nodes` là kết quả **cùng solver_version, thứ tự mở rộng đã cố định**; đổi thuật toán phải bump `difficulty_version` và đánh lại nhãn dataset.
- Nếu solver không `SOLVED`, **không có score hợp lệ để publish**.
- `score` là ước lượng **độ khó đối với thuật toán**, không phải đánh giá người dùng. Gameplay telemetry để hiệu chỉnh band là P1.
- Tăng dần: khi người dùng chọn `next`, hệ thống không giảm **band yêu cầu**; trong cùng band có thể chọn level score cao hơn, hoặc `NONDECREASING_SCORE_UNAVAILABLE` nếu cạn. Không cam kết mọi cặp map tiếp theo đều thực sự khó hơn với tất cả người chơi.
- Các threshold là giả định khởi đầu; cần phân tích histogram sau batch thử nghiệm tuần 3. Nếu không tạo đủ Hard, báo KPI chưa đạt hoặc điều chỉnh không gian ruleset bằng quyết định versioned, không gắn nhãn giả.

## 5. Hash, version & uniqueness

### 5.1. `level_hash`

`SHA-256(canonical_json(ruleset_id, grid, sorted_walls, ordered_worm_segments, sorted_apples, exit))`.

- Không bao gồm `level_id`, `seed`, `generator_version`, metadata, `certificate` trong content hash.
- Serialize canonical JSON: UTF-8, sorted keys, no whitespace, integer coords only.
- `canonical_hash` = hash nhỏ nhất về lexicographic trên 8 phép biến đổi đối xứng của **board vuông**, gồm 4 phép quay và 4 phép phản xạ; biến đổi cả tường, worm segment order (giữ đúng head→tail), táo và exit. Vì luật hướng đối xứng, các biến thể này được coi là trùng puzzle.
- Khóa unique `(ruleset_id, canonical_hash)` trong DB cho published level. `level_hash` và `canonical_hash` là hai trường riêng.
- Level trả về mới phải có `canonical_hash` chưa được phục vụ trong request sequence của người dùng nếu chế độ no-repeat được bật; không đủ map thì trả `503`.

### 5.2. Certificate replay

```json
{
  "level_hash": "<sha256>",
  "ruleset_id": "worm-grid-v1",
  "solver_version": "bfs-v1",
  "verifier_version": "worm-grid-v1.0",
  "actions": ["RIGHT", "RIGHT", "RIGHT"],
  "optimal_steps": 3,
  "expanded_nodes": "<measured integer>",
  "verified_by_replay": true
}
```

Ví dụ actions tương ứng với sample map trong SPEC-01; `expanded_nodes` **không điền bừa** vì phụ thuộc count convention đã log của solver. Certificate phải kiểm tra level hash phù hợp map đang được yêu cầu.

## 6. Generator acceptance criteria

| AC | Điều kiện |
|---|---|
| AC-GEN-01 | Mọi published level có `SOLVED`, `optimal_steps ≤ max_steps`, `verified_by_replay=true`. |
| AC-GEN-02 | Có thể chứng minh independent replay từ serialized level + actions đưa tới `WON`. |
| AC-GEN-03 | `LIMIT` và `EXHAUSTED` không bao giờ xuất hiện trong API như `verified=true`. |
| AC-GEN-04 | Cùng seed/config/ruleset/generator_version sinh cùng danh sách candidate trên cùng build. |
| AC-GEN-05 | DB unique canonical hash chặn map đối xứng bị phát hành lần nữa. |
| AC-GEN-06 | Score/band thuộc khoảng đúng theo công thức versioned. |
| AC-GEN-07 | Batch 1000 yêu cầu: mọi level trả về có solution, không trùng; mọi yêu cầu không đáp ứng trả lỗi rõ ràng, không giả level. |
| AC-GEN-08 | Với board nhỏ (5×5, 1 apple), BFS có optimal steps bằng solver độc lập tham chiếu trên ít nhất 100 fixtures. |

## 7. Phát triển sau MVP

- **P1:** mẫu graph motifs, loops, trap patterns; mở rộng map size, difficulty rating dùng human play.
- **P2:** PCGRL hoặc evolutionary search tối ưu mục tiêu difficulty/novelty, nhưng **vẫn** bắt buộc exact verifier (hoặc bounded certificate checker phù hợp).
- **P2:** adversarial generator chọn map RL thất bại làm curriculum; phải giữ tập test cố định, tránh benchmark contamination.
- **P2:** rulesets mới cho trọng lực, khối đẩy và công tắc: tạo solver/state machine riêng và kiểm tra lại tính hữu hạn.
