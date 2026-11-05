# SPEC-06 — Test strategy, traceability & release gates

## 1. Mục tiêu QA

Chứng minh: (1) game rules đúng và deterministic, (2) map published có lời giải thật trong luật đang chạy, (3) RL evaluation không leakage và không tự dùng BFS, (4) demo end-to-end hoạt động và lỗi minh bạch.

**Test layers:** `unit` → `property-based` → `solver differential` → `API integration` → `web e2e` → `performance` → `ML evaluation`.

## 2. Unit & property tests — engine

| Test ID | Yêu cầu | Input / kỳ vọng |
|---|---|---|
| T-CORE-01 | FR-001 | 4 action delta đúng trục; cùng state + action cho output bằng nhau |
| T-CORE-02 | FR-002 | Đâm tường/biên/body => hình học unchanged, step_count tăng 1 |
| T-CORE-03 | FR-002 | Ăn táo => apple remove, body length +1, head tại apple |
| T-CORE-04 | FR-003 | Hết táo rồi đi vào exit => WON; đi vào exit sớm => không WON |
| T-CORE-05 | FR-003 | Thắng ở bước `max_steps` ưu tiên TIMEOUT; sai action đến max_steps => truncated |
| T-CORE-06 | FR-001/002 | 10k state có segments liên thông/không chồng và nằm trong board sau legal move |
| T-CORE-07 | FR-004 | Mask 4 phần tử khớp engine validity (≥10k sampled states) |
| T-CORE-08 | FR-001 | Same seed/actions replay 100 lần cho final state hash bằng nhau |
| T-CORE-09 | FR-003 | Step sau terminal raise `EPISODE_FINISHED` |
| T-CORE-10 | FR-002 | Đầu vào invalid board/entity/segment bị reject rõ ràng |
| T-CORE-11 | FR-003 | Không còn legal action => LOST deadlock, không trả mask rỗng ở trạng thái PLAYING |

### 2.1. Fixture vàng

Level demo `SPEC-01`: state ban đầu `[head=(1,2), body=(1,3),(1,4)]`, apple `(3,2)`, exit `(4,2)`.

- `RIGHT, RIGHT, RIGHT` → `WON`, `step_count=3`, remaining apples `[]`, body length 4.
- `DOWN` ở state ban đầu → invalid, step_count=1, body giữ nguyên.
- `UP` ở state ban đầu → hợp lệ (cell `(1,1)` trống).
- Ban đầu action mask `[true, true, false, true]` theo `UP,RIGHT,DOWN,LEFT`.

## 3. Solver & generator correctness

| Test ID | Liên quan | Pass condition |
|---|---|---|
| T-SOLVER-01 | FR-005 | Với 100 small fixtures, BFS có optimal_steps bằng independent exhaustive graph search |
| T-SOLVER-02 | FR-005 | BFS `EXHAUSTED` chỉ khi queue empty, không vượt budgets |
| T-SOLVER-03 | FR-005/006 | BFS `LIMIT` không được đánh dấu UNSOLVABLE/SOLVED |
| T-SOLVER-04 | FR-005/006 | Mọi `SOLVED` path replay thắng và len(path) ≤ max_steps |
| T-GEN-01 | FR-006 | 1000 published levels mọi tier có certificate + replay pass |
| T-GEN-02 | FR-007 | 1000 published levels không trùng canonical hash, kể cả quay/lật |
| T-GEN-03 | FR-008 | `difficulty_label` khớp threshold của score versioned |
| T-GEN-04 | FR-007 | Lặp generator seed/ruleset/config trong cùng build sinh candidate hash stream giống |
| T-GEN-05 | FR-006 | Hết candidates/timeout/pool => reject/503; không tự trả map unverified |
| T-GEN-06 | FR-006 | Dữ liệu certificate giả/corrupted => verify failure, không publish |

**Phân biệt:** `1000 published levels` là test trên số level thành công phát hành, không tuyên bố mọi candidate đều khả giải. Ghi thêm acceptance rate theo tier và số `LIMIT` để phát hiện generator bị bias/cạn nguồn.

## 4. RL training & evaluation tests

| Test ID | Liên quan | Pass condition |
|---|---|---|
| T-RL-01 | FR-004 | Gymnasium check_env, observation `(6,8,8)` float32 `[0,1]` |
| T-RL-02 | FR-009 | MaskablePPO train smoke test, checkpoint save/load, same ruleset mapping |
| T-RL-03 | FR-010 | Train/val/test không share canonical hash, test manifest immutable |
| T-RL-04 | FR-010 | Test report có per-tier sample count, solved, mean/std 3 seeds, baselines |
| T-RL-05 | FR-010/012 | Disable solver access trong `rl_inference` path vẫn chạy và trả chính outcome của policy |
| T-RL-06 | FR-004 | State `LOST` hoặc `TIMEOUT` không crash masked action selection |
| T-RL-07 | FR-014 | Checkpoint metadata seed, config hash, code hash và dataset hash đầy đủ |

### 4.1. Chỉ số test không được che giấu

- Evaluate 300 certified held-out maps: 100 easy, 100 medium, 100 hard **mỗi checkpoint**.
- 3 independent train seeds 11/22/33, không báo một run duy nhất là kết quả toàn bộ.
- Baselines Random/Greedy/BFS trên cùng dataset, reported separate.
- Kỳ vọng PRD easy ≥95%, medium ≥80%, hard ≥60% (hard stretch). **Không đạt ngưỡng phải ghi fail KPI, không được thay bằng BFS**.
- Thời gian training/hardware phải log; benchmark engine/generator không phụ thuộc tốc độ GPU.

## 5. API & web e2e

| Test ID | Liên quan | Pass condition |
|---|---|---|
| T-API-01 | FR-011/013 | Chọn difficulty → API `201` chỉ với published verified đúng tier |
| T-API-02 | FR-011 | Tạo session, 3 action fixture, hiển thị win, restart reset |
| T-API-03 | FR-012 | `Watch RL` kết quả riêng, `Show BFS` replay certificate, nhãn phân biệt rõ |
| T-API-04 | FR-013 | 422 input sai, 409 terminal/mismatch, 503 hết pool, 504 inference timeout |
| T-API-05 | FR-014 | Metrics có model ID, dataset hash, seed, sample count |
| T-UX-01 | NFR-007/010 | Keyboard, touch trên viewport 360px, focus indicator, không phụ thuộc màu |
| T-UX-02 | FR-011 | Refresh session unavailable hiển thị reset/new level rõ ràng, không đứng màn trắng |

## 6. Performance & reliability

Thiết bị tham chiếu tạm thời: `[4 vCPU / 8 GB RAM / local Docker — cần xác nhận]`.

- 1000 move requests tuần tự, `p95 ≤150ms` cho backend không có train GPU cạnh tranh tài nguyên.
- 1000 `generate` request từ pool có sẵn, `p95 ≤3s` và tier/verified đúng.
- Mở/tắt worker generator 100 lần không gây mất dữ liệu published đã commit.
- 1000 level published liên tiếp qua replay verification; `internal_verification_failures = 0`.
- Inference RL phải dừng ở terminal hoặc max_steps hoặc timeout 2s, không block vô hạn.

## 7. Traceability matrix

| PRD requirement | Test(s) |
|---|---|
| FR-001–003 | T-CORE-01–11 |
| FR-004 | T-CORE-07, T-RL-01, T-RL-06 |
| FR-005 | T-SOLVER-01–04 |
| FR-006 | T-GEN-01, T-GEN-05–06 |
| FR-007 | T-GEN-02, T-GEN-04 |
| FR-008 | T-GEN-03 |
| FR-009 | T-RL-02, T-RL-07 |
| FR-010 | T-RL-03–05 |
| FR-011 | T-API-02, T-UX-01–02 |
| FR-012 | T-API-03, T-RL-05 |
| FR-013 | T-API-01, T-API-04 |
| FR-014 | T-API-05, T-RL-07 |
| FR-015–016 | P1/P2, không phải release gate MVP |

## 8. Release decision

### Gate G1 — Core correctness (**bắt buộc**)

- Tất cả tests T-CORE pass; seed/action deterministic.
- Không có state invalid sau legal move.

### Gate G2 — Generation correctness (**bắt buộc**)

- T-SOLVER/T-GEN pass.
- Không có map published chưa verify; `LIMIT` không bị coi là UNSOLVABLE.
- Đủ 30 level certified mỗi tier cho demo, hoặc tuyên bố tier tương ứng **chưa hỗ trợ** và không quảng bá tier đó.

### Gate G3 — Demo & observability (**bắt buộc**)

- Test API/E2E và NFR trọng yếu pass, không crash.
- Có checkpoint RL, replay của RL và BFS với nhãn rõ, có report sample counts.

### Gate G4 — RL research target (**KPI độc lập**)

- Đo đủ 3 seeds × 300 maps, báo trung thực success per tier.
- Đạt/chưa đạt các threshold ghi rõ trong release notes; **không** giả định G1–G3 pass tương đương RL đã giải được mọi map.

## 9. Reporting template

```markdown
## Release QA report — <commit> / <date>
- Ruleset / generator / solver versions: ...
- Core unit/property tests: PASS/FAIL, counts: ...
- Published certified levels by tier: ...
- Generator candidate acceptance rate and LIMIT count: ...
- Map uniqueness / canonical collisions: ...
- API p95 move / generation: ...
- RL test dataset hash: ...
- RL model IDs & seeds: ...
- RL success Easy/Medium/Hard: ... (successes/100 per model)
- Baseline Random/Greedy/BFS: ...
- G1/G2/G3/G4 status: ...
- Known issues / release decision: ...
```
