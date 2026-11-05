# PRD — WormLab RL: AI giải và sinh puzzle theo độ khó

| Thuộc tính | Giá trị |
|---|---|
| Mã dự án | WLRL |
| Phiên bản | 0.1.0 — Proposed |
| Ngày | 2026-11-05 |
| Chủ sở hữu | `[Product Owner — TBD]` |
| Nhóm triển khai | `[ML Engineer, Full-stack Engineer, QA — TBD]` |
| Mốc mục tiêu | MVP trong **8 tuần** kể từ ngày bắt đầu dự án, phụ thuộc nguồn lực |
| Phạm vi | Game puzzle độc lập, RL solver, generator được chứng nhận, demo web |

## 1. Product overview

### 1.1. Vấn đề

Puzzle casual dạng lưới thường có tập màn hữu hạn do con người thiết kế. Việc mở rộng nội dung khó nếu cần cả hai điều kiện: **map thực sự có lời giải** và **độ khó tăng hợp lý**. Đồng thời, chỉ dùng giải thuật tìm kiếm để giải map không chứng minh được một mô hình RL đã học cách khái quát hóa sang bài toán mới.

### 1.2. Đề xuất sản phẩm

Phát triển **WormLab RL**, trong đó người chơi hoặc agent điều khiển một con sâu ăn táo và tìm đường ra. Hệ thống gồm hai năng lực có thể kiểm định độc lập:

1. **RL Solver:** học chính sách chọn hành động từ quan sát bản đồ; đánh giá trên level chưa xuất hiện trong huấn luyện.
2. **Level Generator:** sinh nhiều level theo mức khó được yêu cầu, dùng solver tìm kiếm chính xác và replay để bảo đảm level được phát hành có ít nhất một cách thắng.

Ứng dụng web minh họa tương tác của người chơi, giải pháp AI và các chỉ số huấn luyện. Generator không cần chờ RL đạt tỉ lệ thắng cao mới có thể chứng nhận level.

### 1.3. Cơ sở thiết kế và ranh giới IP

Game gốc thuộc thể loại sâu ăn táo/cổng thoát có nhiều biến thể luật; MVP **chủ động giản lược** sang grid puzzle không trọng lực. Không sao chép màn, assets hay tên thương mại. Mục tiêu "giải toàn bộ map Hungry Worm" được chuyển thành mục tiêu đo được: **đạt ngưỡng tỉ lệ giải trên tập level WormLab chưa nhìn thấy**. Việc tương thích một game bên ngoài cần một project nhánh với rule adapter và bộ dữ liệu hợp lệ.

## 2. Objectives & non-goals

### 2.1. Mục tiêu sản phẩm

- **O1 — Chơi được:** người dùng giải level trên web bằng bàn phím/touch, hiển thị thắng và có thể chơi lại.
- **O2 — Học được:** quy trình training tạo checkpoint RL, được đánh giá tách biệt với tập train và benchmark baseline.
- **O3 — Sinh được:** mỗi map xuất bản có lời giải hợp lệ; sinh theo 3 band Easy/Medium/Hard trong miền hỗ trợ.
- **O4 — Quan sát được:** xem agent replay, số bước, tỉ lệ thắng theo tier, phiên bản model và seed.
- **O5 — Tái lập được:** cùng `ruleset_version`, config và seed tạo ra trạng thái/đường đi giống nhau trong cùng bản build.

### 2.2. Không thuộc MVP

- Không cam kết RL giải 100% mọi puzzle hoặc mọi màn của game thương mại.
- Không có trọng lực, vật thể đẩy, bẫy chuyển động, multiplayer, vật phẩm monetization.
- Không sinh "vô hạn map độc nhất" trên tập kích thước/luật cố định; chỉ **stream tạo map on-demand**.
- Không huấn luyện mô hình ngay trong trình duyệt hoặc cung cấp hạ tầng GPU tự động mở rộng.
- Không tuyên bố cấp độ "Hard" đồng nhất với đánh giá của mọi người chơi; nhãn là proxy do thuật toán tính.

## 3. Target audience & use cases

| Persona | Nhu cầu chính | Tình huống thành công |
|---|---|---|
| Người chơi casual | Màn mới, điều khiển dễ, độ khó tăng | Chọn Easy → nhận map hợp lệ → hoàn thành → nhận map tiếp |
| ML practitioner | Thí nghiệm RL có benchmark tái lập | Train checkpoint, so sánh 3 seed và test set chưa thấy |
| Game designer | Có công cụ tạo màn với solution | Sinh map Medium/Hard, xem replay và các chỉ số khó |
| Người review/demo | Thấy rõ giá trị AI | Quan sát RL và BFS giải cùng map, thấy khoảng cách hiệu năng |

## 4. Product scope & MVP prioritization

| Module | Yêu cầu | Độ ưu tiên | MVP |
|---|---|---|---|
| Deterministic game core | Chuyển trạng thái đúng theo luật, replay | P0 | Có |
| Exact BFS solver | Trả về đường đi ngắn nhất hoặc trạng thái EXHAUSTED/LIMIT | P0 | Có |
| Generator | Candidate → solver verify → dedup → publish | P0 | Có |
| Gymnasium environment | `reset/step`, observation, action mask, seed | P0 | Có |
| RL training | MaskablePPO, checkpoint, metrics, held-out evaluation | P0 | Có |
| Playable web demo | Chơi tay, chọn độ khó, AI replay | P0 | Có |
| Dashboard căn bản | Success rate theo tier và model version | P0 | Có |
| Export/import level | File JSON có version, validate khi import | P1 | Chưa |
| Human-based difficulty model | Xếp hạng dựa trên play telemetry | P1 | Chưa |
| AI generator / PCGRL | Model học tự thiết kế màn | P2 | Chưa |
| Rule adapter cho game khác | Hỗ trợ gameplay khác Hungry Worm | P2 | Chưa |

## 5. User stories & acceptance criteria

| ID | User story | Điều kiện nghiệm thu |
|---|---|---|
| US-01 | Là người chơi, tôi muốn điều khiển sâu để hoàn thành puzzle. | AC-01: thao tác 4 hướng; cấm move không hợp lệ; thắng chỉ khi hết táo và đầu ở cổng. |
| US-02 | Là người chơi, tôi muốn nhận màn mới đúng tier. | AC-02: mỗi map phản hồi có `difficulty_label` theo band được yêu cầu và `verified=true` trên server; nếu không có trả lỗi rõ ràng. |
| US-03 | Là designer, tôi muốn xem lời giải map. | AC-03: solver xuất action list replay từ reset tới thắng, số bước khớp với metadata. |
| US-04 | Là người nghiên cứu, tôi muốn train RL nhiều seed. | AC-04: CLI nhận config/seed, lưu checkpoint + config hash + metrics và có thể đánh giá lại. |
| US-05 | Là người review, tôi muốn xem RL giải một level mới. | AC-05: giao diện phân biệt `RL policy` với `BFS oracle`, báo `solved/failed/timeout` mà không thay thế thất bại của RL bằng BFS. |
| US-06 | Là QA, tôi muốn chắc level xuất bản không vô nghiệm. | AC-06: 100% level phát hành có solver status SOLVED và đường đi replay thắng. |
| US-07 | Là ML engineer, tôi cần so sánh năng lực khái quát hóa. | AC-07: tập test tách train bằng canonical hash, báo theo tier và kèm số mẫu/seed/model ID. |

## 6. Functional requirements

| ID | Priority | Yêu cầu kiểm thử được |
|---|---|---|
| FR-001 | P0 | Engine thực thi đúng bốn hướng `UP/RIGHT/DOWN/LEFT`, một action/tick; cùng trạng thái+action cho cùng kết quả. |
| FR-002 | P0 | Engine kiểm tra biên, tường, thân sâu, hướng quay 180° và xử lý phát triển thân khi ăn táo theo `01-game-rules.md`. |
| FR-003 | P0 | Trạng thái thắng đúng khi không còn táo **và** đầu sâu đứng trên cổng; deadlock kết thúc `LOST`, hết giới hạn bước kết thúc bằng `truncated`. |
| FR-004 | P0 | Env cung cấp Gymnasium `reset(seed)` và `step(action)` theo API 5-tuple, observation cố định shape, mask hành động hợp lệ. |
| FR-005 | P0 | Solver BFS tìm đường đi ngắn nhất khi trả `SOLVED`; phân biệt `EXHAUSTED` (đã duyệt hết) và `LIMIT` (không kết luận). |
| FR-006 | P0 | Generator chỉ publish map sau khi solver trả `SOLVED` và simulator xác nhận replay tới thắng. |
| FR-007 | P0 | Generator tách map trùng bằng canonical hash, lưu seed, ruleset, generator_version và certificate. |
| FR-008 | P0 | Generator hỗ trợ Easy/Medium/Hard dựa trên công thức độ khó định nghĩa trong `03-generator-solver.md`. |
| FR-009 | P0 | Pipeline RL huấn luyện MaskablePPO với checkpoint, seed cố định và `run_manifest`. |
| FR-010 | P0 | Evaluation đo success, steps, timeout theo tier trên tập train/validation/test tách biệt, và Random/Greedy/BFS baseline. |
| FR-011 | P0 | Web demo hiển thị bản đồ, status, control bằng keyboard/touch, restart level và màn thắng. |
| FR-012 | P0 | Người dùng có thể yêu cầu AI replay với lựa chọn rõ `RL` hoặc `BFS`, và xem từng hành động. |
| FR-013 | P0 | API validate dữ liệu map/action/tier; không có đường dẫn nào trả map chưa có certificate thành map "verified". |
| FR-014 | P0 | Hệ thống hiển thị run ID, model ID, evaluation dataset version, timestamp và thông số tối thiểu của metric. |
| FR-015 | P1 | Cho phép import/export level JSON, chạy validator trước khi chấp nhận. |
| FR-016 | P2 | Hỗ trợ bộ luật khác qua engine adapter, benchmark riêng từng ruleset. |

## 7. Non-functional requirements

| ID | Thuộc tính | Mục tiêu đề xuất / phép đo |
|---|---|---|
| NFR-001 | Correctness | 100% level publish có certificate replay thắng; không nhận `LIMIT` là solvable. |
| NFR-002 | Determinism | Trong cùng build, 100 lần replay cùng state, actions, seed cho final hash giống nhau. |
| NFR-003 | API latency | Local dev machine tham chiếu `[4 vCPU, 8 GB RAM — cần xác nhận]`: p95 move API ≤150ms cho 1000 request tuần tự. |
| NFR-004 | Level latency | p95 `/levels/generate` ≤3s khi có pool level đã verify; hết pool trả 503 minh bạch. |
| NFR-005 | Reliability | 1000 level published liên tiếp đều qua validator, không crash service. |
| NFR-006 | Reproducibility | Mọi training run lưu git commit, config, seed, dataset hash, checkpoint hash, thư viện versions. |
| NFR-007 | Accessibility | Điều khiển keyboard và touch; thông tin không dựa chỉ trên màu; nút có nhãn. |
| NFR-008 | Privacy & security | Demo mặc định localhost, không thu PII; nếu public phải thêm rate limit, auth/admin cho training và lọc input. |
| NFR-009 | Observability | Log request_id, level_id, ruleset, model_id, elapsed_ms, error_code; không log checkpoint nhị phân. |
| NFR-010 | Compatibility | Browser desktop Chrome/Firefox/Edge hiện hành; mobile responsive từ viewport 360px. |

## 8. ML & content success metrics

Các con số bên dưới là **release target**, cần xác nhận sau baseline tuần 2. Đánh giá **mỗi checkpoint trên cùng 300 level test (100/band)** chưa từng train, qua **3 seed huấn luyện độc lập**; báo trung bình và từng run. Chính sách RL chạy độc lập, không được gọi BFS trong lúc inference.

| KPI | Định nghĩa | Mục tiêu MVP |
|---|---|---|
| Valid published levels | Số level phát hành có replay thắng / tổng published | **100%** |
| Generator service availability | Có level sẵn trong pool theo mỗi band tại demo | ≥30 level verified/band trước demo |
| Dedup exact/canonical | Trùng canonical hash trong batch 1000 level trả về | 0 trùng trong batch, hoặc API trả 409/503 nếu cạn |
| RL Easy success | Episodes solved / 100 level Easy held-out, trung bình 3 run | ≥95% |
| RL Medium success | Tương tự với Medium | ≥80% |
| RL Hard success | Tương tự với Hard | ≥60% *(stretch / cần xác nhận khả thi)* |
| Baseline comparisons | Bảng Random, Greedy, BFS, RL trên cùng benchmark | Có đủ, không yêu cầu RL thắng BFS |
| Reproducibility | Lặp seed/level/ruleset trên cùng build cho kết quả giống nhau | 100% deterministic engine; training có thể sai khác nhỏ do GPU |
| Demo completion | Generate → play → AI replay → metrics | 100% flow smoke test |

**Nguyên tắc phát hành:** nếu RL không đạt ngưỡng kỳ vọng, không giả định agent "giải được tất cả". Có thể phát hành **research demo** với số đo thực, đồng thời ghi rõ KPI chưa đạt. Nếu pool Hard chưa đủ, đánh dấu band chưa hỗ trợ thay vì tự ý trả map Medium dưới nhãn Hard.

## 9. Proposed UX & flows

1. Chọn mức độ khó → API trả level **đã được verify** → bàn chơi hiển thị.
2. Điều khiển sâu bằng phím mũi tên/WASD hoặc nút chạm; nhận feedback action không hợp lệ.
3. Nhấn `Watch RL` để chạy inference từ map reset, xem path, kết quả và số bước.
4. Nhấn `Show BFS solution` để xem lời giải chắc chắn, được gắn nhãn oracle.
5. Tab `Experiments` xem chỉ số theo tier, checkpoint, seed, dataset; dữ liệu không có thì hiển thị empty state.

Chi tiết ở [`spec/07-ux.md`](spec/07-ux.md).

## 10. Technical overview

- **Core + ML:** Python, NumPy, Gymnasium; MaskablePPO của sb3-contrib.
- **Solver:** BFS theo toàn trạng thái (vị trí có thứ tự của thân + bitmask táo).
- **Generation:** candidate theo constraints; solve, replay, grade, dedup; prefill pool.
- **API:** FastAPI + Pydantic; dữ liệu level và certificate lưu SQLite ở MVP.
- **Web:** React + TypeScript; UI chỉ gửi action, engine backend là nguồn sự thật để tránh phân kỳ luật.
- **Runtime:** local Docker Compose hoặc chạy service riêng; CPU đủ cho game/generator, GPU chỉ tùy chọn cho train.

Các phiên bản thư viện phải pin trong lockfiles khi bootstrap repo, không mặc định khóa theo tài liệu này.

## 11. Timeline & milestones

| Tuần | Deliverable | Exit gate |
|---|---|---|
| 1 | Game core + schemas + unit tests | Chạy pass rule test, deterministic |
| 2 | BFS exact + level fixture | BFS trả path đúng, replay và kiểm tra optimality ở board nhỏ |
| 3 | Generator + 3 band + pool | 100% level published verified, dedup |
| 4 | Gymnasium env + baseline + benchmark | Benchmark train/val/test tách biệt |
| 5 | RL training pipeline + checkpoints | Train và eval nhiều seed, lưu manifest |
| 6 | Curriculum + tuning + so sánh | Có báo cáo KPI thực tế theo tier |
| 7 | API + web play/replay/dashboard | End-to-end demo đầy đủ |
| 8 | QA, docs, profiling, release | Đạt technical gates; công bố kết quả RL thực tế |

Phụ thuộc và backlog tại [`PLAN.md`](PLAN.md).

## 12. Dependencies, risks & mitigation

| Rủi ro | Ảnh hưởng | Giảm thiểu |
|---|---|---|
| Sparse reward làm RL không học được Hard | KPI mô hình không đạt | Curriculum dễ → khó, action masking; chỉ nghiên cứu imitation nếu cần |
| BFS bùng nổ không gian trạng thái | Sinh Hard chậm hoặc cạn pool | Giới hạn node/time; reject `LIMIT`; giới hạn kích thước; precompute pool |
| Độ khó thuật toán khác độ khó người chơi | Label gây hiểu nhầm | Gắn "algorithmic difficulty"; telemetry người chơi là P1 |
| Leakage giữa train/test qua đối xứng | Báo cáo RL quá lạc quan | Canonical hash khi chia split, cố định test manifest |
| Yêu cầu "vô hạn độc nhất" bất khả thi trong miền hữu hạn | Kỳ vọng sai | Dùng on-demand, kiểm soát trùng, mở rộng ruleset/kích thước hậu MVP |
| Khác biệt mechanics của game tham khảo | Không dùng được map bên ngoài | Game độc lập; lập adapter sau khi có ruleset và dữ liệu hợp lệ |
| Thiếu compute/time | Không đạt target RL đúng hạn | Ưu tiên end-to-end correctness, công bố metric thực thay vì hứa 100% |

## 13. Assumptions & open decisions

- **A-01:** Chấp nhận game độc lập gần thể loại Hungry Worm, thay vì sao chép chính xác trò chơi cụ thể.
- **A-02:** MVP giới hạn 5×5–8×8 và 3 tier; Hard là **stretch objective**, không chặn việc nghiệm thu engine/generator.
- **A-03:** Demo localhost hoặc môi trường nội bộ; public deployment cần hardening.
- **A-04:** Có thể tiếp cận máy để train ở mức `[CPU/GPU/budget — TBD]`.
- **Q-01:** Có cần hỗ trợ trọng lực, vật thể đẩy, cạm bẫy như một số biến thể game sâu trong v1 không?
- **Q-02:** Ưu tiên mục tiêu khoa học (benchmark RL) hay mục tiêu game thương mại (retention/người chơi)?
- **Q-03:** Mốc 8 tuần và nhân sự có khả thi với nguồn lực thực tế không?

## 14. Definition of Done (MVP)

MVP được nghiệm thu về kỹ thuật khi các P0 liên quan engine, generator, UI, API và đánh giá model đạt test trong [`spec/06-tests-and-acceptance.md`](spec/06-tests-and-acceptance.md); dữ liệu benchmark được công bố trung thực. Thành tích RL theo tier là **KPI chất lượng nghiên cứu riêng**, không được thay bằng lời giải BFS và không được tuyên bố đạt nếu chưa đo được.

## 15. References

- Gymnasium Env API: https://gymnasium.farama.org/main/api/env/
- SB3-Contrib algorithms: https://sb3-contrib.readthedocs.io/en/master/guide/algos.html
- PCGRL research: https://doi.org/10.1609/aiide.v16i1.7416
- Hungry Worm: Logic Puzzle listing (tham chiếu thể loại): https://play.google.com/store/apps/details?id=com.hungryworm.logicpuzzle
