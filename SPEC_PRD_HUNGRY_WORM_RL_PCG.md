# PRD & Technical Specification v1.0
## Hungry Worm AI Ecosystem — RL Solver, PCGRL Map Generator & Interactive Tooling

- **Trạng thái**: Draft Specification (Ready for Implementation)
- **Thư mục gốc làm việc**: `C:\Su\tools\Attack_law\test\`
- **Nguồn dữ liệu nền tảng**: `hungry_worm_levels.json` (240 màn chơi gốc + lời giải chuẩn), `hungry_worm_elements.json` (11 mã thực thể)
- **Nền tảng công nghệ đề xuất**: Python 3.10+, PyTorch, Gymnasium, Stable-Baselines3 (`sb3-contrib` MaskablePPO), NumPy/Numba, FastAPI + HTML5 Canvas

---

# PHẦN I. PRODUCT REQUIREMENTS DOCUMENT (PRD)

## 1. Product Vision & Tổng quan hệ thống

Hệ thống **Hungry Worm AI Ecosystem** biến bộ dữ liệu tĩnh 240 màn chơi trích xuất từ game *Hungry Worm - Greedy Worm* thành một nền tảng AI khép kín gồm **4 trụ cột cốt lõi** và **hệ sinh thái mở rộng**:

1. **Deterministic Physics Engine (`WormEnv`)**: Môi trường giả lập vật lý lưới `22 × 12` khớp tuyệt đối theo cơ chế gốc của game (trọng lực toàn thân sâu, đẩy đá, ăn táo dài thân, rơi tự do, va chạm bẫy, chui cổng hố đen) chạy tốc độ cao (`>50.000 steps/s` trên CPU), chuẩn hoá theo `gymnasium.Env`.
2. **Hybrid Solver (`RL Solver + Exact Search Oracle`)**:
   - **RL Solver Agent**: Huấn luyện bằng **Behavioral Cloning (BC)** từ 240 chuỗi `hints` kết hợp tinh chỉnh bằng **MaskablePPO / AlphaZero-lite** để tự giải màn chơi mới.
   - **Exact State-Space Solver (BFS / A\*)**: Đóng vai trò "Trọng tài / Oracle" kiểm chứng 100% một Map mới có giải được hay không, đo độ dài lời giải ngắn nhất $L^*$ và số nhánh bẫy (dead-ends).
3. **Procedural Map Generator (`PCGRL — RL Level Designer`)**: Mô hình RL học cách đặt gạch, táo, đá, bẫy và vị trí sâu để sinh ra vô hạn Map mới có điều kiện (theo độ khó, chủ đề cơ chế, hoặc khung hình dáng), đảm bảo **100% giải được** và có chiều sâu giải đố.
4. **Developer & Player Studio ("Các hướng phát triển mở rộng")**: Web Studio vừa tự tay chơi thử (playable), vừa xem AI giải trực tiếp, vừa bấm nút sinh Map mới, chấm điểm độ khó tự động, sinh Map hình chữ/logo, và đóng gói ngược thành bản Mod XAPK cài lên điện thoại Android.

---

## 2. Phân tích Dữ liệu Nền tảng (Baseline Dataset Analytics)

Số liệu thực nghiệm đo trực tiếp trên `C:\Su\tools\Attack_law\test\hungry_worm_levels.json`:

| Chỉ số thống kê | Pack A (`LevelConfigs` - Chính) | Pack B (`LevelConfigs_B` - Phụ) | Ý nghĩa thiết kế Model |
| :--- | :--- | :--- | :--- |
| **Số lượng màn chơi** | 210 màn (`id: 1..210`) | 30 màn (`id: 1..30`) | Tổng **240 màn** có sẵn `hints` chuyên gia để warm-start bằng Imitation Learning |
| **Kích thước lưới (`W × H`)** | `22 × 12` (100% đồng nhất) | `22 × 12` (100% đồng nhất) | Cố định Tensor không gian đầu vào `(C, 12, 22)`, không cần padding |
| **Độ dài sâu ban đầu** | Min `2`, Max `21` (`3` đốt chiếm 165/210 màn) | Cố định `3` đốt (30/30 màn) | Phải lưu chuỗi đốt sâu có thứ tự `deque[(x,y)]` từ đầu đến đuôi |
| **Số lượng Táo (`id=4`)** | Min `0`, Max `47`, TB `4.64` | Min `0`, Max `2`, TB `0.97` | Chiều dài sâu cực đại trong ván có thể lên tới `~68` đốt |
| **Số lượng Đá rơi (`id=2`)** | Min `0`, Max `30` (98/210 màn có đá) | Min `0`, Max `3` (15/30 màn có đá) | Cần mô phỏng trọng lực đồng thời và cơ chế đẩy ngang có điều kiện |
| **Số lượng Bẫy (`id=3,7,9,10,11`)** | Min `0`, Max `48` (167/210 màn có bẫy) | Min `0`, Max `11` (8/30 màn có bẫy) | Mật độ trạng thái chết cao $\rightarrow$ bắt buộc dùng **Invalid Action Masking** |
| **Độ dài lời giải (`hints`)** | Min `11`, Max `200`, TB `53.8` bước | Min `11`, Max `83`, TB `33.0` bước | Đặt `max_episode_steps = 250` bao phủ 100% phân phối màn chơi |


---

## 3. Product Goals & Chỉ tiêu nghiệm thu

| ID | Mục tiêu sản phẩm | Chỉ tiêu nghiệm thu (Acceptance Criteria) | Ưu tiên |
| :--- | :--- | :--- | :-: |
| **PG-01** | **Bit-exact Game Simulator** | Tái hiện chuỗi `hints` trên 240 màn gốc đạt tỷ lệ khớp qua màn `>= 98%` (chuẩn xác trọng lực toàn thân sâu, ăn táo, đẩy đá và rơi tự do) | **P0** |
| **PG-02** | **Exact Search Verifier (Oracle)** | Xác định chính xác `Solvable / Unsolvable`, đường đi ngắn nhất $L^*$ và điểm độ khó cho Map mới trong `< 2 giây/map` | **P0** |
| **PG-03** | **RL Solver Agent** | Agent `BC + MaskablePPO` giải thành công `>= 90%` tập huấn luyện và `>= 75%` tập màn mới chưa từng gặp (zero-shot) | **P0** |
| **PG-04** | **RL Map Generator (PCGRL)** | Sinh tự động Map `22 × 12` mới đảm bảo: **100% giải được**, không tầm thường ($L^* \ge 15$), có ít nhất 1 tương tác giải đố (ăn táo nối cầu / kê đá / né bẫy) | **P0** |
| **PG-05** | **Controllable Difficulty & Style** | Điều khiển sinh Map theo mức độ khó (`Easy / Medium / Hard / Extreme`) hoặc theo chủ đề cơ chế (`Apple-Bridge`, `Stone-Sokoban`, `Precision-Trap`) | **P1** |
| **PG-06** | **Interactive Web Studio** | Chơi trực tiếp bằng bàn phím trên trình duyệt, bật AI chơi mẫu, vẽ/sửa Map có AI kiểm tra tức thì, xuất chuẩn `hungry_worm_levels.json` | **P1** |
| **PG-07** | **Ecosystem Extensions** | Sinh Map theo chữ/pixel-art, đóng gói ngược vào XAPK chạy trên Android, tiến hoá Map đánh lừa tư duy (Deceptive Puzzles) | **P2** |

---

## 4. Cấu trúc Thư mục Dự án Đề xuất (Trong `C:\Su\tools\Attack_law\test\`)

Toàn bộ mã nguồn, dữ liệu huấn luyện, model checkpoints và công cụ hiển thị được tổ chức gọn trong `C:\Su\tools\Attack_law\test\`:

```text
C:\Su\tools\Attack_law\test\
├── hungry_worm_levels.json              # [Đã có] 240 màn gốc + lời giải chuẩn
├── hungry_worm_elements.json            # [Đã có] 11 loại vật thể chuẩn
├── hungry_worm_map_viewer.html          # [Đã có] Trình xem Map offline hiện tại
├── SPEC_PRD_HUNGRY_WORM_RL_PCG.md       # [File này] Tài liệu PRD & Đặc tả kỹ thuật
│
├── worm_ai\                             # Package cốt lõi cho RL & Map Generation
│   ├── __init__.py
│   ├── env\
│   │   ├── physics.py                   # Lõi vật lý tất định (tái tạo chuỗi sâu, trọng lực, đẩy đá)
│   │   ├── solver_env.py                # Gymnasium Env cho RL Solver (chơi game)
│   │   └── generator_env.py             # Gymnasium Env cho PCGRL (thiết kế sinh Map mới)
│   │
│   ├── oracle\
│   │   ├── bfs_astar_solver.py          # Bộ giải chính xác A* / Zobrist Hashing kiểm định Map
│   │   └── difficulty_scorer.py         # Chấm điểm độ khó, nhánh cụt, mức độ đánh lừa
│   │
│   ├── models\
│   │   ├── spatial_resnet.py            # Mạng xương sống ResNet-CNN + CoordConv + Self-Attention
│   │   ├── bc_pretrainer.py             # Học bắt chước (Behavioral Cloning) từ 240 hints
│   │   ├── ppo_solver.py                # Huấn luyện & suy luận MaskablePPO Solver
│   │   └── pcgrl_designer.py            # Huấn luyện RL Generator sinh Map mới + Novelty Archive
│   │
│   ├── generators\
│   │   ├── reverse_play_gen.py          # Sinh Map ngược đảm bảo 100% giải được (Constructive Baseline)
│   │   ├── evolutionary_qd.py           # MAP-Elites / Quality-Diversity tinh chỉnh Map cực khó
│   │   └── shape_conditioned_gen.py     # Sinh Map theo hình chữ cái / Pixel-art mask
│   │
│   └── exporter\
│       ├── json_exporter.py             # Xuất gói Map mới ra định dạng LevelConfigs + tự điền hints
│       └── xapk_patcher.py              # Ghi đè TextAsset mới vào bản sao XAPK để chơi trên Android
│
├── checkpoints\                         # Lưu trọng số models (.pt / .zip) & TensorBoard logs
├── generated_maps\                      # Kho lưu các bộ Map mới sinh ra (.json + preview HTML)
├── studio\                              # Web App tương tác (Playable + AI Co-pilot + Map Editor)
└── tests_worm\                          # Unit tests kiểm định vật lý trên toàn bộ 240 màn gốc
```

---

# PHẦN II. TECHNICAL SPECIFICATION — LÕI VẬT LÝ & MÔI TRƯỜNG (`WormEnv`)

## 5. Đặc tả Luật Vật lý Tất định (Deterministic Physics Specification)

Để mô hình RL học đúng và Map sinh ra chơi được thật trong game gốc, bộ giả lập `physics.py` **bắt buộc** tuân thủ chính xác 6 quy tắc sau (đã đối chiếu từ dữ liệu thực tế của 240 màn):

### 5.1. Khởi tạo Chuỗi Đốt Sâu từ Dữ liệu Gốc (Ordered Worm Reconstruction)
- **Vấn đề thực tế trong JSON gốc**: Trong mảng `bricks` của `hungry_worm_levels.json` (ví dụ Level 20 và Level 50), các đốt thân `id=6` được lưu theo thứ tự quét toạ độ lưới chứ **không theo thứ tự từ cổ đến đuôi**.
- **Thuật toán chuẩn hoá bắt buộc (`SIM-01`)**:
  1. Tìm duy nhất ô đầu sâu $H = (x_0, y_0)$ có `id = 5`.
  2. Thực hiện tìm kiếm đường đi đơn (DFS/Backtracking trên đồ thị 4-láng giềng Manhattan $|dx| + |dy| = 1$) đi qua tất cả các ô có `id = 6` sao cho tạo thành một đường đi không tự cắt $W = [p_0, p_1, \dots, p_{k-1}]$ với $p_0 = H$.
  3. Lưu trạng thái sâu dưới dạng `collections.deque` để thao tác thêm đầu / cắt đuôi đạt độ phức tạp $O(1)$.

### 5.2. Chu trình 1 Bước Di chuyển (`step(action)`)
Không gian hành động rời rạc $\mathcal{A}_{\text{play}} = \{1, 2, 3, 4\}$ khớp trực tiếp mã `d` trong `hints`:
- `1`: Trái ($\Delta x = -1, \Delta y = 0$)
- `2`: Phải ($\Delta x = +1, \Delta y = 0$)
- `3`: Lên ($\Delta x = 0, \Delta y = +1$)
- `4`: Xuống ($\Delta x = 0, \Delta y = -1$)

Mỗi bước gọi `step(d)` thực thi tuần tự 4 pha tất định:

1. **Pha 1 — Kiểm tra Hợp lệ & Tương tác Đẩy Đá (`Move & Push Check`)**:
   - Gọi $p_{\text{next}} = (x_0 + \Delta x, y_0 + \Delta y)$.
   - **Không thể đi ngược vào cổ**: Nếu $p_{\text{next}} == p_1$ $\rightarrow$ Hành động không hợp lệ (bị chặn bởi Action Mask).
   - **Chặn bởi Tường**: Nếu `grid[p_next] == WALL (1)` $\rightarrow$ Không hợp lệ.
   - **Chặn bởi Thân sâu**: Nếu $p_{\text{next}} \in \{p_1, \dots, p_{k-2}\}$ $\rightarrow$ Không hợp lệ. *(Lưu ý: Đi vào đúng vị trí đuôi hiện tại $p_{k-1}$ là **hợp lệ** nếu ô đó không có táo, vì đuôi co lại cùng lúc đầu tiến tới).*
   - **Tương tác Đẩy Đá (`STONE = 2`)**:
     - Chỉ đẩy được theo phương ngang ($\Delta y == 0, |\Delta x| == 1$) khi ô ngay sau viên đá $p_{\text{push}} = (x_0 + 2\Delta x, y_0)$ hoàn toàn trống (`EMPTY`, không có tường, táo, bẫy, cổng, đá khác hay thân sâu).
     - Nếu đẩy được: Dịch viên đá sang $p_{\text{push}}$ trước khi đầu sâu tiến vào $p_{\text{next}}$. Nếu không đẩy được (hoặc đi lên/xuống gặp đá) $\rightarrow$ Hành động bị chặn.

2. **Pha 2 — Dịch chuyển Thân & Ăn Táo (`Advance & Grow`)**:
   - Nếu $p_{\text{next}}$ chứa `APPLE (4)`: Xoá táo khỏi bản đồ, chèn $p_{\text{next}}$ vào đầu `deque` (giữ nguyên đuôi $p_{k-1}$ $\rightarrow$ độ dài sâu tăng lên $k+1$).
   - Ngược lại: Chèn $p_{\text{next}}$ vào đầu `deque` và cắt bỏ đốt đuôi $p_{k-1}$.
   - Kiểm tra tức thì:
     - Nếu $p_{\text{next}} == p_{\text{end}}$ (`END = 8`) $\rightarrow$ **THẮNG (`terminated = True, is_win = True`)**.
     - Nếu $p_{\text{next}}$ chứa Bẫy (`3, 7, 9, 10, 11`) $\rightarrow$ **CHẾT (`terminated = True, is_dead = True`)**.

3. **Pha 3 — Vòng lặp Trọng lực Rơi từng Nấc (`Iterative Gravity Settlement`)**:
   Sau mỗi bước di chuyển, cả **toàn bộ cơ thể Sâu** (dạng khối cứng liên thông — rigid chain) và các **viên Đá (`STONE`)** có thể mất điểm tựa và rơi xuống theo từng ô $\Delta y = -1$ cho đến khi chạm đáy đỡ hoặc kết thúc ván:
   - **Điều kiện đỡ của Sâu**: Sâu được đỡ (không rơi) nếu tồn tại ít nhất một đốt $p_i = (x_i, y_i) \in W$ mà ô ngay dưới nó $(x_i, y_i - 1)$ là `WALL (1)`, `APPLE (4)`, `END (8)` (nếu quy định cổng đỡ thân, hoặc kiểm chứng tự động qua `hints`), hoặc `STONE (2)` đang đứng vững, hoặc chạm bẫy. *(Chi tiết cờ `apple_supports_body` và `portal_supports_body` sẽ được tự động hiệu chuẩn bằng Unit Test chạy lại toàn bộ 240 chuỗi `hints` để đạt độ khớp 100% với code C# gốc).*
   - **Điều kiện đỡ của mỗi viên Đá**: Đá tại $(x_s, y_s)$ rơi xuống $(x_s, y_s - 1)$ nếu ô dưới trống. Đá có thể rơi đè lên thân sâu (hoặc bị thân sâu đỡ tuỳ hiệu chuẩn `hints`).
   - **Kiểm tra trong từng nấc rơi ($\Delta y = -1$)**:
     - Nếu bất kỳ đốt nào của sâu chạm Bẫy (`3, 7, 9, 10, 11`) $\rightarrow$ **CHẾT**.
     - Nếu đầu sâu $p_0$ rơi trúng `END (8)` $\rightarrow$ **THẮNG**.
     - Nếu toàn bộ sâu rơi xuống dưới biên bản đồ ($y < -2$) $\rightarrow$ **CHẾT (Rơi vực)**.
     - Nếu viên đá rơi xuống $y < 0$ $\rightarrow$ Xoá viên đá khỏi bản đồ.

4. **Pha 4 — Kiểm tra Kẹt cứng (`Deadlock Detection`)**:
   - Nếu tại trạng thái ổn định mới, cả 4 hướng đi đều bị chặn hợp lệ $\rightarrow$ `terminated = True, is_dead = True` (sâu tự cuộn tròn kẹt cứng).

---

# PHẦN III. THIẾT KẾ MÔ HÌNH AI GIẢI MÀN CHƠI (`RL Solver & Exact Oracle`)

## 6. Bộ Giải Chính xác (`Exact BFS / A* Oracle`)

Trước khi huấn luyện RL và để làm hàm chấm điểm cho bộ sinh Map (PCGRL), hệ thống cần một bộ giải chính xác `bfs_astar_solver.py`:

- **Mã hoá trạng thái gọn nhẹ (Canonical State Hashing - 64-bit Zobrist Hash)**:
  $$\mathcal{S} = \big(\text{tuple}(W),\; \text{bitmask}_{\text{apples}},\; \text{tuple}(\text{sorted}(\text{stones}))\big)$$
  Vì tường, bẫy và cổng hoàn toàn bất biến trong suốt ván chơi, trạng thái động chỉ gồm toạ độ các đốt sâu, bitmask các quả táo còn lại và toạ độ đá!
- **Heuristic dẫn đường cho A\***:
  $$h(\mathcal{S}) = |x_{\text{head}} - x_{\text{end}}| + |y_{\text{head}} - y_{\text{end}}|$$
  *(Admissible vì mỗi bước chủ động đầu sâu dịch ngang tối đa 1 ô; với chiều dọc rơi tự do có thể nhanh hơn nên dùng Manhattan theo trục $x$ + khoảng cách leo lên $\max(0, y_{\text{end}} - y_{\text{head}})$ để đảm bảo 100% admissible, tìm đúng lời giải tối ưu ngắn nhất $L^*$).
- **Ngân sách tìm kiếm (`Node Budget`)**: Cho phép cấu hình `max_expanded_nodes = 50_000` (chạy `< 0.5s` trong Python/Numba) khi dùng bên trong vòng lặp huấn luyện PCGRL, và `500_000` khi xuất lời giải chuẩn cho gói Map cuối cùng.


---

## 7. Đặc tả Mô hình Học Tăng Cường Giải Màn Chơi (`RL Solver Agent`)

### 7.1. Không gian Quan sát (`Observation Space`: Tensor `11 × 12 × 22`)
Thay vì truyền ID số nguyên thô (gây sai lệch thứ tự đại số giữa các loại gạch), trạng thái lưới `12 × 22` được biểu diễn thành Tensor đa kênh `float32` kích thước `(11, 12, 22)` mang trọn vẹn thông tin hình học và cấu trúc nối đốt của sâu:

| Kênh | Tên kênh | Giá trị mã hoá tại ô $(y, x)$ | Ý nghĩa vật lý cho CNN/Attention |
| :-: | :--- | :--- | :--- |
| `0` | `wall_mask` | `{0.0, 1.0}` | Vị trí tường/đất cố định (`id=1`) |
| `1` | `stone_mask` | `{0.0, 1.0}` | Vị trí các viên đá động (`id=2`) |
| `2` | `apple_mask` | `{0.0, 1.0}` | Vị trí các quả táo hiện còn (`id=4`) |
| `3` | `hazard_mask` | `{0.0, 1.0}` | Hợp nhất mọi bẫy chết người (`id ∈ {3, 7, 9, 10, 11}`) |
| `4` | `portal_mask` | `{0.0, 1.0}` | Vị trí cổng hố đen đích (`id=8`) |
| `5` | `worm_head` | `{0.0, 1.0}` | One-hot vị trí đầu sâu $p_0$ |
| `6` | `worm_tail` | `{0.0, 1.0}` | One-hot vị trí đuôi sâu $p_{k-1}$ (cực kỳ quan trọng vì đuôi sắp rút đi!) |
| `7` | `worm_gradient` | $[0.0, 1.0]$ | Với đốt thứ $i \in \{0..k-1\}$, gán giá trị $1.0 - \frac{i}{k}$ $\rightarrow$ CNN nhận biết chính xác thứ tự thân sâu nối từ đầu tới đuôi! |
| `8` | `support_margin` | $[0.0, 1.0]$ | Đánh dấu các đốt đang trực tiếp chạm điểm tựa bên dưới (giúp Agent "cảm nhận" khi nào nhích thêm 1 bước sẽ bị rơi xuống vực) |
| `9` | `coord_x` | $[-1.0, +1.0]$ | Toạ độ chuẩn hoá trục ngang (`CoordConv`) |
| `10` | `coord_y` | $[-1.0, +1.0]$ | Toạ độ chuẩn hoá trục dọc (`CoordConv` — hướng trọng lực) |

### 7.2. Không gian Hành động & Action Masking (`Invalid Action Masking`)
- $\mathcal{A}_{\text{solver}} = \text{Discrete}(4)$ tương ứng `[Trái, Phải, Lên, Xuống]`.
- Hàm `action_masks() -> bool[4]` trả về `False` đối với các hướng đâm trực tiếp vào tường, thân sâu hoặc đá bị kẹt cứng (không dịch chuyển được). Việc loại bỏ hành động vô nghĩa giúp tốc độ hội tụ của `MaskablePPO` tăng gấp **4–6 lần** trên game giải đố lưới.

### 7.3. Thiết kế Hàm Thưởng (`Potential-Based Reward Shaping`)
Tránh hiện tượng *Reward Hacking* (sâu cứ đi qua đi lại để kiếm điểm gần cổng hoặc ăn táo tự sát):

$$R(s, a, s') = R_{\text{terminal}}(s') + R_{\text{event}}(s, s') + \gamma \Phi(s') - \Phi(s) - c_{\text{step}}$$

- **$R_{\text{terminal}}(s')$**:
  - Chạm cổng đích (`Win`): **`+10.0`** + thưởng tiết kiệm bước $+2.0 \times \left(1 - \frac{t}{T_{\max}}\right)$
  - Chạm bẫy / rơi vực / kẹt cứng (`Dead`): **`-5.0`**
- **$R_{\text{event}}(s, s')$**:
  - Ăn 1 quả táo mới: **`+0.5`** (chỉ nhận tối đa 1 lần cho mỗi quả táo)
  - Lặp lại đúng trạng thái băm $\mathcal{S}$ đã đi qua trong cùng ván (`State Revisitation Penalty`): **`-0.15`** (triệt tiêu hành vi đi lặp vòng tròn!)
- **Hàm thế năng $\Phi(s)$**: $\Phi(s) = -0.05 \times \text{BFS\_Wall\_Dist}(p_{\text{head}}, p_{\text{end}})$ (khoảng cách đường đi ngắn nhất trên lưới tránh tường tĩnh). Nhờ dạng $\gamma \Phi(s') - \Phi(s)$, chính sách tối ưu toàn cục không bị méo (Policy Invariance Theorem - Ng et al.).
- **Chi phí mỗi bước $c_{\text{step}}$**: **`0.01`**.

### 7.4. Kiến trúc Mạng Neural (`Spatial ResNet + Multi-Head Self-Attention`)
Game *Hungry Worm* đòi hỏi suy luận **tầm xa (long-range spatial dependency)** — ví dụ nhìn thấy cổng ở góc phải trên nhưng phải quay sang góc trái ăn 2 quả táo và đẩy hòn đá trước. CNN thuần có vùng cảm nhận hẹp sẽ thất bại ở màn khó; do đó kiến trúc chuẩn là **Hybrid ResNet-Transformer**:
1. **Stem Conv**: `Conv2d(11 -> 64, kernel=3, padding=1) + GroupNorm + SiLU`
2. **3 Residual Blocks**: Mỗi block gồm `2 × Conv2d(64 -> 64, 3×3)` giữ nguyên độ phân giải `12 × 22` (264 spatial tokens).
3. **Spatial Transformer Layer**: Gộp `(B, 64, 12, 22) -> (B, 264, 64)`, đi qua `2 × MultiheadAttention(embed_dim=64, num_heads=4)` để mọi ô trên bản đồ nhìn thấy trực tiếp đầu sâu, táo và cổng hố đen.
4. **Actor & Critic Heads**:
   - Pooling tại token vị trí đầu sâu + Global Average/Max Pooling $\rightarrow$ vector `192` chiều.
   - `Actor MLP(192 -> 128 -> 4)` (kết hợp `action_masks`).
   - `Critic MLP(192 -> 128 -> 1)` ước lượng $V(s)$.
   - *Tổng tham số*: ~**320K parameters** — cực nhẹ, train mượt trên cả CPU hoặc 1 GPU phổ thông trong vài chục phút!

### 7.5. Quy trình Huấn luyện 3 Giai đoạn cho Solver
1. **Stage 1 — Behavioral Cloning (Supervised Warm-start, ~5 phút)**:
   - Chạy `hints` của 240 màn gốc qua `WormEnv` $\rightarrow$ thu được `~12.300` cặp trạng thái-hành động chuẩn $(s_t, a_t^*)$.
   - Tăng cường dữ liệu (**Horizontal Flip Augmentation**): Lật ngang bản đồ `x' = 21 - x` và đảo hướng Trái $\leftrightarrow$ Phải (gai xoay `9 ↔ 11`), nhân đôi tập mẫu lên **`~24.600` mẫu chuyên gia**!
   - Huấn luyện Cross-Entropy Loss cho mạng Actor để đạt độ chính xác dự đoán nước đi tiếp theo `~75–85%` ngay trước khi chạy RL.
2. **Stage 2 — Curriculum MaskablePPO trên 240 màn gốc + màn biến thể**:
   - Bắt đầu từ các màn có `len(hints) <= 25`, tăng dần ngưỡng lên `50 -> 100 -> 200` khi tỷ lệ thắng vượt `80%`.
3. **Stage 3 — Co-Evolution (Đồng tiến hoá cùng bộ sinh Map PCGRL)**:
   - Đưa các Map mới do PCGRL sinh ra vào tập train của Solver để mở rộng khả năng tổng quát hoá.


---

# PHẦN IV. THIẾT KẾ MÔ HÌNH SINH MAP MỚI (`PCGRL — RL Level Designer`)

## 8. Kiến trúc Sinh Map Tự động bằng Học Tăng Cường (`PCGRL`)

Sinh Map game giải đố khó hơn sinh Map game đi cảnh thông thường rất nhiều: **chỉ lệch 1 viên gạch là Map trở thành không thể giải được (Unsolvable) hoặc quá dễ (đi thẳng 3 bước tới đích)**. Để giải quyết triệt để bài toán này, hệ thống kết hợp **PCGRL (Procedural Content Generation via RL)** với **Oracle Verification**.

### 8.1. Biểu diễn MDP cho Tác tử Thiết kế Map (`GeneratorEnv`)
Áp dụng mô hình **Iterative Editing PCGRL (Narrow / Wide Representation — Khalifa et al.)**:

- **Khởi tạo Episode (`reset(target_spec)`)**:
  - Khởi tạo từ một khung sàn cơ bản ngẫu nhiên (hoặc lấy một Map mẫu rồi nhiễu hoá 40% số ô) cùng 1 sâu dài 3 đốt đứng vững và 1 cổng đích `END (8)`.
  - Kèm theo vector điều kiện mục tiêu $\mathbf{c} = [L_{\text{target}}, N_{\text{apples}}, N_{\text{stones}}, N_{\text{hazards}}]$ (ví dụ yêu cầu AI thiết kế 1 màn độ khó *Hard*: cần khoảng $L_{\text{target}} = 45$ bước giải, $3$ quả táo, $1$ hòn đá, $4$ cái bẫy).
- **Không gian Hành động của Designer Agent**:
  - **Chế độ `Narrow` (Khuyên dùng cho MVP — không gian hành động nhỏ, học rất nhanh)**:
    - Mỗi bước, con trỏ quét qua toạ độ $(x_t, y_t)$ trong vùng tích cực `x ∈ [2..19], y ∈ [1..10]`.
    - Designer chọn `action ∈ Discrete(8)`:
      `0: Giữ nguyên`, `1: Xoá trống`, `2: Đặt Tường (1)`, `3: Đặt Đá (2)`, `4: Đặt Táo (4)`, `5: Đặt Gai/Bánh răng (3/7)`, `6: Dời Cổng Đích tới đây`, `7: Dời Sâu tới đây`.
  - **Ràng buộc Hợp lệ Tức thì (`Hard Validity Constraints`)**:
    - Luôn duy trì đúng **1 đầu sâu nối thân hợp lệ** đang có điểm tựa đỡ và đúng **1 cổng đích `id=8`**.
    - Gai nhọn (`3, 9, 10, 11`) tự động xoay hướng gắn vào mặt tường liền kề gần nhất; nếu lơ lửng giữa không trung thì tự chuyển thành Bánh răng (`id=7`) giống hệt phong cách thiết kế của 210 màn gốc!

### 8.2. Hàm Thưởng Thiết kế Map (`Designer Reward Function`)
Gọi $\mathcal{M}_t$ là bản đồ tại bước chỉnh sửa $t$. Để không làm chậm quá trình train, bộ giải nhanh `FastAStarOracle(budget=15_000 nodes)` được gọi ở cuối chuỗi chỉnh sửa (hoặc mỗi $K=5$ thay đổi) để tính điểm chất lượng $Q(\mathcal{M})$:

$$Q(\mathcal{M}) = w_{\text{valid}} S_{\text{valid}} + w_{\text{solv}} S_{\text{solv}} + w_{\text{len}} S_{\text{len}} + w_{\text{mech}} S_{\text{mech}} + w_{\text{decep}} S_{\text{decep}} - P_{\text{clutter}}$$

Trong đó 6 thành phần điểm mang ý nghĩa thiết kế game chuyên sâu:

1. **Tính hợp lệ hình thái ($S_{\text{valid}}$)**: Sâu đứng trên mặt đất không rơi chết ngay ở frame 0; số lượng vật thể nằm trong ngưỡng tự nhiên (`bricks <= 65`).
2. **Tính giải được ($S_{\text{solv}}$)**:
   - Nếu `Oracle` tìm ra lời giải $\pi^*$: **`+5.0`**.
   - Nếu không giải được: thưởng dẫn đường mềm dựa trên khoảng cách gần nhất tới cổng mà `Oracle` chạm tới được $-\,0.2 \times d_{\min}(\text{reachable}, p_{\text{end}})$ (giúp RL Designer biết "xây bớt tường chắn đi hoặc thêm 1 quả táo bắc cầu nữa là giải được!", tránh bị *sparse reward* toàn điểm 0).
3. **Khớp Độ dài Lời giải Mục tiêu ($S_{\text{len}}$)**:
   $$S_{\text{len}} = 4.0 \times \exp\left(-\frac{|L^*(\mathcal{M}) - L_{\text{target}}|}{15.0}\right)$$
   Thưởng tối đa khi độ dài đường đi ngắn nhất $L^*(\mathcal{M})$ khớp với độ khó yêu cầu $L_{\text{target}}$ (loại bỏ hoàn toàn các màn đi thẳng 3 bước là thắng).
4. **Thưởng Cơ chế Tương tác Bắt buộc ($S_{\text{mech}}$ — Anti-Fake-Object Rule!)**:
   - Một lỗi kinh điển của AI sinh Map là đặt 5 quả táo và 2 cục đá ở góc bản đồ cho đủ số lượng, nhưng lời giải tối ưu $\pi^*$ **chẳng cần ăn quả táo nào cũng bò thẳng vào cổng được**!
   - Quy tắc vàng: So sánh số táo được ăn trong lời giải tối ưu $A_{\text{used}}(\pi^*)$ với tổng số táo trên bản đồ $A_{\text{total}}$:
     - Thưởng **`+1.5`** cho mỗi quả táo **bắt buộc phải ăn** để qua màn (nếu xoá quả táo đó khỏi Map thì Map trở thành `Unsolvable`!).
     - Phạt **`-1.0`** cho mỗi quả táo/viên đá "trang trí thừa" không tham gia vào lời giải hoặc không tạo bẫy đánh lừa!
5. **Thưởng Độ thú vị / Bẫy Tư duy ($S_{\text{decep}}$ — "Aha-moment" Score)**:
   - Một màn giải đố hay là màn mà nếu người chơi đi tham lam (cứ lao thẳng về phía cổng hoặc ăn đại quả táo ngay trước mắt) thì sẽ **bị kẹt hoặc rơi chết**!
   - Đo bằng tỷ lệ: `DeadEndRatio` trong cây tìm kiếm A* hoặc khi `GreedyAgent` thất bại nhưng `OptimalSolver` thành công $\rightarrow$ thưởng **`+3.0`**!
6. **Phạt Rác thẩm mỹ ($P_{\text{clutter}}$)**:
   - Phạt các khối gạch lơ lửng rời rạc 1 ô không cần thiết, khuyến khích tường liên thông thành cụm nền tảng đẹp mắt như người thiết kế thật.

### 8.3. Bộ Đa dạng hoá Map (`MAP-Elites Quality-Diversity Archive`)
Để tránh việc RL Generator tìm ra 1 mẫu Map điểm cao rồi cứ sinh đi sinh lại 1 kiểu (Mode Collapse), toàn bộ Map hợp lệ sinh ra được lưu vào lưới lưu trữ **2D MAP-Elites Archive**:
- **Trục X (Hành vi 1)**: Độ dài lời giải tối ưu $L^* \in [10..20, 21..35, 36..60, 61..100, 100+]$.
- **Trục Y (Hành vi 2)**: Cơ chế chủ đạo (`Pure-Climbing`, `Apple-Growth-Bridge`, `Stone-Pushing`, `High-Hazard-Precision`, `Hybrid-Puzzle`).
- Mỗi ô trong lưới lưu giữ các Map có điểm chất lượng $Q(\mathcal{M})$ cao nhất nhưng khác biệt cấu trúc (`Hamming / Edit Distance >= 25%`), đảm bảo xuất ra **hàng trăm Map mới không màn nào trùng ý tưởng màn nào**!

