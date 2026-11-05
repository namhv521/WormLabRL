# SPEC-02 — RL environment, training, evaluation

## 1. Mục tiêu và giới hạn tuyên bố

Train **một RL policy** tự chọn action để hoàn thành `worm-grid-v1` trên những map mới. RL **không dùng** BFS certificate làm feature hoặc gọi solver ở inference. Mức thành công đo bằng success rate trên tập test held-out; không mặc định 100%.

**P0:** Gymnasium env, action mask, MaskablePPO training, checkpoint/reproduce, 3 baseline, curriculum, test leakage guard, eval report.

## 2. MDP definition

| Thành phần | Định nghĩa |
|---|---|
| State (ẩn trong core) | `level_static`, `segments_ordered`, `remaining_apples`, `step_count` |
| Action | `Discrete(4)` mapping 0 UP / 1 RIGHT / 2 DOWN / 3 LEFT |
| Transition | Đồng nhất `SPEC-01` |
| Episode end | `terminated` khi `WON` hoặc `LOST` (deadlock); `truncated` khi hết `max_steps` |
| Reward | Định nghĩa ở §4 |
| Observation | Tensor cố định shape `(6,8,8)`, dtype `float32`, range `[0,1]` |
| Mask | `action_masks() -> bool[4]`, cũng trả qua `info["action_mask"]` để debug |

### 2.1. Observation channels (channel-first)

| Kênh | Giá trị trên cell |
|---|---|
| 0 `walls` | 1 nếu WALL, còn lại 0 |
| 1 `worm_order` | Với segment thứ `i` tính từ 0 (head), giá trị `1 - i/10`; nơi không có sâu là 0. Giữ đủ thứ tự thân vì độ dài ≤9. |
| 2 `apples` | 1 nếu còn táo tại cell |
| 3 `exit` | 1 tại cổng |
| 4 `in_bounds` | 1 ở cell board hợp lệ, 0 ở phần padding tới 8×8 |
| 5 `remaining_step_fraction` | `(max_steps-step_count)/max_steps` tại mọi ô trong board, 0 ở vùng padding |

- Dữ liệu thực được padding ở cạnh phải/dưới, không scale board làm thay đổi liên kết cell.
- **Shape chính thức P0: `(6,8,8)`**. Với kênh số bước còn lại, cùng ruleset, policy có đủ thông tin vị trí, thứ tự thân, táo và horizon để suy ra chuyển trạng thái.
- Mọi model/checkpoint phải xác thực observation shape khi load.

### 2.2. Environment protocol

```python
obs, info = env.reset(seed=seed, options={"level_id": "..."})
# obs: np.ndarray (6,8,8) float32
# info: {"level_id": str, "action_mask": [bool, bool, bool, bool], ...}
next_obs, reward, terminated, truncated, info = env.step(action_id)
# info: {"ate_apple": bool, "won": bool, "invalid_action": bool,
#        "step_count": int, "action_mask": list[bool], "level_id": str}
```

- `reset(seed)` dùng RNG của Gymnasium và seed level sampler; nếu `options.level_id` được cung cấp thì load đúng level đó.
- `action_mask` luôn dựa vào **state hiện tại**. Trong ruleset canonical, nếu một legal move khiến mask của state mới thành toàn `False` và chưa thắng, core kết thúc ngay bằng `LOST` (`terminated=True`). Nhờ đó MaskablePPO không bị yêu cầu sample action từ mask rỗng ở state `PLAYING`. Evaluation tính `LOST` là thất bại; không giả vờ có action hợp lệ.
- Kiểm tra tương thích SB3-Contrib mask wrapper trước train; dùng `MaskablePPO` (không nhầm PPO thường).

## 3. Policy và thuật toán

- **Thuật toán P0:** `sb3_contrib.MaskablePPO` với `MlpPolicy` (flatten 6×8×8).
- **Lý do:** discrete action, tránh thử các nước không hợp lệ, dễ triển khai baseline.
- **Hyperparameter khởi tạo (tuning, không phải cam kết tối ưu):** `learning_rate=3e-4`, `n_steps=1024`, `batch_size=256`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.01`, `seed=[11,22,33]`, `total_timesteps ≤2_000_000/run` theo compute budget.
- **Nâng cấp P1:** CNN extractor nhỏ cho grid hoặc imitation warm start từ solution BFS; cần đánh giá ablation riêng để phân biệt RL-only với hybrid.
- Dùng MaskableEvalCallback hoặc custom evaluator có hỗ trợ mask; không dùng callback PPO thường nếu nó bỏ qua action mask.

## 4. Reward design

Cho mỗi action đã thử:

```text
reward = -0.01                          # time/step penalty
       + (1.00 if ate_apple else 0)
       + (10.00 if won else 0)
       + (-0.05 if invalid_action else 0)
```

- `timeout` không cộng reward thắng và không tự thêm penalty khác.
- Với action mask đúng, `invalid_action` chỉ có thể đến từ người chơi/random baseline hoặc lỗi tích hợp, không phải mẫu policy bình thường.
- Thành tích RL **phải báo success rate**, không dùng tổng reward thay thế tỉ lệ giải.
- Reward shaping có sử dụng khoảng cách/solver là thử nghiệm P1 và phải bật/tắt được, có audit về potential reward hacking.

## 5. Data splits & leakage prevention

### 5.1. Dataset manifest

- Sinh các pool tách biệt `train`, `validation`, `test` với namespace seed và config riêng.
- Mỗi map lưu `canonical_hash` gồm board+tường+sâu+táo+exit+ruleset; chuẩn hóa qua đối xứng quay/lật hợp lệ.
- **Không được** có cùng canonical hash trong hai splits; khi va chạm thì bỏ candidate khỏi split đang sinh.
- Validation/test được khóa ở một `dataset_manifest.json` versioned, checkpoint không được train trên hai tập này.
- Evaluation test MVP cần **300 level (100/tier)** đã certified, phân tầng theo `difficulty_score`; nếu thiếu level thì báo thiếu và không công bố metric tier đó.

### 5.2. Curriculum (P0)

| Giai đoạn | Pool training | Quy tắc chuyển |
|---|---|---|
| C0 | Level Easy, ít tường, 1 apple | Validation Easy success ≥80% trong 3 lần đánh giá liên tiếp |
| C1 | Mix 60% Easy, 40% Medium | Validation Medium success ≥60% trong 3 lần đánh giá liên tiếp |
| C2 | Mix 30% Easy, 40% Medium, 30% Hard | Cho tới hết ngân sách training, ghi checkpoint tốt nhất |

- Chuyển giai đoạn theo **validation**, không dùng test.
- Nếu không vượt gate trong ngân sách time/steps, dừng ở tier đã đạt và báo chưa vượt gate.
- Không đổi phân phối của test set theo hiệu năng policy.

## 6. Training CLI contract

Dự kiến CLI (chưa có code trong bộ tài liệu):

```bash
python -m ml.train --config configs/train/maskable_ppo.yaml --seed 11
python -m ml.evaluate --checkpoint artifacts/models/<model_id>.zip --dataset data/manifests/test-v1.json
python -m ml.baselines --dataset data/manifests/test-v1.json --agents random,greedy,bfs
```

Mỗi run xuất:

- `run_manifest.json`: `run_id`, `git_commit`, `ruleset_id`, `config_hash`, `dataset_hash`, `seed`, versions, start/end, device, steps.
- `checkpoints/`: best validation + final checkpoint với sha256.
- `metrics.csv` hoặc JSONL: timesteps, average reward, success từng tier, episode length, invalid action rate.
- `evaluation.json`: model ID, tier, số episode, successes, success rate, average steps only over successful episodes, failure classes.

Model registry **không cho phép** dùng checkpoint có `ruleset_id/observation_shape/action_mapping` không khớp với env.

## 7. Baselines, inference & evaluation

### 7.1. Baseline definitions

- **Random valid:** chọn ngẫu nhiên đều trong action mask hợp lệ; nếu mask rỗng thì failure.
- **Greedy Manhattan:** chọn action hợp lệ làm giảm khoảng cách Manhattan từ đầu sâu tới táo gần nhất; khi hết táo đi tới exit, tie-break theo thứ tự action ID; không xem trước thân/tương lai ngoài 1 bước.
- **BFS oracle:** exact solver tìm path tối ưu dưới luật core, chỉ dùng khi tạo data/chấm benchmark/hiển thị *oracle*, không hỗ trợ RL inference.

### 7.2. Evaluation protocol

- Test mỗi checkpoint từ `reset` từng level trên test manifest. Inference deterministic (`deterministic=True`), action mask phải được áp dụng.
- Thành công khi `terminated=True` và `info.won=True` trước/đúng `max_steps`.
- Báo metrics theo tier, overall, `success_count/episode_count`, `median_steps_solved`, `p95_inference_latency_ms`, `timeouts`, `deadlocks`.
- 3 independent training seeds 11/22/33, từng seed chạy trên cùng 300 test levels, báo mean, std, từng run; giới hạn kết luận theo sample.
- 95% confidence interval bằng Wilson interval cho tỷ lệ ở từng run (optional); nếu thiếu hard data phải để `N/A`, không thay bằng phỏng đoán.
- Cùng test batch dùng cho 3 baseline; random baseline chạy `3 eval rng seeds` để tránh phụ thuộc một đường đi ngẫu nhiên.

## 8. Training acceptance criteria

| AC | Điều kiện |
|---|---|
| AC-RL-01 | `check_env` tương thích Gymnasium; 100 random resets tạo tensor `(6,8,8)` hợp lệ và mask length 4. |
| AC-RL-02 | Mask luôn nhất quán với valid actions của core trên ≥10k state lấy mẫu. |
| AC-RL-03 | Cùng level + action list cho reward/transition deterministic. |
| AC-RL-04 | CLI train tối thiểu 1 run, lưu và load checkpoint để inference trong process mới. |
| AC-RL-05 | 3 run seeds được log và không truy cập test set trong quá trình train/tune. |
| AC-RL-06 | Report RL có thành tích thực so với Random/Greedy/BFS trên cùng test map, không tự dùng BFS khi RL fail. |
| AC-RL-07 | Map deadlock/mask all-false được xử lý bằng failure rõ ràng, không crash loop sampling. |

## 9. Rủi ro nghiên cứu

- RL có thể yếu ở map dài/hẹp với reward thưa dù BFS luôn tìm được path.
- Tăng hard score không nhất thiết tăng độ khó dự đoán bởi policy; cần lập biểu đồ success vs `difficulty_score`.
- Map bị lọc bởi BFS budget tạo selection bias; báo miền hỗ trợ và acceptance rate.
- GPU, nondeterminism backend và phiên bản thư viện có thể khiến kết quả training lệch; engine/dataset phải tái lập hoàn toàn, training ghi seeds và môi trường phần mềm.

## 10. Phụ thuộc

- Luật chuẩn: [`01-game-rules.md`](01-game-rules.md).
- Solver/data: [`03-generator-solver.md`](03-generator-solver.md).
- Contract/metrics: [`05-api-and-data.md`](05-api-and-data.md), [`06-tests-and-acceptance.md`](06-tests-and-acceptance.md).
