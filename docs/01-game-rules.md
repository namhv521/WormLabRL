# SPEC-01 — Gameplay, state machine & ruleset `worm-grid-v1`

## 1. Mục đích và phạm vi

Đặc tả chính thức của **game độc lập WormLab RL**, là nguồn sự thật cho Python engine, Gymnasium env, BFS verifier và backend. Mọi bước logic phải đi qua **cùng một hàm chuyển trạng thái**, không tự viết một luật khác ở frontend.

**Phạm vi P0:** 1 con sâu, tường tĩnh, táo, cổng, 4 hướng, không trọng lực, không bẫy/khối đẩy, bàn vuông 5–8 ô mỗi cạnh.

## 2. Tọa độ & cấu trúc dữ liệu

- Trục tọa độ: `x` tăng sang phải; `y` tăng xuống dưới; gốc `(0,0)` ở góc trên bên trái.
- Board `width = height ∈ [5,8]` ở MVP. Một cell thuộc một trong các terrain: `EMPTY`, `WALL`, `EXIT`; táo và sâu là entities nằm trên cell nền (không tồn tại trên tường).
- `worm.segments`: mảng có thứ tự **đầu → đuôi** `[(hx,hy), ..., (tx,ty)]`, ban đầu có 2–5 cell, 4-neighbor liền nhau, không tự chồng ô.
- `apples`: 1–4 tọa độ duy nhất, không chồng với tường/sâu/exit lúc khởi tạo. Sau khi ăn, số lượng giảm; thân dài thêm 1.
- `exit`: đúng 1 tọa độ, không nằm trên tường; cổng có thể đi qua cả khi chưa ăn hết táo, nhưng chưa tính là thắng.
- Độ dài sâu tối đa của map MVP: 9 ô (5 ban đầu + 4 táo).
- `state`: `(segments_ordered, remaining_apples, step_count, status)`; `walls/exit/board` là immutable level definition.
- `max_steps = 4 × width × height + 20`; giới hạn thời gian theo **số action được thử**, không phải thời gian thực.

## 3. Action space & transition

| Action ID trong Gym | Tên API | Delta (dx,dy) |
|---|---|---|
| 0 | `UP` | `(0,-1)` |
| 1 | `RIGHT` | `(1,0)` |
| 2 | `DOWN` | `(0,+1)` |
| 3 | `LEFT` | `(-1,0)` |

### 3.1. Kiểm tra hợp lệ (theo thứ tự)

Từ đầu sâu hiện tại `h`, tính `new_head = h + delta`. Action **không hợp lệ** nếu bất kỳ điều kiện nào đúng:

1. Đi ra ngoài phạm vi board.
2. Đi vào tường.
3. Đổi hướng 180° trực tiếp vào đốt thứ hai (cấm quay ngược đối với thân có ít nhất 2 ô).
4. Va vào một cell đang bị thân chiếm **ngoại trừ ô đuôi cuối cùng sắp rời đi** ở action không ăn táo. Khi ăn táo, đuôi không rời đi nên mọi cell thân đều cấm.

`valid_action_mask` gồm 4 boolean theo thứ tự action IDs. Quy tắc mask là **cùng một logic** engine dùng để xác nhận bước đi.

### 3.2. Action hợp lệ

1. Tăng `step_count` lên 1.
2. Nếu `new_head` có táo: cập nhật `segments' = [new_head] + segments`, xóa táo đã ăn.
3. Nếu không có táo: `segments' = [new_head] + segments[:-1]`.
4. Nếu `new_head == exit` **và** sau cập nhật không còn táo: `status = WON`, `terminated = true`.
5. Nếu chưa thắng và không còn bất kỳ action hợp lệ nào trong trạng thái mới: `status = LOST` (deadlock), `terminated = true`.
6. Nếu chưa thắng, chưa deadlock và `step_count ≥ max_steps`: `status = TIMEOUT`, `truncated = true`.
7. Còn lại `status = PLAYING`, `terminated = truncated = false`.

### 3.3. Action không hợp lệ

- Board, thân, táo **không đổi**; `step_count += 1` để tránh spam action vô hạn.
- `info.invalid_action = true`, áp dụng penalty; nếu đạt `max_steps`, `truncated = true`.
- Action sai không trực tiếp tạo `LOST` vì hình học không đổi; episode có thể `TIMEOUT` khi chạm ngân sách bước.
- Nếu episode đã `terminated` hoặc `truncated`, phải `reset()` trước khi `step()` lại.
- Action không hợp lệ ở state `PLAYING` không làm thay đổi legality mask, vì hình học không đổi; vì vậy không sinh `LOST` mới, chỉ có thể timeout.

### 3.4. Reward do RL env áp dụng

Game core trả sự kiện `ate_apple`, `won`, `invalid_action`, `timeout`; công thức reward nằm trong `02-rl-training.md`. Game core không tự tích lũy học máy.

## 4. Terminal conditions & các trạng thái khó

| Tình huống | Kết quả |
|---|---|
| Hết táo và đầu đi tới exit | `WON` (`terminated=true`) |
| Đứng ở exit nhưng còn táo | Không thắng, có thể tiếp tục di chuyển |
| Ăn táo cuối ở ô kề exit | Chưa thắng tới khi đi vào exit |
| Hết `max_steps` khi chưa thắng | `TIMEOUT` (`truncated=true`) |
| Không còn action hợp lệ sau một bước hợp lệ | `LOST` (deadlock), `terminated=true`; không để policy nhận mask toàn false ở state PLAYING |
| Solver xác định không còn đường tới đích từ map ban đầu | Level không được publish |

**Lưu ý:** để chứng nhận level khả giải, solver tìm đường từ trạng thái ban đầu trên **không gian trạng thái không giới hạn step_count**; nếu tìm được solution dài hơn `max_steps` thì map vẫn **không được publish** ở MVP. Vì vậy solver bắt buộc đối chiếu độ dài lời giải `≤ max_steps`. Phép tính độ khó sử dụng BFS trên trạng thái logic, không tính action sai hay chờ đợi.

## 5. Map hợp lệ ban đầu

- `width==height` và trong `[5,8]`.
- Mỗi tọa độ nguyên trong biên; không trùng các thực thể bắt buộc khác loại.
- Ít nhất 1 apple và đúng 1 exit; có 2–5 segments, 4-neighbor từng cặp, không lặp.
- Tường chiếm tối đa 25% cell, nhưng đây là ràng buộc **generator**; map nhập tay P1 có thể có tường dày hơn sau khi được valid.
- Ít nhất 1 action hợp lệ ở trạng thái ban đầu.
- Tất cả ruleset fields được kiểm tra khi khởi tạo; input sai trả `INVALID_LEVEL` chứ không cố sửa im lặng.

## 6. Example level xác định, có solution thật

```json
{
  "schema_version": "1.0",
  "ruleset_id": "worm-grid-v1",
  "level_id": "demo-5x5-001",
  "grid": {"width": 5, "height": 5},
  "walls": [],
  "worm": {"segments": [[1,2], [1,3], [1,4]]},
  "apples": [[3,2]],
  "exit": [4,2],
  "seed": 42,
  "generator_version": "manual-fixture-v1"
}
```

**Replay kiểm tra:** `RIGHT, RIGHT, RIGHT`.

- Bước 1: đầu `(2,2)`, không ăn, đuôi rời `(1,4)`.
- Bước 2: đầu `(3,2)`, ăn 1 táo, thân dài từ 3 lên 4.
- Bước 3: đầu `(4,2)` trùng exit; không còn táo ⇒ `WON`.

Đây là đường hợp lệ; với board không tường và chỉ 1 táo ở `(3,2)`, nó đồng thời là lời giải ngắn nhất dài 3 bước.

## 7. Edge cases phải có unit test

- Va tường/biên; đi vào thân giữa; đầu đâm vào đuôi sẽ rời đi khi không ăn táo (hợp lệ nếu không quay 180°).
- Hành vi vào ô đuôi: được phép nếu ô đuôi rời đi và không vi phạm cấm quay ngược; không thể đồng thời ăn táo vì táo không chồng body.
- Exit được đi qua trước khi ăn táo cuối, không kết thúc sớm.
- Còn 1 action trước max_steps; thắng ở bước cuối phải báo `WON` ưu tiên `TIMEOUT`.
- Hướng quay ngược bị cấm kể cả khi thân chỉ dài 2.
- Sau thắng/deadlock/timeout, gọi step phải raise error có mã `EPISODE_FINISHED`.
- Reset cùng level và seed phải trả cùng state ban đầu, không tự thay đổi level.

## 8. Out of scope / versioning

Cơ chế **trọng lực, đá đẩy, bẫy, nhiều worm, công tắc** phải tạo ruleset mới (`worm-grid-v2` hoặc rule plugin riêng), kèm bộ unit test, solver và benchmark mới. Không dùng level v2 với model v1 mà không có bước chuyển đổi tương thích được kiểm chứng.
