# WormLab RL — Bộ tài liệu yêu cầu & đặc tả

**Phiên bản tài liệu:** 0.1.0  
**Ngày:** 2026-11-05  
**Trạng thái:** Proposed / cần xác nhận trước khi phát triển  
**Ngôn ngữ:** Tiếng Việt

## 1. Ý tưởng được đề xuất

**WormLab RL** là nền tảng nghiên cứu và trải nghiệm puzzle casual dạng lưới: điều khiển một con sâu ăn hết táo rồi đi đến cổng. Dự án kết hợp:

1. **Game engine xác định (deterministic)**, có thể chơi bằng tay trên web.
2. **RL agent** học giải các map chưa gặp trong huấn luyện.
3. **Exact solver** tìm và xác minh đường đi; là nguồn chứng nhận map hợp lệ, không đánh đồng với năng lực RL.
4. **Procedural level generator** sinh map mới theo mức độ khó đã đo bằng solver; có thể sinh nối tiếp theo yêu cầu.
5. **Evaluation dashboard** so sánh agent, baseline và bộ giải.

Đây là game và bộ luật **độc lập**, chỉ lấy cảm hứng từ thể loại puzzle sâu/ăn táo/cổng thoát. Không mặc định tái hiện chính xác mọi cơ chế, tài nguyên hoặc toàn bộ màn trong game thương mại Hungry Worm. Muốn kiểm thử trên bộ màn của trò chơi cụ thể phải có dữ liệu map và một bản mô tả luật tương ứng.

## 2. Tài liệu trong bộ này

| File | Nội dung | Vai trò |
|---|---|---|
| [`PRD.md`](PRD.md) | Mục tiêu, người dùng, scope MVP, yêu cầu, KPI, timeline, rủi ro | Product owner / PM |
| [`spec/01-game-rules.md`](spec/01-game-rules.md) | Quy tắc game, chuyển trạng thái, thắng/thua, sample map | Game/engine developer |
| [`spec/02-rl-training.md`](spec/02-rl-training.md) | RL environment, reward, policy, curriculum, benchmark | ML engineer |
| [`spec/03-generator-solver.md`](spec/03-generator-solver.md) | Sinh map, BFS, chứng nhận, xếp độ khó, khử trùng lặp | Algorithm engineer |
| [`spec/04-system-architecture.md`](spec/04-system-architecture.md) | Module, luồng dữ liệu, cấu trúc repo, vận hành | Tech lead |
| [`spec/05-api-and-data.md`](spec/05-api-and-data.md) | Hợp đồng REST, schema map, phiên chơi, mã lỗi | Backend/frontend |
| [`spec/06-tests-and-acceptance.md`](spec/06-tests-and-acceptance.md) | Test plan, traceability, tiêu chí nghiệm thu | QA / tech lead |
| [`spec/07-ux.md`](spec/07-ux.md) | Hành trình và trạng thái giao diện web | Product designer / frontend |
| [`PLAN.md`](PLAN.md) | Roadmap tám tuần, thứ tự triển khai, exit gate | Project team |

## 3. Các quyết định nền tảng (MVP)

- Bàn cờ hình vuông **5×5 đến 8×8**, tường tĩnh, một con sâu, 1–4 quả táo và một cổng thoát.
- Ăn táo làm sâu dài thêm một ô. Thắng khi **đã ăn hết táo và đưa đầu sâu vào cổng**.
- Game MVP **không có trọng lực, bẫy di động, khối đẩy hay nhiều con sâu**.
- RL: **MaskablePPO** cho action space 4 hướng; có baseline Random/Greedy và BFS.
- Generator: **constructive candidate + exact BFS certification + độ khó từ số bước tối ưu/độ phức tạp tìm kiếm**.
- Web: xem/chơi level, yêu cầu sinh level, xem agent giải và thống kê; huấn luyện chạy qua CLI ở backend.
- Cụm từ "map vô hạn" được hiểu là **có thể tạo liên tục khi người dùng yêu cầu**, không phải bảo đảm vô hạn map phân biệt trên tập luật và kích thước hữu hạn.

## 4. Định nghĩa thành công

- Mọi map **được xuất bản** đều có một chuỗi hành động đã được engine replay và xác minh thắng.
- RL được đánh giá trên map **chưa thấy trong train**, tách tập bằng hash canonical; không chỉ đánh giá map đã thuộc lòng.
- Độ khó có thước đo tái lập; tăng theo từng band **theo proxy của solver**, không tuyên bố tương đương độ khó cảm nhận của con người.
- Có bản demo chạy cục bộ từ đầu đến cuối: generate → verify → play → watch agent → evaluate.

## 5. Quy ước độ ưu tiên và tài liệu

- **P0:** bắt buộc để nghiệm thu MVP.
- **P1:** có giá trị cao nhưng có thể chuyển sang bản tiếp theo.
- **P2:** nghiên cứu/roadmap.
- Mã `FR-*`, `NFR-*`, `US-*`, `AC-*` là định danh tham chiếu liên tài liệu.
- Các ngưỡng KPI là **mục tiêu đề xuất**, không phải kết quả benchmark đã đo.
- Khi tài liệu có bất đồng: `01-game-rules.md` là nguồn chân lý luật chơi; `05-api-and-data.md` là nguồn chân lý giao tiếp; `PRD.md` là nguồn chân lý phạm vi/ưu tiên.

## 6. Nguồn tham khảo kỹ thuật

- Gymnasium custom environments: https://gymnasium.farama.org/main/api/env/
- SB3-Contrib / MaskablePPO: https://sb3-contrib.readthedocs.io/en/master/guide/algos.html
- PCGRL (hướng nghiên cứu hậu MVP): https://doi.org/10.1609/aiide.v16i1.7416
- Ví dụ thể loại tham khảo, **không phải đặc tả ràng buộc**: https://play.google.com/store/apps/details?id=com.hungryworm.logicpuzzle
