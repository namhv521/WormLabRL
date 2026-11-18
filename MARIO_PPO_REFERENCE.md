# Đối chiếu Super Mario Bros PPO với WormLabRL

## Repo Mario huấn luyện như thế nào

Repo chọn **một world và một stage mỗi lần train**. Tất cả môi trường song song trong lần chạy đó đều dùng cùng màn. Script test tải checkpoint có tên tương ứng với world và stage, chọn hành động có xác suất cao nhất và xuất video. Thư mục checkpoint có 31 tệp, còn README ghi nhận 31/32 màn đã giải. Con số này mô tả độ phủ của các checkpoint chuyên từng màn. [Train](https://github.com/vietnh1009/Super-mario-bros-PPO-pytorch/blob/ab4248d715346c6adc33c2157455e2b98c130bcc/train.py#L18-L65) · [Test](https://github.com/vietnh1009/Super-mario-bros-PPO-pytorch/blob/ab4248d715346c6adc33c2157455e2b98c130bcc/test.py#L24-L57) · [Các checkpoint](https://github.com/vietnh1009/Super-mario-bros-PPO-pytorch/tree/ab4248d715346c6adc33c2157455e2b98c130bcc/trained_models)

| Thành phần | Mã Mario đang làm |
| --- | --- |
| State | Đổi ảnh RGB thành grayscale 84 × 84, chuẩn hóa về khoảng 0–1, xếp 4 khung hình. |
| Action | Chọn nhóm nút tay cầm `right`, `simple` hoặc `complex`. Wrapper mặc định giữ một hành động qua tối đa bốn frame emulator hoặc tới khi lượt kết thúc. |
| Reward | Lấy reward từ game, cộng chênh lệch điểm chia 40, cộng 50 khi chạm cờ hoặc trừ 50 khi phiên kết thúc mà chưa chạm cờ. Toàn bộ reward được chia 10 trước khi trả về. Màn 4-4 và 7-4 có luật phạt và kết thúc lượt dựa trên tọa độ. |
| Policy | Bốn lớp convolution, đặc trưng chung 512 chiều, hai đầu ra actor và critic. Actor lấy mẫu categorical khi train và chọn argmax khi test. |
| Rollout | Mặc định 8 môi trường × 512 bước = 4.096 quyết định trước mỗi đợt cập nhật. Dùng GAE, PPO clipping 0,2, Huber loss cho critic, entropy 0,01 và gradient clipping 0,5. |
| Cập nhật | 10 epoch qua mỗi batch rollout, chia 16 minibatch. Adam với learning rate mặc định 1e-4. Hệ số discount 0,9 và GAE lambda 1,0. |
| Dữ liệu | Policy lấy mẫu hành động và học từ lượt chơi mới do emulator sinh ra. Vòng PPO không nạp demo hay checkpoint để warm-start. |

[Nguồn state, action và reward](https://github.com/vietnh1009/Super-mario-bros-PPO-pytorch/blob/ab4248d715346c6adc33c2157455e2b98c130bcc/src/env.py#L27-L135) · [Mạng actor–critic](https://github.com/vietnh1009/Super-mario-bros-PPO-pytorch/blob/ab4248d715346c6adc33c2157455e2b98c130bcc/src/model.py#L7-L31) · [Vòng cập nhật PPO](https://github.com/vietnh1009/Super-mario-bros-PPO-pytorch/blob/ab4248d715346c6adc33c2157455e2b98c130bcc/train.py#L105-L143)

## Điểm cần chú ý khi dùng làm tài liệu tham khảo

- Tác giả ghi nhận đã thử nhiều learning rate giữa các màn. Màn 1-3 được báo cáo thành công với `7e-5` sau nhiều lần thử. Không thể lấy số lần retry đó làm mức compute cố định cho WormLabRL. [README](https://github.com/vietnh1009/Super-mario-bros-PPO-pytorch/blob/ab4248d715346c6adc33c2157455e2b98c130bcc/README.md)
- `--num_global_steps=5e6` xuất hiện trong tham số, nhưng vòng train dùng `while True`. Mốc được đọc trong tiến trình đánh giá để reset. Lệnh lưu checkpoint định kỳ và lưu khi chạm cờ cũng đang được comment trong snapshot này. Vì vậy không có căn cứ lấy 5 triệu làm số bước chạy đã tạo ra từng checkpoint. [Train](https://github.com/vietnh1009/Super-mario-bros-PPO-pytorch/blob/ab4248d715346c6adc33c2157455e2b98c130bcc/train.py#L24-L72) · [Tiến trình đánh giá](https://github.com/vietnh1009/Super-mario-bros-PPO-pytorch/blob/ab4248d715346c6adc33c2157455e2b98c130bcc/src/process.py#L27-L50)
- Trong GAE của Mario, `done` chặn giá trị bootstrap ở TD residual, còn phép tích lũy GAE ở dòng trước không nhân với `1-done`. Giữ cách WormLabRL đã dùng là chặn cả bootstrap lẫn GAE qua biên episode, vì các slot được reset và có thể chuyển map. [GAE trong Mario](https://github.com/vietnh1009/Super-mario-bros-PPO-pytorch/blob/ab4248d715346c6adc33c2157455e2b98c130bcc/train.py#L105-L120) · [GAE trong WormLabRL](train_worm_ppo.py#L125)

## Đối chiếu WormLabRL

WormLabRL hiện học một policy dùng chung trên 141 map PPO và dùng 23 map để chọn checkpoint. Mặc định có 32 slot × 64 bước = 2.048 quyết định mỗi rollout, 3 epoch PPO và có thêm loss bắt chước. Thống kê 11/40 của PPO vòng 4 đo một checkpoint trên 40 map giữ lại và cho chạy một lượt greedy từ đầu từng map. Đây là phép đo khác với dùng checkpoint dành riêng cho mỗi màn. [Vòng PPO hiện tại](train_worm_ppo.py#L142) · [Số liệu PPO](worm_ppo_run/metrics.json)

Ý tưởng áp dụng trực tiếp là **mỗi worker lặp lại cùng map khi kiểm tra khả năng học một bài cụ thể**. WormLabRL dùng tensor lưới và thứ tự đốt sâu từ simulator. Giữ một lệnh cho mỗi lượt puzzle, vì mỗi lần di chuyển làm thay đổi trạng thái và có thể kích hoạt trọng lực. Repo Mario sử dụng ảnh và frame skip phục vụ emulator. Reward phụ thuộc tọa độ biết trước của hai màn Mario cũng không thích hợp làm đặc trưng cho phép đo map mới.

## Pilot đề xuất

1. Chốt trước 3 map đã xác minh trong tập train mà PPO vòng 4 từng thua. Chọn một ca rơi, một ca lặp và một ca khó hơn nếu có đủ cả ba nhóm. Mỗi job dùng 8 slot trên cùng map, reset từ đầu map.
2. Chạy thử `1e-4` và `2.5e-4` với cùng số quyết định môi trường. Dùng checkpoint PPO vòng 4 cho nhóm tiếp tục train. Có thể thêm một cấu hình khởi tạo mới với đúng ngân sách để biết cần bao nhiêu bước nếu học từ đầu. Giữ kiến trúc, luật, reward và giới hạn 250 bước cho phép so sánh đầu tiên.
3. Log số bước môi trường thật, số lượt thắng, kết quả greedy theo checkpoint và đường đi đã kiểm chứng. So pilot chuyên từng map với PPO dùng chung trên chính ba map đó.
4. Nếu policy học được từng map ổn định, thử chia curriculum thành nhóm map nhỏ trước khi train lại policy dùng chung. Nghiệm thu khả năng tổng quát bằng một checkpoint và danh sách map giữ riêng, đồng thời báo số map từng thắng bị mất khi chuyển curriculum.

Phần phân tích dựa trên mã nguồn Mario ở commit `ab4248d715346c6adc33c2157455e2b98c130bcc`. Kết quả 31/32 là kết quả tác giả báo cáo. Pilot WormLabRL nêu trên là thiết kế thí nghiệm.