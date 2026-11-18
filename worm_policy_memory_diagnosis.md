# Trí nhớ của mô hình WormLabRL và tỷ lệ tự thắng

## Mô hình nhớ gì?

Checkpoint `worm_run/policy.pt` lưu trọng số của một MLP nhỏ. Ở mỗi bước, mô hình chấm bốn hướng đi từ 23 đặc trưng trên mỗi hướng. Các đặc trưng mô tả trạng thái hiện tại và kết quả dự đoán của một bước trong bộ giả lập. Mô hình dùng cùng trọng số trên các map. Nó không có trạng thái ẩn truyền qua thời gian, bộ nhớ đọc/ghi trong khi chơi hay cơ chế xây dựng kế hoạch nhiều bước. [Mã policy](train_worm.py#L47)

Vòng chơi có bộ nhớ ngắn hạn bên ngoài mô hình. `visited` lưu trạng thái đã đi qua trong lượt để nhận diện vòng lặp. Khi huấn luyện, `penalties` lưu một số cặp trạng thái–hành động gây lỗi giữa các lượt thử của cùng map. `visited` được tạo lại mỗi lượt, còn `penalties` được tạo lại khi sang map. Bài kiểm tra tự thắng dùng một lượt tham lam mỗi map, không dùng bảng phạt học khi thử lại. Bộ nhớ mẫu huấn luyện giúp cập nhật trọng số, rồi được bỏ khi kết thúc tiến trình. [Vòng chơi](train_worm.py#L105) · [Huấn luyện](train_worm.py#L322)

## Tại sao khớp 70,6% nhưng chỉ thắng 7/40 map?

Độ khớp 70,6% đo trên 2.301 **trạng thái trong lời giải mẫu** của 40 map giữ lại. Ở các trạng thái đó, mô hình được đưa về đúng vị trí trong lời giải trước mỗi dự đoán. Tỷ lệ tự thắng yêu cầu mô hình tự tạo ra toàn bộ chuỗi nước đi. Khi mô hình rẽ sang nhánh khác, nó phải tiếp tục từ trạng thái do chính nó tạo ra. Dữ liệu bắt chước chủ yếu đến từ các đường giải đã xác minh, còn dữ liệu sau khi rẽ sai chủ yếu nhận tín hiệu thưởng/phạt và có ít chỉ dẫn cách quay về một đường tới đích.

Ví dụ ở **A10**, hai bước đầu trùng lời giải. Ở bước 3, từ `(7, 7)` mô hình đi **xuống**, trong khi lời giải đi **phải**. Cả hai hướng đều hợp lệ ngay tại bước đó. Lượt của mô hình kết thúc vì lặp trạng thái sau 35 bước. Ví dụ này cho thấy kiểm tra hợp lệ trước khi đi chỉ xử lý được một phần vấn đề. Hai hướng đi hợp lệ có thể dẫn đến khả năng hoàn thành rất khác nhau.

## Thử nghiệm với cùng checkpoint

Mỗi map chạy một lượt tham lam, tối đa 250 bước. Với biến thể dùng quy tắc chặn, nếu còn ít nhất một hướng được phép thì loại các hướng vi phạm trước khi chọn. Nếu cả bốn bị loại, giữ cách chọn ban đầu. Thử nghiệm dùng lại 40 map đã báo cáo, mang tính chẩn đoán.

| Cách chọn nước đi | Số map thắng | Quan sát |
| --- | ---: | --- |
| Checkpoint gốc | 7/40 | 12 lỗi vướng thân, 7 rơi, 5 lặp, 4 vướng tường, 2 vướng đá, 2 chạm bẫy và 1 hết bước |
| Chặn tường, thân và đá | 7/40 | 22 lượt kết thúc vì rơi |
| Chặn thêm nước gây chết ngay hoặc quay lại trạng thái đã đi | 7/40 | Tập map thắng giữ nguyên |
| Checkpoint kết hợp tìm kiếm nhiều bước | 30/40 | Ngân sách tối đa 4.000 trạng thái tìm kiếm mỗi map |

Con số 30/40 phản ánh cả mô hình, bộ giả lập và tìm kiếm. Cùng ngân sách tìm kiếm, mô hình chưa train cũng giải được 28/40. Vì vậy, phần lớn số map tăng thêm so với lượt tự chơi hiện đến từ việc mở nhiều nhánh và kiểm tra hậu quả nhiều bước. Số liệu gốc nằm trong [metrics.json](worm_run/metrics.json).

## Phần còn thiếu so với spec

Bản hiện tại là baseline MLP huấn luyện theo lời giải mẫu và thưởng/phạt. Nó chưa triển khai ResNet + Self-Attention và MaskablePPO ở [mục 7.4–7.5](SPEC_PRD_HUNGRY_WORM_RL_PCG.md#74-kiến-trúc-mạng-neural-spatial-resnet--multi-head-self-attention). Self-Attention trong spec là chú ý theo không gian của bản đồ. Trí nhớ xuyên bước sẽ cần cơ chế riêng, chẳng hạn trạng thái hồi quy hoặc lưu các trạng thái và nhánh đã thử ở tầng solver.

Nếu mục tiêu là tăng tỷ lệ tự thắng, cần đo riêng khả năng chọn nhánh và khả năng hồi phục. Một kiểm chứng tiếp theo là dùng bộ giải tạo nước đi phục hồi từ những trạng thái chính mô hình thường rẽ sai, huấn luyện lại, rồi chạy đúng chế độ một lượt tham lam trên các map giữ lại. Tỷ lệ thắng với tìm kiếm vẫn được báo riêng.

