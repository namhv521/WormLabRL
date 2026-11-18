# Thiết kế WormLabRL lấy cảm hứng từ SuperMimic

## Quyết định thiết kế

Dùng một policy chung cho nhiều map. Bộ huấn luyện thu thập quỹ đạo do chính policy đi, dùng simulator để nhìn trước nhiều bước từ các trạng thái ấy, rồi dạy policy những hướng đã được kiểm chứng có thể tới cổng. Ở chế độ tự giải, policy nhận bản đồ hiện tại, tự chọn bốn hướng di chuyển và không nhận lời giải, đường đi mẫu hay kết quả tìm kiếm nhiều bước.

SuperMimic thu thập chuyển trạng thái từ simulator, học một world model khả vi, rồi cập nhật policy qua các rollout ngắn trong world model với critic ước lượng phần còn lại. Bài gốc điều khiển robot bằng hành động liên tục. WormLabRL có bốn hành động rời rạc và một simulator tất định dùng được trực tiếp. Thiết kế ưu tiên dùng simulator làm mô hình chuyển trạng thái và chuyển kết quả tìm kiếm thành nhãn học. World model neural là thí nghiệm riêng nếu phép đo cho thấy simulator là nút thắt. Đây là bản chuyển thể model-based cho bài toán lưới. [SuperMimic, mục 3.2–3.3](https://arxiv.org/pdf/2610.28480)

## Bằng chứng trước khi triển khai

PPO hiện tại đạt 11/40 map giữ lại ở cả vòng 4, 5 và 6, cùng đúng 11 map. Trên 23 map validation đã dùng trong vòng PPO, thử checkpoint vòng 4 với bộ tìm kiếm có sẵn:

| Giới hạn trạng thái tìm kiếm mỗi map | Policy tự thắng | Số map thắng tổng cộng |
| --- | ---: | ---: |
| 0 | 6/23 | 6/23 |
| 256 | 6/23 | 13/23 |
| 1.024 | 6/23 | 16/23 |
| 4.000 | 6/23 | 18/23 |

Các đường tìm được đều được phát lại theo từng bước bằng simulator và kết thúc ở cổng. Kết quả đo khả năng tạo dữ liệu dạy. Mức tăng của policy sau khi học từ dữ liệu đó còn phải đo riêng. [Tìm kiếm hiện tại](train_worm.py#L163) · [Metric vòng 5–6](worm_ppo_extension/metrics.json)

## State, observation và action

- **Trạng thái simulator** gồm tường, bẫy và cổng của map cùng thứ tự đốt sâu từ đầu đến đuôi, vị trí táo và đá còn lại. Mã simulator giữ tọa độ nguyên kể cả các đốt tạm vượt biên vẽ. Bộ đếm bước thuộc vòng chơi. Mỗi bước xử lý va chạm, đẩy đá, tăng độ dài và trọng lực theo đúng thứ tự của `WormMap.step`.
- **Quan sát policy** dùng tensor `11 × 12 × 22` từ PRD. Các kênh gồm tường, đá, táo, bẫy, cổng, đầu, đuôi, thứ tự đốt, điểm tựa và hai kênh tọa độ. Thêm `[x, y, thứ_tự, loại, đang_ngoài_lưới]` cho thực thể ngoài vùng vẽ, gộp bằng MLP và pooling có mask. Nối số bước còn lại đã chuẩn hóa với vector map. Với map ngoài kích thước đã hỗ trợ, báo rõ lỗi kích thước đầu vào.
- **Mục tiêu** là đưa đầu sâu vào một cổng và sống sót. Cổng là kênh điều kiện mục tiêu. Lời giải chỉ xuất hiện trong bộ huấn luyện để tạo nhãn.
- **Hành động** giữ mã game `1 = trái, 2 = phải, 3 = lên, 4 = xuống`. PyTorch dùng chỉ số 0–3 nội bộ, cộng 1 trước khi gọi simulator. Policy trả bốn logits. Action mask chặn các hướng bị tường, thân hoặc đá cản. Nếu cả bốn hướng bị chặn, kết thúc lượt với trạng thái kẹt. Hướng làm rơi hoặc chạm bẫy vẫn hợp lệ về mặt điều khiển và sinh reward chết. Phép so sánh chính chạy greedy từ đầu map, không tìm kiếm và dùng cùng quy tắc action mask cho cả baseline lẫn policy mới. Báo bổ sung kết quả greedy không mask để khớp cách đo 11/40 trước đây.

[Luật simulator](worm_physics.py) · [Observation gốc trong PRD](SPEC_PRD_HUNGRY_WORM_RL_PCG.md#71-không-gian-quan-sát-observation-space-tensor-11--12--22)

## Reward và điều kiện kết thúc

| Sự kiện | Reward chính | Xử lý |
| --- | ---: | --- |
| Một bước hợp lệ | −0,01 | Tính cả bước đi tới cổng |
| Đầu sâu vào cổng | +10 | Thắng và dừng |
| Rơi, chạm bẫy, bị nghiền hoặc kẹt | −5 | Thua và dừng |
| Hết 250 bước | −1 | Hết lượt và dừng |
| Tường, thân, đá chặn trong log cũ | −2 | Ghi khi phát lại log cũ hoặc chạy chế độ không mask |

Các khoản áp dụng trong cùng một bước được cộng. Ăn táo, đẩy đá và số lần quay lại trạng thái cũ được log để phân tích. Vòng lặp có chi phí khi tiêu tốn thêm bước. Trong phép đo so sánh với PPO cũ, dừng tại trạng thái lặp đầu tiên theo giao thức cũ.

Đặt `gamma = 0,99`. Có thể thử một nhánh reward shaping với `Phi(s) = −0,02 × min(50, d(s))`, trong đó `d(s)` là khoảng cách tĩnh đến cổng qua các ô tránh tường và bẫy. Nếu chưa có đường tĩnh hoặc đầu nằm ngoài lưới, dùng 50. Đặt `Phi = 0` ở mọi trạng thái kết thúc, kể cả hết bước. Reward thêm là `gamma × Phi(s') − Phi(s)`. Khởi chạy thí nghiệm chính với hệ số shaping bằng 0 và dùng nhãn nhìn trước để cung cấp tín hiệu tầm xa. Điều này giữ cho phép thử đo được tác động của dữ liệu tìm kiếm.

## Policy, value và bộ tạo nhãn

- **Actor–critic** dùng encoder chung gồm Conv 64 kênh, ba residual block, hai lớp self-attention bốn head trên 264 ô. Ghép vector ở ô đầu sâu, global pooling, vector ngoài lưới và thời gian còn lại. Actor trả bốn logits. Critic trả một số ước lượng return. Đây là bộ nhớ không gian của trạng thái hiện tại. So sánh một biến thể GRU 128 chiều chỉ sau khi đã đo được policy không hồi quy. Nếu thử GRU, reset hidden state mỗi episode và lấy chuỗi liên tiếp có mask để train.
- **Dữ liệu ban đầu** gồm các quỹ đạo mẫu phát lại thành công, mỗi dòng lưu `map, episode, state, action, next_state, event, source`. Tạo ảnh lật ngang chỉ khi đổi hướng trái/phải và phát lại toàn bộ thành công. Nhóm map trùng hoặc lật tương đương phải nằm cùng một tập. Kiểm tra cả số bước lẫn trạng thái khi tái tạo kết quả.
- **Dữ liệu sửa sai** lấy từ trạng thái do policy tự tạo, ưu tiên điểm rẽ trước khi gặp bẫy, vòng lặp hoặc mất đường. Tại một trạng thái, thử từng hướng hợp lệ bằng simulator. Với mỗi nhánh, tìm đường tới cổng trong ngân sách 256 trạng thái, tăng lên 1.024 cho các ca khó đã lấy mẫu. Chỉ ghi nhãn thắng khi đường đã được phát lại tới cổng trong số bước còn lại. Phân biệt `win_verified`, `dead_immediate` và `unknown_budget`. Hết ngân sách mang nghĩa chưa biết, không tạo nhãn thua.
- **Loss policy** dùng cross-entropy để bắt chước một lựa chọn đã được xác minh. Nếu có nhiều hướng thắng, chuẩn hóa trọng số `exp(-(L - L_min)/4)` trên các hướng đã biết về đích, với `L` là chiều dài đường đã tìm được. So phân phối mục tiêu ấy với softmax trên tất cả hướng hợp lệ để vẫn có gradient khi chỉ tìm được một hướng thắng. Hướng chưa rõ sẽ không nhận mass trong nhãn bắt chước của lượt đó và không được ghi là thất bại trong bộ dữ liệu. Bỏ qua loss sửa sai khi chưa tìm được hướng thắng nào. Chỉ gọi một đường là ngắn nhất nếu phép tìm kiếm đã chứng minh điều đó.
- **Loss critic** dùng Huber trên discounted return của đường phát lại từ trạng thái gốc, bao gồm chi phí bước và terminal reward. Duy trì replay buffer riêng cho lời giải mẫu và dữ liệu sửa sai. Lấy batch mặc định 50% mỗi nguồn, đổi tỷ lệ chỉ dựa trên validation. Actor cập nhật từ nhãn đã kiểm chứng. Critic có thể hỗ trợ xếp ưu tiên nhánh cho vòng tạo nhãn kế tiếp.
- **Triển khai** giữ hai cách chơi. Policy tự chơi chọn một nước ở mỗi bước bằng actor. Policy có planner gọi simulator nhiều lần, ghi riêng node mở rộng và độ dài lời giải. Validation chọn checkpoint dựa trên tự thắng, rồi xét thời gian giải trên những map cùng thắng khi cần phá hòa.

## Quy trình huấn luyện và nghiệm thu

1. **Xác nhận dữ liệu và baseline.** Giữ 141 map cho train, 23 map theo dõi validation và 40 map so sánh lịch sử. Trong dữ liệu đã kiểm tra, 8 map từng lệch giữa hint và simulator nằm ngoài huấn luyện, 28 bản trùng hình học không tính thành map độc lập. Tập 23 map đã xuất hiện ở giai đoạn huấn luyện baseline cũ, nên chỉ dùng để chọn cấu hình cho đợt mới. Tập 40 map đã được xem kết quả nhiều lần. Muốn xác nhận tổng quát hóa mới cần thêm tập map chưa từng được dùng để phát triển mô hình. Chốt danh sách theo nhóm hình học trước run.
2. **Warm start bằng lời giải.** Huấn luyện CNN–attention bằng dữ liệu bắt chước, tối đa 20 epoch, AdamW với learning rate `3e-4`, batch 128. Lưu checkpoint theo số map tự thắng trên validation. Đo cả baseline cũ và checkpoint mới với cùng simulator và giao thức.
3. **Học từ sai lầm.** Chạy 4 vòng, mỗi vòng tối đa 2.000 trạng thái mới trên tập train. Lúc đầu lấy thêm ca gần cuối lời giải để phủ đích. Các vòng sau tăng tỷ lệ tự chơi từ đầu map. Tạo nhãn bằng simulator, trộn vào replay buffer rồi cập nhật actor và critic. Chạy ablation với cùng encoder, số lượt cập nhật và seed để so warm start thuần với warm start cộng nhãn sửa sai. Tắt nhánh GRU, shaping và world model ở phép so sánh này.
4. **Đánh giá theo map.** Sau mỗi vòng đo thắng/thua, lỗi chết/kẹt/lặp, độ dài đường trên nhóm map cùng thắng, node tìm kiếm với chế độ có planner và thời gian thực. Chạy ba seed trước khi kết luận cải thiện ổn định. Chỉ thử PPO bổ sung khi nhãn sửa sai đã mang lại hiệu quả hoặc cần ablation riêng. Mốc quyết định mở rộng là tăng ít nhất 2/23 trên validation so với baseline cùng giao thức, ưu tiên mức tăng lặp lại giữa các seed. Con số này là mốc thử nghiệm, không phải bảo đảm kết quả.
5. **Replay và truy vết.** Lưu từng episode gồm hash dữ liệu, seed, map, điểm khởi đầu, hành động và kết quả. Với mẫu sửa sai, lưu hành động policy chọn, hành động được dạy, lý do, node mở rộng và quỹ đạo xác minh. Hiển thị lại các lượt trước và sau train cùng từng bước. Chốt checkpoint theo validation trước khi mở tập kiểm tra mới.

## Khi nào thử world model học được

Xây một world model nhỏ nếu profiling cho thấy tìm kiếm hoặc chuyển trạng thái chi phối thời gian. Input là state, action one-hot và luật map. Dự đoán vị trí đầu, delta các occupancy map động, event kết thúc, action mask tiếp theo và latent của trạng thái mới. Train với các đoạn liên tiếp 4 bước từ buffer thật, ưu tiên lấy mẫu đủ các trường hợp đẩy đá, rơi và vào cổng. Báo độ chính xác theo từng event, độ đúng hành động hợp lệ và sai số tích lũy 1/4/8 bước trên nhóm map validation.

Cho world model xếp hạng nhánh trong cửa sổ 4 bước trước, tăng tới 8 bước khi benchmark cho thấy tốc độ và độ chính xác cho phép. Simulator phát lại toàn bộ đường được chấp nhận để tạo nhãn thắng. Với bốn hành động, nhân gradient qua chỉ số `argmax` không khả thi. Thiết kế chính dùng tìm kiếm có nhãn và distillation. Nếu nghiên cứu tối ưu qua world model khả vi, có thể so sánh với actor categorical tối ưu bằng ước lượng gradient rời rạc hoặc dùng xấp xỉ Gumbel-Softmax và kiểm tra lại trên simulator. Chi phí và sai lệch của nhánh đó là một thí nghiệm bổ sung.

## Ngân sách GPU đề xuất

Máy hiện tại nhận Intel UHD. PyTorch 2.8 cho kết quả CUDA `False` và XPU `False`, đều có 0 thiết bị. Mô phỏng và tìm kiếm chạy được trên CPU. Để chạy kế hoạch GPU cần chuyển sang một máy có GPU mà PyTorch nhận được. Thời gian dưới đây là ngân sách, cần đo tốc độ trước khi chốt số bước.

| Thời gian dự kiến | Công việc |
| --- | --- |
| 0–15 phút | Kiểm tra một batch forward/backward thực sự chạy trên GPU, ghi RAM/VRAM, tốc độ tạo nhãn bằng CPU và tốc độ cập nhật model |
| 15–45 phút | Warm start, đo baseline cùng giao thức và kiểm tra replay |
| 45–150 phút | Chạy bốn vòng tạo nhãn sửa sai và cập nhật policy, lưu checkpoint từng vòng |
| 150–180 phút | Đánh giá, tái tạo replay và xác nhận checkpoint tải lại được |

CPU tiếp tục chạy trò chơi và khâu tìm nhãn. GPU xử lý batch encoder, actor và critic. Nếu GPU chờ dữ liệu, cần gom batch hoặc tăng tốc bước tạo nhãn trước khi kéo dài thời lượng train. Thử nghiệm gốc của SuperMimic dùng H100 cho bài toán robot, nên ngân sách của WormLabRL được quyết định bằng benchmark riêng. [Cấu hình và thời gian trong bài gốc](https://supermimic.github.io/#compute)
