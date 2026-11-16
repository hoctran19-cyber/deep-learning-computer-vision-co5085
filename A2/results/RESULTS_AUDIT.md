# Kiểm toán kết quả A2

Tài liệu này ghi lại những kết luận được các hiện vật trong repository hỗ trợ. Không tạo lại hay thay đổi dataset, model, prediction, dòng history hoặc metric nào.

## Dataset và manifest E0

`e0_report.json` ghi nhận các kiểm tra ảnh/nhãn đã đạt: 18.000 ảnh train, 2.000 ảnh validation và 5.000 ảnh test; không có nhãn bị thiếu, nhãn mồ côi hoặc annotation không hợp lệ. Phạm vi của trường `all_valid` hiện được nêu rõ là chỉ kiểm tra tính toàn vẹn ảnh/nhãn; việc xác minh tổng thể chưa hoàn tất cho đến khi đối chiếu được các đường dẫn trong manifest.

So sánh tập đường dẫn trong báo cáo không tìm thấy đường dẫn nào khớp ở bất kỳ split nào (train 18.000/18.000, validation 2.000/2.000, test 5.000/5.000). Ví dụ cho thấy khác biệt về cách biểu diễn: đường dẫn train trong manifest bắt đầu bằng `ccpd_base/<stem>`, còn đường dẫn ảnh đã chuẩn bị chỉ là `<stem>`; đường dẫn test trong manifest bắt đầu bằng `ccpd_blur/<stem>`, còn ảnh đã chuẩn bị dùng `ccpd_blur__<stem>`. Notebook so sánh chuỗi đường dẫn tương đối sau khi bỏ một số tiền tố `images/<split>` hoặc `<split>` và phần mở rộng file; notebook không xử lý việc làm phẳng thư mục subset hoặc bí danh basename.

Repository không có manifest nguồn hoặc cây thư mục ảnh gốc/đã chuẩn bị. Notebook trỏ đến manifest trên Drive và dataset đã chuẩn bị trong Colab; cả hai đều không có sẵn tại đây. Vì vậy, đã xác nhận được sự không khớp, nhưng chưa thể xác định đó chỉ là biến đổi tiền tố/làm phẳng thư mục, khác thư mục gốc/nội dung, hay do file thiếu/khác nhau. Không chuẩn hóa các đường dẫn để tạo kết quả khớp trước khi chạy lại với cả hai nguồn gốc và xác nhận mapping từ `output_path` đến file ảnh thực tế (tốt nhất là kiểm tra danh tính ở cấp file).

## Số epoch E2 và lựa chọn lần chạy

Phần siêu tham số PDLPR trong notebook nhận dạng đặt `epochs=40`; biến toàn cục `EPOCHS=30` là giá trị mặc định cho CRNN. Output của cell huấn luyện được lưu trong notebook ghi cấu hình 40 epoch, đủ từng epoch từ 1 đến 40 cho cả hai biến thể E2 và không có thông báo resume (code sẽ in thông báo khi nạp `last.pt`). Bằng chứng này phù hợp với một lần chạy mới đủ 40 epoch, không phải chạy tiếp từ checkpoint. Hàm ghi kết quả test đã ghi nhầm giá trị toàn cục vào `epochs_cfg` cho mọi kiến trúc. Cả hai history E2 chính đều có epoch 1–40, vì vậy metadata cấu hình đã được sửa thành 40 và `epochs_run` cũng là 40. Checkpoint tốt nhất vẫn ở epoch 34 (`e2_pdlpr`) và 39 (`e2_pdlpr_memq`); theo logic chỉ lưu khi có cải thiện nghiêm ngặt của notebook, các epoch sau có validation bằng điểm không thay thế checkpoint được lưu trước đó.

Notebook hỗ trợ resume từ `last.pt`, nhưng các file checkpoint không có trong repository. Output nhúng trong notebook là bằng chứng cho kết luận lần chạy mới; không còn log huấn luyện hoặc checkpoint riêng để đối chứng độc lập.

Kết quả chính nằm trong `results/`: history có 40 epoch và các CSV dự đoán E2 có 5.000 dòng với prediction không rỗng. Lần chạy E2 riêng trong `results_run1_plateau/` thất bại/không hội tụ: history có 9 epoch với `val_exact=0`, cả 5.000 prediction nhận dạng đều rỗng/sai và cả 5.000 prediction end-to-end đều rỗng. Các file này được giữ lại để kiểm toán; những file so sánh chính tiếp tục dùng dữ liệu trong `results/`.

## Protocol nhận dạng và end-to-end

CSV nhận dạng đánh giá 5.000 crop được cắt theo nhãn YOLO ground-truth, với margin 10%. Đếm lại điều kiện `pred == gt` cho kết quả 3.144/5.000 = 0.6288.

CSV end-to-end đánh giá toàn bộ 5.000 ảnh test sau khi detector E0 chạy (confidence 0.25, NMS IoU 0.45), chọn detection có confidence cao nhất rồi cắt theo box dự đoán với margin 10% tương tự. Metric này còn yêu cầu IoU > 0.6. Đếm lại điều kiện `(iou > 0.6) and (e2_pdlpr_pred == gt)` cho kết quả 3.254/5.000 = 0.6508. Độ chính xác nhận dạng có điều kiện IoU > 0.7 là metric khác (0.68416). Đây là các protocol hợp lệ nhưng khác nhau, không thể dùng thay thế cho nhau như cùng một giá trị accuracy nhận dạng.

Cả hai CSV có 5.000 dòng cùng thứ tự và cùng giá trị `(subset, gt)`, phù hợp với việc notebook dùng chung một bảng test. CSV end-to-end hiện tại không có đường dẫn crop/mẫu duy nhất và một số cặp `(subset, gt)` bị lặp, nên không thể chứng minh phép nối từng dòng là duy nhất chỉ từ các CSV đã lưu. Notebook hiện sẽ ghi `crop_rel` cho các output end-to-end trong tương lai. Code nạp `best.pt` của từng recognizer cho cả hai lần đánh giá, nhưng JSON end-to-end hiện có không ghi epoch/hash của checkpoint; vì vậy chưa xác minh được checkpoint đã dùng cho các metric hiện tại.

| Metric | Số mẫu | Điều kiện | Nguồn |
|---|---:|---|---|
| Recognition exact match | 5.000 | Crop theo nhãn ground-truth; khớp chính xác toàn bộ ký tự | `recognition/results/e2_pdlpr_test_preds.csv` (`pred`, `gt`) |
| End-to-end exact match có điều kiện detector | 5.000 | IoU của box dự đoán > 0.6 và khớp toàn bộ ký tự; mẫu số gồm mọi dòng test | `end_to_end/results/e2e_per_image.csv` (`iou`, `e2_pdlpr_pred`, `gt`) |
| Recognition exact match khi detection đúng | 4.401 mẫu đủ điều kiện | Khớp chính xác chuỗi ký tự với điều kiện IoU > 0.7 | `end_to_end/results/e2e_per_image.csv`; tổng hợp trong `end_to_end/results/e2e_e2_pdlpr.json` |

## Cần xác minh lại

1. Mount/truy cập manifest CCPD gốc và dataset đã chuẩn bị, ánh xạ từng `output_path` trong manifest tới ảnh thực tế rồi chạy lại kiểm tra E0. Chỉ sau đó mới quyết định có cần quy tắc chuẩn hóa đường dẫn hay không.
2. Để xác định history 40 epoch có resume hay không, cần lưu log huấn luyện và metadata checkpoint (`last.pt`/`best.pt`) của lần chạy đó.
3. Để kiểm toán chính xác checkpoint và từng mẫu giữa recognition/E2E, cần lưu hash/epoch của checkpoint và thêm `crop_rel` vào từng dòng kết quả end-to-end; không cần chạy lại inference cho cuộc kiểm toán này.
