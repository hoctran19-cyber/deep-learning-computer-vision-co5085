# Báo cáo A1-2

Thư mục này chứa báo cáo Markdown tiếng Việt và các hình ảnh được tham chiếu bởi báo cáo.

## Cấu trúc

- [`report.md`](./report.md): báo cáo A1-2 gồm giới thiệu, dataset và preprocessing, methodology, experimental setup, so sánh bảy cấu hình, đánh giá Test, phân tích lỗi theo lớp, thảo luận và kết luận.
- `figures/training/`: loss/accuracy theo epoch, so sánh Validation Accuracy/Macro-F1 và thời gian.
- `figures/evaluation/`: confusion matrix và ví dụ dự đoán trên Test.
- `figures/error_analysis/`: biểu đồ Top 10 lớp F1 thấp, Recall thấp và cặp lớp thường nhầm lẫn.
- Metrics nguồn: `../outputs/report_artifacts/metrics/`; summary và log theo epoch: `../outputs/experiments/<experiment>/`; cấu hình gốc: `../configs/` và `../outputs/experiments/<experiment>/effective_config.yaml`.

## Xem báo cáo

Mở `report.md` trong VS Code hoặc trình xem Markdown có hỗ trợ ảnh tương đối. Các hình huấn luyện được sao chép từ `A1-2/outputs/figures/`; confusion matrix, ví dụ dự đoán và biểu đồ phân tích lỗi dùng artifacts đã sửa mapping trong `A1-2/outputs/repaired_artifacts/`. Hình được sao chép nguyên trạng, không chỉnh sửa dữ liệu hoặc vẽ lại. Đường dẫn hình tính tương đối từ thư mục `A1-2/report/`.

## Ghi chú

Các chỉ số và bảng được đối chiếu với JSON/CSV trong `A1-2/outputs/`. Mapping tên lớp của prediction đã được tái tạo từ nhãn số và `classes.txt`; trạng thái đối chiếu được lưu trong `outputs/repaired_artifacts/mapping_audit.json`. Báo cáo cũng nêu rõ nguồn hiện có chưa xác định license mở cụ thể cho ảnh Food-101. Những nội dung còn thiếu được đánh dấu `[CẦN BỔ SUNG]` hoặc `[CẦN XÁC NHẬN]`. Chưa tạo PDF.
