# Báo cáo A1-2

Thư mục này chứa báo cáo Markdown tiếng Việt và các hình ảnh được tham chiếu bởi báo cáo.

## Cấu trúc

- [`report.md`](./report.md): báo cáo A1-2 gồm giới thiệu, dataset và preprocessing, methodology, experimental setup, so sánh bảy cấu hình, đánh giá Test, phân tích lỗi theo lớp, thảo luận và kết luận.
- `figures/training/`: loss/accuracy theo epoch, so sánh Validation Accuracy/Macro-F1 và thời gian.
- `figures/evaluation/`: confusion matrix và ví dụ dự đoán trên Test.
- `figures/error_analysis/`: biểu đồ Top 10 lớp F1 thấp, Recall thấp và cặp lớp thường nhầm lẫn.
- Metrics nguồn: `../outputs/report_artifacts/metrics/`; summary và log theo epoch: `../outputs/experiments/<experiment>/`; cấu hình gốc: `../configs/` và `../outputs/experiments/<experiment>/effective_config.yaml`.

## Xem báo cáo

Mở `report.md` trong VS Code hoặc trình xem Markdown có hỗ trợ ảnh tương đối. Các hình được sao chép nguyên trạng từ `A1-2/outputs/figures/` và `A1-2/outputs/report_artifacts/figures/`; không chỉnh sửa dữ liệu hoặc vẽ lại biểu đồ. Đường dẫn hình tính tương đối từ thư mục `A1-2/report/`.

## Ghi chú

Các chỉ số và bảng được đối chiếu với JSON/CSV trong `A1-2/outputs/`. Báo cáo ghi rõ giới hạn mapping tên lớp trong prediction export và tình trạng license Food-101 cần xác nhận. Những nội dung còn thiếu được đánh dấu `[CẦN BỔ SUNG]` hoặc `[CẦN XÁC NHẬN]`. Chưa tạo PDF.
