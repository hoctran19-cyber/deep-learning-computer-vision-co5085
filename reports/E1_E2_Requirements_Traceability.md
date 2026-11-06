# Ma trận truy xuất yêu cầu E1 và E2

## Phạm vi và tình trạng tài liệu đặc tả

Branch hiện tại là `main`; working tree sạch trước khi bắt đầu tác vụ báo cáo
này. Repository có README bài tập, source code, notebook, cấu hình, kết quả và
hình trực quan hóa. Không tìm thấy PDF đề bài E1/E2 chính thức,
`exercise-vne.pdf` hoặc rubric chấm điểm trong repository hay thư mục bài tập
lân cận. Đã kiểm tra danh sách file được theo dõi ở một branch khác; branch đó
cũng không có tài liệu đặc tả chính thức riêng.

Do đó, các dòng bên dưới truy xuất những yêu cầu và deliverable được ghi nhận
trong repository và phần triển khai đã kiểm tra. Bảng này **không thay thế
rubric chính thức của môn học**. Yêu cầu nào phụ thuộc vào tài liệu chính thức
chưa tìm thấy đều được đánh dấu `Not verifiable`.

Các trạng thái được phép: `Completed`, `Partially completed`, `Not completed`,
`Not verifiable`.

## Yêu cầu chung và tài liệu chính thức

| Bài tập | Yêu cầu | File bằng chứng | Kết quả xác minh | Trạng thái | Ghi chú |
|---|---|---|---|---|---|
| E1 và E2 | Có nội dung đề bài chính thức và rubric chấm điểm để đối chiếu | Không tìm thấy PDF hoặc rubric chính thức; mô tả trong `index.html`, `E1/README.md`, `E2/README.md` | Đã kiểm tra repository, thư mục lân cận và danh sách file của branch được theo dõi; không có tài liệu chính thức cụ thể | Not verifiable | README cục bộ và mô tả trên landing page không chứng minh đã bao quát toàn bộ tiêu chí chính thức. |
| E1 và E2 | Sử dụng chung dataset và protocol train/validation/test có thể tái lập | `E1/configs/config.yaml`, `E1/src/datasets.py`, `E2/configs/default.yaml`, `E2/src/data.py` | Cả hai mặc định dùng Fashion-MNIST, seed 42, validation 10%; test set được nạp riêng | Completed | Không có snapshot cấu hình chính xác theo từng lần chạy cho mọi bản ghi metric E1. |
| E1 và E2 | Báo cáo cấu hình/môi trường thực nghiệm có thể tái lập | Cấu hình E1/E2 và các file metric JSON | Có cấu hình mặc định và phần lớn tham số; metric E2 ghi nhận CUDA, metric E1 không ghi thiết bị | Partially completed | Chưa xác minh được thiết bị E1 và phiên bản package/runtime thống nhất từ các tư liệu kết quả. |

## Truy xuất yêu cầu E1

| Bài tập | Yêu cầu | File bằng chứng | Kết quả xác minh | Trạng thái | Ghi chú |
|---|---|---|---|---|---|
| E1 | Nạp Fashion-MNIST và tạo train/validation/test loader | `E1/src/datasets.py`, `E1/configs/config.yaml` | Dataset mặc định là Fashion-MNIST; validation split ngẫu nhiên có seed chiếm 10%; test set riêng | Completed | Có hỗ trợ MNIST và CIFAR-10, nhưng metric E1 đã lưu thuộc Fashion-MNIST. |
| E1 | Tiền xử lý và normalize ảnh đầu vào | `E1/src/datasets.py` | Chuyển thành tensor và normalization với mean/std 0.5 cho dữ liệu grayscale | Completed | Pipeline mặc định không cấu hình augmentation khác. |
| E1 | Triển khai Softmax classifier | `E1/src/models.py` | Flatten ảnh rồi dùng một linear classifier tạo logits | Completed | Training sử dụng cross-entropy. |
| E1 | Triển khai MLP classifier | `E1/src/models.py`, `E1/configs/config.yaml` | Flatten, các hidden layer fully connected `[128, 64]`, ReLU và output layer | Completed | Dropout mặc định bằng 0.0. |
| E1 | Triển khai CNN classifier | `E1/src/models.py`, `E1/configs/config.yaml` | Hai convolution/ReLU/max-pooling block và classifier head; channels mặc định `[32, 64]` | Completed | Đã xác minh architecture trong source code. |
| E1 | Sử dụng training loop PyTorch tự triển khai | `E1/src/train.py` | Thực hiện tường minh zero-grad, forward, loss, backward, optimizer step và validation pass | Completed | Cấu hình E1 dùng Adam, learning rate 0.001, batch size 128, 10 epoch, seed 42. |
| E1 | Đánh giá cả ba classifier trên test data | `E1/results/metrics/model_comparison.csv`, `E1/results/metrics.json`, các file JSON riêng của model | Có test accuracy, best validation accuracy, số parameters và training time cho cả ba model | Completed | Dữ liệu CSV và JSON khớp nhau. |
| E1 | Phân tích training curve, confusion matrix và lỗi dự đoán | `E1/results/analysis.md`, `E1/results/figures/` | Có curve, confusion matrix, hình ví dụ dự đoán sai và nội dung phân tích cho các model | Partially completed | Bảng trong phần tường thuật dùng các giá trị cũ, không khớp metric CSV/JSON hiện tại; cần đối chiếu trước khi nộp bản cuối. |
| E1 | Deliverable có thể cần thêm: thống kê dataset theo lớp và sơ đồ architecture | `E1/src/datasets.py`, `E1/src/models.py`, `E1/results/analysis.md` | Source có tên lớp và cấu trúc model; chưa xác minh được bảng thống kê dataset hoặc sơ đồ architecture riêng | Not verifiable | Đây là bước rà soát tính đầy đủ, chưa được xác nhận là yêu cầu chính thức; cần tài liệu đề bài để biết có bắt buộc hay không. |
| E1 | Xác minh tiêu chí chấm điểm và định dạng báo cáo chính thức của E1 | Không có spec/rubric chính thức trong repository | Không thể đối chiếu deliverable với tiêu chí chính xác | Not verifiable | Cần có tài liệu bài tập chính thức để hoàn tất việc kiểm tra này. |

## Truy xuất yêu cầu E2

| Bài tập | Yêu cầu | File bằng chứng | Kết quả xác minh | Trạng thái | Ghi chú |
|---|---|---|---|---|---|
| E2 | Hiện thực MSA từ các phép toán cơ bản | `E2/src/attention.py` | Module custom có Q/K/V projection, scaled dot-product, softmax, aggregation, ghép head và output projection | Completed | Đã kiểm tra phần triển khai trong source. |
| E2 | Hiện thực phiên bản MSA đối chiếu bằng PyTorch | `E2/src/attention.py` | Wrapper sử dụng `nn.MultiheadAttention(batch_first=True)` | Completed | Đã lưu kết quả so sánh MSA chính trong metric. |
| E2 | Tích hợp attention vào image classifier | `E2/src/model.py` | Có tokenizer, class token, positional embedding, encoder block nông và classification head | Completed | Đây là classifier thực nghiệm gọn nhẹ, không phải ViT pretrained quy mô đầy đủ. |
| E2 | Cung cấp tối thiểu hai chiến lược image-tokenization | `E2/src/tokenizers.py`, `E2/tests/test_tokenizers_and_model.py` | Có patch, row và CNN-stem tokenizer; test kiểm tra kích thước token đầu ra | Completed | Có automated test nhưng không chạy lại trong quá trình tạo báo cáo. |
| E2 | Training và evaluation trên Fashion-MNIST | `E2/configs/default.yaml`, `E2/outputs/metrics/comparison.csv`, các file JSON từng lần chạy | Tư liệu chính ghi nhận năm lần chạy trên Fashion-MNIST, seed 42, CUDA | Completed | Các lần chạy chính gồm hai biến thể attention và ba biến thể tokenizer. |
| E2 | So sánh custom attention với PyTorch attention | `E2/outputs/metrics/comparison.csv`, `E2/outputs/metrics/msa_custom_patch.json`, `E2/outputs/metrics/msa_torch_patch.json` | Các lần chạy patch-tokenizer có test accuracy, validation accuracy, số parameters, training time và inference time | Completed | Mỗi cách hiện thực chỉ có một lần chạy được lưu; không có ước lượng độ bất định. |
| E2 | So sánh các biến thể tokenization | `E2/outputs/metrics/comparison.csv`, `E2/outputs/metrics/tok_patch_custom.json`, `E2/outputs/metrics/tok_row_custom.json`, `E2/outputs/metrics/tok_cnn_stem_custom.json` | Có metric và hình trực quan hóa đã lưu cho patch, row và CNN-stem | Completed | Mỗi biến thể có một lần chạy được lưu. |
| E2 | Cung cấp bằng chứng định lượng về accuracy và efficiency | `E2/outputs/metrics/comparison.csv`, `E2/outputs/figures/` | Có test/best-validation accuracy, số parameters, training/inference time, confusion matrix và learning curve | Completed | Không tìm thấy Macro-F1 hoặc thống kê qua nhiều lần chạy. |
| E2 | Deliverable có thể cần thêm: so sánh trực tiếp attention classifier với baseline E1 | `E1/results/metrics/model_comparison.csv`, `E2/outputs/metrics/comparison.csv`, `E2/README.md` | Có kết quả của cả hai bài nhưng chưa ghi nhận bảng/protocol so sánh chung được cân bằng | Not verifiable | Đây là khả năng cần xem xét liên bài, chưa được xác nhận là yêu cầu chính thức; optimizer và số epoch khác nhau. |
| E2 | Deliverable có thể cần thêm: phân tích lỗi bằng văn bản theo từng lớp | `E2/outputs/figures/`, `E2/README.md` | Có confusion matrix và hình; không tìm thấy tài liệu riêng phân tích lỗi chi tiết theo từng lớp | Not verifiable | Không thể xác định có bắt buộc tường thuật theo lớp hay không nếu thiếu đặc tả chính thức. |
| E2 | Xác minh tiêu chí chấm điểm và định dạng báo cáo chính thức của E2 | Không có spec/rubric chính thức trong repository | Không thể đối chiếu deliverable với tiêu chí chính xác | Not verifiable | Cần có tài liệu bài tập chính thức để hoàn tất việc kiểm tra này. |

## Sai khác chưa được giải quyết và bằng chứng còn thiếu

1. `E1/results/analysis.md` ghi các giá trị cũ (bao gồm CNN test accuracy
   91.05% và training time 118.44/118.44/122.53 giây). CSV so sánh và JSON E1
   hiện tại ghi 91.11% và 104.96/106.99/112.99 giây. Thời gian đã làm tròn
   trong README E1 khớp với metric hiện tại. Repository không có thông tin
   nguồn gốc lần chạy để xác định nguyên nhân sai khác.
2. Không có nội dung yêu cầu E1/E2 chính thức và rubric chấm điểm chính xác.
3. Các tư liệu metric E1 hiện tại không ghi thiết bị đã dùng.
4. Không có thống kê qua nhiều seed hoặc confidence interval trong các tư
   liệu kết quả đã kiểm tra của cả hai bài.
5. Không tìm thấy Macro-F1 trong metric E1/E2 đã lưu.
6. Chưa có tài liệu về phép so sánh trực tiếp E1-vs-E2 với protocol cân bằng.
7. Có các file unit test cho E2 nhưng không chạy test suite trong tác vụ lập
   báo cáo này.
