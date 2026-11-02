# A1 — CNN vs Transformer trên tập ảnh lớn

**Môn học:** CO5085 — Deep Learning and its Applications in Computer Vision  
**Sinh viên:** Trần Thái Học  
**MSSV:** 2570426  
**Trạng thái:** Đang thực hiện  
**Phiên bản:** Interim report

## 1. Giới thiệu

Bài tập A1 hướng đến nghiên cứu và so sánh mạng tích chập (CNN) với Vision
Transformer trên một bộ dữ liệu ảnh quy mô lớn hơn các tập dữ liệu nhỏ thường
dùng trong bài tập nhập môn. Protocol hiện chọn ResNet-50 đại diện cho CNN và
ViT-B/16 đại diện cho Vision Transformer, trên bài toán phân loại 101 lớp của
Food-101.

Ở giai đoạn hiện tại, tài liệu A1.1 về dataset và protocol đã được chuẩn bị
trong [`A1-1/README.md`](A1-1/README.md). Pipeline A1.1 chưa được triển khai,
dataset chưa được tải, và các mô hình chưa được huấn luyện hoặc đánh giá. Vì
vậy báo cáo này ghi lại thiết kế nghiên cứu hiện có, không trình bày kết quả
thực nghiệm hay khẳng định toàn bộ bài A1 đã hoàn thành.

## 2. Dataset — Food-101

Food-101 là bộ dữ liệu phân loại ảnh món ăn gồm 101 lớp và 101.000 ảnh, tương
đương 1.000 ảnh mỗi lớp. Ảnh màu có cạnh dài nhất không vượt quá 512 pixel.
Training set gốc có thể chứa ảnh nhiễu và nhãn không chính xác; 250 ảnh test
mỗi lớp đã được kiểm tra nhãn thủ công.

Food-101 được chọn vì quy mô và số lớp lớn hơn các dataset đồ chơi thường dùng
trong bài tập nhập môn, đồng thời tạo một bài toán phân loại đa lớp chung để
khảo sát hai kiến trúc dự kiến. Đây là lý do lựa chọn theo protocol, không phải
kết luận rằng dataset hoặc kiến trúc nào phù hợp hơn dựa trên thực nghiệm.

Trang dataset chính thức của ETH Zurich cung cấp thông tin và tải dataset. Tài
liệu A1.1 chưa xác minh được một license mở riêng cho ảnh hoặc metadata/nhãn;
ảnh gốc không thuộc ETH Zurich. Một dataset card thứ cấp của Voxel51 ghi nhận
nguồn ảnh Foodspotting và lưu ý về điều kiện sử dụng, nhưng đây không phải giấy
phép do chủ sở hữu ảnh cấp. Do đó, báo cáo không kết luận ảnh được tự do sử dụng
thương mại hoặc tái phân phối. Không đưa archive hoặc ảnh Food-101 vào
repository; việc sử dụng cần tuân thủ các quyền và điều khoản áp dụng.

## 3. Data split và preprocessing

### 3.1. Split gốc của Food-101

| Tập dữ liệu gốc | Ảnh mỗi lớp | Số ảnh | Ghi chú |
|---|---:|---:|---|
| Training set | 750 | 75.750 | Nhãn có thể nhiễu |
| Test set | 250 | 25.250 | Nhãn được kiểm tra thủ công |
| **Tổng** | **1.000** | **101.000** | **101 lớp** |

### 3.2. Split dự kiến cho A1

| Tập dữ liệu dự án | Ảnh mỗi lớp | Số ảnh | Nguồn/cách tạo |
|---|---:|---:|---|
| Training set | 675 | 68.175 | 90% Training set gốc mỗi lớp |
| Validation set | 75 | 7.575 | 10% Training set gốc mỗi lớp |
| Test set | 250 | 25.250 | Test set gốc, giữ nguyên |
| **Tổng** | **1.000** | **101.000** | **101 lớp** |

Validation set dự kiến được trích phân tầng từ Training set gốc; Test set gốc
được giữ nguyên. Đây là split theo kế hoạch: manifest và các tập con chưa được
tạo.

### 3.3. Preprocessing

Các bước dưới đây được đề xuất trong protocol và chưa được chạy hoặc kiểm
chứng:

1. Đọc ảnh RGB và giữ nhãn trong 101 lớp.
2. Resize cạnh ngắn về 256 pixel. Với Training set, áp dụng random crop
   `224 × 224` và random horizontal flip với xác suất `0.5`.
3. Với Validation set và Test set, resize cạnh ngắn về 256 pixel rồi center
   crop `224 × 224`; không dùng augmentation ngẫu nhiên.
4. Chuẩn hóa theo ImageNet mean `(0.485, 0.456, 0.406)` và std
   `(0.229, 0.224, 0.225)`.

Protocol dự kiến dùng cùng preprocessing cho cả hai kiến trúc và mọi training
regime. Không dự kiến dùng color jitter mạnh; nếu weights được chọn đòi hỏi
transform khác, thay đổi cần được ghi lại và áp dụng nhất quán.

### 3.4. Reproducibility

Kế hoạch chia dữ liệu là sắp xếp tên lớp và định danh ảnh tương đối theo thứ tự
từ điển, sau đó dùng NumPy seed `42` để xáo trộn ảnh riêng trong từng lớp và
lấy 75 ảnh/lớp làm validation. Dự kiến lưu phiên bản NumPy cùng manifest gồm
định danh tương đối, nhãn và split; mọi model/regime sẽ dùng chung manifest.
Manifest hiện chưa được tạo.

Seed huấn luyện dự kiến là `42` cho Python, NumPy, PyTorch và sampler/worker
DataLoader trong phạm vi thiết bị hỗ trợ. Việc tái lập hoàn toàn có thể bị ảnh
hưởng bởi phép toán và thiết bị không xác định. Các bước này là kế hoạch; chưa
có pipeline chạy để xác nhận.

## 4. Models và experimental protocol

Hai kiến trúc dự kiến được xây dựng bằng model builder phù hợp của Torchvision,
với classification head 101 đầu ra. Tên và phiên bản pretrained weights cần
được ghi cố định khi triển khai.

| Kiến trúc | From scratch | Frozen backbone / train head | Partial fine-tuning | Full fine-tuning |
|---|---|---|---|---|
| ResNet-50 (CNN) | Khởi tạo ngẫu nhiên và huấn luyện toàn mạng | Dùng ImageNet pretrained weights; chỉ huấn luyện head | Mở `layer4` và head; đóng băng các stage trước | Huấn luyện toàn mạng và head |
| ViT-B/16 (Vision Transformer) | Khởi tạo ngẫu nhiên và huấn luyện toàn mạng | Dùng ImageNet pretrained weights; chỉ huấn luyện head | Mở block Transformer cuối cùng và head; đóng băng các block trước | Huấn luyện toàn encoder và head |

Các chế độ và pretrained weights trong bảng là kế hoạch, chưa được triển khai.
Protocol đề xuất các cấu hình huấn luyện khởi điểm sau:

| Thành phần | Giá trị/nguyên tắc dự kiến |
|---|---|
| Optimizer | AdamW cho mọi cấu hình |
| Learning rate | `1e-3` cho from-scratch và frozen head; `1e-4` cho backbone và `1e-3` cho head trong partial/full fine-tuning, dùng parameter groups riêng |
| Batch size | 32 ảnh trên mỗi thiết bị nếu bộ nhớ cho phép; ghi lại nếu cần giảm hoặc dùng gradient accumulation |
| Epoch tối đa | 30 |
| Checkpoint/early stopping | Chọn theo Macro-F1 validation; dùng Accuracy validation để phân xử hòa; dừng sau 5 epoch liên tiếp không cải thiện hoặc khi đạt giới hạn epoch |
| Seed | 42; ghi rõ thiết bị và các hạn chế về tính xác định |

Đây là các giá trị khởi điểm đề xuất, chưa phải cấu hình đã chạy và có thể cần
được chốt trước thực nghiệm. Nếu giới hạn phần cứng buộc phải đổi batch size,
epoch budget, precision hoặc phần backbone được mở, các thay đổi cần được ghi
lại cho từng run.

Nguyên tắc so sánh là dùng chung manifest, preprocessing, định nghĩa metrics
và quy tắc chọn checkpoint. Khi báo cáo cần ghi cấu hình thực tế, phiên bản
PyTorch/Torchvision, pretrained weights, seed, thiết bị và mọi khác biệt. Đây
là protocol kiểm soát các yếu tố so sánh, không bảo đảm công bằng tuyệt đối
nếu ngân sách tính toán hoặc cấu hình khác nhau đáng kể.

## 5. Evaluation plan

Các chỉ số và phân tích dự kiến:

- **Accuracy:** tỷ lệ dự đoán đúng trên split được đánh giá.
- **Macro-F1:** tính F1 riêng cho mỗi lớp rồi lấy trung bình không trọng số
  trên 101 lớp, để phản ánh hiệu suất theo lớp.
- **Training time:** ghi thời gian huấn luyện theo cùng quy tắc đo, đồng thời
  nêu thiết bị và phạm vi đo.
- **Trainable parameters:** ghi số tham số được cập nhật sau khi áp dụng chính
  sách đóng băng/mở backbone, nếu khả thi.
- **Confusion matrix và phân tích lỗi:** xem precision/recall/F1 theo lớp, các
  cặp lớp thường bị nhầm và các lỗi đại diện nếu thực hiện được. Không lưu ảnh
  Food-101 vào repository khi quyền tái phân phối chưa được xác nhận.

Checkpoint và siêu tham số dự kiến được chọn trên Validation set theo
Macro-F1, dùng Accuracy validation để phân xử hòa. Test set chỉ được dùng cho
đánh giá cuối sau khi chốt cấu hình. Chưa có metric hoặc phân tích nào được
tính cho A1 trong báo cáo này.

## 6. Tiến độ thực hiện

| Hạng mục | Trạng thái hiện tại |
|---|---|
| A1.1 — Dataset + Protocol | Đã chuẩn bị tài liệu protocol trong `A1-1/README.md`; pipeline, manifest và xác minh dữ liệu tải thực tế chưa hoàn thành |
| A1.2 — Implementation / Experiments | Chưa xác nhận hoàn thành; chưa có kết quả A1 trong tài liệu được kiểm tra |
| Phân tích kết quả | Chưa thực hiện |
| Báo cáo kết quả cuối cùng | Chưa hoàn thành; sẽ cập nhật sau khi có thực nghiệm được kiểm chứng |

## 7. Kết quả thực nghiệm

Chưa có kết quả thực nghiệm được báo cáo trong phiên bản này. Phần này sẽ
được cập nhật sau khi các thí nghiệm được thực hiện và kiểm chứng.

| Kiến trúc | Training regime | Accuracy | Macro-F1 | Training time | Trainable parameters |
|---|---|---:|---:|---:|---:|
| Chưa chạy | — | — | — | — | — |

## 8. Thảo luận và hạn chế

Các rủi ro và giới hạn đã được ghi nhận trong protocol:

- **License:** chưa xác minh license mở riêng cho ảnh hoặc metadata/nhãn.
  Không mặc định ảnh được tự do tái phân phối; nếu yêu cầu bài tập cần license
  mở rõ ràng, cần xác nhận trước khi sử dụng.
- **Dung lượng và tải dữ liệu:** archive chính thức được README ghi khoảng
  5 GB, ngoài ra cần dung lượng để giải nén. Chưa tải để kiểm tra.
- **Nhiễu nhãn:** Training set gốc có thể chứa ảnh nhiễu hoặc nhãn sai, có thể
  ảnh hưởng kết quả và cần được cân nhắc khi diễn giải.
- **Khác biệt phiên bản:** metadata, đường dẫn tải và pretrained weights có
  thể khác giữa thư viện hoặc bản sao dataset; cần ghi rõ phiên bản thư viện,
  weights và nguồn dữ liệu khi triển khai.
- **Tài nguyên tính toán:** ResNet-50 và ViT-B/16 có thể đòi hỏi tài nguyên
  lớn hơn các mô hình E1/E2. Chưa đánh giá thời gian chạy, mức dùng bộ nhớ hay
  khả năng đáp ứng batch size đề xuất.
- **Công bằng so sánh:** khác biệt về ngân sách huấn luyện, batch size hoặc
  cấu hình có thể ảnh hưởng diễn giải. Chưa đánh giá được mức độ ảnh hưởng
  trước khi chạy thực nghiệm.

Các hạn chế phụ thuộc kết quả, bao gồm xu hướng nhầm lẫn giữa lớp và hiệu suất
tương đối của hai kiến trúc, chưa được đánh giá.

## 9. Kết luận và công việc tiếp theo

A1.1 đã xác định Food-101, split dự kiến, preprocessing, hai kiến trúc,
training regimes và quy tắc evaluation dưới dạng protocol. Đây là công tác
chuẩn bị; chưa có dữ liệu được tải, pipeline được triển khai hoặc mô hình được
đánh giá. Vì vậy chưa thể kết luận CNN hay Vision Transformer tốt hơn.

Các bước tiếp theo theo protocol:

1. Xác nhận điều kiện sử dụng dữ liệu và tải Food-101 từ nguồn chính thức khi
   được phép.
2. Triển khai split phân tầng seed `42`, lưu manifest và kiểm tra số lượng/
   nhãn từng split.
3. Triển khai preprocessing và pipeline huấn luyện cho ResNet-50, ViT-B/16 và
   các training regimes đã đề xuất.
4. Chốt và ghi lại phiên bản thư viện, pretrained weights, thiết bị và cấu
   hình chạy thực tế; thay đổi do giới hạn phần cứng cần được nêu rõ.
5. Chọn checkpoint trên Validation set, sau đó đánh giá Test set theo protocol;
   lưu metrics, thời gian và phân tích lỗi nếu thực hiện được.
6. Cập nhật báo cáo này bằng kết quả đã kiểm chứng, thảo luận hạn chế và kết
   luận dựa trên dữ liệu.

## Tài liệu tham khảo

1. ETH Zurich, Computer Vision Lab. [Food-101 dataset — trang chính thức](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).
2. L. Bossard, M. Guillaumin và L. Van Gool. “Food-101 — Mining Discriminative Components with Random Forests.” ECCV 2014. [PDF từ ETH Zurich](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/static/bossard_eccv14_food-101.pdf).
3. Voxel51. [Food-101 dataset card](https://docs.voxel51.com/dataset_zoo/datasets_hf/food101.html). Nguồn thứ cấp được README A1.1 dẫn để cung cấp ngữ cảnh về nguồn ảnh và điều kiện sử dụng; không phải license do chủ sở hữu ảnh cấp.
