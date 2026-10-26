# A1.1 — Dataset + Protocol

## Trạng thái

Đây là **protocol dự kiến** cho A1.1, không phải báo cáo kết quả thực nghiệm.
Hiện chưa triển khai pipeline A1.1, chưa tải dataset và chưa huấn luyện hoặc
đánh giá mô hình. Không có số liệu accuracy, macro-F1 hay thời gian huấn luyện
được báo cáo trong tài liệu này.

## Dataset overview và lý do lựa chọn

Food-101 là bộ dữ liệu phân loại ảnh gồm 101 lớp món ăn và 101.000 ảnh, với
1.000 ảnh mỗi lớp. Ảnh có định dạng màu và được chuẩn hóa bởi tác giả dataset
sao cho cạnh dài nhất không vượt quá 512 pixel. Tập train có ảnh nhiễu và nhãn
có thể không chính xác; 250 ảnh test mỗi lớp được rà soát thủ công.

Food-101 được chọn vì có quy mô và số lớp lớn hơn các dataset đồ chơi thường
dùng trong bài tập nhập môn, đủ để khảo sát cả CNN và Vision Transformer trên
cùng bài toán phân loại đa lớp.

Nguồn thông tin và tải:

- [Trang Food-101 của ETH Zurich](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
  mô tả dataset, cung cấp tải chính thức `food-101.tar.gz` (trang ghi 5 GB) và
  hướng dẫn trích dẫn.
- Paper gốc: Bossard, Guillaumin và Van Gool, *Food-101 — Mining Discriminative
  Components with Random Forests*, ECCV 2014. [PDF từ ETH Zurich](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/static/bossard_eccv14_food-101.pdf).

### License và hạn chế chia sẻ

Trang dataset của ETH Zurich không nêu một license mở riêng cho ảnh trong nội
dung đã kiểm tra. Một [dataset card của Voxel51](https://docs.voxel51.com/dataset_zoo/datasets_hf/food101.html)
cho biết ảnh bắt nguồn từ Foodspotting, không thuộc ETH Zurich, và việc sử dụng
ngoài “scientific fair use” cần được thỏa thuận với chủ sở hữu ảnh theo điều
khoản Foodspotting. Đây là nguồn thứ cấp, không phải giấy phép cấp quyền từ chủ
sở hữu ảnh.

Vì vậy, tài liệu này **không kết luận** ảnh được cấp phép tự do, dùng thương
mại hoặc tái phân phối. License riêng của metadata/nhãn cũng chưa được xác
minh. Nếu sử dụng dataset cho bài tập, chỉ tải từ nguồn chính thức và dẫn paper;
không đưa archive, ảnh dataset hoặc bản sao ảnh vào repository. Việc sử dụng
phải tuân thủ quy định môn học và các quyền áp dụng; thông tin hiện có không
thay thế tư vấn pháp lý.

## Split protocol

Giữ nguyên test split do Food-101 cung cấp. Khi tạo split, sắp xếp tên lớp và
định danh ảnh tương đối theo thứ tự từ điển; khởi tạo một bộ sinh ngẫu nhiên
NumPy với seed `42`; lần lượt xáo trộn danh sách ảnh của từng lớp và lấy 75 ảnh
mỗi lớp làm validation. 675 ảnh còn lại mỗi lớp là train. Như vậy validation
được lấy phân tầng và seed `42` quyết định lựa chọn ảnh. Ghi phiên bản NumPy
cùng manifest để có thể kiểm tra/tái tạo phép chia; mọi model và regime phải
dùng chính manifest đó thay vì chia lại độc lập.

| Split | Ảnh mỗi lớp | Tổng ảnh | Nguồn/cách tạo |
|---|---:|---:|---|
| Train | 675 | 68.175 | 90% train gốc mỗi lớp |
| Validation | 75 | 7.575 | 10% train gốc mỗi lớp, seed `42` |
| Test | 250 | 25.250 | Test gốc, giữ nguyên |
| **Tổng** | **1.000** | **101.000** | |

Phép tính: `101 × 675 = 68.175`, `101 × 75 = 7.575` và
`101 × 250 = 25.250`. Khi triển khai, nên lưu manifest gồm đường dẫn/định danh
tương đối của từng ảnh, nhãn và split (train, validation hoặc test); manifest
cần được dùng lại cho mọi cấu hình. Manifest **chưa được tạo**. Không đưa ảnh
dataset vào Git; chỉ lưu manifest nếu không chứa nội dung ảnh và tuân theo các
hạn chế sử dụng dataset. Test chỉ dùng đánh giá cuối cùng, không dùng chọn
checkpoint, siêu tham số hay preprocessing.

## Preprocessing và augmentation dự kiến

Các bước sau là protocol cần triển khai, chưa được chạy hoặc kiểm chứng:

1. Đọc ảnh RGB và giữ nguyên nhãn trong tập 101 lớp.
2. Resize cạnh ngắn về 256 pixel; ở train, lấy random crop `224 × 224` và
   random horizontal flip với xác suất `0.5`.
3. Ở validation/test, dùng resize cạnh ngắn 256 pixel rồi center crop
   `224 × 224`; không áp dụng augmentation ngẫu nhiên.
4. Chuẩn hóa từng kênh bằng mean `(0.485, 0.456, 0.406)` và std
   `(0.229, 0.224, 0.225)`. Dùng cùng preprocessing cho hai kiến trúc và mọi
   training regime để so sánh nhất quán.

Không dự kiến dùng color jitter mạnh vì thay đổi màu có thể làm mất đặc trưng
hữu ích của món ăn. Có thể điều chỉnh preprocessing nếu weights được chọn yêu
cầu transform khác, nhưng mọi thay đổi cần được ghi lại và áp dụng nhất quán.

## Mô hình và training regimes

Dùng các model builder phù hợp trong Torchvision; thay classifier head thành
101 đầu ra. Tên/version weights pretrained cần được ghi cố định trong cấu hình
khi triển khai, vì alias “default” có thể thay đổi theo phiên bản thư viện.

| Kiến trúc | From scratch | Pretrained: frozen backbone | Pretrained: partial fine-tuning | Pretrained: full fine-tuning |
|---|---|---|---|---|
| ResNet-50 | Khởi tạo ngẫu nhiên; huấn luyện toàn bộ mạng | Khởi tạo từ ImageNet; đóng băng backbone, chỉ huấn luyện head 101 lớp | Khởi tạo từ ImageNet; mở `layer4` và head, đóng băng các stage trước | Khởi tạo từ ImageNet; huấn luyện toàn bộ mạng và head |
| ViT-B/16 | Khởi tạo ngẫu nhiên; huấn luyện toàn bộ mạng | Khởi tạo từ ImageNet; đóng băng encoder backbone, chỉ huấn luyện head 101 lớp | Khởi tạo từ ImageNet; mở block Transformer cuối cùng và head, đóng băng các block trước | Khởi tạo từ ImageNet; huấn luyện toàn bộ encoder và head |

Với mọi regime, head phân loại phải được thay/khởi tạo cho 101 lớp. So sánh cần
dùng cùng split, preprocessing, metric và quy tắc chọn checkpoint; ghi lại cấu
hình thực tế để kết quả có thể diễn giải. Số tham số trainable sẽ được tính sau
khi áp dụng chính sách đóng băng/mở từng phần.

### Cấu hình huấn luyện dự kiến

Các giá trị khởi điểm dưới đây là **đề xuất để chốt trước thực nghiệm**, chưa
phải cấu hình đã chạy. Có thể chọn learning rate khác cho các regime; cần chọn
trên validation và ghi rõ cấu hình thực tế.

| Thành phần | Giá trị/nguyên tắc khởi điểm |
|---|---|
| Optimizer | AdamW cho tất cả cấu hình để tạo một baseline chung |
| Learning rate | Scratch: `1e-3`; frozen head: `1e-3`; partial/full fine-tuning: backbone `1e-4`, head `1e-3` với parameter groups riêng. Đây là giá trị khởi điểm đề xuất, cần chốt trước khi chạy |
| Batch size | `32` ảnh trên mỗi thiết bị nếu bộ nhớ cho phép; nếu phải giảm, ghi batch size và gradient accumulation thực tế |
| Epoch tối đa | `30` epoch cho mỗi cấu hình |
| Early stopping/checkpoint | Theo dõi macro-F1 validation; lưu checkpoint tốt nhất, dùng accuracy validation để phân xử hòa; dừng nếu không cải thiện sau `5` epoch liên tiếp hoặc đạt epoch tối đa |
| Seed huấn luyện | `42` cho Python, NumPy, PyTorch CPU/CUDA/MPS nếu hỗ trợ, sampler và worker DataLoader. Ghi nhận rằng một số phép toán/thiết bị có thể không hoàn toàn xác định |
| Thiết bị và thời gian | Ghi thiết bị cụ thể (CPU, CUDA hoặc MPS), phiên bản runtime và batch size. Đo wall-clock từ batch train đầu đến optimizer update cuối, gồm thời gian nạp batch nhưng loại trừ validation, tải dữ liệu ban đầu và test; đồng bộ accelerator trước/sau phép đo nếu hỗ trợ |

Nếu giới hạn phần cứng buộc phải thay batch size, epoch budget, precision hoặc
phần backbone được mở, ghi lại thay đổi đó cho từng run.

## Metrics và evaluation protocol

- **Accuracy:** tỷ lệ dự đoán đúng trên split được đánh giá.
- **Macro-F1:** tính F1 riêng từng lớp rồi lấy trung bình không trọng số trên
  101 lớp.
- Ghi nhận thời gian huấn luyện cho mỗi regime và số tham số trainable nếu khả
  thi; dùng cùng cách đo và nêu rõ thiết bị/phạm vi thời gian được tính.
- Chọn checkpoint theo macro-F1 validation cao nhất; nếu bằng nhau, dùng
  accuracy validation để phân xử. Không xem test trong quá trình chọn mô hình.
- Sau khi chốt checkpoint/cấu hình trên validation, đánh giá test một lần cho
  mỗi cấu hình được báo cáo. Báo cáo accuracy và macro-F1 test cùng cách tính.
- Các mô hình dùng chung manifest split, định nghĩa metrics, preprocessing và
  quy tắc chọn checkpoint. Ghi cấu hình thực tế, phiên bản PyTorch/Torchvision,
  định danh chính xác pretrained weights, seed, thiết bị và mọi thay đổi do
  giới hạn phần cứng.
- Đây là so sánh có kiểm soát theo protocol, không mặc nhiên là công bằng tuyệt
  đối: nếu ngân sách huấn luyện, thời gian, batch size hoặc cấu hình khác nhau
  đáng kể, phải nêu rõ giới hạn khi diễn giải kết quả.

### Phân tích lỗi dự kiến

Sau đánh giá, xem confusion matrix, precision/recall/F1 theo lớp và các cặp
lớp thường bị nhầm. Có thể kiểm tra định tính các lỗi đại diện để nhận diện
nhầm lẫn thị giác hoặc ảnh/nhãn nhiễu. Không đưa ảnh Food-101 vào repository;
chỉ lưu phân tích tổng hợp hoặc hình/bảng không tái phân phối ảnh nếu quyền sử
dụng chưa được xác nhận.

## Giới hạn, rủi ro và xử lý

- **License ảnh chưa rõ:** không tái phân phối ảnh; ghi nguồn và giới hạn như
  trên. Nếu yêu cầu môn học đòi hỏi license mở rõ ràng, xác nhận với giảng viên
  trước khi sử dụng dataset.
- **Dung lượng/tải:** archive chính thức khoảng 5 GB và cần thêm dung lượng để
  giải nén. Dùng link ETH, kiểm tra hoàn tất tải/đọc ảnh; không lưu dataset
  trong Git.
- **Nhiễu nhãn:** train gốc có thể chứa ảnh nhiễu hoặc nhãn sai, ảnh hưởng kết
  quả; giữ nguyên dữ liệu để protocol nhất quán và đề cập trong diễn giải.
- **Khác biệt phiên bản:** thư viện có thể thay đổi URL tải, metadata hoặc
  pretrained weights. Ghi phiên bản Torch/Torchvision và định danh weights khi
  triển khai; không giả định các bản sao Kaggle/Hugging Face có cùng cấu trúc,
  split hoặc điều khoản.
- **Chi phí tính toán:** ResNet-50 và ViT-B/16 lớn hơn mô hình trong E1/E2.
  Chọn batch size/thiết bị phù hợp nhưng giữ chung cấu hình so sánh khi khả thi;
  ghi lại mọi khác biệt bắt buộc.

## Trạng thái đáp ứng A1.1

Trang assignment trong repository mô tả A1.1 là chọn dataset ảnh lớn và định
nghĩa dataset, splits, metrics cùng protocol so sánh CNN và Transformer qua
nhiều training regimes. Không tìm thấy rubric/marking scheme chi tiết hơn trong
repository; bảng dưới đây đối chiếu với yêu cầu đó và các thông số đã thống
nhất.

| Yêu cầu | Trạng thái |
|---|---|
| Chọn dataset ảnh không phải MNIST, Fashion-MNIST hoặc CIFAR-10 | Đã xác định: Food-101 |
| Mô tả dataset, split và lý do chọn | Đã mô tả trong protocol; chưa kiểm tra dataset tải thực tế |
| Nêu nguồn và tình trạng license/hạn chế | Đã ghi nguồn và các điểm chưa xác minh; chưa có xác nhận cấp phép riêng |
| Định nghĩa preprocessing/augmentation | Đã đề xuất; chưa triển khai |
| So sánh CNN và Transformer | Đã xác định ResNet-50 và ViT-B/16 |
| Nêu from-scratch và các chế độ pretrained | Đã xác định scratch, frozen, partial và full fine-tuning |
| Metrics/evaluation và phân tích lỗi | Đã định nghĩa dự kiến; chưa chạy đánh giá |
| Kết quả thực nghiệm | Chưa có và không được suy diễn trong A1.1 |
