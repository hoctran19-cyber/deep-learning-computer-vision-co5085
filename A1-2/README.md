# A1.2 — Thí nghiệm CNN và Transformer

Thư mục này chứa phần triển khai và quy trình thực nghiệm cho bài tập A1.2.

## Môi trường thực hiện

- **Mac cục bộ (Intel):** chỉnh sửa mã nguồn, kiểm tra các thành phần nhỏ và quản lý Git.
- **Google Colab:** chuẩn bị Food-101, huấn luyện mô hình, kiểm định và đánh giá cuối cùng trên tập test.
- **Google Drive:** lưu bộ dữ liệu, checkpoint, log, các chỉ số và hình ảnh được tạo ra.
- **GitHub:** quản lý phiên bản mã nguồn, cấu hình, notebook và báo cáo. Không commit kho ảnh Food-101 hoặc các checkpoint lớn.

## Các thí nghiệm dự kiến

Mỗi kiến trúc trong hai kiến trúc được đánh giá theo bốn chế độ huấn luyện:

| Kiến trúc | Chế độ huấn luyện |
|---|---|
| ResNet-50 | huấn luyện từ đầu, đóng băng backbone, tinh chỉnh một phần (layer4 + head), tinh chỉnh toàn bộ |
| ViT-B/16 | huấn luyện từ đầu, đóng băng backbone, tinh chỉnh một phần (khối encoder cuối + head), tinh chỉnh toàn bộ |

Quy trình thực nghiệm, chiến lược chia dữ liệu, tiền xử lý và các chỉ số được trình bày trong [A1-1](../A1-1/README.md). Phần triển khai phải tuân theo quy trình đó, trừ khi có thay đổi được phê duyệt và ghi lại.

## Cấu trúc repository

- `configs/config.yaml`: cấu hình chung cho thí nghiệm.
- `notebooks/A1_2_experiment.ipynb`: notebook khởi chạy trên Google Colab.
- `src/`: các mô-đun có thể tái sử dụng cho dataset, biến đổi dữ liệu, mô hình, huấn luyện, đánh giá và tiện ích.
- `scripts/`: các công cụ hỗ trợ chạy từ dòng lệnh.
- `outputs/`: cấu trúc lưu kết quả cục bộ; các tệp lớn được tạo ra không được Git theo dõi.

## Quy trình trên Colab

1. Mở `notebooks/A1_2_experiment.ipynb` trên Google Colab.
2. Chọn môi trường chạy có GPU nếu có và kiểm tra bằng `torch.cuda.is_available()`.
3. Gắn Google Drive và thiết lập đường dẫn dự án, dữ liệu và đầu ra trong notebook.
4. Clone hoặc pull nhánh `assignment_A1`.
5. Triển khai và kiểm tra pipeline trước khi chạy toàn bộ thí nghiệm.
6. Lưu checkpoint và các kết quả thí nghiệm lên Google Drive.
7. Xuất các bảng kết quả và hình ảnh gọn cho báo cáo.

## Dữ liệu và giấy phép

Ảnh Food-101 không được đưa vào repository này. Hãy xem các lưu ý về việc sử dụng dữ liệu và giấy phép trong [A1-1/README.md](../A1-1/README.md) trước khi tải xuống hoặc phân phối lại dữ liệu. Không tải ảnh gốc lên GitHub.

## Kiểm tra manifest Food-101

Từ thư mục gốc của repository, kiểm tra manifest độc lập với metadata chính
thức (không cần tệp ảnh hoặc thư viện bên thứ ba):

```bash
python3 A1-2/src/validate_manifest.py \
  --dataset-root A1-2/data/food-101-metadata \
  --manifest A1-2/data/manifests/food101_split_seed42.csv
```

`--dataset-root` phải chứa `meta/classes.txt`, `meta/train.json` và
`meta/test.json`. Công cụ kiểm tra thứ tự lớp và nhãn, đường dẫn, số lượng theo
từng tập, dữ liệu trùng lặp và sự khớp chính xác với các tập chính thức. Công
cụ in kết quả từng kiểm tra và trả về mã thoát khác 0 nếu thất bại. Công cụ
không kiểm tra sự tồn tại của tệp ảnh.

Chạy các kiểm thử tích hợp với dữ liệu giả lập bằng lệnh:

```bash
python3 -m unittest discover -s A1-2/tests -p 'test_validate_manifest.py'
```

## Kiểm thử nhanh DataLoader của Food-101

Sau khi cài đặt các phiên bản tương thích của PyTorch, Torchvision và Pillow,
và giải nén đầy đủ Food-101 với `meta/classes.txt` và `images/`, chạy:

```bash
python3 A1-2/src/test_dataloader.py \
  --dataset-root /path/to/food-101 \
  --manifest A1-2/data/manifests/food101_split_seed42.csv \
  --batch-size 8
```

Kiểm thử xác nhận số lượng mẫu của cả ba tập, mẫu ảnh RGB, hoạt động của
sampler, các batch tensor và batch cuối. Ảnh được đổi kích thước về 224×224
trước `ToTensor()` vì kích thước ảnh gốc khác nhau. Kiểm thử duyệt các chỉ số
của sampler để kiểm tra độ bao phủ, nhưng chỉ giải mã batch đầu, batch cuối và
một mẫu. Khi khởi tạo dataset, chương trình kiểm tra sự tồn tại của tất cả ảnh
trong tập tương ứng. Chương trình không huấn luyện hoặc tải dữ liệu; nếu thất
bại, mã thoát sẽ khác 0.

## Mô hình ResNet-50 và ViT-B/16

Hàm `build_resnet50(num_classes=101, pretrained=False)` trong
`src/models.py` tạo ResNet-50 với lớp phân loại phù hợp. Đặt
`pretrained=True` để dùng trọng số ImageNet mặc định của Torchvision; lần đầu
có thể cần tải trọng số. Chạy kiểm thử mà không tải trọng số:

Hàm `build_vit_b16(num_classes=101, pretrained=False)` trong cùng mô-đun
tạo ViT-B/16 với lớp phân loại có thể cấu hình. Tương tự, đặt
`pretrained=True` để dùng trọng số ImageNet mặc định của Torchvision, có thể
cần tải xuống khi chạy lần đầu.

Hàm `build_model(model_name, num_classes=101, pretrained=False)` chọn một
trong hai kiến trúc `resnet50` hoặc `vit_b16`. Cấu hình trong
`configs/config.yaml` liệt kê các kiến trúc và đặt `pretrained: false` mặc
định; hãy ghi đè giá trị này cho từng chế độ cần trọng số pretrained.
Chạy kiểm thử cho cả hai mô hình:

```bash
python3 -m unittest discover -s A1-2/tests -p 'test_models.py'
```

## Cấu hình tham số huấn luyện

`count_parameters(model)` trả về `(tổng số, số có thể huấn luyện)`;
`freeze_parameters(module)` và `unfreeze_parameters(module)` đóng băng hoặc
mở toàn bộ tham số của mô-đun. Dùng `configure_finetuning(model, regime)`
để áp dụng một trong các chế độ đã liệt kê trong cấu hình:

- `from_scratch` và `full_finetuning`: tất cả tham số có thể huấn luyện.
- `frozen_backbone`: chỉ head phân loại (`fc` của ResNet hoặc `heads` của ViT).
- `partial_finetuning`: `layer4` và `fc` của ResNet; block encoder cuối và
  `heads` của ViT. Các tham số còn lại, kể cả encoder norm cuối của ViT,
  được đóng băng.

Các hàm chỉ thay đổi `requires_grad`, không khởi tạo lại trọng số hoặc chuyển
chế độ train/eval. Khi huấn luyện từ đầu, cần tạo mô hình với
`pretrained=False`. Với ResNet, các thống kê BatchNorm vẫn có thể thay đổi
trong chế độ train ngay cả khi tham số của backbone đã đóng băng; vòng huấn
luyện sau này cần quản lý chế độ train/eval nếu muốn cố định các thống kê đó.

## Optimizer và scheduler

`src/optim.py` cung cấp `build_optimizer(model, config)` và
`build_scheduler(optimizer, config)`, nhận toàn bộ cấu hình YAML. Trước khi
tạo optimizer, đặt `training.regime` trùng chế độ truyền vào
`configure_finetuning(model, regime)`. Chế độ `from_scratch` dùng
`scratch_learning_rate` cho cả head và backbone; các chế độ khác dùng
`head_learning_rate` và `backbone_learning_rate` riêng. Chỉ tham số có
`requires_grad=True` được đưa vào optimizer; cần tạo lại optimizer sau khi
thay đổi các tham số được mở hoặc đóng băng.

Chọn `training.optimizer` là `AdamW` hoặc `SGD` (SGD dùng `momentum`).
Chọn `training.scheduler.name` là `CosineAnnealingLR` hoặc `StepLR`, với
tham số tương ứng trong các mục `cosine_annealing` và `step`. `T_max` và
`step_size` tính theo epoch; gọi `scheduler.step()` sau mỗi epoch, sau
`optimizer.step()`. Với cosine, `eta_min` không được lớn hơn learning rate
thấp nhất của các nhóm; StepLR dùng `0 < gamma <= 1`.

```bash
python3 -m unittest discover -s A1-2/tests -p 'test_optim.py'
```

## Huấn luyện một epoch

`train_one_epoch(model, dataloader, criterion, optimizer, device)` trong
`src/train.py` trả về loss trung bình theo mẫu, accuracy top-1 theo tỷ lệ
`[0, 1]` và `num_samples`. Trước khi gọi, chuyển model sang `device` và
chuẩn bị optimizer theo chế độ fine-tuning. DataLoader cần trả về batch
`(images, labels)` gồm các tensor, với nhãn `torch.int64` một chiều.

Criterion phải trả về loss vô hướng trung bình theo mẫu (`reduction="mean"`).
Khi dùng class weights hoặc `ignore_index`, caller phải bảo đảm loss vẫn là
trung bình theo tất cả mẫu của batch để loss epoch được tính đúng. Hàm không
thực hiện validation, scheduler, lưu checkpoint hay dừng sớm.

```bash
python3 -m unittest discover -s A1-2/tests -p 'test_train.py'
```

## Trạng thái hiện tại

Phần tải dữ liệu và tạo mô hình ResNet-50, ViT-B/16 đã được triển khai.
Vòng huấn luyện một epoch đã có; quy trình huấn luyện đầy đủ, đánh giá và
kết quả thí nghiệm vẫn chưa được triển khai.
