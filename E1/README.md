# CO5085 E1 — Softmax, MLP và CNN

Bài tập này so sánh ba bộ phân loại ảnh PyTorch được huấn luyện thủ công trên
cùng một cách chia train/validation/test của Fashion-MNIST: bộ phân loại
Softmax tuyến tính, MLP nhận vector ảnh đã làm phẳng và mạng nơ-ron tích chập
(CNN). Có thể chọn MNIST hoặc CIFAR-10 trong file cấu hình.

Các kết quả trong tài liệu này lấy từ lần chạy thật bằng
`notebooks/E1_experiment.ipynb`, được lưu tại `results/`.

## Cấu trúc dự án

`src/datasets.py` xử lý việc chia dữ liệu cố định và DataLoader; `src/models.py`
chứa ba mô hình; `src/train.py` chứa vòng lặp huấn luyện tường minh
`zero_grad → forward → loss → backward → step` và giao diện dòng lệnh;
`src/evaluate.py` chứa các hàm kiểm thử, dự đoán và phân tích lỗi; `src/utils.py`
cung cấp hạt giống ngẫu nhiên, tự chọn thiết bị, checkpoint và vẽ biểu đồ. Cấu
hình nằm trong `configs/config.yaml`; notebook chỉ là nơi chạy thí nghiệm bằng
cách import các module này, không lặp lại mã nguồn.

## Cài đặt và phát triển cục bộ

Chạy từ thư mục E1:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Khi chạy cục bộ, code tự dùng CPU nếu máy không có CUDA. Dữ liệu được lưu trong
thư mục `data/` của dự án; đặt `dataset.download: true` để tải dữ liệu khi chạy
thí nghiệm thật.

## Chạy thí nghiệm

Tất cả lệnh chạy từ thư mục `E1/`:

```bash
python3 -m src.train --model softmax
python3 -m src.train --model mlp
python3 -m src.train --model cnn
```

Có thể ghi đè tham số mà không cần sửa YAML:

```bash
python3 -m src.train --model cnn --epochs 10 --batch-size 128 --lr 0.001 \
  --config configs/config.yaml
```

Để so sánh công bằng, giữ nguyên seed, cách chia dữ liệu và cấu hình cho cả ba
lần chạy. Thiết bị mặc định được chọn bằng
`cuda if torch.cuda.is_available() else cpu`, không yêu cầu gói hay GPU riêng.

## Chạy trên Google Colab

1. Mở Colab và chọn runtime GPU nếu muốn.
2. Clone repository:

```bash
!git clone https://github.com/hoctran19-cyber/deep-learning-computer-vision-co5085.git
%cd deep-learning-computer-vision-co5085/E1
!pip install -r requirements.txt
```

3. Chạy ba lệnh ở trên (hoặc notebook `notebooks/E1_experiment.ipynb`). Code tự
   dùng GPU nếu có, nếu không thì dùng CPU.
4. Xem hoặc tải thư mục `results/`, sau đó commit và push các kết quả cần nộp.

GitHub đồng bộ mã nguồn và cấu hình giữa máy cục bộ và Colab. Không commit
checkpoint lớn nếu không có yêu cầu.

## Kết quả sinh ra và khả năng tái lập

Mỗi lần chạy ghi kết quả vào `results/metrics/<model>.json`, cập nhật
`results/metrics/model_comparison.csv`, lưu biểu đồ trong `results/figures/` và
lưu checkpoint trong `results/checkpoints/`. Bảng so sánh gồm tên mô hình, độ
chính xác kiểm thử, số tham số, độ chính xác validation tốt nhất và thời gian
huấn luyện. `data/` và checkpoint mặc định bị Git bỏ qua; các file JSON/CSV/PNG
nhỏ vẫn có thể commit.

Seed cấu hình được dùng cho Python, NumPy, PyTorch, bộ sinh chia dữ liệu và
DataLoader. Kết quả vẫn có thể khác nhau chút ít giữa các thiết bị.

## Kết quả thực nghiệm

Cấu hình: Fashion-MNIST, validation split 10%, seed 42, optimizer Adam,
`batch_size=128`, `epochs=10`, `learning_rate=0.001`. Số liệu lấy từ
`results/metrics/model_comparison.csv`.

| Mô hình | Test accuracy | Best validation accuracy | Số tham số | Thời gian huấn luyện (s) |
|---|---:|---:|---:|---:|
| Softmax | 0,8419 | 0,8525 | 7.850 | 105,0 |
| MLP | 0,8792 | 0,8867 | 109.386 | 107,0 |
| CNN | **0,9111** | **0,9203** | 50.186 | 113,0 |

Biểu đồ đường cong huấn luyện, ma trận nhầm lẫn và các mẫu bị phân loại sai của
từng mô hình nằm trong `results/figures/`. Phân tích chi tiết xem
[results/analysis.md](results/analysis.md).

## Nhận xét

CNN đạt độ chính xác kiểm thử cao nhất (91,11%), tiếp theo là MLP (87,92%) và
Softmax (84,19%). CNN hơn Softmax 6,92 điểm phần trăm và hơn MLP 3,19 điểm phần
trăm.

CNN dùng ít tham số hơn MLP (50.186 so với 109.386) nhưng chính xác hơn, vì phép
tích chập khai thác cấu trúc không gian 2D và chia sẻ trọng số, trong khi MLP
làm phẳng ảnh và mất thông tin vị trí lân cận. Softmax nhỏ nhất (7.850 tham số)
nhưng bị giới hạn bởi ranh giới quyết định tuyến tính.

Thời gian huấn luyện của ba mô hình khá gần nhau; CNN lâu hơn một chút.

## Hạn chế

Mỗi mô hình chỉ chạy một lần (seed 42, 10 epoch), nên chênh lệch nhỏ có thể nằm
trong dao động ngẫu nhiên. Các con số trong `results/analysis.md` được tổng hợp
từ một lần chạy trước và lệch nhẹ so với `model_comparison.csv` (ví dụ CNN
91,05%); bảng ở trên theo lần chạy mới nhất.
