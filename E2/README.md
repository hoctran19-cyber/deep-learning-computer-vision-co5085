# CO5085 E2 - Transformer: MSA and Tokenization

E2 triển khai mô hình phân loại ảnh nhỏ dùng Transformer trên cùng quy ước dữ liệu của E1. Dataset mặc định là Fashion-MNIST, validation split 10%, seed 42 và normalize ảnh bằng mean/std 0.5.

Không có kết quả thực nghiệm được bịa sẵn. Các bảng trong tài liệu này là khung để điền sau khi chạy thật trên local hoặc Google Colab.

## Cấu trúc

`src/attention.py`: Custom Multi-Head Self-Attention và wrapper PyTorch `nn.MultiheadAttention`.

`src/tokenizers.py`: Patch, Row và CNN-stem tokenization.

`src/model.py`: Transformer image classifier nhỏ.

`src/data.py`: tải Fashion-MNIST/MNIST/CIFAR-10 và chia train/validation/test.

`src/train.py`: training loop PyTorch tự viết, lưu checkpoint, metrics CSV/JSON và hình.

`tests/`: unit tests cho attention, tokenizer, forward/backward.

`notebooks/e2_experiments.ipynb`: notebook Colab để chạy thí nghiệm.

`configs/default.yaml`: cấu hình mặc định.

`outputs/`: nơi sinh metrics, checkpoint và biểu đồ khi chạy.

## Custom MSA

`CustomMultiHeadSelfAttention` nhận input shape `(batch_size, num_tokens, embedding_dim)` và thực hiện:

1. Chiếu tuyến tính thành Query, Key, Value.
2. Chia embedding thành `num_heads` head.
3. Tính `QK^T`.
4. Chia cho `sqrt(d_k)`.
5. Softmax để lấy attention weights.
6. Nhân attention weights với Value.
7. Ghép các head.
8. Chiếu đầu ra.

Module này không dùng `nn.MultiheadAttention` hoặc Transformer dựng sẵn. Code kiểm tra `embedding_dim` chia hết cho `num_heads`.

## PyTorch MSA

`TorchMultiHeadSelfAttention` dùng `nn.MultiheadAttention(batch_first=True)` với cùng input/output shape như custom MSA. Hai phiên bản dùng chung cấu hình quan trọng: `embedding_dim`, `num_heads`, `dropout`.

## Tokenization

Patch tokenization: ảnh được chia thành patch vuông không chồng lấp. Với Fashion-MNIST `1x28x28` và `patch_size=7`, số token là `4x4=16`, mỗi patch có 49 giá trị trước khi chiếu sang `embedding_dim`.

Row tokenization: mỗi hàng pixel là một token. Với Fashion-MNIST, số token là 28, mỗi token có `channels * width = 28` giá trị trước khi chiếu. Cách này giữ cấu trúc theo hàng ngang và khác patch tokenization.

CNN-stem tokenization: CNN nhỏ tạo feature map, sau đó mỗi vị trí không gian của feature map là một token. Với Fashion-MNIST và hai convolution stride 2, feature map khoảng `7x7`, tức 49 token. Cách này thêm inductive bias cục bộ của CNN nhưng có thêm tham số.

Chi phí attention tỷ lệ xấp xỉ `O(num_tokens^2 * embedding_dim)`, nên tokenizer tạo nhiều token sẽ tốn bộ nhớ/thời gian hơn.

## Cài đặt local

Chạy từ thư mục `E2/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Kiểm tra nhanh:

```bash
pytest
```

Smoke test 1 epoch trên Fashion-MNIST:

```bash
python -m src.train --config configs/default.yaml --experiment-name smoke_custom_patch --epochs 1 --batch-size 64
```

Code tự chọn thiết bị theo thứ tự CUDA -> MPS -> CPU. MacBook không cần CUDA.

## Chạy thí nghiệm

Thí nghiệm A - so sánh implementation MSA, giữ tokenizer patch:

```bash
python -m src.train --config configs/default.yaml --experiment-name msa_custom_patch --attention custom --tokenizer patch
python -m src.train --config configs/default.yaml --experiment-name msa_torch_patch --attention torch --tokenizer patch
```

Thí nghiệm B - so sánh tokenization, giữ custom MSA:

```bash
python -m src.train --config configs/default.yaml --experiment-name tok_patch_custom --attention custom --tokenizer patch
python -m src.train --config configs/default.yaml --experiment-name tok_row_custom --attention custom --tokenizer row
python -m src.train --config configs/default.yaml --experiment-name tok_cnn_stem_custom --attention custom --tokenizer cnn_stem
```

Các hyperparameters chính nằm trong `configs/default.yaml`: `embedding_dim`, `num_heads`, `num_blocks`, `batch_size`, `learning_rate`, `epochs`, `patch_size`, `dropout`.

## Google Colab

Notebook `notebooks/e2_experiments.ipynb` được thiết kế để chạy trên Colab. Nếu clone từ GitHub, dùng URL thật của repo hiện tại:

```bash
!git clone git@github.com:hoctran19-cyber/deep-learning-computer-vision-co5085.git
%cd deep-learning-computer-vision-co5085/E2
```

Nếu dùng HTTPS trong Colab, đổi URL SSH trên thành URL HTTPS tương ứng của cùng repository.

Google Drive là tùy chọn. Nếu muốn lưu kết quả bền vững sau khi runtime bị ngắt, mount Drive và đặt `output.copy_to_dir` hoặc copy thủ công thư mục `outputs/` sang Drive. Nếu không mount Drive, kết quả vẫn lưu cục bộ trong runtime Colab và cần tải về trước khi ngắt phiên.

## Kết quả sinh ra

Mỗi lần chạy tạo:

`outputs/metrics/<experiment>.json`: config, history, test metrics, training time, inference time.

`outputs/metrics/comparison.csv`: bảng so sánh các lần chạy.

`outputs/checkpoints/<experiment>.pt`: checkpoint tốt nhất theo validation accuracy.

`outputs/figures/<experiment>_learning_curves.png`: loss/accuracy theo epoch.

`outputs/figures/<experiment>_confusion_matrix.png`: confusion matrix trên test set.

## Bảng kết quả cần điền sau khi chạy thật

| Nhóm | Experiment | MSA | Tokenizer | Test accuracy | Best validation accuracy | Params | Training time | Inference time |
|---|---|---|---|---:|---:|---:|---:|---:|
| A | msa_custom_patch | custom | patch | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy |
| A | msa_torch_patch | torch | patch | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy |
| B | tok_patch_custom | custom | patch | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy |
| B | tok_row_custom | custom | row | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy |
| B | tok_cnn_stem_custom | custom | cnn_stem | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy |

## Nhận xét dự kiến cần kiểm chứng

Patch tokenizer thường ít token hơn row/cnn-stem trong cấu hình hiện tại nên attention rẻ hơn.

Row tokenizer đơn giản, dễ giải thích, nhưng token theo hàng có thể kém phù hợp với cấu trúc cục bộ 2D.

CNN-stem tokenizer có thêm đặc trưng cục bộ trước attention, nhưng nhiều phép tính hơn và không còn là tokenization tuyến tính thuần túy.

Các nhận xét định lượng chỉ nên viết sau khi đã chạy thí nghiệm thật và đọc `comparison.csv`.

## Hạn chế

Mô hình nhỏ để phù hợp Colab và smoke test local, không nhằm đạt SOTA.

Kết quả có thể khác giữa CUDA, MPS và CPU.

Notebook và README không chứa số liệu thực nghiệm nếu chưa chạy thật.
