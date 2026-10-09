"""Các mô hình phân loại ảnh PyTorch được sử dụng trong bài tập E1.

Mô-đun cung cấp ba kiến trúc: bộ phân loại tuyến tính trên ảnh đã làm phẳng,
mạng perceptron nhiều lớp (MLP) và mạng tích chập (CNN). Hàm ``get_model``
dựa trên cấu hình để khởi tạo kiến trúc tương ứng.
"""

from typing import Any, Sequence

import torch
from torch import Tensor, nn


class SoftmaxClassifier(nn.Module):
    """Làm phẳng ảnh đầu vào rồi ánh xạ tuyến tính tới các lớp phân loại.

    Lớp tuyến tính tạo ra một điểm số (logit) cho mỗi lớp. Không áp dụng
    Softmax ở đây; các hàm mất mát như ``CrossEntropyLoss`` thường nhận trực
    tiếp logits và tự xử lý bước chuẩn hóa cần thiết.
    """

    def __init__(self, input_shape: Sequence[int], num_classes: int) -> None:
        super().__init__()
        # Chuyển ảnh dạng (kênh, cao, rộng) thành vector đặc trưng một chiều.
        self.flatten = nn.Flatten()
        
        # Mỗi pixel là một đầu vào của lớp tuyến tính; đầu ra có num_classes điểm số.
        self.classifier = nn.Linear(_input_size(input_shape), num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Trả về logits phân loại cho mỗi ảnh trong lô đầu vào."""
        return self.classifier(self.flatten(x))


class MLPClassifier(nn.Module):
    """MLP gồm đầu vào làm phẳng, các lớp ẩn kết nối đầy đủ và lớp đầu ra.

    Số lượng/nút của lớp ẩn được cấu hình qua ``hidden_dims``. Mỗi lớp ẩn
    dùng ReLU; Dropout chỉ được thêm sau lớp ẩn khi tỷ lệ ``dropout`` lớn hơn 0.
    """

    def __init__(self, input_shape: Sequence[int], num_classes: int,
                 hidden_dims: Sequence[int] = (128, 64), dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        # Kích thước đầu vào của lớp đầu tiên bằng tổng số phần tử trong một ảnh.
        in_features = _input_size(input_shape)
        for hidden_dim in hidden_dims:
            # Mỗi khối ẩn gồm lớp tuyến tính và hàm kích hoạt phi tuyến ReLU.
            layers.extend([nn.Linear(in_features, hidden_dim), nn.ReLU()])
            if dropout > 0:
                # Dropout ngẫu nhiên tắt một phần nút khi huấn luyện để hạn chế quá khớp.
                layers.append(nn.Dropout(dropout))
            # Đầu ra của lớp hiện tại trở thành đầu vào của lớp kế tiếp.
            in_features = hidden_dim
        # Lớp cuối tạo một logit cho mỗi lớp, không kèm hàm kích hoạt.
        layers.append(nn.Linear(in_features, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Truyền lô ảnh qua toàn bộ mạng tuần tự để tạo logits."""
        return self.network(x)


class CNNClassifier(nn.Module):
    """CNN gồm hai khối tích chập/thu gọn và một đầu phân loại.

    Mỗi khối tích chập dùng kernel 3x3, đệm để giữ kích thước không gian trước
    khi gộp cực đại 2x2. Kích thước đầu vào cho lớp phân loại được suy ra từ
    ``input_shape``, nên đầu phân loại phù hợp với kích thước ảnh đã cấu hình.
    """

    def __init__(self, input_shape: Sequence[int], num_classes: int,
                 channels: Sequence[int] = (32, 64), dropout: float = 0.0) -> None:
        super().__init__()
        # Hai phần tử lần lượt xác định số kênh đầu ra của hai lớp tích chập.
        first, second = channels
        self.features = nn.Sequential(
            # padding=1 giữ nguyên chiều cao/rộng sau tích chập kernel 3x3;
            # MaxPool2d(2) sau đó giảm một nửa mỗi chiều không gian.
            nn.Conv2d(input_shape[0], first, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(first, second, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        # Chạy một ảnh giả để xác định số đặc trưng sau các lớp tích chập/gộp,
        # tránh phải tính thủ công theo kích thước đầu vào.
        with torch.no_grad():
            feature_size = self.features(torch.zeros(1, *input_shape)).flatten(1).shape[1]
        # Làm phẳng bản đồ đặc trưng rồi ánh xạ sang logits của các lớp.
        head: list[nn.Module] = [nn.Flatten(), nn.Linear(feature_size, num_classes)]
        if dropout > 0:
            # Đặt Dropout giữa bước làm phẳng và lớp phân loại cuối.
            head.insert(1, nn.Dropout(dropout))
        self.classifier = nn.Sequential(*head)

    def forward(self, x: Tensor) -> Tensor:
        """Trích xuất đặc trưng không gian rồi tính logits cho từng ảnh."""
        return self.classifier(self.features(x))


def _input_size(input_shape: Sequence[int]) -> int:
    """Tính tổng số phần tử của một ảnh từ dạng (kênh, cao, rộng)."""
    size = 1
    for dimension in input_shape:
        # Nhân các chiều để thu được số đặc trưng sau khi làm phẳng.
        size *= dimension
    return size


def get_model(name: str, config: dict[str, Any], input_shape: Sequence[int] = (1, 28, 28),
              num_classes: int = 10) -> nn.Module:
    """Khởi tạo mô hình theo tên và các tham số trong cấu hình.

    ``name`` nhận một trong ba giá trị ``softmax``, ``mlp`` hoặc ``cnn``.
    Cấu hình ``model.dropout`` được dùng cho MLP/CNN; các kích thước lớp ẩn
    và số kênh tích chập lần lượt đọc từ ``mlp_hidden_dims`` và ``cnn_channels``.
    ``input_shape`` có dạng (kênh, cao, rộng), còn ``num_classes`` là số lớp đầu ra.
    """
    model_config = config.get("model", {})
    model_name = name.lower()
    # Dùng cùng một thiết lập Dropout từ cấu hình cho các kiến trúc có hỗ trợ.
    kwargs = {"dropout": float(model_config.get("dropout", 0.0))}
    if model_name == "softmax":
        return SoftmaxClassifier(input_shape, num_classes)
    if model_name == "mlp":
        return MLPClassifier(input_shape, num_classes, model_config.get("mlp_hidden_dims", [128, 64]), **kwargs)
    if model_name == "cnn":
        return CNNClassifier(input_shape, num_classes, model_config.get("cnn_channels", [32, 64]), **kwargs)
    raise ValueError("Unknown model. Choose softmax, mlp, or cnn.")
