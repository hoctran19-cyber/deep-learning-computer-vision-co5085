"""Các tiện ích dùng chung cho tái lập thí nghiệm, lưu trữ và trực quan hóa.

Mô-đun tập trung những thao tác phụ trợ được dùng ở nhiều nơi trong dự án:
thiết lập seed, chọn thiết bị, đọc cấu hình, lưu/tải checkpoint và chỉ số,
cũng như tạo biểu đồ phục vụ phân tích kết quả mô hình.
"""

import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn


def set_seed(seed: int) -> None:
    """Đặt seed cho bộ sinh số ngẫu nhiên của Python, NumPy và PyTorch.

    Việc này giúp các thao tác ngẫu nhiên có kết quả ổn định hơn giữa những
    lần chạy cùng cấu hình. Nếu CUDA khả dụng, seed cũng được đặt cho mọi GPU.
    """
    # Đồng bộ seed giữa các thư viện thường dùng để tạo số ngẫu nhiên.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # Đặt seed cho toàn bộ thiết bị CUDA, không chỉ GPU mặc định.
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Chọn CUDA nếu có sẵn; nếu không thì dùng CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: str | Path) -> dict[str, Any]:
    """Đọc tệp cấu hình YAML và trả về nội dung dưới dạng từ điển.

    Tệp được mở với UTF-8 để hỗ trợ nội dung Unicode. ``safe_load`` chỉ phân
    tích các kiểu dữ liệu YAML an toàn thay vì khởi tạo đối tượng tùy ý.
    """
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_path(path: str | Path, project_root: Path) -> Path:
    """Chuyển đường dẫn tương đối thành đường dẫn tính từ thư mục dự án.

    Nếu ``path`` đã là đường dẫn tuyệt đối thì giữ nguyên; nhờ đó chương trình
    không phụ thuộc vào thư mục làm việc hiện tại của từng máy.
    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def count_trainable_parameters(model: nn.Module) -> int:
    """Đếm số phần tử tham số đang được phép cập nhật trong mô hình.

    Chỉ các tham số có ``requires_grad=True`` mới được tính; đây là số lượng
    phần tử vô hướng, không phải số lượng tensor tham số.
    """
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_checkpoint(model: nn.Module, path: str | Path, optimizer: torch.optim.Optimizer | None = None,
                    epoch: int | None = None) -> None:
    """Lưu trạng thái mô hình và metadata tùy chọn vào checkpoint PyTorch.

    Thư mục cha được tạo nếu chưa tồn tại. Checkpoint luôn chứa trọng số mô
    hình; trạng thái optimizer và số epoch chỉ được thêm khi được truyền vào.
    """
    # Tạo thư mục đích trước khi ghi tệp để hỗ trợ cả đường dẫn mới.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        # Lưu thêm trạng thái optimizer để có thể tiếp tục quá trình huấn luyện.
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    torch.save(checkpoint, path)


def load_checkpoint(model: nn.Module, path: str | Path, device: torch.device) -> dict[str, Any]:
    """Nạp trọng số từ checkpoint vào mô hình và trả lại toàn bộ metadata.

    ``map_location`` ánh xạ các tensor sang thiết bị được chỉ định, giúp có
    thể nạp checkpoint được tạo trên GPU vào môi trường CPU (hoặc ngược lại).
    Trạng thái optimizer/epoch nếu có sẽ được trả về nhưng không tự khôi phục
    vào optimizer tại đây.
    """
    checkpoint = torch.load(path, map_location=device)
    # Tải riêng trạng thái mô hình; các metadata khác được giữ trong kết quả trả về.
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    """Ghi các chỉ số ra tệp JSON, đồng thời tạo thư mục đích nếu cần.

    Tham số ``default`` chuyển đối tượng có phương thức ``tolist`` (ví dụ
    mảng NumPy hoặc tensor) thành kiểu dữ liệu Python mà JSON có thể ghi.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as file:
        # Thụt lề giúp tệp JSON dễ đọc; các kiểu thông thường vẫn do json xử lý.
        json.dump(metrics, file, indent=2, default=lambda value: value.tolist())


def plot_training_curves(history: dict[str, list[float]], path: str | Path) -> None:
    """Vẽ và lưu đường cong loss, accuracy của train và validation.

    ``history`` cần có bốn danh sách chỉ số, mỗi phần tử tương ứng với một
    epoch. Biểu đồ được lưu vào ``path``; figure được đóng sau khi lưu để
    tránh giữ tài nguyên khi tạo nhiều biểu đồ trong cùng tiến trình.
    """
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Biểu đồ bên trái theo dõi loss; bên phải theo dõi độ chính xác.
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["validation_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[1].plot(history["train_accuracy"], label="Train")
    axes[1].plot(history["validation_accuracy"], label="Validation")
    axes[1].set_title("Accuracy")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
    figure.tight_layout()
    # Tạo thư mục cha nếu đường dẫn lưu biểu đồ chưa tồn tại.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], path: str | Path) -> None:
    """Trực quan hóa ma trận nhầm lẫn đã được tính từ trước.

    Trục ngang biểu thị nhãn dự đoán, trục dọc biểu thị nhãn thật; mỗi ô cho
    biết số mẫu thuộc cặp nhãn tương ứng. Hàm chỉ vẽ dữ liệu được truyền vào,
    không tự tính lại các chỉ số đánh giá.
    """
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
             xticklabels=class_names, yticklabels=class_names,
             xlabel="Predicted label", ylabel="True label")
    # Xoay nhãn trục ngang để tên lớp dài không chồng lấn nhau.
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def plot_misclassified_samples(samples: list[tuple[torch.Tensor, int, int]],
                               class_names: list[str], path: str | Path,
                               max_samples: int = 16) -> None:
    """Vẽ tối đa ``max_samples`` ảnh bị phân loại sai để kiểm tra trực quan.

    Mỗi mẫu gồm tensor ảnh, nhãn thật và nhãn dự đoán. Các ảnh dư trong lưới
    4x4 sẽ bị ẩn; nếu không có mẫu sai thì hàm kết thúc mà không tạo tệp.
    """
    # Chọn các mẫu đầu tiên theo thứ tự nhận được, giới hạn số ảnh hiển thị.
    selected = samples[:max_samples]
    if not selected:
        return
    figure, axes = plt.subplots(4, 4, figsize=(8, 8))
    for axis, (image, true_label, predicted_label) in zip(axes.flat, selected):
        # Bỏ chiều kênh dư; ảnh màu PyTorch có dạng (kênh, cao, rộng),
        # trong khi imshow cần (cao, rộng, kênh).
        display_image = image.squeeze().numpy()
        if display_image.ndim == 3:
            display_image = np.transpose(display_image, (1, 2, 0))
        axis.imshow(display_image, cmap="gray" if display_image.ndim == 2 else None)
        # T: nhãn thật (true), P: nhãn dự đoán (predicted).
        axis.set_title(f"T: {class_names[true_label]}\nP: {class_names[predicted_label]}", fontsize=8)
        axis.axis("off")
    # Tắt các ô không dùng khi số mẫu sai ít hơn kích thước lưới.
    for axis in axes.flat[len(selected):]:
        axis.axis("off")
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
