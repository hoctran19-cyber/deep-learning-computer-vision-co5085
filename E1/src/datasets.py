"""Các tiện ích xử lý Dataset và DataLoader cho E1.

Tập dữ liệu mặc định là Fashion-MNIST. Có thể tái sử dụng cùng các chỉ số
chia tập cho cả ba mô hình để việc so sánh được công bằng.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


# Các tập dữ liệu được hỗ trợ trong cấu hình.
DATASET_NAMES = {"fashion_mnist", "mnist", "cifar10"}


@dataclass
class DatasetBundle:
    """Gom các tập train, validation, test cùng thông tin cơ bản về ảnh."""

    train: Dataset
    validation: Dataset
    test: Dataset
    input_shape: tuple[int, int, int]
    num_classes: int
    class_names: list[str]


def _dataset_class(name: str) -> type:
    """Trả về lớp dataset của torchvision tương ứng với tên đã cấu hình."""
    normalized = name.lower()
    if normalized not in DATASET_NAMES:
        raise ValueError(f"Unsupported dataset '{name}'. Choose from {sorted(DATASET_NAMES)}.")
    
    # Chuẩn hóa tên để không phân biệt chữ hoa, chữ thường.
    return {
        "fashion_mnist": datasets.FashionMNIST,
        "mnist": datasets.MNIST,
        "cifar10": datasets.CIFAR10,
    }[normalized]


def _metadata(name: str) -> tuple[tuple[int, int, int], int, list[str]]:
    """Trả về kích thước ảnh, số lớp và tên nhãn của tập dữ liệu."""
    if name.lower() == "cifar10":
        return (3, 32, 32), 10, [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ]
    classes = (
        ["t-shirt/top", "trouser", "pullover", "dress", "coat",
         "sandal", "shirt", "sneaker", "bag", "ankle boot"]
        if name.lower() == "fashion_mnist"
        else [str(index) for index in range(10)]
    )
    return (1, 28, 28), 10, classes


def _transform(name: str) -> transforms.Compose:
    """Tạo chuỗi biến đổi ảnh dùng chung cho tập dữ liệu được chọn."""
    
    # CIFAR-10 có ba kênh màu; MNIST và Fashion-MNIST chỉ có một kênh xám.
    normalize = ((0.5,), (0.5,)) if name.lower() != "cifar10" else (
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    )
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(*normalize)])


def load_datasets(config: dict[str, Any]) -> DatasetBundle:
    """Tải các tập train/test và chia train thành train/validation có thể tái lập.

    Cần đặt ``dataset.download`` thành true để torchvision tải các tệp còn thiếu.
    Hàm này không được gọi trong quá trình thiết lập dự án.
    """
    dataset_config = config["dataset"]
    name = dataset_config.get("name", "fashion_mnist").lower()
    dataset_class = _dataset_class(name)
    transform = _transform(name)
    root = dataset_config.get("data_dir", dataset_config.get("root", "data"))
    download = bool(dataset_config.get("download", False))

    # Tải riêng dữ liệu huấn luyện và kiểm thử; phép biến đổi được áp dụng khi đọc ảnh.
    full_train = dataset_class(root=root, train=True, transform=transform, download=download)
    test = dataset_class(root=root, train=False, transform=transform, download=download)
    
    # Đọc tỷ lệ validation từ cấu hình experiment trước, rồi mới dùng giá trị cấp cao nhất.
    validation_size = int(len(full_train) * float(
        config.get("experiment", {}).get("validation_split",
                    config.get("validation_split", 0.1))))
    train_size = len(full_train) - validation_size
    
    # Dùng seed cấu hình để kết quả chia tập giống nhau giữa các lần chạy.
    generator = torch.Generator().manual_seed(int(config.get("seed", config.get("random_seed", 42))))
    train, validation = random_split(full_train, [train_size, validation_size], generator=generator)
    shape, classes, class_names = _metadata(name)
    return DatasetBundle(train, validation, test, shape, classes, class_names)


def create_dataloaders(bundle: DatasetBundle, config: dict[str, Any]) -> dict[str, DataLoader]:
    """Tạo DataLoader cho ba phần train, validation và test."""
    
    batch_size = int(config.get("training", {}).get("batch_size", config.get("batch_size", 128)))
    workers = int(config["dataset"].get("num_workers", 2))
    return {
        # Xáo trộn dữ liệu train; giữ nguyên thứ tự khi validation và test.
        "train": DataLoader(bundle.train, batch_size=batch_size, shuffle=True, num_workers=workers),
        "validation": DataLoader(bundle.validation, batch_size=batch_size, shuffle=False, num_workers=workers),
        "test": DataLoader(bundle.test, batch_size=batch_size, shuffle=False, num_workers=workers),
    }
