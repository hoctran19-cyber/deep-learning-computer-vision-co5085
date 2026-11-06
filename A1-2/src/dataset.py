"""Tải dữ liệu Food-101 từ tệp manifest và tạo DataLoader cho PyTorch.

Manifest dùng chung thông tin đường dẫn ảnh, tên lớp, nhãn và tập dữ liệu
theo quy ước chia tập của A1-1.
"""

import csv
from pathlib import Path
from typing import Optional, Callable

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch


class Food101Dataset(Dataset):
    """
    Bộ nạp dữ liệu Food-101 dựa trên một tệp manifest dùng chung.

    Các cột bắt buộc trong manifest:
        image_path, class_name, label, split

    image_path là đường dẫn tương đối tính từ dataset_root.
    """

    VALID_SPLITS = {"train", "validation", "test"}

    def __init__(
        self,
        dataset_root: str | Path,
        manifest_path: str | Path,
        split: str,
        transform: Optional[Callable] = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.transform = transform

        # Chỉ chấp nhận ba tập dữ liệu đã được quy ước.
        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"Invalid split: {split}. "
                f"Expected one of {self.VALID_SPLITS}"
            )

        if not self.dataset_root.is_dir():
            raise FileNotFoundError(
                f"Dataset root not found: {self.dataset_root}"
            )

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )

        # Danh sách lớp định nghĩa thứ tự ánh xạ từ tên lớp sang nhãn số.
        classes_file = self.dataset_root / "meta" / "classes.txt"

        if not classes_file.is_file():
            raise FileNotFoundError(
                f"Classes file not found: {classes_file}"
            )

        with open(classes_file, "r", encoding="utf-8") as file:
            self.classes = [
                line.strip()
                for line in file
                if line.strip()
            ]

        if len(self.classes) != 101:
            raise ValueError(
                f"Expected 101 classes, got {len(self.classes)}"
            )

        if len(set(self.classes)) != len(self.classes):
            raise ValueError("Class names in classes.txt must be unique")

        if self.classes != sorted(self.classes):
            raise ValueError(
                "Class names in classes.txt must be alphabetically sorted"
            )

        # enumerate giữ nhãn nhất quán với thứ tự lớp trong classes.txt.
        self.class_to_idx = {
            class_name: idx
            for idx, class_name in enumerate(self.classes)
        }

        self.samples = self._load_manifest()

        if not self.samples:
            raise ValueError(
                f"No samples found for split: {split}"
            )

    def _load_manifest(self):
        samples = []
        seen_image_paths = set()

        # Kiểm tra cấu trúc manifest trước khi đọc từng mẫu.
        required_columns = {
            "image_path",
            "class_name",
            "label",
            "split",
        }

        with open(
            self.manifest_path,
            "r",
            encoding="utf-8",
            newline="",
        ) as file:

            reader = csv.DictReader(file)

            if not required_columns.issubset(
                set(reader.fieldnames or [])
            ):
                raise ValueError(
                    f"Manifest must contain columns: "
                    f"{required_columns}"
                )

            # Kiểm tra toàn bộ manifest, kể cả trùng ảnh giữa các tập.
            for row_number, row in enumerate(reader, start=2):
                for column in required_columns:
                    value = row.get(column)
                    if value is None or not value.strip():
                        raise ValueError(
                            f"Manifest row {row_number}: "
                            f"field '{column}' must not be empty"
                        )

                split = row["split"]
                if split not in self.VALID_SPLITS:
                    raise ValueError(
                        f"Manifest row {row_number}: "
                        f"invalid split '{split}'"
                    )

                image_path = Path(row["image_path"])
                class_name = row["class_name"]

                if class_name not in self.class_to_idx:
                    raise ValueError(
                        f"Unknown class: {class_name}"
                    )

                try:
                    label = int(row["label"])
                except ValueError as error:
                    raise ValueError(
                        f"Manifest row {row_number}: label must be an integer"
                    ) from error

                if not 0 <= label <= 100:
                    raise ValueError(
                        f"Manifest row {row_number}: "
                        "label must be between 0 and 100"
                    )

                expected_label = self.class_to_idx[class_name]

                # Phát hiện nhãn trong manifest không khớp với danh sách lớp.
                if label != expected_label:
                    raise ValueError(
                        f"Label mismatch for {class_name}: "
                        f"expected {expected_label}, got {label}"
                    )

                if image_path.is_absolute():
                    raise ValueError(
                        "Manifest image_path must be relative "
                        "to dataset_root"
                    )

                # Chuẩn hóa đường dẫn rồi đảm bảo ảnh vẫn nằm trong thư mục dữ liệu.
                full_path = (
                    self.dataset_root / image_path
                ).resolve()

                if not full_path.is_relative_to(
                    self.dataset_root.resolve()
                ):
                    raise ValueError(
                        f"Image path escapes dataset root: "
                        f"{image_path}"
                    )

                if full_path in seen_image_paths:
                    raise ValueError(
                        f"Manifest row {row_number}: duplicate image path "
                        f"'{image_path}'"
                    )
                seen_image_paths.add(full_path)

                # Các bản ghi hợp lệ của split khác cũng được kiểm tra,
                # nhưng chỉ nạp ảnh thuộc split được yêu cầu.
                if split != self.split:
                    continue

                if not full_path.is_file():
                    raise FileNotFoundError(
                        f"Image not found: {full_path}"
                    )

                samples.append({
                    "image_path": full_path,
                    "class_name": class_name,
                    "label": label,
                })

        return samples

    def __len__(self):
        # PyTorch dùng giá trị này để biết số mẫu của dataset.
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # Chuyển ảnh sang RGB để mọi mẫu có cùng ba kênh màu.
        with Image.open(sample["image_path"]) as img:
            image = img.convert("RGB")

        # Không áp dụng tiền xử lý mặc định; transform do caller quyết định.
        if self.transform is not None:
            image = self.transform(image)

        label = sample["label"]

        return image, label


def create_dataloader(
    dataset: Food101Dataset,
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: Optional[bool] = None,
):
    """
    Tạo DataLoader của PyTorch cho dataset đã cho.

    Tập train được xáo trộn; validation và test giữ nguyên thứ tự. Khi
    ``transform=None``, mỗi ảnh là đối tượng PIL nên default collate của
    PyTorch không thể gom ảnh thành batch; hãy truyền một transform tương
    thích, ví dụ ``torchvision.transforms.ToTensor()``, khi dùng DataLoader.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    # Mặc định bật pin_memory khi có CUDA để hỗ trợ truyền dữ liệu lên GPU.
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # Giữ nguyên thứ tự validation/test và không bỏ batch cuối.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(dataset.split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
