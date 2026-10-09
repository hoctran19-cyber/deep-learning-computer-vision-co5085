"""Vòng lặp huấn luyện PyTorch và điểm bắt đầu chạy thí nghiệm qua dòng lệnh."""

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .datasets import create_dataloaders, load_datasets
from .evaluate import evaluate_model, misclassified_samples
from .models import get_model
from .utils import (count_trainable_parameters, get_device, load_config,
                    plot_confusion_matrix, plot_misclassified_samples,
                    plot_training_curves, project_path, save_checkpoint,
                    save_metrics, set_seed)


def _run_accuracy(output: torch.Tensor, target: torch.Tensor) -> int:
    """Đếm số dự đoán đúng trong một lô dữ liệu.

    ``output`` chứa logits cho từng lớp; chỉ số có logit lớn nhất được xem
    là lớp mô hình dự đoán. Hàm trả về số lượng mẫu đúng (số nguyên), không
    phải tỷ lệ chính xác, để bên gọi cộng dồn qua nhiều lô.
    """
    return int((output.argmax(dim=1) == target).sum().item())


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: Optimizer, device: torch.device) -> tuple[float, float]:
    """Chạy một lượt huấn luyện trên toàn bộ dữ liệu của ``loader``.

    Với mỗi lô, hàm chuyển ảnh và nhãn tới thiết bị tính toán, tính đầu ra và
    loss, rồi lan truyền ngược gradient để optimizer cập nhật trọng số. Loss
    trả về là trung bình theo số mẫu (không phải trung bình các trung bình của
    từng lô); độ chính xác là tỷ lệ mẫu được dự đoán đúng trên cả epoch.
    """
    model.train()
    total_loss = total_correct = total_items = 0
    for images, targets in loader:
        # Dữ liệu và mô hình phải ở cùng thiết bị, chẳng hạn CPU hoặc GPU.
        images, targets = images.to(device), targets.to(device)
        # Xóa gradient còn lại từ lần cập nhật trước trước khi tính gradient mới.
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, targets)
        # Lan truyền ngược để tính gradient, sau đó cập nhật trọng số mô hình.
        loss.backward()
        optimizer.step()
        # Cộng loss theo số mẫu để cuối epoch tính trung bình trên toàn bộ dữ liệu.
        total_loss += loss.item() * targets.size(0)
        total_correct += _run_accuracy(output, targets)
        total_items += targets.size(0)
    return total_loss / total_items, total_correct / total_items


def validate_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                       device: torch.device) -> tuple[float, float]:
    """Tính loss và độ chính xác trên tập validation mà không cập nhật mô hình.

    ``eval()`` chuyển các lớp có hành vi khác nhau giữa huấn luyện và suy luận
    (ví dụ Dropout) sang chế độ đánh giá. Kết quả trả về là loss trung bình
    theo số ảnh và tỷ lệ dự đoán đúng trên toàn bộ tập validation.
    """
    model.eval()
    total_loss = total_correct = total_items = 0
    # Không cần gradient khi đánh giá, giúp giảm bộ nhớ và chi phí tính toán.
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            output = model(images)
            loss = criterion(output, targets)
            # Nhân loss trung bình của lô với kích thước lô để cộng dồn chính xác,
            # kể cả khi lô cuối có ít mẫu hơn các lô còn lại.
            total_loss += loss.item() * targets.size(0)
            total_correct += _run_accuracy(output, targets)
            total_items += targets.size(0)
    return total_loss / total_items, total_correct / total_items


def train_model(model: nn.Module, loaders: dict[str, DataLoader], config: dict[str, Any],
                device: torch.device) -> dict[str, list[float]]:
    """Huấn luyện mô hình qua nhiều epoch và trả về lịch sử các chỉ số.

    Hàm dùng CrossEntropyLoss cho bài toán phân loại nhiều lớp, chọn Adam hoặc
    SGD theo cấu hình, rồi lần lượt chạy pha huấn luyện và validation ở mỗi
    epoch. Kết quả gồm loss và accuracy của cả hai pha theo thứ tự epoch; hàm
    không đánh giá tập test và không tự lưu checkpoint.
    """
    criterion = nn.CrossEntropyLoss()
    training = config.get("training", {})
    optimizer_setting = config.get("optimizer", "adam")
    # Hỗ trợ cấu hình optimizer dưới dạng tên trực tiếp hoặc một từ điển có khóa "name".
    optimizer_name = (optimizer_setting.get("name", "adam")
                      if isinstance(optimizer_setting, dict) else optimizer_setting).lower()
    learning_rate = float(training.get("learning_rate", config.get("learning_rate", 1e-3)))
    if optimizer_name == "adam":
        # Adam tự điều chỉnh bước cập nhật cho từng tham số dựa trên gradient.
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "sgd":
        # SGD cập nhật tham số theo gradient với tốc độ học đã cấu hình.
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Unsupported optimizer. Choose adam or sgd.")
    model.to(device)
    history = {"train_loss": [], "validation_loss": [], "train_accuracy": [], "validation_accuracy": []}
    # Ưu tiên số epoch trong training; nếu thiếu thì dùng cấu hình cấp cao nhất,
    # và cuối cùng mặc định 10 epoch.
    # Mỗi epoch lần lượt huấn luyện rồi đánh giá, sau đó lưu các chỉ số vào lịch sử.
    for _ in range(int(training.get("epochs", config.get("epochs", 10)))):
        train_loss, train_accuracy = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        validation_loss, validation_accuracy = validate_one_epoch(
            model, loaders["validation"], criterion, device
        )
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_accuracy"].append(validation_accuracy)
    return history


def parse_args() -> argparse.Namespace:
    """Khai báo và đọc các tùy chọn dòng lệnh của chương trình huấn luyện.

    ``--model`` là bắt buộc và giới hạn ở các kiến trúc được hỗ trợ. Tệp cấu
    hình có đường dẫn mặc định; các tùy chọn epoch, batch size và learning
    rate là tùy chọn, khi được truyền sẽ ghi đè giá trị tương ứng trong cấu hình.
    """
    parser = argparse.ArgumentParser(description="Train one E1 classifier.")
    parser.add_argument("--model", choices=["softmax", "mlp", "cnn"], required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    return parser.parse_args()


def main() -> None:
    """Điều phối toàn bộ thí nghiệm cho một kiến trúc mô hình.

    Quy trình gồm đọc cấu hình và ghi đè bằng tham số dòng lệnh, cố định seed,
    chuẩn bị dữ liệu/mô hình, huấn luyện, rồi đo kết quả trên tập test. Cuối
    cùng, các chỉ số, trọng số, biểu đồ và bảng so sánh được ghi vào các vị trí
    xác định trong cấu hình đầu ra.
    """
    args = parse_args()
    # Các đường dẫn tương đối trong cấu hình được tính từ thư mục làm việc hiện tại.
    project_root = Path.cwd()
    config = load_config(args.config)
    config.setdefault("training", {})
    # Các tùy chọn được truyền qua dòng lệnh sẽ ghi đè giá trị trong tệp cấu hình.
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    set_seed(int(config.get("seed", 42)))
    device = get_device()
    bundle = load_datasets(config)
    loaders = create_dataloaders(bundle, config)
    model = get_model(args.model, config, bundle.input_shape, bundle.num_classes)
    # Chỉ đo thời gian của quá trình huấn luyện, không tính thời gian tải dữ liệu
    # hoặc đánh giá test.
    start = time.perf_counter()
    history = train_model(model, loaders, config, device)
    training_time = time.perf_counter() - start
    # Chỉ đánh giá trên tập test sau khi hoàn tất huấn luyện.
    evaluation = evaluate_model(model, loaders["test"], device)
    output = config["output"]
    # Chuẩn hóa các thư mục đầu ra theo thư mục gốc dự án.
    metrics_dir = project_path(output["metrics_dir"], project_root)
    figures_dir = project_path(output["figures_dir"], project_root)
    checkpoints_dir = project_path(output["checkpoints_dir"], project_root)
    # Lưu chỉ số, checkpoint và các biểu đồ vào những thư mục đã cấu hình.
    save_metrics({"model": args.model, "history": history,
                  "test_accuracy": evaluation["accuracy"],
                  "num_parameters": count_trainable_parameters(model),
                  "best_val_accuracy": max(history["validation_accuracy"]),
                  "training_time": training_time}, metrics_dir / f"{args.model}.json")
    save_checkpoint(model, checkpoints_dir / f"{args.model}.pt")
    # Lưu biểu đồ đường học, ma trận nhầm lẫn và các ví dụ bị phân loại sai.
    plot_training_curves(history, figures_dir / f"{args.model}_learning_curves.png")
    plot_confusion_matrix(evaluation["confusion_matrix"], bundle.class_names,
                          figures_dir / f"{args.model}_confusion_matrix.png")
    plot_misclassified_samples(misclassified_samples(evaluation), bundle.class_names,
                               figures_dir / f"{args.model}_misclassified.png")
    _update_comparison(metrics_dir / "model_comparison.csv", args.model, evaluation,
                       history, training_time)
    print(f"Completed {args.model} on {device}; test accuracy: {evaluation['accuracy']:.4f}")


def _update_comparison(path: Path, model_name: str, evaluation: dict[str, Any],
                       history: dict[str, list[float]], training_time: float) -> None:
    """Ghi kết quả của một mô hình vào bảng CSV so sánh chung.

    Nếu tệp đã tồn tại, các dòng hiện có được đọc và giữ lại để không làm mất
    kết quả của những mô hình khác. Dòng của ``model_name`` được thay bằng
    kết quả mới (hoặc được thêm nếu chưa có), sau đó toàn bộ bảng được ghi lại.
    """
    rows: dict[str, dict[str, Any]] = {}
    if path.exists():
        # Đọc các kết quả cũ để giữ lại những mô hình chưa được chạy lại.
        with path.open(newline="", encoding="utf-8") as file:
            rows = {row["model"]: row for row in csv.DictReader(file)}
    # Gán theo tên mô hình để cập nhật kết quả cũ hoặc thêm một mô hình mới.
    rows[model_name] = {"model": model_name, "test_accuracy": evaluation["accuracy"],
                       "num_parameters": evaluation["trainable_parameters"],
                       "best_val_accuracy": max(history["validation_accuracy"]),
                       "training_time": training_time}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["model", "test_accuracy", "num_parameters", "best_val_accuracy", "training_time"])
        writer.writeheader()
        writer.writerows(rows.values())


if __name__ == "__main__":
    # Chỉ chạy quy trình huấn luyện khi tệp được gọi trực tiếp.
    main()
