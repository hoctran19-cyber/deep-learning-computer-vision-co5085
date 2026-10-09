"""Các hàm hỗ trợ đánh giá mô hình và tổng hợp kết quả so sánh.

Các hàm trong mô-đun này thu thập nhãn thật, dự đoán và ảnh đầu vào từ
DataLoader, sau đó tính các thông tin phục vụ đánh giá như độ chính xác,
ma trận nhầm lẫn và số lượng tham số có thể huấn luyện.
"""

from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from .utils import count_trainable_parameters


def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[torch.Tensor]]:
    """Thu thập nhãn thật, nhãn dự đoán và ảnh tương ứng từ một DataLoader.

    Mô hình được chuyển sang chế độ đánh giá để các lớp như Dropout hoạt động
    đúng khi suy luận. Gradient không được tính vì bước này chỉ cần dự đoán,
    không cập nhật trọng số. Ảnh được đưa lên ``device`` để chạy mô hình, còn
    ảnh và dự đoán được giữ trên CPU để thuận tiện cho xử lý, hiển thị về sau.

    Trả về ba danh sách/ mảng theo cùng thứ tự: nhãn thật, nhãn dự đoán và ảnh.
    """
    
    model.eval()
    labels, predictions, images = [], [], []
    
    # Tắt theo dõi gradient để giảm bộ nhớ và chi phí tính toán khi đánh giá.
    with torch.no_grad():
        for batch_images, targets in loader:
            output = model(batch_images.to(device))
            # Lưu nhãn thật và lớp có điểm đầu ra cao nhất cho mỗi ảnh.
            labels.extend(targets.numpy())
            predictions.extend(output.argmax(dim=1).cpu().numpy())
            # Giữ ảnh gốc trên CPU để có thể dùng lại khi phân tích kết quả.
            images.extend(batch_images.cpu())
    return np.array(labels), np.array(predictions), images


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    """Đánh giá mô hình trên dữ liệu trong ``loader`` mà không cập nhật trọng số.

    Độ chính xác được biểu diễn dưới dạng tỷ lệ từ 0 đến 1. Kết quả còn bao
    gồm nhãn thật, nhãn dự đoán, ảnh đầu vào, ma trận nhầm lẫn và số tham số
    hiện đang được bật ``requires_grad`` trong mô hình.
    """
    labels, predictions, images = collect_predictions(model, loader, device)
    return {
        "accuracy": float((labels == predictions).mean()),
        "labels": labels,
        "predictions": predictions,
        "images": images,
        # Hàng biểu thị nhãn thật, cột biểu thị nhãn mô hình dự đoán.
        "confusion_matrix": confusion_matrix(labels, predictions),
        "trainable_parameters": count_trainable_parameters(model),
    }


def misclassified_samples(evaluation: dict[str, Any]) -> list[tuple[torch.Tensor, int, int]]:
    """Lọc các mẫu dự đoán sai từ kết quả của :func:`evaluate_model`.

    Mỗi phần tử trả về có dạng ``(ảnh, nhãn_thật, nhãn_dự_đoán)``. Các ảnh,
    nhãn thật và dự đoán được ghép theo cùng chỉ số trong kết quả đánh giá.
    """
    labels, predictions = evaluation["labels"], evaluation["predictions"]
    return [(image, int(label), int(prediction)) for image, label, prediction in zip(
        evaluation["images"], labels, predictions) if label != prediction
    ]


def compare_results(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Tạo danh sách tóm tắt để tiện so sánh nhiều mô hình đã đánh giá.

    Đầu vào là ánh xạ tên mô hình tới kết quả do ``evaluate_model`` trả về.
    Mỗi phần tử đầu ra chỉ giữ tên mô hình, độ chính xác và số tham số có thể
    huấn luyện; danh sách vẫn theo thứ tự các mô hình trong đầu vào.
    """
    return [
        {"model": name, "accuracy": values["accuracy"],
         "trainable_parameters": values["trainable_parameters"]}
        for name, values in results.items()
    ]
