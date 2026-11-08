"""Theo dõi tiến bộ validation và lựa chọn checkpoint tốt nhất."""

import math
from numbers import Real


class EarlyStopping:
    """Theo dõi macro-F1, với accuracy phân xử khi macro-F1 bằng nhau.

    Checkpoint tốt nhất luôn được chọn theo macro-F1 cao nhất, rồi accuracy
    cao nhất nếu bằng nhau. Patience được đặt lại chỉ khi macro-F1 tăng hơn
    ``min_delta`` so với mốc tiến bộ gần nhất; hòa accuracy hoặc cải thiện nhỏ
    vẫn có thể chọn checkpoint mới nhưng không đặt lại patience.
    """

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        if isinstance(patience, bool) or not isinstance(patience, int) or patience <= 0:
            raise ValueError("patience must be a positive integer")
        if (
            isinstance(min_delta, bool)
            or not isinstance(min_delta, Real)
            or not math.isfinite(min_delta)
            or min_delta < 0
        ):
            raise ValueError("min_delta must be a finite nonnegative number")
        self.patience = patience
        self.min_delta = float(min_delta)
        self.best_epoch: int | None = None
        self.best_score: float | None = None
        self.best_accuracy: float | None = None
        self.progress_score: float | None = None
        self.bad_epochs = 0
        self.should_stop = False
        self._last_epoch = 0

    def update(self, epoch: int, macro_f1: float, accuracy: float) -> bool:
        """Cập nhật chỉ số validation; trả True nếu checkpoint tốt hơn."""
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= self._last_epoch:
            raise ValueError("epoch must be a positive, strictly increasing integer")
        for name, value in (("macro_f1", macro_f1), ("accuracy", accuracy)):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.should_stop:
            raise RuntimeError("early stopping has already been triggered")

        improved = (
            self.best_score is None
            or macro_f1 > self.best_score
            or (
                macro_f1 == self.best_score
                and self.best_accuracy is not None
                and accuracy > self.best_accuracy
            )
        )
        if improved:
            self.best_epoch = epoch
            self.best_score = float(macro_f1)
            self.best_accuracy = float(accuracy)
        if self.progress_score is None or macro_f1 > self.progress_score + self.min_delta:
            self.progress_score = float(macro_f1)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        self.should_stop = self.bad_epochs >= self.patience
        self._last_epoch = epoch
        return improved
