
import csv
import json
from pathlib import Path

# Xac dinh thu muc E1
BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"


def inspect_json(path: Path) -> None:
    """Doc va hien thi noi dung file JSON."""
    print("\n" + "=" * 60)
    print(f"FILE: {path.relative_to(BASE_DIR)}")
    print("=" * 60)

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    print(json.dumps(data, indent=2, ensure_ascii=False))


def inspect_csv(path: Path) -> None:
    """Doc va hien thi noi dung file CSV."""
    print("\n" + "=" * 60)
    print(f"FILE: {path.relative_to(BASE_DIR)}")
    print("=" * 60)

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        rows = list(reader)

    if not rows:
        print("File CSV dang trong.")
        return

    for row in rows:
        print(" | ".join(row))


def main() -> None:
    metrics_dir = RESULTS_DIR / "metrics"

    # Doc metrics cua tung model
    for model_name in ("softmax", "mlp", "cnn"):
        path = metrics_dir / f"{model_name}.json"

        if path.exists():
            inspect_json(path)
        else:
            print(f"Khong tim thay file: {path}")

    # Doc bang so sanh model
    comparison_path = metrics_dir / "model_comparison.csv"

    if comparison_path.exists():
        inspect_csv(comparison_path)
    else:
        print(f"Khong tim thay file: {comparison_path}")

    # Doc metrics tong hop
    summary_path = RESULTS_DIR / "metrics.json"

    if summary_path.exists():
        inspect_json(summary_path)
    else:
        print(f"Khong tim thay file: {summary_path}")


if __name__ == "__main__":
    main()
