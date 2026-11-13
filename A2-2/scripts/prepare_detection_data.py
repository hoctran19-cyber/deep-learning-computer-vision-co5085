#!/usr/bin/env python3
"""Prepare CCPD YOLO detection data from the authoritative manifest.

Expected manifest fields (case-insensitive aliases are supported):
- split: train / val / validation / test
- image path: output_path, image_path, path, or filepath
- filename: filename or name (optional; otherwise derived from image path)

CCPD filename format stores the two plate-box corners in the third hyphen-
separated field, e.g. ...-x1&y1_x2&y2-.... Coordinates are converted to YOLO
normalized xywh. The script deliberately does not decode plate text.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from tqdm import tqdm


def first_existing(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lookup = {c.lower(): c for c in columns}
    return next((lookup[c] for c in candidates if c in lookup), None)


def normalize_split(value: Any) -> str:
    value = str(value).strip().lower()
    aliases = {"validation": "val", "valid": "val", "train": "train",
               "training": "train", "test": "test"}
    result = aliases.get(value, value)
    if result not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split value: {value!r}")
    return result


def parse_bbox_from_ccpd_name(filename: str) -> tuple[int, int, int, int]:
    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) < 3:
        raise ValueError(f"Not a CCPD filename (expected >=3 fields): {filename}")
    try:
        left_top, right_bottom = parts[2].split("_", 1)
        x1, y1 = (int(v) for v in left_top.split("&"))
        x2, y2 = (int(v) for v in right_bottom.split("&"))
    except Exception as exc:
        raise ValueError(f"Cannot parse CCPD bbox from {filename}: {exc}") from exc
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid/non-positive bbox in {filename}: {(x1,y1,x2,y2)}")
    return x1, y1, x2, y2


def safe_resolve(root: Path, raw_path: str) -> Path:
    p = Path(str(raw_path))
    if p.is_absolute():
        return p
    return root / p


def relative_image_path(raw_path: str, image_path: Path, root: Path, split: str) -> Path:
    """Preserve the manifest's path after removing its dataset-root prefix.

    If the manifest path is already rooted elsewhere, use split/basename to
    avoid accidental traversal and collisions are caught by duplicate checks.
    """
    p = Path(str(raw_path))
    if p.is_absolute():
        try:
            rel = image_path.relative_to(root)
        except ValueError:
            rel = Path(image_path.name)
    else:
        rel = p
    # Avoid duplicate split component in generated directory layout.
    if rel.parts and rel.parts[0].lower() == split:
        rel = Path(*rel.parts[1:])
    if not rel.parts or ".." in rel.parts:
        raise ValueError(f"Unsafe or empty relative image path: {raw_path}")
    return rel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path, help="Dataset root")
    ap.add_argument("--manifest", required=True, type=Path, help="CSV manifest")
    ap.add_argument("--output", required=True, type=Path, help="Output YOLO dataset directory")
    ap.add_argument("--link-images", action="store_true", help="Symlink images instead of copying")
    ap.add_argument("--expected-counts", default="train=18000,val=2000,test=5000",
                    help="Expected counts; set empty string to disable")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest}")

    df = pd.read_csv(manifest)
    split_col = first_existing(list(df.columns), ("split", "subset_split"))
    path_col = first_existing(list(df.columns), ("output_path", "image_path", "path", "filepath", "relative_path"))
    name_col = first_existing(list(df.columns), ("filename", "name"))
    if not split_col or not path_col:
        raise ValueError(f"Manifest needs split and image path columns; found: {list(df.columns)}")

    expected: dict[str, int] = {}
    if args.expected_counts.strip():
        for item in args.expected_counts.split(","):
            k, v = item.split("=", 1)
            expected[normalize_split(k)] = int(v)

    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    (output / "labels").mkdir(exist_ok=True)
    counts = {"train": 0, "val": 0, "test": 0}
    errors: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing YOLO labels"):
        raw_path = str(row[path_col])
        try:
            split = normalize_split(row[split_col])
            image_path = safe_resolve(root, raw_path)
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")
            filename = str(row[name_col]) if name_col and pd.notna(row[name_col]) else image_path.name
            x1, y1, x2, y2 = parse_bbox_from_ccpd_name(filename)
            with Image.open(image_path) as im:
                width, height = im.size
                im.verify()
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"BBox {(x1,y1,x2,y2)} outside image {width}x{height}")

            rel = relative_image_path(raw_path, image_path, root, split)
            key = (split, rel.as_posix())
            if key in seen:
                raise ValueError(f"Duplicate output image path: {key}")
            seen.add(key)
            image_out = output / "images" / split / rel
            label_out = output / "labels" / split / rel.with_suffix(".txt")
            image_out.parent.mkdir(parents=True, exist_ok=True)
            label_out.parent.mkdir(parents=True, exist_ok=True)

            if args.link_images:
                if image_out.exists() or image_out.is_symlink():
                    image_out.unlink()
                os.symlink(image_path, image_out)
            else:
                shutil.copy2(image_path, image_out)

            xc = ((x1 + x2) / 2) / width
            yc = ((y1 + y2) / 2) / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            label_out.write_text(f"0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}\n", encoding="utf-8")
            counts[split] += 1
        except Exception as exc:
            errors.append({"path": raw_path, "error": str(exc)})

    data_yaml = (
        f"path: {output.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n  0: license_plate\n"
    )
    (output / "data.yaml").write_text(data_yaml, encoding="utf-8")
    summary = {"manifest": str(manifest), "root": str(root), "output": str(output),
               "total_manifest_rows": len(df), "successful": sum(counts.values()),
               "counts": counts, "errors": len(errors), "link_images": args.link_images}
    (output / "preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "preparation_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    mismatches = {k: {"expected": v, "actual": counts[k]} for k, v in expected.items() if counts[k] != v}
    if errors or mismatches:
        print("DATA PREPARATION FAILED")
        if mismatches:
            print("Split count mismatches:", json.dumps(mismatches))
        return 2
    print("DATA PREPARATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
