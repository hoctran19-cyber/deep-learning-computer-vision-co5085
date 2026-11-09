"""Integration tests for the standalone Food-101 manifest validator."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "src" / "validate_manifest.py"
FIELDS = ("image_path", "class_name", "label", "split")


class ManifestValidatorTests(unittest.TestCase):
    """Run the CLI against synthetic metadata without creating image files."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp_dir.name)
        meta = cls.root / "meta"
        meta.mkdir()
        classes = [f"class_{index:03d}" for index in range(101)]
        (meta / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
        train = {name: [f"{name}/{index}" for index in range(750)] for name in classes}
        test = {name: [f"{name}/{index}" for index in range(750, 1000)] for name in classes}
        (meta / "train.json").write_text(json.dumps(train), encoding="utf-8")
        (meta / "test.json").write_text(json.dumps(test), encoding="utf-8")
        cls.manifest = cls.root / "manifest.csv"
        with cls.manifest.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            for label, name in enumerate(classes):
                for index in range(1000):
                    split = "train" if index < 675 else "validation" if index < 750 else "test"
                    writer.writerow({"image_path": f"images/{name}/{index}.jpg", "class_name": name, "label": label, "split": split})

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def run_validator(self, manifest: Path | None = None, root: Path | None = None) -> subprocess.CompletedProcess[str]:
        """Run the same CLI users invoke."""
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--dataset-root", str(root or self.root), "--manifest", str(manifest or self.manifest)],
            text=True,
            capture_output=True,
            check=False,
        )

    def modified_manifest(self, changes: dict[int, dict[str, str] | None]) -> Path:
        """Copy the fixture, applying changes at zero-based row indices."""
        target = self.root / "modified.csv"
        with self.manifest.open(encoding="utf-8", newline="") as source, target.open("w", encoding="utf-8", newline="") as output:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            for index, row in enumerate(reader):
                if index in changes:
                    replacement = changes[index]
                    if replacement is None:
                        continue
                    row.update(replacement)
                writer.writerow(row)
        return target

    def assert_failure(self, result: subprocess.CompletedProcess[str], message: str) -> None:
        """Check both exit status and the reported reason."""
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("VALIDATION FAILED", result.stdout)
        self.assertIn(message, result.stdout + result.stderr)

    def test_valid_fixture_passes_without_images(self) -> None:
        result = self.run_validator()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("VALIDATION PASSED", result.stdout)
        self.assertIn("[PASS] Train + validation match official train.json", result.stdout)
        self.assertIn("[PASS] Test matches official test.json", result.stdout)

    def test_split_class_label_and_path_errors(self) -> None:
        manifest = self.modified_manifest({
            0: {"split": "invalid"},
            1: {"class_name": "unknown"},
            2: {"label": "1"},
            3: {"image_path": "images/class_000/../3.jpg"},
        })
        result = self.run_validator(manifest)
        for check in ("Split names", "Class names and coverage", "Labels match", "Relative image path"):
            self.assert_failure(result, f"[FAIL] {check}")

    def test_duplicate_and_cross_split_overlap(self) -> None:
        manifest = self.modified_manifest({675: {"image_path": "images/class_000/0.jpg"}})
        result = self.run_validator(manifest)
        self.assert_failure(result, "[FAIL] Duplicate image paths")
        self.assert_failure(result, "[FAIL] Images in multiple splits")
        self.assert_failure(result, "[FAIL] Train + validation match")

    def test_membership_even_when_counts_are_correct(self) -> None:
        manifest = self.modified_manifest({
            0: {"image_path": "images/class_000/not_official.jpg"},
            750: {"image_path": "images/class_000/not_test.jpg"},
        })
        result = self.run_validator(manifest)
        self.assert_failure(result, "[FAIL] Train + validation match")
        self.assert_failure(result, "[FAIL] Test matches")
        self.assertIn("[PASS] Per-class split counts", result.stdout)
        self.assertIn("[PASS] Split and overall totals", result.stdout)

    def test_missing_row_fails_counts(self) -> None:
        result = self.run_validator(self.modified_manifest({0: None}))
        self.assert_failure(result, "[FAIL] Per-class split counts")
        self.assert_failure(result, "[FAIL] Split and overall totals")

    def test_missing_file_and_invalid_header_are_reported(self) -> None:
        result = self.run_validator(self.root / "missing.csv")
        self.assert_failure(result, "[FAIL] Input error")
        bad_csv = self.root / "bad.csv"
        bad_csv.write_text("image_path,class_name,label,label\n", encoding="utf-8")
        self.assert_failure(self.run_validator(bad_csv), "expected CSV columns")

    def test_invalid_metadata_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            (root / "meta" / "classes.txt").write_text("only_one_class\n", encoding="utf-8")
            self.assert_failure(self.run_validator(root=root), "expected 101 distinct")


if __name__ == "__main__":
    unittest.main()
