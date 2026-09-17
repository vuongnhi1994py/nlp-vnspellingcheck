"""Build small final results from the read-only Kaggle backup."""

import argparse
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BACKUP = Path.home() / "VN-Spelling-Check-Backup/kaggle-output/output"
COUNT = 3003


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, 1):
            row = json.loads(line)
            if set(row) != {"image", "gt", "pred"} or not all(isinstance(row[key], str) for key in row):
                raise ValueError(f"Unexpected JSONL schema: {path}:{number}")
            rows.append({key: row[key] for key in ("image", "gt", "pred")})
    return rows


def read_csv(path, columns):
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != columns:
            raise ValueError(f"Unexpected CSV schema in {path}: {reader.fieldnames}")
        rows = list(reader)
        if any(None in row or any(row[key] is None for key in columns) for row in rows):
            raise ValueError(f"Malformed CSV row in {path}")
    return rows


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", type=Path, default=DEFAULT_BACKUP)
    backup = parser.parse_args().backup
    sources = {
        "baseline": backup / "baseline_PP-OCRv5_mobile_rec/pred_test_baseline.jsonl",
        "epoch2": backup / "vi_PP-OCRv5_mobile_rec/pred_test_finetuned.jsonl",
        "epoch4": backup / "vi_PP-OCRv5_mobile_rec_ep4/epoch4_predictions.csv",
        "experiment": backup / "epoch2_vs_epoch4.csv",
        "errors": backup / "vi_PP-OCRv5_mobile_rec_ep4/error_analysis_20_classified.csv",
    }
    # Check all inputs before changing any result file.
    baseline = read_jsonl(sources["baseline"])
    epoch2 = read_jsonl(sources["epoch2"])
    csv4 = read_csv(sources["epoch4"],
                    ["image", "ground_truth", "prediction", "score", "exact_match"])
    epoch4 = [{"image": row["image"], "gt": row["ground_truth"],
               "pred": row["prediction"]} for row in csv4]
    read_csv(sources["experiment"], ["config", "dataset", "samples", "acc", "norm_edit_dis"])
    read_csv(sources["errors"], ["image", "ground_truth", "prediction", "score", "exact_match", "error_type"])
    for name, rows in (("baseline", baseline), ("epoch2", epoch2), ("epoch4", epoch4)):
        if len(rows) != COUNT:
            raise ValueError(f"Expected {COUNT} rows in {sources[name]}, got {len(rows)}")
    reference = [(row["image"], row["gt"]) for row in baseline]
    for name, rows in (("epoch2", epoch2), ("epoch4", epoch4)):
        if [(row["image"], row["gt"]) for row in rows] != reference:
            raise ValueError(f"Image/ground truth order differs in {sources[name]}")

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    for name, rows in (("baseline", baseline), ("epoch2", epoch2), ("epoch4", epoch4)):
        write_jsonl(results / f"pred_test_{name}.jsonl", rows)
    shutil.copyfile(results / "pred_test_epoch4.jsonl", results / "pred_test_finetuned.jsonl")
    shutil.copyfile(sources["experiment"], results / "epoch2_vs_epoch4.csv")
    shutil.copyfile(sources["errors"], results / "error_analysis_20_classified.csv")
    with (results / "model_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "samples", "acc", "norm_edit_dis", "training_time"])
        writer.writerow(["PaddleOCR original", COUNT, "0.04229104229104229", "0.7951043080461289", "0"])
        writer.writerow(["Fine-tuned PP-OCRv5 Mobile (4 epochs)", COUNT,
                         "0.3633033633033633", "0.9513367698593498", "18:55:50"])
    print(f"Finalized artifacts in {results}")


if __name__ == "__main__":
    main()
