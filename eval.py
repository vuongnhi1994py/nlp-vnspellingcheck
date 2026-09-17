"""Evaluate OCR predictions with exact match and normalized edit distance."""

import argparse
import csv
import json
from pathlib import Path


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance, using one row of memory."""
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def norm_edit(gt: str, pred: str) -> float:
    if not gt and not pred:
        return 1.0
    return 1 - edit_distance(gt, pred) / max(len(gt), len(pred), 1)


def evaluate(pairs):
    samples = exact = 0
    ned = 0.0
    for gt, pred in pairs:
        samples += 1
        exact += gt == pred
        ned += norm_edit(gt, pred)
    if not samples:
        raise ValueError("Prediction file is empty")
    return {"samples": samples, "acc": exact / samples,
            "norm_edit_dis": ned / samples}


def eval_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not {"ground_truth", "prediction"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing ground_truth or prediction column: {path}")
        return evaluate((row["ground_truth"], row["prediction"]) for row in reader)


def eval_jsonl(path):
    def pairs():
        with open(path, encoding="utf-8") as file:
            for number, line in enumerate(file, 1):
                row = json.loads(line)
                if not all(isinstance(row.get(key), str) for key in ("image", "gt", "pred")):
                    raise ValueError(f"Invalid image/gt/pred at {path}:{number}")
                yield row["gt"], row["pred"]
    return evaluate(pairs())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    args = parser.parse_args()
    print(eval_csv(args.file) if args.file.suffix == ".csv" else eval_jsonl(args.file))
