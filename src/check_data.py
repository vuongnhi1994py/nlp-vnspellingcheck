import argparse
from pathlib import Path

from .paths import DATA_CACHE_DIR


SAMPLE_SIZE = 3


def count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as file:
        return sum(1 for _ in file)


def image_path_from_record(record: str, label_file: Path, line_number: int) -> Path:
    image_path, separator, _ = record.partition("\t")
    image_path = image_path.strip()
    relative_path = Path(image_path)
    if not separator or not image_path:
        raise SystemExit(
            f"Dataset check failed. Invalid record in {label_file} "
            f"at line {line_number}: expected '<image_path>\\t<label>'."
        )
    if not relative_path.parts or relative_path.is_absolute() or ".." in relative_path.parts:
        raise SystemExit(
            f"Dataset check failed. Image path must be relative to the dataset root: "
            f"{image_path} ({label_file}, line {line_number})"
        )
    return relative_path


def inspect_labels(split: str, label_file: Path) -> tuple[int, set[str]]:
    count = 0
    folders = set()
    print(f"{split} sample records:")
    with label_file.open(encoding="utf-8") as file:
        for count, record in enumerate(file, start=1):
            image_path = image_path_from_record(record, label_file, count)
            folders.add(image_path.parts[0])
            if count <= SAMPLE_SIZE:
                print(f"  {count}: {image_path}")

    if count == 0:
        raise SystemExit(f"Dataset check failed. Label file is empty: {label_file}")
    return count, folders


def verify_images(dataset_root: Path, split: str, label_file: Path) -> None:
    image_folder = dataset_root / split
    if not image_folder.is_dir():
        raise SystemExit(f"Dataset check failed. Missing image folder: {image_folder}")

    with label_file.open(encoding="utf-8") as file:
        for line_number, record in enumerate(file, start=1):
            image_path = image_path_from_record(record, label_file, line_number)
            image_file = dataset_root / image_path
            if not image_file.is_file():
                raise SystemExit(
                    f"Dataset check failed. Missing image referenced by {label_file} "
                    f"at line {line_number}: {image_file}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the OCR dataset structure.")
    parser.add_argument("dataset_root", nargs="?", type=Path, default=DATA_CACHE_DIR)
    dataset_root = parser.parse_args().dataset_root.resolve()
    data_files = {
        "train": dataset_root / "rec_train.txt",
        "val": dataset_root / "rec_val.txt",
        "test": dataset_root / "rec_test.txt",
        "dictionary": dataset_root / "vi_dict.txt",
    }

    print(f"Dataset root: {dataset_root}")
    for name, path in data_files.items():
        print(f"{name}: {path}")

    missing = [path for path in data_files.values() if not path.is_file()]
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"Dataset check failed. Missing files:\n{missing_list}")

    for split in ("train", "val", "test"):
        line_count, folders = inspect_labels(split, data_files[split])
        expected_folder = {split}
        displayed_folders = ", ".join(f"{folder}/" for folder in sorted(folders))
        print(f"{split} image folders: {displayed_folders}")
        print(f"{split} lines: {line_count}")
        if folders != expected_folder:
            raise SystemExit(
                f"Dataset check failed. Expected only {split}/ paths in "
                f"{data_files[split]}, found: {displayed_folders}"
            )
        verify_images(dataset_root, split, data_files[split])

    print(f"dictionary characters: {count_lines(data_files['dictionary'])}")


if __name__ == "__main__":
    main()
