"""Pre-training audit for labels, files, class balance, and patient leakage."""

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from config import DISEASE_LABELS, TRAIN_CSV, TRAIN_IMG_DIR, VAL_CSV, VAL_IMG_DIR
from dataset import (
    SimpleDataset, audit_train_val_leakage, detect_format, find_study_images,
    get_val_transforms, load_labels_long_format,
)


def print_distribution(values):
    for name, rate, count in zip(DISEASE_LABELS, values.mean(axis=0), values.sum(axis=0)):
        print(f"  {name:<28} positives={int(count):5d} rate={rate:.2%}")


def audit_split(name, csv_path, image_dir):
    data_format = detect_format(csv_path, image_dir)
    print(f"\n{name}: format={data_format}")
    if data_format == "study":
        labels = load_labels_long_format(csv_path)
        images = find_study_images(image_dir)
        label_keys = set(zip(labels["Subject_id"].astype(str), labels["Study_id"].astype(str)))
        missing = label_keys - set(images)
        orphan = set(images) - label_keys
        image_count = sum(len(paths) for paths in images.values())
        print(f"  subjects={labels['Subject_id'].nunique()} studies={len(label_keys)} images={image_count}")
        print(f"  labeled_without_images={len(missing)} image_studies_without_labels={len(orphan)}")
        values = labels[list(range(len(DISEASE_LABELS)))].values.astype(np.float32)
    else:
        dataset = SimpleDataset(csv_path, image_dir, get_val_transforms())
        missing = [image_id for image_id in dataset.get_image_ids() if not (Path(image_dir) / Path(image_id).name).exists()]
        print(f"  images={len(dataset)} missing_images={len(missing)}")
        values = dataset.df[DISEASE_LABELS].values.astype(np.float32)
    print_distribution(values)
    return values


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def duplicate_report(directories):
    groups = defaultdict(list)
    for directory in directories:
        for path in Path(directory).rglob("*"):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                groups[file_digest(path)].append(path)
    duplicates = [paths for paths in groups.values() if len(paths) > 1]
    print(f"\nExact duplicate groups: {len(duplicates)}")
    for paths in duplicates[:10]:
        print("  " + " | ".join(str(path) for path in paths))


def main(args):
    train_values = audit_split("train", TRAIN_CSV, TRAIN_IMG_DIR)
    val_values = audit_split("val", VAL_CSV, VAL_IMG_DIR)
    audit_train_val_leakage()
    print("\nPatient leakage: none")
    for split_name, values in [("train", train_values), ("val", val_values)]:
        invalid_auc = [DISEASE_LABELS[index] for index in range(values.shape[1]) if len(np.unique(values[:, index])) < 2]
        if invalid_auc:
            print(f"[WARN] {split_name} classes without both labels: {invalid_auc}")
    if args.hash_duplicates:
        duplicate_report([TRAIN_IMG_DIR, VAL_IMG_DIR])
    print("\nAudit complete")


def parse_args():
    parser = argparse.ArgumentParser(description="Audit competition data before training")
    parser.add_argument("--hash-duplicates", action="store_true", help="Slow exact image duplicate check")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
