"""Batch inference and Study-level submission generation."""

import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from config import (
    BATCH_SIZE, BEST_MODEL_PATH, DEFAULT_THRESHOLD, DISEASE_LABELS,
    IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, SUBMISSION_PATH, TEST_IMG_DIR,
)
from dataset import ResizeWithPad, get_val_transforms
from evaluate import get_device, load_checkpoint_model
from model import get_model_size_mb

IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")


def find_test_studies(test_dir):
    test_dir = Path(test_dir)
    if not test_dir.exists():
        return []
    direct_images = []
    for pattern in IMAGE_PATTERNS:
        direct_images.extend(test_dir.glob(pattern))
    nested_studies = []
    for subject_dir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
        for study_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            images = []
            for pattern in IMAGE_PATTERNS:
                images.extend(study_dir.glob(pattern))
            if images:
                nested_studies.append({
                    "subject_id": str(subject_dir.name),
                    "study_id": str(study_dir.name),
                    "images": sorted(set(images)),
                })
    if direct_images and nested_studies:
        raise ValueError(
            f"{test_dir} mixes flat images with Subject/Study directories; "
            "separate the two formats before prediction"
        )
    if direct_images:
        return [
            {"subject_id": "", "study_id": path.stem, "images": [path]}
            for path in sorted(set(direct_images))
        ]
    return nested_studies


def build_tta_transforms(image_size, enabled):
    pipelines = [get_val_transforms(image_size)]
    if enabled:
        pipelines.append(transforms.Compose([
            ResizeWithPad(image_size, content_scale=0.92),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]))
    return pipelines


class InferenceImageDataset(Dataset):
    def __init__(self, studies, transform_list):
        self.transforms = transform_list
        self.records = []
        for study in studies:
            key = f"{study['subject_id']}::{study['study_id']}"
            for image_path in study["images"]:
                self.records.append((image_path, key))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_path, key = self.records[index]
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            views = torch.stack([pipeline(image) for pipeline in self.transforms])
        return {"views": views, "study_key": key}


@torch.no_grad()
def predict_studies(models, studies, transform_list, device, batch_size=16, workers=0):
    dataset = InferenceImageDataset(studies, transform_list)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )
    grouped_logits = OrderedDict()
    amp_enabled = device.type == "cuda"
    for batch in tqdm(loader, desc="Predicting images"):
        views = batch["views"]
        batch_count, view_count = views.shape[:2]
        images = views.flatten(0, 1).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = torch.stack([model(images).float() for model in models]).mean(dim=0)
            logits = logits.reshape(batch_count, view_count, -1).mean(dim=1)
        for key, logit in zip(batch["study_key"], logits.cpu().numpy()):
            grouped_logits.setdefault(key, []).append(logit)

    metadata = {f"{study['subject_id']}::{study['study_id']}": study for study in studies}
    results = []
    for key, image_logits in grouped_logits.items():
        mean_logits = np.mean(image_logits, axis=0)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(mean_logits, -80.0, 80.0)))
        results.append({
            "subject_id": metadata[key]["subject_id"],
            "study_id": metadata[key]["study_id"],
            "probs": probabilities,
        })
    return results


def generate_submission(results, output_path, filter_threshold=None):
    rows = []
    for result in results:
        for class_id in range(NUM_CLASSES):
            probability = float(result["probs"][class_id])
            if filter_threshold is not None and probability < filter_threshold:
                continue
            rows.append({
                "Subject_id": result["subject_id"],
                "Study_id": result["study_id"],
                "Predict_class": class_id,
                "Probability": f"{probability:.8f}",
            })
    rows.sort(key=lambda row: (row["Subject_id"], row["Study_id"], row["Predict_class"]))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["Subject_id", "Study_id", "Predict_class", "Probability"],
        )
        writer.writeheader()
        writer.writerows(rows)
    expected = len(results) * NUM_CLASSES
    print(f"submission={output_path} studies={len(results)} rows={len(rows)}")
    if filter_threshold is None and len(rows) != expected:
        raise RuntimeError(f"Expected {expected} complete probability rows, got {len(rows)}")


def main(args):
    device = get_device()
    models = []
    checkpoints = []
    image_sizes = set()
    for checkpoint_path in args.ckpt:
        model, checkpoint, image_size = load_checkpoint_model(checkpoint_path, device)
        models.append(model)
        checkpoints.append(checkpoint)
        image_sizes.add(image_size)
    if len(image_sizes) != 1:
        raise ValueError(f"Ensemble checkpoints use different image sizes: {sorted(image_sizes)}")
    image_size = image_sizes.pop()
    total_size = sum(get_model_size_mb(model) for model in models)
    print(f"device={device} models={len(models)} total_model_size={total_size:.1f}MB image_size={image_size}")
    print("checkpoint AUCs:", [checkpoint.get("macro_auc", float("nan")) for checkpoint in checkpoints])
    if total_size > 500:
        print("[WARN] Ensemble model size exceeds the 500MB competition limit")
    studies = find_test_studies(args.test_dir)
    if not studies:
        raise RuntimeError(f"No Subject/Study/image structure found under {args.test_dir}")
    results = predict_studies(
        models, studies, build_tta_transforms(image_size, args.tta), device,
        batch_size=args.batch_size, workers=args.workers,
    )
    all_probs = np.stack([result["probs"] for result in results])
    print("Prediction distribution (0.5 is diagnostic only):")
    for index, name in enumerate(DISEASE_LABELS):
        print(f"  {name:<28} mean={all_probs[:, index].mean():.3f} rate@{args.report_threshold:.2f}={(all_probs[:, index] >= args.report_threshold).mean():.1%}")
    generate_submission(results, args.output, args.filter_threshold)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Study-level probability submission")
    parser.add_argument("--ckpt", type=Path, nargs="+", default=[BEST_MODEL_PATH])
    parser.add_argument("--test-dir", type=Path, default=TEST_IMG_DIR)
    parser.add_argument("--output", type=Path, default=SUBMISSION_PATH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--tta", action="store_true", help="Average two center-crop scales")
    parser.add_argument("--report-threshold", "--threshold", dest="report_threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--filter-threshold", type=float, default=None, help="Only use if the official format requires filtered rows")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
