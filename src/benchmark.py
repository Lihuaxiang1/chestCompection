"""Measure model-only and true file-to-probability single-image latency."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from PIL import Image

from config import BEST_MODEL_PATH, TEST_IMG_DIR
from dataset import get_val_transforms
from evaluate import get_device, load_checkpoint_model
from model import get_model_size_mb


def summarize(times):
    values = np.asarray(times)
    return {
        "mean": values.mean(), "std": values.std(), "min": values.min(),
        "max": values.max(), "p50": np.percentile(values, 50),
        "p95": np.percentile(values, 95), "p99": np.percentile(values, 99),
    }


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark_model(model, tensor, device, runs, warmup):
    tensor = tensor.unsqueeze(0).to(device)
    for _ in range(warmup):
        torch.sigmoid(model(tensor))
    synchronize(device)
    times = []
    for _ in range(runs):
        synchronize(device)
        start = time.perf_counter()
        torch.sigmoid(model(tensor))
        synchronize(device)
        times.append((time.perf_counter() - start) * 1000)
    return summarize(times)


@torch.no_grad()
def benchmark_pipeline(image_path, model, transform, device, runs, warmup):
    def run_once():
        with Image.open(image_path) as opened:
            tensor = transform(opened.convert("RGB")).unsqueeze(0).to(device)
        probabilities = torch.sigmoid(model(tensor))
        synchronize(device)
        return probabilities

    for _ in range(warmup):
        run_once()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        run_once()
        times.append((time.perf_counter() - start) * 1000)
    return summarize(times)


def print_stats(title, stats):
    print(f"\n{title}")
    print(f"  mean={stats['mean']:.2f}ms p50={stats['p50']:.2f}ms p95={stats['p95']:.2f}ms p99={stats['p99']:.2f}ms")


def find_image(path):
    path = Path(path)
    if path.is_file():
        return path
    for extension in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
        match = next(path.rglob(extension), None) if path.exists() else None
        if match:
            return match
    return None


def main(args):
    device = get_device()
    model, _, image_size = load_checkpoint_model(args.ckpt, device)
    image_path = find_image(args.image)
    if image_path is None:
        raise FileNotFoundError(f"No benchmark image found under {args.image}")
    transform = get_val_transforms(image_size)
    with Image.open(image_path) as opened:
        tensor = transform(opened.convert("RGB"))
    model_stats = benchmark_model(model, tensor, device, args.runs, args.warmup)
    pipeline_stats = benchmark_pipeline(image_path, model, transform, device, args.runs, args.warmup)
    print(f"device={device} image={image_path} input={image_size} model={get_model_size_mb(model):.1f}MB")
    print_stats("Model-only latency", model_stats)
    print_stats("End-to-end file latency", pipeline_stats)
    status = "PASS" if pipeline_stats["p95"] <= 100 else "FAIL"
    print(f"\n{status}: end-to-end P95 {pipeline_stats['p95']:.2f}ms / 100ms")


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end inference benchmark")
    parser.add_argument("--ckpt", type=Path, default=BEST_MODEL_PATH)
    parser.add_argument("--image", type=Path, default=TEST_IMG_DIR)
    parser.add_argument("--runs", "--n-runs", dest="runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
