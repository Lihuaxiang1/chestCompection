"""Search F1 thresholds on Study-level validation predictions."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.metrics import f1_score

from config import BATCH_SIZE, BEST_MODEL_PATH, DISEASE_LABELS
from dataset import build_dataloaders
from evaluate import get_device, load_checkpoint_model
from metrics import collect_study_predictions


def best_threshold(probabilities, targets, thresholds):
    scores = [
        f1_score(targets, probabilities >= threshold, zero_division=0)
        for threshold in thresholds
    ]
    index = int(np.argmax(scores))
    return float(thresholds[index]), float(scores[index])


def main(args):
    device = get_device()
    model, checkpoint, image_size = load_checkpoint_model(args.ckpt, device)
    _, val_loader, _, _ = build_dataloaders(
        batch_size=args.batch_size, num_workers=args.workers, image_size=image_size,
    )
    probs, targets, study_keys = collect_study_predictions(model, val_loader, device)
    thresholds = np.linspace(args.minimum, args.maximum, args.steps)

    global_scores = [
        f1_score(targets, probs >= threshold, average="macro", zero_division=0)
        for threshold in thresholds
    ]
    global_index = int(np.argmax(global_scores))
    print(f"Studies: {len(study_keys)}; checkpoint AUC: {checkpoint.get('macro_auc', float('nan')):.4f}")
    print(f"Global threshold={thresholds[global_index]:.4f} Macro-F1={global_scores[global_index]:.4f}")
    print("CLASS_THRESHOLDS = {")
    for class_id, name in enumerate(DISEASE_LABELS):
        threshold, score = best_threshold(probs[:, class_id], targets[:, class_id], thresholds)
        print(f'    "{name}": {threshold:.4f},  # F1={score:.4f}')
    print("}")
    print("Note: thresholds affect F1, not ROC-AUC.")


def parse_args():
    parser = argparse.ArgumentParser(description="Study-level F1 threshold search")
    parser.add_argument("--ckpt", type=Path, default=BEST_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--minimum", type=float, default=0.05)
    parser.add_argument("--maximum", type=float, default=0.95)
    parser.add_argument("--steps", "--n-thresholds", dest="steps", type=int, default=181)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
