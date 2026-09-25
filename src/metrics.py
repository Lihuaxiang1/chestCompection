"""Study-level prediction aggregation and multi-label metrics."""

from collections import OrderedDict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def aggregate_study_logits(logits, targets, study_keys):
    """Average image logits per study and keep one target vector per study."""
    grouped = OrderedDict()
    for logit, target, key in zip(logits, targets, study_keys):
        key = str(key)
        if key not in grouped:
            grouped[key] = {"logits": [], "target": target}
        elif not np.array_equal(grouped[key]["target"], target):
            raise ValueError(f"Study {key} has inconsistent labels across images")
        grouped[key]["logits"].append(logit)

    mean_logits = np.stack(
        [np.mean(item["logits"], axis=0) for item in grouped.values()]
    )
    study_targets = np.stack([item["target"] for item in grouped.values()])
    study_probs = 1.0 / (1.0 + np.exp(-np.clip(mean_logits, -80.0, 80.0)))
    return study_probs, study_targets, list(grouped.keys())


@torch.no_grad()
def collect_study_predictions(model, data_loader, device, use_amp=True):
    """Run image batches, then aggregate outputs to the competition Study unit."""
    model.eval()
    all_logits = []
    all_targets = []
    all_keys = []
    amp_enabled = use_amp and device.type == "cuda"

    for batch in tqdm(data_loader, desc="Inference"):
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
        all_logits.append(logits.float().cpu().numpy())
        all_targets.append(batch["target"].numpy())
        all_keys.extend(batch["study_key"])

    return aggregate_study_logits(
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_targets, axis=0),
        all_keys,
    )


def compute_auc_metrics(probs, targets, class_names):
    """Return per-class and macro AUC, requiring both classes for a valid AUC."""
    per_class = {}
    valid = []
    for index, name in enumerate(class_names):
        values = np.unique(targets[:, index])
        if len(values) < 2:
            per_class[name] = float("nan")
            continue
        auc = float(roc_auc_score(targets[:, index], probs[:, index]))
        per_class[name] = auc
        valid.append(auc)
    return per_class, float(np.mean(valid)) if valid else float("nan")
