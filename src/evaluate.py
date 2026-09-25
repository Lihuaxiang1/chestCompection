"""Evaluate the deployment checkpoint at the competition Study level."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve

from config import (
    BATCH_SIZE, BEST_MODEL_PATH, DISEASE_LABELS, DROPOUT, FIGURES_DIR,
    GRADCAM_TARGET_LAYER, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
)
from dataset import build_dataloaders
from metrics import collect_study_predictions, compute_auc_metrics
from model import build_model

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


class GradCAM:
    """Grad-CAM for the final ConvNeXt feature stage."""

    def __init__(self, model, target_layer_name=None):
        self.model = model
        target_name = target_layer_name or GRADCAM_TARGET_LAYER
        target_layer = dict(model.named_modules())[target_name]
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inputs, output):
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor.requires_grad_(True))
        if target_class is None:
            target_class = int(output.argmax(dim=1).item())
        self.model.zero_grad(set_to_none=True)
        output[0, target_class].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False,
        ).squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam, output, target_class


def _denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = tensor.detach().cpu() * std + mean
    return (image.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def plot_roc_curves(probs, targets, per_class_auc, save_path):
    n_cols = 5
    n_rows = int(np.ceil(len(DISEASE_LABELS) / n_cols))
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2.8 * n_rows))
    axes = np.asarray(axes).reshape(-1)
    for index, name in enumerate(DISEASE_LABELS):
        axis = axes[index]
        auc = per_class_auc[name]
        if np.isfinite(auc):
            fpr, tpr, _ = roc_curve(targets[:, index], probs[:, index])
            axis.plot(fpr, tpr, lw=1.5, label=f"AUC={auc:.3f}")
            axis.legend(fontsize=7, loc="lower right")
        axis.plot([0, 1], [0, 1], "k--", lw=0.6)
        axis.set_title(name, fontsize=8)
        axis.tick_params(labelsize=6)
    for axis in axes[len(DISEASE_LABELS):]:
        axis.axis("off")
    figure.suptitle("Study-level ROC curves")
    figure.tight_layout()
    figure.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def load_checkpoint_model(checkpoint_path, device):
    """Rebuild the exact inference model stored in a deployment checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"{checkpoint_path} has no model_state_dict")
    saved_config = checkpoint.get("config", {})
    architecture = saved_config.get("architecture", "convnext_tiny")
    if architecture != "convnext_tiny":
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
    num_classes = int(saved_config.get("num_classes", len(DISEASE_LABELS)))
    if num_classes != len(DISEASE_LABELS):
        raise ValueError(
            f"Checkpoint has {num_classes} classes, but config expects {len(DISEASE_LABELS)}"
        )
    class_names = saved_config.get("class_names")
    if class_names is not None and list(class_names) != list(DISEASE_LABELS):
        raise ValueError("Checkpoint class order does not match config.DISEASE_LABELS")
    model = build_model(
        num_classes=num_classes,
        pretrained=False,
        dropout=float(saved_config.get("dropout", DROPOUT)),
    )
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} is incompatible with the current "
            "Dropout + Linear classification head"
        ) from error
    return model.to(device).eval(), checkpoint, int(saved_config.get("image_size", IMAGE_SIZE))


def evaluate(args):
    device = get_device()
    model, checkpoint, image_size = load_checkpoint_model(args.ckpt, device)
    print(f"device={device} checkpoint_epoch={checkpoint.get('epoch', '?')} image_size={image_size}")
    _, val_loader, _, _ = build_dataloaders(
        batch_size=args.batch_size, num_workers=args.workers, image_size=image_size,
    )
    if val_loader is None:
        raise RuntimeError("Validation data not found")
    probs, targets, study_keys = collect_study_predictions(model, val_loader, device)
    per_class_auc, macro_auc = compute_auc_metrics(probs, targets, DISEASE_LABELS)
    print(f"Study count: {len(study_keys)}")
    print(f"Macro AUC: {macro_auc:.4f}")
    for name, auc in sorted(per_class_auc.items(), key=lambda item: np.nan_to_num(item[1], nan=-1), reverse=True):
        print(f"  {name:<28} {auc:.4f}" if np.isfinite(auc) else f"  {name:<28} N/A")
    output_path = FIGURES_DIR / "roc_curves.png"
    plot_roc_curves(probs, targets, per_class_auc, output_path)
    print(f"ROC curves: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Study-level model evaluation")
    parser.add_argument("--ckpt", type=Path, default=BEST_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
