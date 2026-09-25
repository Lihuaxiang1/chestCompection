"""Study-aware training for multi-label chest X-ray classification."""

import argparse
import copy
import csv
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import (
    BACKBONE_LR, BATCH_SIZE, BEST_MODEL_PATH, DISEASE_LABELS, DROPOUT,
    EARLY_STOP_PATIENCE, EMA_DECAY, FREEZE_EPOCHS, HEAD_LR, IMAGE_SIZE,
    LAST_MODEL_PATH, LOGS_DIR, NUM_EPOCHS, NUM_WORKERS, POS_WEIGHT_MAX,
    POS_WEIGHT_POWER, SEED, WARMUP_EPOCHS, WEIGHT_DECAY,
)
from dataset import build_dataloaders
from metrics import aggregate_study_logits, compute_auc_metrics
from model import (
    build_model, count_parameters, get_model_size_mb, get_parameter_groups,
    set_backbone_trainable,
)


def resolve_pretrained_mode(pretrained_arg):
    """
    解析 --pretrained 参数。
    'auto' → 本地有权重就用，没有就不用
    'true' → 强制用，没有就下载
    'false' → 不用
    返回 bool：最终是否使用预训练
    """
    if isinstance(pretrained_arg, bool):
        return pretrained_arg
    if pretrained_arg == "true":
        return True
    if pretrained_arg == "false":
        return False
    # auto 模式：检查本地有没有权重
    from model import _find_local_weights
    return _find_local_weights("convnext_tiny") is not None


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ModelEMA:
    """Exponential moving average of model weights for more stable validation."""

    def __init__(self, model, decay=EMA_DECAY):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        self.updates = 0
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        # A fixed 0.999 decay makes short runs mostly reproduce initialization.
        decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        source = model.state_dict()
        for name, value in self.module.state_dict().items():
            if value.is_floating_point():
                value.mul_(decay).add_(source[name].detach(), alpha=1.0 - decay)
            else:
                value.copy_(source[name])

    def state_dict(self):
        return {
            "model_state_dict": self.module.state_dict(),
            "updates": self.updates,
            "decay": self.decay,
        }

    def load_state_dict(self, state):
        self.module.load_state_dict(state["model_state_dict"], strict=True)
        self.updates = int(state.get("updates", 0))
        self.decay = float(state.get("decay", self.decay))


class AsymmetricLoss(nn.Module):
    """ASL for imbalanced multi-label classification, returned element-wise."""

    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        positive = torch.sigmoid(logits)
        negative = 1.0 - positive
        if self.clip:
            negative = (negative + self.clip).clamp(max=1.0)
        log_loss = targets * torch.log(positive.clamp(min=self.eps))
        log_loss += (1.0 - targets) * torch.log(negative.clamp(min=self.eps))
        probability = positive * targets + negative * (1.0 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        return -log_loss * torch.pow(1.0 - probability, gamma)


def build_criterion(loss_name, raw_pos_weights, device):
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss(reduction="none"), torch.ones_like(raw_pos_weights)
    if loss_name == "weighted-bce":
        weights = raw_pos_weights.pow(POS_WEIGHT_POWER).clamp(max=POS_WEIGHT_MAX)
        return nn.BCEWithLogitsLoss(pos_weight=weights.to(device), reduction="none"), weights
    if loss_name == "asl":
        return AsymmetricLoss(), torch.ones_like(raw_pos_weights)
    raise ValueError(f"Unknown loss: {loss_name}")


def reduce_study_balanced_loss(elementwise_loss, sample_weights):
    per_image = elementwise_loss.mean(dim=1)
    return (per_image * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)


def train_epoch(model, loader, criterion, optimizer, scaler, ema, device, epoch, use_amp):
    model.train()
    total_loss = 0.0
    total_weight = 0.0
    progress = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        weights = batch["sample_weight"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = reduce_study_balanced_loss(criterion(logits, targets), weights)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        batch_weight = float(weights.sum().item())
        total_loss += float(loss.item()) * batch_weight
        total_weight += batch_weight
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(total_weight, 1e-8)


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    all_logits, all_targets, all_keys = [], [], []
    total_loss = 0.0
    total_weight = 0.0
    for batch in tqdm(loader, desc="[Valid]"):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        weights = batch["sample_weight"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = reduce_study_balanced_loss(criterion(logits, targets), weights)
        batch_weight = float(weights.sum().item())
        total_loss += float(loss.item()) * batch_weight
        total_weight += batch_weight
        all_logits.append(logits.float().cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_keys.extend(batch["study_key"])

    probs, targets, _ = aggregate_study_logits(
        np.concatenate(all_logits), np.concatenate(all_targets), all_keys,
    )
    per_class_auc, macro_auc = compute_auc_metrics(probs, targets, DISEASE_LABELS)
    return total_loss / max(total_weight, 1e-8), macro_auc, per_class_auc


def build_scheduler(optimizer, epochs, warmup_epochs):
    min_factor = 0.03

    def schedule(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        span = max(1, epochs - warmup_epochs - 1)
        progress = min(1.0, max(0.0, (epoch - warmup_epochs) / span))
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def checkpoint_config(args):
    return {
        "architecture": "convnext_tiny",
        "num_classes": len(DISEASE_LABELS),
        "class_names": list(DISEASE_LABELS),
        "image_size": args.image_size,
        "dropout": args.dropout,
        "loss": args.loss,
        "pretrained": args.pretrained,  # "auto" / "true" / "false"
        "backbone_lr": args.effective_backbone_lr,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "freeze_epochs": args.freeze_epochs,
        "ema_decay": args.ema_decay,
        "seed": args.seed,
    }


def save_history(rows):
    path = LOGS_DIR / "training_history.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def apply_resume_config(args, checkpoint):
    """Restore settings that must remain compatible with a training checkpoint."""
    saved = checkpoint.get("config", {})
    fields = (
        "image_size", "dropout", "loss", "pretrained", "head_lr",
        "weight_decay", "warmup_epochs", "freeze_epochs", "ema_decay", "seed",
    )
    for field in fields:
        if field in saved:
            old_value = getattr(args, field)
            new_value = saved[field]
            if old_value != new_value:
                print(f"[resume] {field}: {old_value} -> {new_value}")
            setattr(args, field, new_value)
    if "backbone_lr" in saved:
        args.backbone_lr = float(saved["backbone_lr"])


def train(args):
    set_seed(args.seed)
    device = get_device()

    # 解析预训练模式（auto/true/false → bool）
    use_pretrained_config = resolve_pretrained_mode(args.pretrained)
    print(f"预训练模式: {args.pretrained} → {'启用' if use_pretrained_config else '不使用'}")

    resume_checkpoint = None
    if args.resume is not None:
        resume_checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if "model_state_dict" not in resume_checkpoint:
            raise KeyError(f"{args.resume} has no model_state_dict")
        apply_resume_config(args, resume_checkpoint)
        set_seed(args.seed)
        # resume 后重新解析（配置可能被 checkpoint 覆盖了）
        use_pretrained_config = resolve_pretrained_mode(args.pretrained)
    use_amp = args.amp and device.type == "cuda"
    if use_pretrained_config:
        print("[WARN] ImageNet pretraining must be confirmed as legal with the organizer.")

    print("=" * 60)
    print("Study-aware chest X-ray training")
    print(f"device={device} image_size={args.image_size} batch={args.batch_size}")
    print(f"loss={args.loss} AMP={use_amp} EMA={args.ema_decay}")
    print("=" * 60)

    train_loader, val_loader, _, raw_pos_weights = build_dataloaders(
        batch_size=args.batch_size, num_workers=args.workers,
        seed=args.seed, image_size=args.image_size,
    )
    if train_loader is None or val_loader is None:
        raise RuntimeError("Both training and validation data are required")

    model = build_model(
        pretrained=args.pretrained if resume_checkpoint is None else False,
        dropout=args.dropout,
    ).to(device)
    print(f"parameters={count_parameters(model)['total_M']:.2f}M size={get_model_size_mb(model):.1f}MB")
    criterion, effective_weights = build_criterion(args.loss, raw_pos_weights, device)
    print("effective positive weights:", effective_weights.tolist())

    args.effective_backbone_lr = (
        args.backbone_lr
        if args.backbone_lr is not None
        else (BACKBONE_LR if use_pretrained_config else args.head_lr)
    )
    print(f"configured backbone_lr={args.effective_backbone_lr:.2e} head_lr={args.head_lr:.2e}")
    optimizer = torch.optim.AdamW(
        get_parameter_groups(model, args.effective_backbone_lr, args.head_lr),
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = ModelEMA(model, args.ema_decay)
    best_auc = -float("inf")
    best_epoch = 0
    no_improve = 0
    history = []
    start_epoch = 1

    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        completed_epoch = int(resume_checkpoint.get("epoch", 0))
        start_epoch = completed_epoch + 1
        if "scheduler_state_dict" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        else:
            print("[WARN] Legacy checkpoint has no scheduler state; advancing one step")
            scheduler.step()
        if "scaler_state_dict" in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
        if "ema_state_dict" in resume_checkpoint:
            ema.load_state_dict(resume_checkpoint["ema_state_dict"])
        else:
            print("[WARN] Legacy checkpoint has no EMA state; restarting EMA from model")
            ema = ModelEMA(model, args.ema_decay)
        best_auc = float(resume_checkpoint.get("best_auc", resume_checkpoint.get("macro_auc", -float("inf"))))
        best_epoch = int(resume_checkpoint.get("best_epoch", completed_epoch))
        no_improve = int(resume_checkpoint.get("no_improve", 0))
        history = list(resume_checkpoint.get("history", []))
        print(f"Resuming {args.resume} from epoch {completed_epoch}; next epoch={start_epoch}")
        if start_epoch > args.epochs:
            raise ValueError(
                f"Checkpoint already completed epoch {completed_epoch}; "
                f"set --epochs to at least {start_epoch}"
            )

    for epoch in range(start_epoch, args.epochs + 1):
        freeze_backbone = use_pretrained_config and epoch <= args.freeze_epochs
        set_backbone_trainable(model, not freeze_backbone)
        start = time.time()
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, ema,
            device, epoch, use_amp,
        )
        val_loss, macro_auc, per_class_auc = validate(
            ema.module, val_loader, criterion, device, use_amp,
        )
        current_lrs = [group["lr"] for group in optimizer.param_groups]
        elapsed = time.time() - start
        print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.0f}s)")
        print(f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} macro_auc={macro_auc:.4f}")
        print(f"backbone_lr={current_lrs[0]:.2e} head_lr={current_lrs[1]:.2e} frozen={freeze_backbone}")
        for name, auc in per_class_auc.items():
            print(f"  {name:<28} {auc:.4f}" if np.isfinite(auc) else f"  {name:<28} N/A")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "macro_auc": macro_auc, "backbone_lr": current_lrs[0],
            "head_lr": current_lrs[1], "seconds": elapsed,
        })
        save_history(history)

        if np.isfinite(macro_auc) and macro_auc > best_auc:
            best_auc, best_epoch, no_improve = macro_auc, epoch, 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": ema.module.state_dict(),
                "macro_auc": macro_auc,
                "val_loss": val_loss,
                "config": checkpoint_config(args),
            }, BEST_MODEL_PATH)
            print(f"[OK] saved best EMA model: AUC={macro_auc:.4f}")
        else:
            no_improve += 1

        scheduler.step()
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "macro_auc": macro_auc,
            "val_loss": val_loss,
            "best_auc": best_auc,
            "best_epoch": best_epoch,
            "no_improve": no_improve,
            "history": history,
            "config": checkpoint_config(args),
        }, LAST_MODEL_PATH)
        if no_improve >= args.patience:
            print(f"Early stopping after {args.patience} epochs without AUC improvement")
            break

    print(f"Training complete: best AUC={best_auc:.4f} at epoch {best_epoch}")
    print(f"Deployment checkpoint: {BEST_MODEL_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Study-aware chest X-ray classifier")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--backbone-lr", type=float, default=None, help="Default: 3e-5 pretrained, otherwise head LR")
    parser.add_argument("--head-lr", "--lr", dest="head_lr", type=float, default=HEAD_LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--freeze-epochs", type=int, default=FREEZE_EPOCHS)
    parser.add_argument("--ema-decay", type=float, default=EMA_DECAY)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--loss", choices=["bce", "weighted-bce", "asl"], default="weighted-bce")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--pretrained", choices=["auto", "true", "false"], default="auto",
                        help="预训练模式：auto（默认，本地有就用）/ true（强制用，没有就下载）/ false（不用）")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--resume", type=Path, nargs="?", const=LAST_MODEL_PATH, default=None,
        help="Resume from a training checkpoint (default: last_model.pth)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
