"""
===============================================================================
  模型定义
===============================================================================
  使用 ConvNeXt-Tiny + ImageNet 预训练
  输出 logits（未 sigmoid），用于 BCEWithLogitsLoss
===============================================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights

from config import DROPOUT, IMAGE_SIZE, NUM_CLASSES, PRETRAINED_DIR


def _find_local_weights(model_name="convnext_tiny"):
    """
    在项目 pretrained/ 目录下找本地预训练权重文件。
    找到就返回路径，没找到返回 None。
    """
    if not PRETRAINED_DIR.exists():
        return None
    for pattern in [f"{model_name}*.pth", f"{model_name.replace('_', '-')}*.pth"]:
        matches = sorted(PRETRAINED_DIR.glob(pattern))
        if matches:
            return matches[0]
    return None


def build_model(num_classes=NUM_CLASSES, pretrained="auto", dropout=DROPOUT):
    """
    构建 ConvNeXt-Tiny 模型。

    Args:
        num_classes: 输出类别数（默认 10）
        pretrained: 预训练模式
            - True: 必须用预训练，本地有就加载，没有就自动下载
            - False: 不用预训练，从头训练
            - "auto"（默认）: 本地有预训练权重就用，没有就不用，也不下载
        dropout: 分类头 dropout 比例

    Returns:
        model: PyTorch 模型
    """
    # 加载预训练模型
    use_pretrained = False
    local_weights = _find_local_weights("convnext_tiny")

    if pretrained == "auto":
        # 自动模式：本地有就用，没有就不用
        if local_weights:
            use_pretrained = True
            print(f"[auto] 检测到本地预训练权重，将启用: {local_weights.name}")
        else:
            use_pretrained = False
            print("[auto] 未检测到本地预训练权重，将从头训练")
    elif pretrained:
        # 强制启用：本地有就用本地的，没有就下载
        use_pretrained = True
        if local_weights:
            print(f"从本地加载预训练权重: {local_weights}")
        else:
            print("从 torchvision 加载 ImageNet 预训练权重（首次会自动下载）")
    else:
        use_pretrained = False
        print("不加载预训练权重（从头训练）")

    if use_pretrained and local_weights:
        model = models.convnext_tiny(weights=None)
        state_dict = torch.load(local_weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    elif use_pretrained:
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = models.convnext_tiny(weights=weights)
    else:
        model = models.convnext_tiny(weights=None)

    # 修改分类头
    # ConvNeXt classifier: LayerNorm2d -> Flatten -> prediction head
    in_features = model.classifier[2].in_features  # 768

    # 替换最后一层
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )

    print(f"模型输出维度: {num_classes}")
    print(f"分类头输入维度: {in_features}")

    return model


def set_backbone_trainable(model, trainable):
    """Freeze or unfreeze the ConvNeXt feature extractor."""
    for parameter in model.features.parameters():
        parameter.requires_grad = trainable


def get_parameter_groups(model, backbone_lr, head_lr):
    """Use a conservative LR for the backbone and a larger LR for the new head."""
    return [
        {"params": model.features.parameters(), "lr": backbone_lr, "name": "backbone"},
        {"params": model.classifier.parameters(), "lr": head_lr, "name": "head"},
    ]


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "total_M": total / 1e6,
        "trainable_M": trainable / 1e6,
    }


def get_model_size_mb(model):
    """计算模型文件大小（MB）"""
    # 参数转成 float32
    param_size = sum(p.numel() * 4 for p in model.parameters())  # 4 bytes per float32
    # buffer (如 BatchNorm 的 running_mean/var)
    buffer_size = sum(b.numel() * 4 for b in model.buffers())

    total_bytes = param_size + buffer_size
    return total_bytes / (1024 ** 2)  # 转成 MB


# ═══════════════════════════════════════════════════════════════════════════════
#  测试
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("测试 Model 模块")
    print("=" * 60)

    # 设备
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"设备: {device}")

    # 构建模型
    print("\n构建模型...")
    model = build_model(pretrained=False)  # 不下载权重，省时间
    model = model.to(device)

    # 统计参数
    print("\n参数统计:")
    stats = count_parameters(model)
    print(f"  总参数量: {stats['total_M']:.2f} M")
    print(f"  可训练参数量: {stats['trainable_M']:.2f} M")

    # 模型大小
    size_mb = get_model_size_mb(model)
    print(f"  模型大小: {size_mb:.1f} MB")

    # 前向传播测试
    print("\n前向传播测试...")
    dummy_input = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"  输入形状: {dummy_input.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  输出范围: [{output.min():.2f}, {output.max():.2f}]")

    # 模拟预测
    probs = torch.sigmoid(output)
    print(f"  Sigmoid 后: [{probs.min():.4f}, {probs.max():.4f}]")

    print("\n[OK] Model 模块测试完成")
