"""
下载预训练权重到项目本地目录。

默认下载到项目根目录的 pretrained/ 文件夹下，
这样打包发给队友的时候一起带上，队友不用自己下载。

用法:
    python src/download_weights.py
    python src/download_weights.py --model convnext_tiny
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torchvision import models

from config import PRETRAINED_DIR


def download_convnext_tiny(save_dir: Path) -> Path:
    """
    下载 ConvNeXt-Tiny ImageNet 预训练权重到本地目录。

    Returns:
        保存的权重文件路径
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "convnext_tiny_imagenet1k_v1.pth"

    if save_path.exists():
        print(f"权重已存在: {save_path}")
        print("如需重新下载，请先删除该文件")
        return save_path

    print("=" * 60)
    print("下载 ConvNeXt-Tiny ImageNet 预训练权重")
    print("=" * 60)
    print(f"目标路径: {save_path}")
    print()
    print("正在从 torchvision 下载... (约 110MB)")
    print("如果下载慢，可以手动下载后放到这个目录下")
    print()

    # 触发下载（torchvision 会先下到缓存，我们再存一份到本地）
    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = models.convnext_tiny(weights=weights)

    # 保存权重到项目本地
    torch.save(model.state_dict(), save_path)

    size_mb = save_path.stat().st_size / (1024 ** 2)
    print(f"\n✅ 下载完成，已保存到: {save_path}")
    print(f"   文件大小: {size_mb:.1f} MB")

    return save_path


def main():
    parser = argparse.ArgumentParser(description="下载预训练权重到项目本地")
    parser.add_argument(
        "--model",
        default="convnext_tiny",
        choices=["convnext_tiny"],
        help="要下载的模型（目前只支持 ConvNeXt-Tiny）",
    )
    args = parser.parse_args()

    if args.model == "convnext_tiny":
        download_convnext_tiny(PRETRAINED_DIR)


if __name__ == "__main__":
    main()
