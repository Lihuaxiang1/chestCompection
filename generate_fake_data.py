"""
高质量模拟胸片生成器
- 胸腔轮廓 + 双侧肺野 + 心影纵隔 + 肺纹理 + 肋骨 + 膈肌
- 灰度医学影像风格
- 多标签，疾病分布模拟真实数据
"""

import csv
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

# 输出目录（假数据测试集）
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "假数据测试集"
IMG_SIZE = 512

DATASETS = {
    "train": 1000,
    "val":   250,
    "test":  100,
}

DISEASE_LABELS = [
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation",
    "Pneumonia", "Edema", "Cardiomegaly", "Atelectasis",
    "Lung Opacity", "Pleural Effusion", "No Finding",
]

SEED = 42


def generate_labels(rng):
    """
    生成多标签，模拟真实疾病关联。
    基础阳性率参考真实胸片数据，疾病之间有一定共现概率。
    """
    # 基础概率
    base_probs = {
        "Enlarged Cardiomediastinum": 0.10,
        "Pneumothorax": 0.05,
        "Consolidation": 0.08,
        "Pneumonia": 0.06,
        "Edema": 0.07,
        "Cardiomegaly": 0.12,
        "Atelectasis": 0.10,
        "Lung Opacity": 0.18,
        "Pleural Effusion": 0.12,
    }

    labels = {}

    # 先决定几个主要疾病
    for disease, prob in base_probs.items():
        labels[disease] = 1 if rng.random() < prob else 0

    # 疾病关联：肺炎 -> 肺部阴影 概率高
    if labels["Pneumonia"] and rng.random() < 0.7:
        labels["Lung Opacity"] = 1

    # 疾病关联：胸腔积液 -> 肺不张 概率升高
    if labels["Pleural Effusion"] and rng.random() < 0.4:
        labels["Atelectasis"] = 1

    # 疾病关联：心影扩大 -> 肺水肿 概率升高
    if labels["Cardiomegaly"] and rng.random() < 0.3:
        labels["Edema"] = 1

    # 疾病关联：实变 -> 肺部阴影 几乎必然
    if labels["Consolidation"]:
        labels["Lung Opacity"] = 1

    # 计算是否有任何疾病
    has_disease = any(labels.values())

    # No Finding：没有任何疾病时为1
    labels["No Finding"] = 0 if has_disease else 1

    # 按固定顺序返回
    return [labels[name] for name in DISEASE_LABELS]


def make_chest_base(size, np_rng, variant=0):
    """生成一张模拟胸片的基础结构。"""
    y, x = np.ogrid[:size, :size]
    cy = size * 0.48  # 中心稍偏上
    cx_left = size * 0.32   # 左肺中心
    cx_right = size * 0.68  # 右肺中心

    # 随机变量
    heart_ratio = 0.42 + np_rng.normal(0, 0.03)  # 心胸比
    heart_ratio = np.clip(heart_ratio, 0.35, 0.55)

    lung_width_factor = 1.0 + np_rng.normal(0, 0.04)
    lung_height_factor = 1.0 + np_rng.normal(0, 0.03)

    # 胸腔整体轮廓（稍扁的椭圆）
    chest_ry = size * 0.42 * lung_height_factor
    chest_rx = size * 0.40 * lung_width_factor
    chest_dist = ((y - cy) / chest_ry) ** 2 + ((x - size * 0.5) / chest_rx) ** 2
    chest_mask = 1.0 - np.clip((chest_dist - 1.0) / 0.08 + 1.0, 0.0, 1.0)

    # 左肺（椭圆）
    left_ry = size * 0.38 * lung_height_factor
    left_rx = size * 0.20 * lung_width_factor
    left_dist = ((y - cy) / left_ry) ** 2 + ((x - cx_left) / left_rx) ** 2
    left_lung = 1.0 - np.clip((left_dist - 1.0) / 0.05 + 1.0, 0.0, 1.0)

    # 右肺（椭圆，稍大一点）
    right_ry = size * 0.38 * lung_height_factor
    right_rx = size * 0.22 * lung_width_factor
    right_dist = ((y - cy) / right_ry) ** 2 + ((x - cx_right) / right_rx) ** 2
    right_lung = 1.0 - np.clip((right_dist - 1.0) / 0.05 + 1.0, 0.0, 1.0)

    # 双肺合并
    lung_mask = np.clip(left_lung + right_lung, 0.0, 1.0)

    # 心影（中间偏左的梨形/椭圆形亮区）
    heart_cy = cy + size * 0.02
    heart_cx = size * (0.5 - 0.03)  # 稍偏左
    heart_ry = size * 0.24
    heart_rx = size * heart_ratio * 0.5
    heart_dist = ((y - heart_cy) / heart_ry) ** 2 + ((x - heart_cx) / heart_rx) ** 2
    heart_mask = 1.0 - np.clip((heart_dist - 1.0) / 0.06 + 1.0, 0.0, 1.0)
    # 心影上窄下宽（梨形）
    heart_top_taper = np.clip(1.0 - (cy - y) / (size * 0.3), 0.3, 1.0)
    heart_mask = heart_mask * np.where(y < heart_cy, heart_top_taper, 1.0)

    # 纵隔（心影上方连接区域）
    mediastinum_width = size * 0.12
    med_mask = np.exp(-((x - size * 0.5) / mediastinum_width) ** 2)
    med_mask = med_mask * np.where(y < cy - size * 0.05, 1.0, 0.0)
    med_mask = med_mask * chest_mask

    # 膈肌（底部弧形亮边）
    diaphragm_y = cy + size * 0.28
    diaphragm_height = size * 0.06
    # 右侧膈肌略高
    dip_right = 1.0 + 0.2 * np.exp(-((x - cx_right) / (size * 0.15)) ** 2)
    dip_left = 1.0 + 0.15 * np.exp(-((x - cx_left) / (size * 0.15)) ** 2)
    dip_factor = dip_right + dip_left - 1.0

    diaphragm_dist = (y - diaphragm_y) / (diaphragm_height * dip_factor)
    diaphragm_mask = np.where(y > diaphragm_y,
                              np.exp(-diaphragm_dist ** 2 * 3),
                              0.0)
    diaphragm_mask = diaphragm_mask * chest_mask * 0.6

    # 肋骨（水平弧线，胸腔上部）
    rib_mask = np.zeros((size, size), dtype=np.float32)
    n_ribs = 8
    for i in range(n_ribs):
        rib_y = size * (0.18 + i * 0.055)
        rib_thickness = size * 0.012 + np_rng.normal(0, size * 0.002)
        # 肋骨弧度（中间低两边高）
        rib_arc = 0.015 * ((x - size * 0.5) / (size * 0.4)) ** 2
        rib_dist = (y - rib_y - rib_arc * size) / rib_thickness
        rib = np.exp(-rib_dist ** 2)
        rib = rib * chest_mask * 0.35
        rib_mask = np.maximum(rib_mask, rib)

    # 肺纹理（从肺门向外的放射状细线纹理）
    texture = np.zeros((size, size), dtype=np.float32)
    # 多尺度噪声叠加
    for scale in [4, 8, 16, 32]:
        small = np_rng.normal(0, 1.0 / scale * 15, (size // scale, size // scale)).astype(np.float32)
        texture += np.array(Image.fromarray(small).resize((size, size), Image.BILINEAR))

    # 肺纹理只在肺野内可见，且从肺门向周边衰减
    hilum_left = (cy, cx_left + size * 0.03)
    hilum_right = (cy, cx_right - size * 0.03)

    dist_left_hilum = np.sqrt(((y - hilum_left[0]) / size) ** 2 + ((x - hilum_left[1]) / size) ** 2)
    dist_right_hilum = np.sqrt(((y - hilum_right[0]) / size) ** 2 + ((x - hilum_right[1]) / size) ** 2)
    dist_hilum = np.minimum(dist_left_hilum, dist_right_hilum)
    texture_grad = np.clip(1.0 - dist_hilum * 2.5, 0.0, 1.0)

    lung_texture = texture * lung_mask * texture_grad * 0.4

    # 基础亮度
    base_brightness = np_rng.integers(155, 185)

    # 构建图像
    img = np.full((size, size), base_brightness, dtype=np.float32)

    # 肺野变暗（含气区域）
    img -= lung_mask * 75

    # 心影和纵隔变亮（软组织密度）
    img += heart_mask * 60
    img += med_mask * 50

    # 膈肌变亮
    img += diaphragm_mask * 40

    # 肋骨稍亮
    img += rib_mask * 30

    # 肺纹理（稍暗的线状影）
    img -= lung_texture

    # 整体大尺度亮度变化（模拟曝光不均）
    big_noise = np_rng.normal(0, 8, (size // 32, size // 32)).astype(np.float32)
    big_noise_img = np.array(Image.fromarray(big_noise).resize((size, size), Image.BILINEAR))
    img += big_noise_img * 0.5

    # 细颗粒噪声（胶片颗粒感）
    grain = np_rng.normal(0, 6, (size, size)).astype(np.float32)
    img += grain

    # 应用胸腔遮罩，外面压黑
    outside_brightness = 8.0
    img = img * chest_mask + outside_brightness * (1 - chest_mask)

    # 裁剪到有效范围
    img = np.clip(img, 0, 255).astype(np.uint8)

    # 转 RGB
    img_rgb = np.stack([img, img, img], axis=-1)
    return img_rgb


def main():
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)

    print("=" * 60)
    print("高质量模拟胸片生成器")
    print("=" * 60)
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"图片尺寸: {IMG_SIZE}x{IMG_SIZE}")
    print()

    total = 0

    for split, count in DATASETS.items():
        img_dir = OUTPUT_DIR / split
        img_dir.mkdir(parents=True, exist_ok=True)
        csv_path = OUTPUT_DIR / f"{split}_labels.csv"

        print(f"[{split}] 生成 {count} 张...")

        rows = []
        for i in range(count):
            img_name = f"{split}_{i:04d}.png"
            img = make_chest_base(IMG_SIZE, np_rng, variant=total + i)
            Image.fromarray(img).save(img_dir / img_name, "PNG", optimize=True)

            labels = generate_labels(rng)
            rows.append([img_name] + labels)

            if (i + 1) % 200 == 0:
                print(f"  已生成 {i+1}/{count} 张")

        # 写 CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Image Index"] + DISEASE_LABELS)
            writer.writerows(rows)

        # 统计
        label_arr = np.array([row[1:] for row in rows], dtype=int)
        pos_rates = label_arr.mean(axis=0)
        n_no_finding = int(label_arr[:, 9].sum())

        print(f"  完成: {count} 张 -> {img_dir}")
        print(f"  No Finding: {n_no_finding} 张 ({pos_rates[9]:.0%})")
        print(f"  平均每张阳性疾病数: {label_arr[:, :9].sum() / count:.2f}")
        print()

        total += count

    print("=" * 60)
    print(f"✅ 全部完成，共 {total} 张高质量模拟胸片")
    print(f"📁 数据目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
