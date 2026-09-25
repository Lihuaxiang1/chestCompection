# 比赛数据存放说明

## 📁 目录结构

```
chest-xray-ai - 修改版/
└── data/
    ├── train_labels.csv       ← 训练集标注文件
    ├── val_labels.csv         ← 验证集标注文件（如果有）
    ├── train/                 ← 训练集图片（直接放图片，不要套文件夹）
    ├── val/                   ← 验证集图片（如果有）
    └── test/                  ← 测试集图片
```

## 📋 标注文件格式

标注 CSV 至少需要包含两列：

| 列名 | 说明 | 示例 |
|---|---|---|
| `Image Index` | 图片文件名 | `00000001_000.png` |
| 每个病名一列 | 0 = 没病，1 = 有病 | `Atelectasis: 1` |

或者标签列叫 `Finding Labels`，用 `|` 分隔多个病名（NIH 格式）：
```
Image Index,Finding Labels
00000001_000.png,Atelectasis|Effusion
```

**如果格式不一样，去 `src/dataset.py` 里的 `_load_labels()` 函数改一下就行。**

## 🏷️ 当前标签猜测

当前 `config.py` 里默认的 10 种病（从常见的 NIH 14 类中挑选）：

1. Atelectasis（肺不张）
2. Cardiomegaly（心脏肥大）
3. Effusion（胸腔积液）
4. Infiltration（浸润）
5. Mass（肿块）
6. Nodule（结节）
7. Pneumonia（肺炎）
8. Pneumothorax（气胸）
9. Consolidation（实变）
10. Edema（水肿）

⚠️ **拿到数据后第一件事：核对这 10 个病名和顺序是否正确！**
- 如果名字不对 → 去 `src/config.py` 改 `DISEASE_LABELS`
- 如果顺序不对 → 重新排列 `DISEASE_LABELS` 的顺序

## 🚀 数据放好之后

```bash
# 开始训练
python src/train.py

# 训练完评估
python src/evaluate.py
```
