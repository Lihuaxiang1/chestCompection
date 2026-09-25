"""
===============================================================================
  配置中心 —— 所有参数都在这里，改一处全部生效
===============================================================================
  拿到比赛数据后需要改的：
    1. DATA_DIR      — 数据文件夹路径
    2. TRAIN_CSV     — 训练集标注文件
    3. VAL_CSV       — 验证集标注文件（如果有）
    4. DISEASE_LABELS — 10个疾病名称（按比赛要求顺序）
===============================================================================
"""

from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
#  路径配置
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"

# 图片目录
TRAIN_IMG_DIR = DATA_DIR / "train"
VAL_IMG_DIR   = DATA_DIR / "val"
TEST_IMG_DIR  = DATA_DIR / "test"

# 标注文件
TRAIN_CSV = DATA_DIR / "train_labels.csv"
VAL_CSV   = DATA_DIR / "val_labels.csv"

# 输出目录
OUTPUTS_DIR     = PROJECT_ROOT / "outputs"
CKPT_DIR        = OUTPUTS_DIR / "checkpoints"
LOGS_DIR        = OUTPUTS_DIR / "logs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
FIGURES_DIR     = OUTPUTS_DIR / "figures"

# 预训练权重目录（放在项目本地，方便打包）
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"

# 创建目录
for d in [CKPT_DIR, LOGS_DIR, PREDICTIONS_DIR, FIGURES_DIR, PRETRAINED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  类别标签（比赛10类，按顺序）
# ═══════════════════════════════════════════════════════════════════════════════

DISEASE_LABELS = [
    "Enlarged Cardiomediastinum",  # 0: 纵隔心影增大
    "Pneumothorax",                # 1: 气胸
    "Consolidation",               # 2: 肺部实变
    "Pneumonia",                   # 3: 肺炎
    "Edema",                       # 4: 肺水肿
    "Cardiomegaly",                # 5: 心影扩大
    "Atelectasis",                 # 6: 肺不张
    "Lung Opacity",                # 7: 肺部阴影
    "Pleural Effusion",            # 8: 胸腔积液
    "No Finding",                  # 9: 无异常发现
]
NUM_CLASSES = len(DISEASE_LABELS)

# ═══════════════════════════════════════════════════════════════════════════════
#  图像参数
# ═══════════════════════════════════════════════════════════════════════════════

IMAGE_SIZE = 320  # 胸片小病灶对分辨率敏感；最终需用 T4 复测端到端延迟

# ImageNet 归一化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Grad-CAM 目标层（ConvNeXt-Tiny 最后一阶段）
GRADCAM_TARGET_LAYER = "features.7"

# ═══════════════════════════════════════════════════════════════════════════════
#  训练参数
# ═══════════════════════════════════════════════════════════════════════════════

SEED = 42

BATCH_SIZE = 16       # 320x320 默认 batch；显存充足时可增大
NUM_WORKERS = 4       # 数据加载线程数

NUM_EPOCHS = 30       # 训练轮数

BACKBONE_LR = 3e-5    # 预训练主干使用较小学习率
HEAD_LR = 3e-4        # 新分类头使用较大学习率
WEIGHT_DECAY = 1e-4   # 权重衰减
WARMUP_EPOCHS = 2
FREEZE_EPOCHS = 2
EMA_DECAY = 0.999
DROPOUT = 0.2

# 稀有类别权重使用 sqrt 并截断，避免原始 neg/pos 权重过度补偿
POS_WEIGHT_POWER = 0.5
POS_WEIGHT_MAX = 10.0

EARLY_STOP_PATIENCE = 8  # 早停耐心值

# ═══════════════════════════════════════════════════════════════════════════════
#  其他
# ═══════════════════════════════════════════════════════════════════════════════

BEST_MODEL_PATH = CKPT_DIR / "best_model.pth"
LAST_MODEL_PATH = CKPT_DIR / "last_model.pth"
SUBMISSION_PATH = PREDICTIONS_DIR / "submission.csv"

# F1 诊断默认阈值；AUC 提交默认输出全部连续概率
DEFAULT_THRESHOLD = 0.5
