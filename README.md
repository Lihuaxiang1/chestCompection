# 胸部 X 光多标签分类比赛版

面向全国高校计算机能力挑战赛人工智能赛道的 Study 级多标签分类代码。

## 快速开始

三步跑通完整流程（先用假数据验证链路）：

```powershell
# 1. 生成模拟胸片假数据（1000 train / 250 val / 100 test）
python generate_fake_data.py

# 2. 训练（2 个 epoch 验证流程，正式训练去掉 --epochs 2）
python src\train.py --epochs 2 --loss weighted-bce --workers 0

# 3. 评估 + 生成提交文件
python src\evaluate.py
python src\predict.py
```

假数据的图像与标签相互独立，仅用于验证程序能否跑通，合理 AUC 接近 0.5。

## 当前设计

- ConvNeXt-Tiny，默认输入 320x320，分类头含 Dropout。
- 图像按原始宽高比缩放并补黑边，不裁掉肺尖和肋膈角。
- 正式数据按 `(Subject_id, Study_id)` 识别检查，使用一个 Study 的全部图片。
- 训练时每张图参与反向传播，但同一 Study 的样本总权重为 1。
- 验证、阈值搜索和提交均先平均 Study 内的 image logits。
- 支持 BCE、截断平方根正样本权重 BCE、Asymmetric Loss。
- 支持主干冻结预热、分层学习率、cosine 调度、AMP 和 EMA。
- 默认提交每个 Study 的全部 10 类连续概率，阈值只用于 Macro-F1。

## 数据目录

正式比赛数据：

```text
data/
├── train/
│   └── Subject_id/Study_id/image.png
├── val/
│   └── Subject_id/Study_id/image.png
├── test/
│   └── Subject_id/Study_id/image.png
├── train_labels.csv
└── val_labels.csv
```

长格式 CSV 至少需要：

```text
Subject_id,Study_id,Predict_class
```

`Predict_class` 必须为 `0..9`。同一 Study 可以有多行阳性类别。

代码也保留了平目录加宽格式 CSV 的模式，用于本地冒烟测试。可通过 `generate_fake_data.py` 生成模拟数据快速验证链路。当前生成的假数据中，图像内容和疾病标签相互独立，因此它只能测试程序是否正常，不能用于评价模型效果；其合理 AUC 接近 0.5。

### 生成假数据

```powershell
python generate_fake_data.py
```

会在假数据测试集目录下生成 1350 张带解剖结构的模拟胸片（train 1000 / val 250 / test 100）以及对应的宽格式标签。将其复制到 `data/` 目录即可用于冒烟测试。

## 环境

```powershell
conda activate cxr
cd "D:\1738\Desktop\计算机能力挑战赛\老师资料\比赛版"
pip install -r requirements.txt
```

主办方规则写明不允许外部标注数据。ImageNet 预训练是否允许需要先向主办方确认。代码默认不使用预训练权重；确认允许后再传入 `--pretrained`。

### 预训练权重

项目支持两种加载方式：

1. **自动下载**：首次加 `--pretrained` 运行训练时，torchvision 会自动下载到系统缓存
2. **本地权重（推荐）**：运行 `python src\download_weights.py` 下载到项目 `pretrained/` 目录，以后都从本地加载

本地权重的好处：打包发给队友的时候一起带上，队友不用自己下载。

## 运行顺序

### 1. 数据审计

```powershell
python src\audit_data.py
python src\audit_data.py --hash-duplicates
```

审计标签范围、类别分布、缺图、无异常标签冲突和训练验证患者泄漏。重复图片哈希检查较慢，因此默认关闭。

### 2. 训练

规则最保守的训练方式：

```powershell
python src\train.py --loss weighted-bce
```

确认允许 ImageNet 权重后：

```powershell
python src\train.py --pretrained --loss weighted-bce
```

常用参数：

```powershell
python src\train.py `
  --pretrained `
  --epochs 30 `
  --batch-size 16 `
  --image-size 320 `
  --backbone-lr 3e-5 `
  --head-lr 3e-4 `
  --freeze-epochs 2 `
  --warmup-epochs 2 `
  --loss weighted-bce
```

部署权重写入 `outputs/checkpoints/best_model.pth`，只包含 EMA 模型和必要配置。`last_model.pth` 额外包含原始模型、EMA、优化器、学习率调度器、AMP scaler、早停状态和训练历史，用于恢复训练，不应作为提交模型。

中断后按原配置续训到指定的总 epoch：

```powershell
python src\train.py --resume --epochs 30
```

也可以通过 `--resume 路径` 指定其他训练 checkpoint。模型结构和训练关键参数会从 checkpoint 恢复。

### 3. Study 级评估

```powershell
python src\evaluate.py
```

输出 Macro-AUC、每类 AUC 和 `outputs/figures/roc_curves.png`。Grad-CAM 在 Gradio Demo 中根据上传图片动态生成。

### 4. F1 阈值搜索

```powershell
python src\threshold_search.py
```

阈值只影响 Macro-F1，不影响 ROC-AUC。

### 5. 生成提交文件

```powershell
python src\predict.py
```

默认每个 Study 输出 10 行概率。可选多尺度 TTA：

```powershell
python src\predict.py --tta
```

多个合规 checkpoint 可以直接平均 logits：

```powershell
python src\predict.py --ckpt outputs\checkpoints\seed1.pth outputs\checkpoints\seed2.pth
```

只有官方明确要求删除低概率类别时才使用：

```powershell
python src\predict.py --filter-threshold 0.5
```

主办方文档对完整概率与 0.5 阈值的表述存在冲突，正式提交前必须根据官方样例确认列名和行数。

平目录测试图片仅用于冒烟测试，输出中的 `Subject_id` 会为空；正式提交必须使用 `test/Subject_id/Study_id/image.png` 结构。脚本会拒绝在同一个测试目录中混用平目录和嵌套目录，避免漏掉 Study。

### 6. 可视化 Demo（Gradio）

```powershell
python app.py
```

启动后浏览器打开本地地址，可上传单张胸片查看：
- 10 类疾病的预测概率
- Grad-CAM 热力图叠加效果
- 高置信度阳性类别列表

用于快速验证模型效果、做演示和展示。

### 7. 端到端测速

```powershell
python src\benchmark.py
```

脚本同时报告模型前向延迟和文件读取、预处理、推理、Sigmoid 的端到端延迟。比赛的 100ms 限制应以后者 P95 为准，并最终在指定 T4 环境复测。

## 目录结构

```text
比赛版/
├── data/                  # 数据集（train/val/test + 标签 CSV）
├── outputs/               # 运行产物（自动生成）
│   ├── checkpoints/       # 模型权重（best_model.pth 用于提交）
│   ├── logs/              # 训练日志
│   ├── predictions/       # 提交文件
│   └── figures/           # ROC 曲线等评估图
├── src/                   # 核心代码
│   ├── config.py          # 全部参数配置
│   ├── dataset.py         # 数据加载 + 增强（自动识别格式）
│   ├── model.py           # ConvNeXt-Tiny 模型定义
│   ├── metrics.py         # Study 级聚合 + AUC 计算
│   ├── train.py           # 训练主循环
│   ├── evaluate.py        # Study 评估 + ROC/Grad-CAM 工具
│   ├── predict.py         # 测试集推理 + 生成提交
│   ├── benchmark.py       # 推理速度测试
│   ├── threshold_search.py # F1 最优阈值搜索
│   ├── audit_data.py      # 数据质量审计
│   └── download_weights.py # 预训练权重下载
├── app.py                 # Gradio 可视化 Demo
├── generate_fake_data.py  # 生成模拟胸片假数据
├── requirements.txt       # 依赖清单
└── README.md              # 本文件
```

## 建议消融顺序

每次只改变一个因素，并记录 Macro-AUC、每类 AUC、Macro-F1、模型大小和端到端延迟：

1. 224 与 320 输入分辨率。
2. BCE、weighted-bce 和 ASL。
3. 是否使用合规的预训练权重。
4. 1、2、3 个冻结预热 epoch。
5. 单尺度与 `--tta`。
6. 不同随机种子。

验证集只有 859 个 Study 时，小幅 AUC 变化可能来自随机波动。重要结论至少用三个随机种子验证。

## 自动化测试

```powershell
python -m unittest discover -s tests -v
```

测试覆盖 Study 聚合、多图权重、患者级多标签划分、EMA warmup 和测试目录格式保护。
