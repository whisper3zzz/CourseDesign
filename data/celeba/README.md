# CelebA 数据目录

这里用于存放 `CelebA` 官方数据和课程设计用的小规模训练子集。

## 为什么切到 CelebA

当前仓库已经验证过 `PubFig` 路线，但它的官方发布方式是图片 URL 清单，今天大量链接已经失效，不适合继续作为真正可落地的训练数据来源。

`CelebA` 更适合当前课程设计的原因：

- 官方长期维护，数据说明明确
- 自带人脸对齐后的图片版本
- 自带身份标注文件
- 容易裁出一个适合本机 CPU 训练的小子集

官方页面：

`https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html`

## 目录结构

建议把官方文件按下面结构放入：

```text
data/celeba/
├── README.md
├── raw/
│   ├── img_align_celeba/
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   ├── identity_CelebA.txt
│   └── list_eval_partition.txt
└── subsets/
```

最少需要：

- `img_align_celeba/`
- `identity_CelebA.txt`

如果你还放了 `list_eval_partition.txt`，脚本也可以按官方划分去生成子集。

## 准备课程设计子集

仓库里已经提供了子集准备脚本：

```bash
uv run python scripts/prepare_celeba_subset.py \
  --raw-root data/celeba/raw \
  --output-root data/celeba/subsets/course_v1 \
  --max-identities 50 \
  --train-per-identity 20 \
  --val-per-identity 5 \
  --test-per-identity 5
```

默认会：

- 选择样本数足够的身份
- 生成 `train/ val/ test/` 目录
- 按 `ImageFolder` 结构组织，方便 `MindSpore` 直接训练
- 额外输出 `meta/identity_map.tsv` 和 `meta/subset_summary.json`

如果你想按官方分区挑图，可以加：

```bash
uv run python scripts/prepare_celeba_subset.py \
  --raw-root data/celeba/raw \
  --output-root data/celeba/subsets/course_v1_official \
  --split-mode official
```

## 本地训练 MobileFaceNet

准备好子集后，可以直接跑轻量训练脚本：

```bash
uv sync --extra mindspore
uv run python scripts/train_mobilefacenet_subset.py \
  --train-dir data/celeba/subsets/course_v1/train \
  --val-dir data/celeba/subsets/course_v1/val \
  --output-dir output/mobilefacenet_course_v1
```

这个脚本默认：

- 使用 `MobileFaceNet`
- 优先加载仓库里的预训练权重
- 在 CPU 上运行
- 默认只训练分类头，避免在本机上训练过慢

注意：

- 当前这台 macOS 机器上已经实测过 `MindSpore 2.8.0`
- `mindspore` 可以导入，普通张量运算也能跑
- 但 `Conv2d` 在 CPU 侧会触发底层 `Xbyak::Error`

所以这个训练脚本现在会先做一次卷积自检：

- 如果当前运行环境能执行 CNN，它会继续训练
- 如果像这台 Mac 一样只能导入但跑不了卷积，它会直接给出明确错误提示

这不影响 `CelebA` 子集准备脚本使用，但会影响本机直接跑 `MindSpore CNN` 训练。要真正训练，优先换到 `Linux/GPU` 或其他兼容环境。

如果你想连 backbone 一起微调，可以追加：

```bash
uv run python scripts/train_mobilefacenet_subset.py \
  --train-dir data/celeba/subsets/course_v1/train \
  --val-dir data/celeba/subsets/course_v1/val \
  --output-dir output/mobilefacenet_course_v1_full \
  --train-backbone
```

## 输出内容

训练脚本会在输出目录中生成：

- `checkpoints/`
- `mobilefacenet_backbone.ckpt`
- `mobilefacenet_classifier.ckpt`
- `label_map.json`
- `train_summary.json`

后续如果要把深度识别接回桌面程序，优先复用 `mobilefacenet_backbone.ckpt` 和本地样本库做 embedding 比对即可。
