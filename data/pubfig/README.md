# PubFig 数据目录

本目录用于存放 PubFig: Public Figures Face Database 的原始清单文件和本地下载子集。

## 目录结构

- `raw/`
  - 官方下载页提供的文本清单，例如 `dev_people.txt`、`dev_urls.txt`、`eval_urls.txt`
- `dev_subset/`
  - 根据 `dev_urls.txt` 实际下载并按人脸框裁剪后的开发集子集
  - `faces_112/<person>/...jpg` 为可直接用于训练的 `112x112` 人脸图
  - `download_report.tsv` 为逐条下载状态
  - `summary.tsv` 为每个身份的下载统计
  - `subset_manifest.tsv` 为成功样本的元数据

## 下载开发集子集

推荐命令：

```bash
./.venv/bin/python scripts/download_pubfig_subset.py \
  --manifest data/pubfig/raw/dev_urls.txt \
  --people-file data/pubfig/raw/dev_people.txt \
  --output-dir data/pubfig/dev_subset \
  --num-people 30 \
  --images-per-person 15
```

## 说明

- PubFig 官方因版权原因不直接分发图片文件，只提供图片 URL 清单。
- 本仓库中的下载脚本会按官方提供的人脸框裁剪，并保存为统一尺寸。
- 建议将 `dev` 集用于算法开发，将 `eval` 集用于最终评测，避免数据泄漏。
