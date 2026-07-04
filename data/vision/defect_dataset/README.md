# 缺陷专用数据集工作区

该目录用于承接“防腐视觉缺陷”样本回收、人工标注和 YOLO 数据集导出。

## 推荐结构

| 目录 | 用途 |
|---|---|
| `raw_results/` | 板端 `vision-image` / `vision-camera --result-json` 保存的原始结果 JSON |
| `label_studio/` | Label Studio 任务文件、导出结果 |
| `yolo_v1/` | 转换后的 Ultralytics YOLO 数据集 |

## 当前首版类别

1. `rust_like_corrosion`
2. `coating_flaking_or_delamination`
3. `chalking_or_powdering`

`CUI` 当前保留为图像级风险标签，不作为首版边界框训练类别。
