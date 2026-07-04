# Phase 5：缺陷专用数据与模型准备链路

适用日期：`2026-06-26`

## 当前目标

| 目标 | 当前做法 |
|---|---|
| 从“官方通用模型 + 回退映射”过渡到“缺陷模型直出” | 先铺好样本回收、人工标注、YOLO 数据集生成、训练导出 ONNX 的闭环 |
| 保持板端部署兼容 | 训练在 PC/GPU 环境完成，部署继续回到 Muse Pi Pro 的 `spacemit_vision` |
| 避免错误标签设计 | `CUI` 保留为图像级风险标签，不作为首版边界框检测类 |

## 首版类别口径

| 类型 | 类别 |
|---|---|
| 边界框训练类 | `rust_like_corrosion`、`coating_flaking_or_delamination`、`chalking_or_powdering` |
| 图像级标签 | `cui_risk_visual_hint`、`not_target_surface`、`uncertain` |

## 已落地文件

| 路径 | 作用 |
|---|---|
| `configs/defect_taxonomy.yaml` | 首版缺陷标注口径 |
| `assets/label_studio/defect_detection_config.xml` | Label Studio 标注界面 |
| `scripts/vision_build_label_studio_tasks.py` | 从板端结果 JSON 生成标注任务 |
| `scripts/vision_convert_labelstudio_to_yolo.py` | 从 Label Studio 导出转 Ultralytics YOLO 数据集 |
| `scripts/vision_train_defect_yolo.py` | 训练并导出 ONNX |
| `configs/vision_spacemit_defect.yaml` | 未来缺陷模型在板端的切换配置 |

## 推荐工作流

### 1. 板端回收样本

```bash
cd /mnt/ssd/spacemit_project
bash scripts/vision.sh vision-camera \
  --result-json /mnt/ssd/data/vision/defect_dataset/raw_results/sample_001.json
```

### 2. Windows 侧生成标注任务

```bash
python scripts/vision_build_label_studio_tasks.py \
  data/vision/defect_dataset/raw_results \
  --output data/vision/defect_dataset/label_studio/tasks.json
```

### 3. 在 Label Studio 中人工修订

- 使用 `assets/label_studio/defect_detection_config.xml`
- 导入 `tasks.json`
- 导出 JSON

### 4. 转成 YOLO 数据集

```bash
python scripts/vision_convert_labelstudio_to_yolo.py \
  data/vision/defect_dataset/label_studio/export.json \
  --dataset-root data/vision/defect_dataset/yolo_v1
```

### 5. 训练并导出 ONNX

```bash
python scripts/vision_train_defect_yolo.py \
  --data data/vision/defect_dataset/yolo_v1/data.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --export-onnx
```

### 6. 板端切换到缺陷模型

训练完成后，把 ONNX 放到：

`/mnt/ssd/models/vision/defect/yolov8_defect.onnx`

然后把识别配置切到：

`configs/vision_spacemit_defect.yaml`

## 外部公开数据的使用策略

| 优先级 | 用法 |
|---|---|
| P0 | 优先积累自己的板端/现场图像，这决定最终泛化能力 |
| P1 | 用涂层/腐蚀相关公开数据做 warm-start 或补类间差异 |
| P2 | 用通用工业缺陷或异常检测数据做预训练、数据增强和鲁棒性补充 |

## 当前技术判断

| 结论 | 说明 |
|---|---|
| 首版模型不要把 `CUI` 直接当 bbox 类 | 它更像风险推断结论，不是稳定的单帧可见对象 |
| 首版优先做检测，不优先做分割 | 标注成本更低，导出到板端也更容易 |
| 首版继续兼容 YOLOv8 | 更容易接回 `spacemit_vision` 的 `YOLOv8Detector` 链路 |
