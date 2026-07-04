# Phase 5：官方模型后端的缺陷语义映射层

适用日期：`2026-06-26`

## 当前定位

| 项目 | 当前做法 | 目的 |
|---|---|---|
| 采集后端 | `MIPI -> mipi_official`，未来 `USB -> usb_v4l2` | 保持两种相机链路工程隔离 |
| 模型后端 | `spacemit_vision + YOLOv8` | 先跑通官方模型后端 |
| 缺陷语义输出 | “官方标签直映射 + 启发式回退” | 在缺陷专用模型到位前，先输出项目可用的缺陷术语 |

## 映射策略

| 场景 | 输出策略 |
|---|---|
| 官方模型直接输出 `rust / corrosion / flaking / chalking / cui` 等缺陷类标签 | 直接映射为项目缺陷术语 |
| 官方模型输出的是 `person / kite / car` 等通用 COCO 标签 | 不强行伪造语义，回退到图像启发式缺陷提示 |
| 官方模型没有检出任何目标框 | 仍保留官方后端，随后回退到启发式缺陷提示 |

## 当前项目输出

| 字段 | 含义 |
|---|---|
| `candidates` | 当前项目真正对外输出的缺陷语义结果 |
| `raw_candidates` | 官方模型的原始检测结果，保留用于调试和后续替换模型 |
| `metrics.mapping_strategy` | 当前是 `direct_label_map` 还是 `heuristic_fallback` |

## USB 预留

| 项目 | 路径 |
|---|---|
| USB 独立配置 | `configs/vision_usb.yaml` |
| USB 独立脚本 | `scripts/vision-usb.sh` |
| USB 独立输出目录 | `/mnt/ssd/logs/spacemit_project/vision_usb` |

## 下一步建议

| 优先级 | 工作项 |
|---|---|
| P1 | 收集防腐缺陷图片，替换当前通用 COCO 模型 |
| P1 | 将 `raw_candidates` 与人工标注结果一起落盘，为后续缺陷模型训练做样本回收 |
| P2 | USB 摄像头到货后只验证 `configs/vision_usb.yaml`，不要改当前 MIPI 配置 |
