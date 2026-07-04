# Phase 5: spacemit_vision Official Backend Upgrade

适用日期：`2026-06-25`

## 当前结论

| 项目 | 状态 | 说明 |
|---|---|---|
| 板端官方 `spacemit_vision` | 已安装 | 已在 `Muse Pi Pro` 的项目虚拟环境 `/mnt/ssd/spacemit_project/.venv` 内完成安装 |
| 官方原生库构建 | 已完成 | 基于 `third_party/model-zoo-vision` 构建 `libvision.so` 与 `_vision_service_cpp` |
| 项目默认识别后端 | 已切换 | `configs/vision.yaml`、`voice.yaml`、`voice_fast.yaml` 默认从 `heuristic_defect` 切到 `spacemit_vision` |
| 默认官方模型 | 已接通 | 当前使用 `YOLOv8n no_dfl` 官方 ONNX 模型 |
| 相机工程隔离 | 保持 | `MIPI` 继续走 `mipi_official`；未来 `USB` 继续走 `usb_v4l2`，两条链路不冲突 |

## 板端落点

| 类型 | 路径 |
|---|---|
| 板端主项目目录 | `/mnt/ssd/spacemit_project` |
| 官方仓库镜像 | `/mnt/ssd/spacemit_project/third_party/model-zoo-vision` |
| 项目虚拟环境 | `/mnt/ssd/spacemit_project/.venv` |
| 官方模型配置 | `/mnt/ssd/spacemit_project/configs/vision_spacemit_yolov8.yaml` |
| 官方模型文件 | `/mnt/ssd/models/vision/yolov8/yolov8n_no_dfl.q.onnx` |

## 说明

| 项目 | 说明 |
|---|---|
| 本次升级内容 | 是“规则基线识别器 -> 官方模型后端”的工程切换，不是“防腐缺陷专用模型训练完成” |
| 当前模型语义 | 仍是通用目标检测标签，项目后续还需要做“官方输出标签 -> 防腐缺陷语义”映射，或替换成自训练缺陷模型 |
| 当前推荐验证顺序 | 先 `vision-image` 静态图验证，再 `vision-camera` 走 MIPI 实拍验证 |
| USB 摄像头后续接入 | 只切换 `vision.backend=usb_v4l2` 或增加独立 USB 配置，不要改动当前 `spacemit_vision` 识别层 |

## 下一步

| 优先级 | 工作项 | 目标 |
|---|---|---|
| P1 | 做标签映射层 | 把 `COCO/官方类别` 映射为项目自己的“锈蚀/剥落/粉化/CUI 风险”等术语 |
| P1 | 引入缺陷专用数据集 | 为后续微调或替换官方通用模型做准备 |
| P2 | 增加 `vision-usb.yaml` | USB 摄像头到货后单独接入，不干扰 MIPI 配置 |
| P2 | 增加结果落盘规范 | 固定保存原图、标注图、JSON 结果，便于答辩与回归 |
