# Phase 5：板端缺陷实验模型接入

适用日期：`2026-06-28`

## 板端落点

| 项目 | 路径 |
|---|---|
| 量化实验模型 | `/mnt/ssd/models/vision/defect/yolov8n_corrosion_warmstart_v1.q.onnx` |
| MIPI 实验配置 | `/mnt/ssd/spacemit_project/configs/vision_defect_exp.yaml` |
| USB 实验配置 | `/mnt/ssd/spacemit_project/configs/vision_usb_defect_exp.yaml` |
| `spacemit_vision` 模型配置 | `/mnt/ssd/spacemit_project/configs/vision_spacemit_defect_exp.yaml` |

## 设计原则

| 原则 | 做法 |
|---|---|
| 不覆盖默认链路 | 默认 `vision.yaml` / `vision_usb.yaml` 保持不变 |
| 显式实验切换 | 只在命令里通过 `--config` 指向实验配置 |
| MIPI / USB 不冲突 | 两套相机入口分别保留独立配置文件 |
| 先量化再连续流 | 板端 `SpaceMITExecutionProvider` 连续流默认使用 `.q.onnx` |

## 常用命令

### 1. USB 实验模型健康检查

```bash
cd /mnt/ssd/spacemit_project
source .venv/bin/activate
python3 -m app.main \
  --config /mnt/ssd/spacemit_project/configs/vision_usb_defect_exp.yaml \
  vision-doctor --backend usb_v4l2 --probe
```

### 2. USB 连续流验证

```bash
cd /mnt/ssd/spacemit_project
source .venv/bin/activate
python3 -m app.main \
  --config /mnt/ssd/spacemit_project/configs/vision_usb_defect_exp.yaml \
  vision-stream \
  --backend usb_v4l2 \
  --interval-seconds 0.0 \
  --max-frames 2 \
  --performance-mode
```

### 3. 缺陷样例单图推理

```bash
cd /mnt/ssd/spacemit_project
source .venv/bin/activate
python3 -m app.main \
  --config /mnt/ssd/spacemit_project/configs/vision_usb_defect_exp.yaml \
  vision-image /mnt/ssd/data/vision/defect_samples/corrosion_bi3q3_vlc_extracted00006.jpg \
  --annotated-output /mnt/ssd/logs/spacemit_project/vision_static_defect/corrosion_bi3q3_vlc_extracted00006_spacemit.png \
  --result-json /mnt/ssd/logs/spacemit_project/vision_static_defect/corrosion_bi3q3_vlc_extracted00006_result.json
```

## 当前结论

| 项目 | 结果 |
|---|---|
| 板端模型加载 | 成功，`SpaceMITExecutionProvider` 正常初始化 |
| `vision-doctor` 预检 | 成功，`preflight_errors` 已清空，不再报“非 q.onnx”阻断 |
| USB 连续流 | 成功跑完 `2` 帧，未再触发 `tcm buffer alloc failed for core id 3` |
| 缺陷样例单图推理 | 成功，`raw_detection_count = 4`，`mapping_strategy = direct_label_map` |
| 实机实时画面 | 已能稳定走通采集 + 推理链路；是否检出真实缺陷仍取决于现场画面本身 |

## 已确认的根因

| 现象 | 判断 |
|---|---|
| 旧版 USB 连续流一切到缺陷实验模型就崩 | 旧模型是普通 `ONNX`，不是板端连续流应使用的 `.q.onnx` |
| 板端报 `tcm buffer alloc failed for core id 3` | 根因是 `SpaceMITExecutionProvider + 非量化 ONNX` 组合触发板端资源问题 |
| 量化后链路恢复 | 说明 Phase 5 的主阻塞点不是相机，而是板端模型形态不符合官方量化运行路径 |

## 当前残留问题

| 优先级 | 问题 | 说明 |
|---|---|---|
| P0 | 板端环境仍混装 `onnxruntime` 与 `spacemit-ort` | `vision-doctor` 仍会给出官方 FAQ 风险警告，后续应清理为单一正式运行时 |
| P1 | 现场实拍未必每帧都能直接出缺陷框 | 已知静态缺陷样例可出框，现场画面需要继续针对真实角度、距离、光照做验证 |
| P1 | MPP/GStreamer 仍有退出期警告 | 当前不影响 2 帧验证通过，但后续长时间流测仍需观察稳定性 |

## 下一步建议

| 优先级 | 动作 | 说明 |
|---|---|---|
| P0 | 清理板端运行时 | 卸掉板端不需要的 `onnxruntime`，保留 `spacemit-ort` 正式链路 |
| P0 | 做更长时间 USB 流测 | 把 `--max-frames 2` 提升到 `30` 或 `100`，验证长时间稳定性 |
| P1 | 采集真实缺陷实拍样本 | 用 Muse Pi Pro 实机拍摄赛题现场素材，评估自训练模型域外泛化 |
| P1 | 视结果继续微调 | 若现场漏检明显，再补充真实样本重训并重新走 xquant 量化流程 |
