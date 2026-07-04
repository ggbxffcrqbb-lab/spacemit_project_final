# Phase 5：官方 MIPI 采集链与视觉识别模块

适用时间：`2026-06-25`

## 1. 当前结论

| 项目 | 当前状态 | 说明 |
|---|---|---|
| 官方 MIPI 探测链 | 已打通 | 项目内已接入 `cam-test + csi*_camera_detect.json -> /tmp/csi*_camera_auto.json` |
| 当前实机传感器 | 已确认 | `csi3 / imx219_spm` |
| 单帧采集闭环 | 已打通 | `cam-test auto.json -> NV12 dump -> PNG` |
| 视觉识别闭环 | 已打通 | `vision-camera` 可输出 `capture + analysis` JSON |
| 当前识别后端 | 基线可用 | 先用 `heuristic_defect` 规则识别器打通链路 |
| 官方模型识别后端 | 已预留接口，尚未启用 | `spacemit_vision` 仍未安装到项目 `.venv` |
| USB 相机支持 | 已做工程隔离 | 预留 `usb_v4l2` 后端，不与 MIPI 冲突 |

## 2. 当前代码结构

| 路径 | 作用 |
|---|---|
| `app/vision/camera_backends.py` | `mipi_official` / `usb_v4l2` 相机后端 |
| `app/vision/recognizer.py` | `heuristic_defect` / `spacemit_vision` 识别后端 |
| `app/vision/image_utils.py` | NV12 -> RGB、图像读写、RGB/BGR 转换 |
| `app/vision/service.py` | 视觉 CLI 服务编排 |
| `app/main.py` | 新增 `vision-doctor / vision-capture-once / vision-image / vision-camera` |
| `configs/vision.yaml` | 视觉默认配置 |
| `scripts/vision.sh` | 板端视觉统一入口 |
| `tests/vision_heuristic_smoke.py` | 基线规则识别 smoke test |

## 3. 已验证命令

板端目录：`/mnt/ssd/spacemit_project`

| 命令 | 结果 |
|---|---|
| `bash scripts/vision.sh vision-doctor --probe` | 成功识别 `csi3 / imx219_spm / /tmp/csi3_camera_auto.json` |
| `bash scripts/vision.sh vision-capture-once` | 成功输出 PNG |
| `bash scripts/vision.sh vision-image /mnt/ssd/logs/spacemit_project/vision/captures/mipi_20260625-011916.png` | 成功输出分析 JSON 与注释图 |
| `bash scripts/vision.sh vision-camera` | 成功输出 `capture + analysis` JSON |

## 4. 当前输出路径

| 类型 | 路径示例 |
|---|---|
| 采集图像 | `/mnt/ssd/logs/spacemit_project/vision/captures/mipi_20260625-012047.png` |
| 注释图像 | `/mnt/ssd/logs/spacemit_project/vision/annotated/mipi_20260625-012047_heuristic.png` |
| 临时 NV12 dump | `/tmp/cpp0_output_1920x1080_s1920.nv12` |
| 临时 RAW dump | `/tmp/raw_output0_1920x1080.raw` |

## 5. 当前工程约束

| 项目 | 约束 |
|---|---|
| MIPI 后端 | 走官方 `cam-test` 链，不走 Python 直连 MIPI 假方案 |
| USB 后端 | 未来单独走 `usb_v4l2`，不修改 MIPI 逻辑 |
| 识别后端 | 采集后端与识别后端解耦，可独立替换 |
| 并发调用 | 不要同时运行两个 `vision-capture-once / vision-camera`，会争抢同一相机资源 |
| 当前识别结论 | `heuristic_defect` 仅用于打通链路，不等价于正式缺陷模型 |

## 6. 下一步建议

| 优先级 | 工作项 | 说明 |
|---|---|---|
| P1 | 在板端安装并接通 `spacemit_vision` | 让识别后端从规则基线升级到官方模型接口 |
| P1 | 选定第一版正式缺陷模型 | 建议先做检测版，再做分割版 |
| P1 | 为 `spacemit_vision` 输出做项目缺陷标签映射 | 把模型标签映射到锈蚀、剥落、粉化、CUI 等项目语义 |
| P2 | 补 `usb_v4l2` 实机验证 | USB 摄像头到位后只验证后端，不动识别器 |
| P2 | 做 `vision-camera --save-json/--session-dir` | 为答辩演示和样本回收留痕 |
| P3 | 再做实时流版 | 当前先稳住单帧链路，后续再上持续帧处理 |

## 7. 官方模型后端接入建议

| 步骤 | 建议 |
|---|---|
| 1 | 板端安装系统依赖：`opencv-spacemit`、`spacemit-onnxruntime`、`libyaml-cpp-dev` |
| 2 | 在板端构建官方 `model-zoo-vision` wheel |
| 3 | 将 wheel 安装到项目 `.venv`，优先尝试 `pip install --no-deps` |
| 4 | 在 `configs/vision.yaml` 中填入 `recognizer.spacemit_vision_config` 与 `spacemit_model_path` |
| 5 | 用 `bash scripts/vision.sh vision-image <image> --recognizer spacemit_vision` 先验静态图 |
| 6 | 再切到 `vision-camera --recognizer spacemit_vision` 做板端实机验证 |
