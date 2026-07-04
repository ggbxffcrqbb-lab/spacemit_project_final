# Phase 5：预览与分析分层说明

适用日期：`2026-06-27`

## 当前结论

| 能力 | 命令 | 作用 |
|---|---|---|
| 实时画面预览 | `vision-preview` | 优先走官方/系统级预览链路，先把摄像头画面稳定显示在屏幕上 |
| 连续分析雏形 | `vision-stream` | 保留后续接实时缺陷框、风险提示和状态页展示的统一入口 |
| 单帧验证 | `vision-camera` | 做一次采集 + 一次推理，适合留痕和调试 |

## 工程原则

| 原则 | 做法 |
|---|---|
| 预览与分析解耦 | `preview` 负责看画面，`stream` 负责持续分析，互不混写 |
| MIPI 优先官方路线 | `mipi_official` 先 `cam-test detect` 生成 `auto.json`，再走 `gst spacemitsrc + waylandsink` |
| USB 预留独立实现 | `usb_v4l2` 的预览与采集单独维护，不改 MIPI 逻辑 |
| 后续实时框能力保留 | `vision-stream` 继续作为后续实时缺陷框的主入口 |

## 当前推荐命令

### 1. MIPI 实时预览

建议直接在 **Muse Pi Pro 本地图形终端** 运行：

```bash
cd /mnt/ssd/spacemit_project
source .venv/bin/activate
PYTHONPATH=. python3 -m app.main \
  --config /mnt/ssd/spacemit_project/configs/vision.yaml \
  vision-preview
```

只查看将要执行的命令：

```bash
cd /mnt/ssd/spacemit_project
source .venv/bin/activate
PYTHONPATH=. python3 -m app.main \
  --config /mnt/ssd/spacemit_project/configs/vision.yaml \
  vision-preview --dry-run
```

### 2. MIPI 连续分析雏形

```bash
cd /mnt/ssd/spacemit_project
source .venv/bin/activate
PYTHONPATH=. python3 -m app.main \
  --config /mnt/ssd/spacemit_project/configs/vision.yaml \
  vision-stream --interval-seconds 1.5
```

### 3. v2 自训练 warm-start 模型切换

```bash
cd /mnt/ssd/spacemit_project
source .venv/bin/activate
PYTHONPATH=. python3 -m app.main \
  --config /mnt/ssd/spacemit_project/configs/vision_defect_exp.yaml \
  vision-stream --interval-seconds 1.5
```

## 当前能力边界

| 项 | 现状 |
|---|---|
| 屏幕实时看画面 | 已有统一 `vision-preview` 入口 |
| 连续分析状态页 | 已有 `vision_status.html/json/txt` |
| 真正实时缺陷框 | 还未完成，当前主要受单帧采集模式限制 |
| USB 预览入口 | 已在代码结构中预留，后续插入 USB 即可继续验证 |

## 当前 MIPI 预览实现

| 步骤 | 实现 |
|---|---|
| 1 | `cam-test /usr/share/camera_json/csi3_camera_detect.json` |
| 2 | 生成 `/tmp/csi3_camera_auto.json` |
| 3 | `gst-launch-1.0 -e spacemitsrc location=/tmp/csi3_camera_auto.json ... ! waylandsink` |
| 4 | `spacemitsrc` 会自动加载 `/usr/share/camera_json/sensor_rear_primary_cpp_preview_setting.data` |

说明：
- 这条链路比直接执行 `cam-test /tmp/csi3_camera_auto.json` 更接近连续实时预览。
- 如果只是在 SSH、串口或非图形 TTY 中执行，链路可能已经进入 `PLAYING`，但你仍然看不到板端桌面窗口。

## 下一步建议

| 优先级 | 工作项 |
|---|---|
| P1 | 继续查官方推荐的连续相机输入/视频流方案，替换当前“每帧重启 cam-test”模式 |
| P1 | 保持 `vision-preview` 只负责看画面，`vision-stream` 只负责分析 |
| P1 | USB 摄像头到位后，优先验证 `vision-preview --backend usb_v4l2` |
| P2 | 在连续采集链路稳定后，再把实时缺陷框叠加接回 `vision-stream` |
