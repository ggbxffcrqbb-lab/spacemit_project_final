# spacemit_project

Muse Pi Pro (K1) 板端正式工程。

## 当前结论

| 项目 | 当前状态 |
|---|---|
| 唯一真源 | `/mnt/ssd/spacemit_project` |
| 当前阶段 | `Phase 5` 板端视觉防腐演示阶段 |
| 当前主线 | USB 摄像头实时两级腐蚀识别演示 |
| 正式运行时 | `spacemit-ort` |
| 板端模型目录 | `/mnt/ssd/models` |
| GitHub 仓库 | `ggbxffcrqbb-lab/spacemit_project_phase1-5` |
| Windows 侧定位 | 辅助整理、副本备份、文档与同步中转，不作为板端最终验证口径 |

## Phase 5 当前重点

当前正式演示入口已经切换到板端视觉链路，重点不再只是 Phase 4 的知识库与语音，而是：

- 图像处理 + 实时显示链路
- 一级 1 类腐蚀分割
- 二级 3 类腐蚀细分分类
- `q.onnx` 量化模型板端推理
- USB 摄像头实时演示入口

## 正式演示入口

板端正式演示脚本：

```bash
cd /mnt/ssd/spacemit_project
bash scripts/launch_corrosion_two_stage_rt_demo.sh
```

对应主配置：

- `configs/vision_usb_corrosion_two_stage_rt.yaml`

该入口实际执行的是：

- `vision-stream`
- `--backend usb_v4l2`
- `--display-competition`

## 当前模型链路

### 一级模型：1 类腐蚀分割

- 配置：`configs/vision_spacemit_corrosion_seg_1cls_v1.yaml`
- 模型：`/mnt/ssd/models/vision/corrosion_two_stage/seg/yolov8n_seg_corrosion_1cls_v1.q.onnx`
- 标签：`assets/labels/corrosion_seg_1cls.txt`
- 类别：`corrosion`

### 二级模型：3 类腐蚀分类

- 模型：`/mnt/ssd/models/vision/corrosion_two_stage/cls/yolov8n_cls_corrosion_3cls_v1.q.onnx`
- 标签：`assets/labels/corrosion_cls_3cls.txt`
- 类别：
  - `crevice_corrosion`
  - `pitting_corrosion`
  - `uniform_corrosion`

## 当前已完成能力

- 板端图像采集、推理、叠框、实时显示链路已经打通
- 一级分割 + 二级分类两级推理已经接入实时摄像头流
- 板端正式运行已切到 `q.onnx + spacemit-ort`
- 演示入口已经针对实时流畅性做了节流与日志降噪处理
- Phase 4 的语音、RAG、状态页能力继续保留，可作为辅助能力调用

## 关键目录

| 路径 | 用途 |
|---|---|
| `app/` | 正式业务代码 |
| `configs/` | 板端运行配置 |
| `scripts/` | 启动、测试、同步、诊断脚本 |
| `docs/` | Phase 5 文档、实验记录、路线说明 |
| `assets/labels/` | 腐蚀分割 / 分类标签 |
| `/mnt/ssd/models` | 板端正式模型目录 |
| `/mnt/ssd/logs/spacemit_project` | 日志与输出目录 |

## 板端常用命令

```bash
cd /mnt/ssd/spacemit_project

# 正式实时演示
bash scripts/launch_corrosion_two_stage_rt_demo.sh

# 查看视觉日志目录
ls /mnt/ssd/logs/spacemit_project/vision_usb_corrosion_two_stage_rt
```

## 说明

- 之前 README 中仍写 `Phase 4`，仅表示文档未及时更新，不代表板端主线仍停留在 Phase 4。
- 当前仓库主线已经进入 `Phase 5`，正式演示口径以板端 `/mnt/ssd/spacemit_project` 和实时视觉链路为准。
