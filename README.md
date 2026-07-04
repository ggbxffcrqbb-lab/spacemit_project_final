# spacemit_project_phase6

Muse Pi Pro (K1) 板端正式工程。

## 当前结论

| 项目 | 当前状态 |
|---|---|
| 唯一真源 | `/mnt/ssd/spacemit_project` |
| 当前阶段 | `Phase 6` 板端多模态比赛工程阶段 |
| 当前主线 | USB 摄像头 + 板端视觉识别 + ASR/TTS/LLM/RAG 多模态闭环 |
| 正式运行时 | `spacemit-ort` + `ollama` + resident ASR/TTS |
| 板端模型目录 | `/mnt/ssd/models` |
| GitHub 仓库 | `ggbxffcrqbb-lab/spacemit_project_phase6` |
| Windows 侧定位 | 辅助整理、同步中转、文档与答辩材料，不作为板端最终验证口径 |

## Phase 6 当前重点

- 板端正式演示入口已经切换到 Phase 6 多模态链路。
- 当前正式默认模式是板端视觉常驻巡检 + 语音问答触发。
- 视觉链路以 `USB + spacemit-ort` 为正式路线。
- 语音链路采用 resident ASR / TTS / LLM worker，降低首轮加载开销。
- RAG、direct_rag、direct_visual、普通问答已经并入统一控制流。
- 原生嵌入预览、语音播报期间的视觉保护与无感恢复已经纳入正式代码。

## 正式演示入口

```bash
cd /mnt/ssd/spacemit_project
bash scripts/launch_phase6_demo.sh
```

Phase 6 常用控制：

- `Enter` / `a`：语音问答
- `t`：文本问答
- `v`：视觉摘要播报
- `s`：导出当前竞赛画面快照
- `q`：退出

## 当前关键能力

| 模块 | 当前状态 |
|---|---|
| 板端多模态正式工程收口 | 已完成 |
| USB 摄像头实时取流与竞赛大屏 | 已完成 |
| 一级腐蚀分割 + 二级细分类 | 已完成 |
| resident ASR / TTS / LLM worker | 已完成 |
| 普通知识问答 / `direct_rag` / `direct_visual` | 已完成 |
| RAG 本地知识库闭环 | 已完成 |
| CPU 亲和性与线程调优 | 已完成一轮板端基准测试 |
| 语音播报期间视觉保护与恢复 | 已完成并进入正式默认值 |

## 目录重点

| 路径 | 作用 |
|---|---|
| `app/` | 正式业务代码 |
| `app/vision/` | Phase 5-6 视觉链路、相机后端、竞赛显示、识别服务 |
| `app/voice/` | ASR / TTS / LLM / RAG 语音链路 |
| `app/core/` | Phase 6 控制流、事件总线、会话状态、CPU 亲和性 |
| `configs/` | 板端正式配置与实验配置 |
| `scripts/` | 启动、预热、自检、训练与数据处理脚本 |
| `benchmarks/` | Phase 6 板端基准测试脚本与结果 |
| `data/knowledge/` | 本地知识库正文与结构化知识卡 |
| `docs/` | 设计文档、实验记录、阶段说明 |
| `tests/` | RAG 回归与视觉 smoke tests |

## 仓库包含什么

当前 GitHub 仓库已经包含写技术文档最关键的比赛工程内容：

- Phase 6 主代码
- Phase 5-6 视觉代码与配置
- 板端多模态控制流
- 正式启动脚本与实验脚本
- 规则卡 / 视觉卡 / 知识库原始资料
- 关键设计文档
- 基准测试脚本
- 已提交的板端基准测试结果

## 仓库没有包含什么

这个仓库已经是“比赛工程真源仓库”，但它不是“整盘 SSD 镜像”。以下内容是有意不入 Git 的：

- `.venv/` 等本地 Python 环境
- `/mnt/ssd/models` 下的正式模型文件
- `assets/bootstrap/` 下的大体积 `.onnx` / `.bin` 运行时权重
- `data/index/*.json` 这类可重建的 RAG 索引缓存
- `.shaders/`、日志、临时输出、运行缓存
- `.bak_*` 调试备份文件
- 本地外部依赖工作树或构建目录
  - `third_party/model-zoo-vision/`
  - `third_party/onnxruntime-extensions/`
  - `third_party/model-zoo-tts/build_backup_*`

因此：

- 如果你要写技术文档，当前仓库已经足够作为“代码、配置、流程、实验结果”的依据。
- 如果你要写“部署复现文档”，还需要额外说明模型文件、板端运行环境和 `/mnt/ssd/models` 的落点。

## 当前推荐的文档依据

建议优先以这些内容为技术文档主依据：

- `README.md`
- `docs/phase6-multimodal-demo-design.md`
- `docs/phase5-spacemit-vision-upgrade.md`
- `docs/phase5-preview-engineering.md`
- `configs/multimodal_demo.yaml`
- `scripts/launch_phase6_demo.sh`
- `benchmarks/phase6_module_bench.py`
- `benchmarks/results/`

## 板端常用命令

```bash
cd /mnt/ssd/spacemit_project

# 正式 Phase 6 演示
bash scripts/launch_phase6_demo.sh

# 重建知识库
PYTHONPATH=. python3 -m app.main --config configs/voice.yaml rag-rebuild

# 文本查询
PYTHONPATH=. python3 -m app.main --config configs/voice.yaml rag-query "起泡范围过大先做什么"

# RAG 回归
PYTHONPATH=. python3 tests/rag_regression_suite.py

# Phase 6 基准测试
PYTHONPATH=. python3 benchmarks/phase6_module_bench.py
```

## 当前文档口径

如果后续你根据本仓库写比赛技术文档，建议统一使用以下表述：

- 正式工程阶段：`Phase 6`
- 正式真源：`/mnt/ssd/spacemit_project`
- 正式仓库：`ggbxffcrqbb-lab/spacemit_project_phase6`
- 正式板端平台：`Muse Pi Pro (K1)`
- 正式演示形态：`板端视觉巡检 + 本地语音问答 + RAG + 原生嵌入预览`
