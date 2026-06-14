# spacemit_project

Muse Pi Pro (K1) 板端正式工程。

## 当前重点

| 项目 | 说明 |
|---|---|
| 当前阶段 | 已进入 `Phase 3 + 最小可用 RAG` |
| 统一入口 | `python -m app.main` |
| 统一终端脚本 | `scripts/voice.sh` |
| 默认模式 | `qwen2.5:0.5b` |
| 极速模式 | `smollm2:135m` |
| 模型目录 | `/mnt/ssd/models` |
| 第三方依赖目录 | `/mnt/ssd/spacemit_project/third_party` |
| 日志目录 | `/mnt/ssd/logs/spacemit_project/voice` |
| 当前 ASR 策略 | 优先使用官方 `model_quant_optimized.onnx`，缺失时回退到 `model_quant.onnx` |
| 当前 TTS 策略 | 默认常驻 `matcha_zh`，保持板端稳定优先 |
| 当前 RAG 策略 | 本地 `data/knowledge` 种子知识卡 + 板端 BM25/关键词检索 + 回答后附引用 |

## Git 与模型资产约定

| 项目 | 约定 |
|---|---|
| canonical repo | `/mnt/ssd/spacemit_project` |
| GitHub 远端 | 只保存源码、脚本、说明、轻量配置/词表 |
| 不再进 Git 的内容 | bootstrap `.onnx` / `.bin` 模型权重 |
| 正式模型落点 | `/mnt/ssd/models` |
| Windows 模型中转 | `D:\spacemit\tmp` |

这意味着 `assets/bootstrap/` 目录可以保留结构与说明，但真正的 bootstrap 权重默认只保留在板端本地或 Windows 中转目录，不再作为 GitHub 仓库内容长期托管。

更完整的维护规则见：

- [docs/git-maintenance.md](docs/git-maintenance.md)
- [docs/project-status-handoff.md](docs/project-status-handoff.md)

## 常用命令

```bash
cd /mnt/ssd/spacemit_project
bash scripts/bootstrap_board_env.sh
bash scripts/prepare_models.sh

# 默认模式
bash scripts/voice.sh doctor
bash scripts/voice.sh warmup
bash scripts/voice.sh voice-console

# 极速模式
bash scripts/voice.sh --mode fast doctor
bash scripts/voice.sh --mode fast warmup
bash scripts/voice.sh --mode fast voice-console

# 默认 vs 极速实测
bash scripts/benchmark_voice_modes.sh

# RAG 调试
python -m app.main rag-rebuild
python -m app.main rag-query "涂层起泡怎么办"
```

兼容脚本仍然保留：

```bash
bash scripts/doctor_voice.sh
bash scripts/warmup_voice.sh
bash scripts/run_voice_console.sh
bash scripts/doctor_voice_fast.sh
bash scripts/warmup_voice_fast.sh
bash scripts/run_voice_console_fast.sh
```

如需启用官方优化版 ASR 模型，板端建议先安装：

```bash
sudo apt install spacemit-sensevoice-model
```

随后执行：

```bash
bash scripts/prepare_models.sh
```

如需离线导入极速模式 LLM，可把官方 `smollm2:135m` 资产先下载到 Windows，再同步到板端 `/mnt/ssd/models/ollama_import/smollm2_135m`，最后导入到 `ollama` 系统仓库。

## 当前目录约定

| 路径 | 作用 |
|---|---|
| `app/` | 正式业务代码 |
| `configs/` | 板端运行配置 |
| `scripts/` | 板端启动、自检、预热、基准脚本 |
| `benchmarks/` | 可复现性能测试脚本与报告 |
| `third_party/model-zoo-tts` | Matcha TTS 原生运行时与构建产物 |
| `data/knowledge/` | 防腐领域本地知识卡与后续知识库原文资料 |
| `data/index/` | 本地 RAG 索引缓存 |
| `assets/bootstrap/` | 首次补齐 `/mnt/ssd/models` 时可选使用的引导目录；权重文件默认不进 Git |

`official_examples/` 已不再作为正式运行依赖目录，后续仅保留在归档区或外部备份中。
