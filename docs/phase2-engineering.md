# Phase 2 工程骨架说明

| 模块 | 当前落点 | 说明 |
|---|---|---|
| 统一入口 | `python -m app.main` | 支持 `voice-console / warmup / doctor`，并支持 `--mode default|fast` |
| 统一终端开关 | `scripts/voice.sh` | 正式推荐入口，不再要求记忆两套脚本 |
| 语音服务 | `app/voice/service.py` | 常驻 ASR / LLM / TTS，内置 TTS 队列与播放队列 |
| 默认配置 | `configs/voice.yaml` | 默认模式，LLM 使用 `qwen2.5:0.5b` |
| 极速配置 | `configs/voice_fast.yaml` | 保留 ASR / TTS 不变，将 LLM 切到 `smollm2:135m` |
| 日志 | `/mnt/ssd/logs/spacemit_project/voice` | 文本日志 + 每轮 JSONL 指标 |
| 样例收编 | `app/voice/asr_runtime`、`app/voice/llm_runtime`、`app/voice/matcha_tts.py` | 从样例中抽取纯代码并纳入正式工程 |
| 第三方落点 | `third_party/model-zoo-tts` | Matcha 原生模块与基准代码的正式依赖目录 |
| 模型归一 | `scripts/prepare_models.sh` | 以 `/mnt/ssd/models` 为正式模型目录 |
| ASR 优化 | `model_quant_optimized.onnx` | 若板端已安装 `spacemit-sensevoice-model`，则优先使用官方优化版 |
| 双模式对比基准 | `benchmarks/voice_mode_compare.py` | 在板端统一口径对比默认模式与极速模式 |
| 本地 RAG | `app/rag/knowledge_base.py`、`data/knowledge/` | 当前已接入种子知识卡与本地检索，作为 Phase 3 向 Phase 4 的过渡实现 |

当前默认策略说明：`preload_mixed_engine` 保持 `false`，优先保证 `matcha_zh` 常驻稳定启动，避免板端在后台预加载 `matcha_zh_en` 时触发不稳定退出。

当前目录治理说明：`official_examples/` 不再作为运行时必经路径，正式工程只保留：

1. `app/` 下的收编代码
2. `third_party/` 下的必要原生依赖
3. `/mnt/ssd/models` 下的正式模型目录

## 板端推荐命令

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

# 统一基准
bash scripts/benchmark_voice_modes.sh
```

兼容脚本 `run_voice_console*.sh / warmup_voice*.sh / doctor_voice*.sh` 目前仍保留，但已经退化为对统一入口的薄包装。

## 当前新增板端调试命令

```bash
cd /mnt/ssd/spacemit_project
python -m app.main rag-rebuild
python -m app.main rag-query "储罐外壁腐蚀如何初判"
```
