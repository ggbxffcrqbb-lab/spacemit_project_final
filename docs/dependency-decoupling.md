# official_examples 脱钩说明

## 目标

把历史验证样例从正式工程运行链路中移除，避免业务代码、模型目录和第三方运行时继续直接依赖 `official_examples/`。

## 当前正式约定

| 路径 | 作用 |
|---|---|
| `/mnt/ssd/spacemit_project/app` | 正式业务代码 |
| `/mnt/ssd/spacemit_project/third_party/model-zoo-tts` | Matcha TTS 原生运行时与基准仓 |
| `/mnt/ssd/spacemit_project/assets/bootstrap` | 可选模型种子 |
| `/mnt/ssd/models` | 正式模型目录 |

## 迁移原则

1. ASR / LLM / TTS Python 逻辑保留在 `app/voice/*`
2. Matcha 原生 `.so` 与基准脚本统一指向 `third_party/model-zoo-tts`
3. `/mnt/ssd/models` 作为唯一正式模型落点
4. `official_examples/` 仅允许进入归档区，不再出现在正式运行配置里

## 板端迁移动作建议

```bash
cd /mnt/ssd/spacemit_project
mkdir -p third_party assets/bootstrap/asr assets/bootstrap/tts
mv official_examples/model-zoo-tts-eval/model-zoo-tts third_party/model-zoo-tts
```

如果还需要保留历史模型种子，可把旧样例中的备份模型目录移动到：

```bash
assets/bootstrap/asr/sensevoice-small
assets/bootstrap/tts/melotts
```

其余 `official_examples/` 历史文件建议移到 `/mnt/ssd/cache/spacemit_project_archive/` 归档，不再留在正式工程根目录。
