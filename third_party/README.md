# Third-Party Runtime Layout

正式工程中的第三方运行时统一放在这里。

当前约定：

| 路径 | 说明 |
|---|---|
| `model-zoo-tts/` | Matcha TTS 原生模块、构建产物与基准依赖 |

业务代码只应依赖 `third_party/`，不再直接依赖 `official_examples/`。
