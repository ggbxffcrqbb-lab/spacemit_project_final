# Bootstrap Assets

这个目录只放“首次补齐 `/mnt/ssd/models` 时可能用到的引导资料”，不是正式运行目录，也不是正式模型仓库。

## 当前规则

| 项目 | 约定 |
|---|---|
| canonical repo | `/mnt/ssd/spacemit_project` |
| 正式模型目录 | `/mnt/ssd/models` |
| Git 中保留的内容 | 配置、说明、脚本、轻量词表/元数据 |
| Git 中不再保留的内容 | `.onnx` / `.bin` 这类 bootstrap 模型权重 |

也就是说：

1. `assets/bootstrap/` 可以保留目录结构和说明
2. 真正的模型权重文件默认只保留在板端本地、Windows 临时中转区，或单独的模型分发位置
3. 不再把 bootstrap 权重直接提交到 GitHub

## 典型来源

| 相对路径 | 说明 |
|---|---|
| `asr/sensevoice-small/` | SenseVoice ASR 引导目录，通常需要 `model_quant.onnx` 等本地文件 |
| `tts/melotts/` | 历史 MeloTTS 引导目录，通常需要 `encoder-zh.onnx` / `decoder-zh.onnx` 等本地文件 |

## 推荐工作流

| 场景 | 推荐做法 |
|---|---|
| Windows 侧收集模型 | 先放到 `D:\spacemit\tmp` |
| 同步到板端 | 用 `scp` / `rsync` 传到 `/mnt/ssd/models` 或板端本地 bootstrap 目录 |
| 板端补齐模型 | 运行 `bash scripts/prepare_models.sh` |

正式运行时统一使用：

- `/mnt/ssd/models/asr/sensevoice-small`
- `/mnt/ssd/models/tts/matcha-tts`
- `/mnt/ssd/models/legacy/melotts`
