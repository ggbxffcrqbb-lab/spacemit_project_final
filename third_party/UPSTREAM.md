# Third-Party Upstream Notes

## `model-zoo-tts`

| Item | Value |
|---|---|
| Upstream | `https://github.com/spacemit-com/model-zoo-tts.git` |
| Snapshot Commit | `787106bb612fdcecbb6240a7ae08218535e529e1` |
| Local Role | Muse Pi Pro 板端 Matcha TTS 原生运行时源码来源 |

本项目已把 `model-zoo-tts` 作为正式工程依赖收编到 `third_party/` 下统一管理。
板端实际运行仍依赖板端已构建好的 `_spacemit_tts` 原生模块，但该构建产物不纳入本仓库版本管理。
