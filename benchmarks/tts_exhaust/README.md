# model-zoo-tts 榨干基准目录

这个目录用于对 `spacemit-com/model-zoo-tts` 做可复现的板端基准与对比。

当前目标：

1. 先确认真实能力边界，不盲信 README 里的 "streaming" 表述。
2. 先跑引擎参数扫描：`preset / provider / num_threads / speech_rate / warmup`。
3. 再比较链路开销：纯合成、保存文件、后续播放链。

约定：

- 正式测试环境是 Muse Pi Pro(K1) 板端。
- 建议通过 `ssh fyp@192.168.3.38` 在板端执行。
- 建议日志落在板端 ` /mnt/ssd/logs/tts_exhaust/ `。
- 不把 Windows 侧结果当最终结论。

已知关键事实：

- 当前仓库的 `StreamingCall()` 在实现上仍是 `Call()` 完成后再回调。
- `StartDuplexStream()` 当前返回 `nullptr`。
- Matcha / Kokoro 后端都声明 `supportsStreaming() == false`。

因此：

- 当前第一阶段重点不是“马上改真双向流式”。
- 而是先把参数上限、引擎热路径、文件 I/O 成本和播放链成本摸清楚。
