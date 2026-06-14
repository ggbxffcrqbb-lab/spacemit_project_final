# 项目现状与交接说明

适用时间：`2026-06-15`

适用目的：给新的 Codex 对话、临时接手开发者、后续维护者快速建立项目上下文，避免重复摸底。

## 1. 一句话总览

| 项目 | 当前状态 |
|---|---|
| 项目名称 | `spacemit_project` |
| 目标平台 | `Muse Pi Pro (K1)` |
| 唯一 canonical repo | `/mnt/ssd/spacemit_project` |
| GitHub 仓库 | `https://github.com/ggbxffcrqbb-lab/spacemit_project.git` |
| 当前主分支 | `main` |
| 当前板端提交 | `f25210a` |
| 当前 GitHub `main` | 已与板端 `f25210a` 同步 |

一句话结论：

项目已经完成“板端正式工程收口 + 基础语音链路 + 本地 RAG + 更强知识导入 + 状态页 + Git 管理治理”，现在已经不是散落样例阶段，而是进入“可以持续维护和迭代的板端正式工程”阶段。

## 2. 当前开发进度

### 2.1 已完成部分

| 模块 | 当前进度 | 说明 |
|---|---|---|
| 板端正式工程目录 | 已完成 | 真源固定为 `/mnt/ssd/spacemit_project` |
| GitHub 同步链路 | 已完成 | GitHub 仓库已建立并同步到当前主线 |
| Git 治理 | 已完成 | 已把 bootstrap 大模型权重从 Git 历史/索引剥离 |
| ASR | 已完成基础接入 | 优先使用官方 `model_quant_optimized.onnx` |
| LLM | 已完成基础接入 | 默认 `qwen2.5:0.5b`，保留 `smollm2:135m` 极速模式 |
| TTS | 已完成基础接入 | Matcha TTS 常驻路径已稳定 |
| 常驻语音服务 | 已完成 | `voice-console / warmup / doctor` 已统一 |
| 本地 RAG | 已完成最小可用版 | 本地知识卡 + BM25/关键词检索 + 引用输出 |
| 知识导入器 | 已完成增强版 | 支持 `.md/.txt/.pdf/.docx`、递归导入、分类/标签 |
| 状态页 | 已完成 | 输出 `html/json/txt` 三份状态文件 |
| 维护文档 | 已完成第一轮 | Git 维护手册已补齐 |

### 2.2 当前阶段判断

| 维度 | 判断 |
|---|---|
| 工程化程度 | 已从“样例拼装”进入“正式项目维护态” |
| 板端开发能力 | 已具备 |
| 本地知识库能力 | 已具备最小闭环 |
| GitHub 对外协作能力 | 已具备 |
| 直接板端推 GitHub | 还未最终打通认证 |
| 真正的领域知识规模 | 还偏小，当前主要是种子知识卡 |
| 外设/实机联调完备度 | 仍需按任务继续推进 |

## 3. 当前已验证可用的能力

以下内容已在板端真实执行过，不是纯 Windows 假验证。

| 能力 | 现状 |
|---|---|
| `bash scripts/voice.sh doctor` | 可运行 |
| `bash scripts/voice.sh warmup` | 可运行 |
| `bash scripts/voice.sh voice-console` | 入口已具备 |
| `python -m app.main rag-query` | 可运行 |
| `python -m app.main rag-rebuild` | 可运行 |
| `python -m app.main knowledge-import` | 可运行 |
| 状态页文件输出 | 可运行 |
| GitHub 同步 | 已可通过 Windows 辅助凭据完成 |

## 4. 板端最新运行状态快照

以下状态基于 `2026-06-15` 当次板端 `doctor` 检查。

| 项目 | 当前值 |
|---|---|
| 项目根目录 | `/mnt/ssd/spacemit_project` |
| 模型目录 | `/mnt/ssd/models` |
| ASR 模型目录存在 | 是 |
| ASR 实际使用模型 | `/mnt/ssd/models/asr/sensevoice-small/model_quant_optimized.onnx` |
| TTS 目录存在 | 是 |
| TTS 当前常驻 preset | `matcha_zh` |
| TTS 初始化耗时 | 约 `24.5s` |
| RAG 已启用 | 是 |
| 知识库目录 | `/mnt/ssd/spacemit_project/data/knowledge` |
| 当前知识文档数 | `5` |
| 当前 chunk 数 | `25` |
| 状态页目录 | `/mnt/ssd/logs/spacemit_project/ui` |
| 状态页格式 | `status_page.html / status_state.json / status_screen.txt` |

## 5. 当前项目结构重点

| 路径 | 作用 |
|---|---|
| `app/main.py` | 统一 CLI 入口 |
| `app/voice/service.py` | 常驻语音服务主逻辑 |
| `app/rag/knowledge_base.py` | 本地检索与索引缓存 |
| `app/rag/importer.py` | 强化版知识导入器 |
| `app/ui/status_page.py` | 文本/HTML/JSON 状态页输出 |
| `configs/voice.yaml` | 默认板端配置 |
| `configs/voice_fast.yaml` | 极速模式配置 |
| `scripts/voice.sh` | 统一脚本入口 |
| `scripts/knowledge_import.sh` | 知识导入快捷入口 |
| `scripts/prepare_models.sh` | 板端模型目录校验与补齐脚本 |
| `data/knowledge/` | 本地知识资料 |
| `data/index/` | RAG 索引缓存 |
| `docs/git-maintenance.md` | Git 管理规则 |
| `docs/phase3-rag-ui.md` | RAG + 状态页说明 |

## 6. 当前 Git 管理现状

| 项目 | 当前规则 |
|---|---|
| 真源仓库 | 只认板端 `/mnt/ssd/spacemit_project` |
| Windows 仓库 | 只是辅助镜像，不是最终发布依据 |
| GitHub 仓库 | 对外同步源码，不承担模型权重托管 |
| bootstrap 模型权重 | 不再进入 Git 历史 |
| 大改前保护 | 先打 `backup/*` 分支 |

当前要特别提醒新的 Codex：

1. 如果看到 `D:\spacemit\spacemit_project` 工作区很脏，不要把它当成权威状态
2. 先看板端 `git status` 和板端 `git log`
3. 最终提交默认发生在板端 canonical repo
4. GitHub 已经和板端同步，但板端本机暂时还没有 GitHub 认证能力

## 7. 当前已知限制与未完项

| 项目 | 当前情况 | 影响 |
|---|---|---|
| 板端直推 GitHub | 未配置 GitHub 认证 | 当前仍需 Windows 辅助推送 |
| 知识库规模 | 只有种子知识卡 | 回答质量上限仍受限 |
| 状态页形态 | 文件输出型，不是完整前端应用 | 更适合看板/调试，产品化还可继续做 |
| TTS 冷启动 | 约 24-25 秒 | 首次启动仍偏慢 |
| 外设联调覆盖 | 需按任务继续补 | 相机/麦克风/更多板载能力仍可继续深入 |

## 8. 当前推荐下一步优先级

| 优先级 | 建议工作 |
|---|---|
| P1 | 给板端配置 GitHub SSH key，让 canonical repo 能直接推 GitHub |
| P1 | 导入真实领域资料，扩充知识库而不是只靠 5 份种子知识卡 |
| P1 | 针对真实任务继续板端语音闭环验证 |
| P2 | 优化 TTS 冷启动时间与常驻流程 |
| P2 | 继续把状态页做成更接近实机展示界面的板端页面 |
| P3 | 根据任务再接相机、外设、多模态能力 |

## 9. 新对话接手时，Codex 应该先做什么

这是给“新对话里的 Codex”最重要的一节。

### 9.1 先相信什么

| 应先相信 | 不应先相信 |
|---|---|
| 板端 `/mnt/ssd/spacemit_project` | Windows 本地脏工作区状态 |
| 板端 `git status` / `git log` | 记忆里的旧目录结构 |
| 板端 `doctor` 结果 | Windows 侧模拟运行结果 |

### 9.2 第一轮必做检查

```bash
ssh fyp@192.168.3.38 "cd /mnt/ssd/spacemit_project && git status --short --branch && git log --oneline --decorate -n 5"
ssh fyp@192.168.3.38 "cd /mnt/ssd/spacemit_project && bash scripts/voice.sh doctor"
```

### 9.3 如果需要继续做功能开发

推荐顺序：

1. 先在板端读当前代码与配置
2. 再决定是否需要从 Windows 侧辅助编辑
3. 修改后回板端验证
4. 通过板端 canonical repo 提交

## 10. 建议给新 Codex 的启动提示词

下面这段可以直接复制到新对话里：

```text
请先把 /mnt/ssd/spacemit_project 视为唯一 canonical repo，不要先相信 D:\spacemit\spacemit_project 的本地 git 状态。先通过 SSH 检查板端 git status、git log 和 bash scripts/voice.sh doctor，再基于 docs/project-status-handoff.md、docs/git-maintenance.md、docs/phase3-rag-ui.md 进入工作。当前 GitHub 仓库是 https://github.com/ggbxffcrqbb-lab/spacemit_project.git，但板端本机暂未配置 GitHub 认证，必要时可通过 Windows 侧凭据辅助推送。模型权重不进 Git，正式模型目录是 /mnt/ssd/models。
```

## 11. 交接时的关键注意事项

| 注意项 | 说明 |
|---|---|
| 不要误把 Windows 仓当真源 | 当前本地工作区 git 历史未完全对齐板端 |
| 不要把模型权重重新提交进 Git | 当前治理刚刚收口 |
| 不要跳过板端验证 | 本项目最终以 Muse Pi Pro 为准 |
| 不要贸然重写历史 | 如果必须这么做，先打 `backup/*` |
| 不要把“命令执行成功”说成“实机体验完全验证” | 板端体验类任务仍要看实机效果 |

## 12. 交接结论

| 结论 | 说明 |
|---|---|
| 项目已进入可维护状态 | 不再是样例堆叠阶段 |
| 板端真源明确 | `/mnt/ssd/spacemit_project` |
| GitHub 已可用 | 当前代码和文档已同步 |
| Phase 3.5 已基本落地 | RAG、导入器、状态页、Git 治理都已到位 |
| 后续重点 | 扩知识库、直连 GitHub、继续板端闭环验证 |

如果新 Codex 只记住四句话，记这四句就够了：

1. 先看板端，不先看 Windows
2. 先确认真源，再修改代码
3. 模型不进 Git，代码才进 Git
4. 任何“已完成”都优先以板端验证为准
