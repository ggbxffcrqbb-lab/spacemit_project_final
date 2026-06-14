# Git 维护手册

适用范围：`spacemit_project` 当前阶段的正式维护、协作开发、板端发布与 GitHub 同步。

## 1. 当前 Git 管理总则

| 项目 | 当前约定 |
|---|---|
| 唯一 canonical repo | `/mnt/ssd/spacemit_project` |
| 主开发环境 | Muse Pi Pro 板端，通过 `ssh fyp@192.168.3.38` 进入 |
| GitHub 远端仓库 | `https://github.com/ggbxffcrqbb-lab/spacemit_project.git` |
| Windows 侧仓库角色 | 辅助镜像、文档编辑、临时整理，不是最终真源 |
| 默认主分支 | `main` |
| 历史保护方式 | 重要重构前先打 `backup/*` 分支 |

一句话原则：

1. 真正的代码真源始终在板端 `/mnt/ssd/spacemit_project`
2. GitHub 是对外同步仓，不反过来定义板端状态
3. Windows 侧可以参与编辑，但最终必须回收到板端仓再提交

## 2. 三个位置的职责边界

| 位置 | 路径 | 职责 | 不能替代什么 |
|---|---|---|---|
| 板端 canonical | `/mnt/ssd/spacemit_project` | 正式代码、正式提交、正式运行验证 | 不能被 Windows 临时改动取代 |
| GitHub 远端 | `ggbxffcrqbb-lab/spacemit_project` | 对外备份、协作入口、历史托管 | 不是模型文件仓库 |
| Windows 辅助侧 | `D:\spacemit\spacemit_project` | 文档整理、临时草改、辅助推送 | 不是最终发布依据 |

## 3. 模型与大文件管理规则

| 类型 | 放哪里 | 是否进 Git |
|---|---|---|
| 正式运行模型 | `/mnt/ssd/models` | 否 |
| Windows 中转模型 | `D:\spacemit\tmp` | 否 |
| `assets/bootstrap/` 里的目录结构、词表、说明 | 仓库内 | 是 |
| `assets/bootstrap/` 里的 `.onnx` / `.bin` 权重 | 板端本地或中转目录 | 否 |
| 日志、缓存、索引产物 | `/mnt/ssd/logs`、`data/index` 等 | 否 |

当前必须牢记：

1. GitHub 不承担模型权重托管职责
2. 不要再把 bootstrap `.onnx` / `.bin` 直接 `git add`
3. 模型同步优先走 `scp` / `rsync` 到 `/mnt/ssd/models`

## 4. 当前推荐日常工作流

### 4.1 纯板端开发

| 步骤 | 命令/动作 | 说明 |
|---|---|---|
| 进入板端 | `ssh fyp@192.168.3.38` | 正式开发优先在板端执行 |
| 进入项目 | `cd /mnt/ssd/spacemit_project` | canonical repo 根目录 |
| 修改代码 | 直接在板端改 | 小改动、脚本、调试优先如此 |
| 板端验证 | `bash scripts/voice.sh doctor` 等 | 以板端结果为准 |
| 查看状态 | `git status` | 确认提交范围 |
| 提交 | `git add ... && git commit -m "..."` | 提交发生在板端 repo |

### 4.2 Windows 侧辅助编辑

| 步骤 | 命令/动作 | 说明 |
|---|---|---|
| Windows 侧编辑 | 在 `D:\spacemit\spacemit_project` 修改 | 适合长文档、批量整理 |
| 同步回板端 | `scp` / `rsync` 到 `/mnt/ssd/spacemit_project` | 先回收至 canonical repo |
| 板端验证 | 在板端执行命令 | 不拿 Windows 结果替代板端 |
| 板端提交 | 只在板端 repo 提交 | 保持真源单点清晰 |

## 5. 当前推荐发布流

| 场景 | 默认做法 |
|---|---|
| 板端已具备 GitHub 认证 | 直接在板端 `git push origin main` |
| 板端未具备 GitHub 认证 | 板端提交后，用 Windows 侧凭据辅助推送 |
| 需要对外查看源码 | 以 GitHub `main` 为准 |
| 需要判断真实可运行状态 | 以板端 `main` + 板端验证为准 |

当前项目已验证可行的“辅助推送”思路：

1. 板端先提交 canonical 改动
2. Windows 侧单独拉一个干净镜像或克隆板端 repo
3. 利用 Windows 上已登录的 GitHub 凭据，把这份板端镜像推到 GitHub

## 6. 常用命令模板

### 6.1 板端查看当前状态

```bash
ssh fyp@192.168.3.38 "cd /mnt/ssd/spacemit_project && git status --short --branch && git log --oneline --decorate -n 5"
```

### 6.2 板端提交

```bash
ssh fyp@192.168.3.38 "cd /mnt/ssd/spacemit_project && git add . && git commit -m 'feat: your change'"
```

### 6.3 Windows 侧把单文件同步回板端

```bash
scp D:\spacemit\spacemit_project\README.md \
  fyp@192.168.3.38:/mnt/ssd/spacemit_project/README.md
```

### 6.4 Windows 侧创建一个干净的板端镜像

```bash
git clone ssh://fyp@192.168.3.38/mnt/ssd/spacemit_project D:\spacemit\tmp\board_mirror
```

### 6.5 Windows 侧用镜像推 GitHub

```bash
git -C D:\spacemit\tmp\board_mirror remote set-url origin https://github.com/ggbxffcrqbb-lab/spacemit_project.git
git -C D:\spacemit\tmp\board_mirror push -u origin main
```

## 7. Windows 辅助仓的维护建议

| 建议 | 说明 |
|---|---|
| 把 Windows 工作区视为辅助镜像 | 不把它当最终真源 |
| Windows 仓很脏时，不要硬 `reset --hard` | 优先另建干净镜像 |
| 需要对比板端变化时，优先新 clone | 比在脏工作区强拉更稳 |
| 发布前先看板端提交 | GitHub 应该反映板端 canonical，而不是反过来 |

推荐做法：

1. 日常长期编辑可以继续用 `D:\spacemit\spacemit_project`
2. 真要发布或对齐历史时，单独创建 `D:\spacemit\tmp\board_mirror_*`
3. 用干净镜像对 GitHub 做推送或校验

## 8. 分支策略

| 分支类型 | 作用 | 是否长期保留 |
|---|---|---|
| `main` | 当前正式开发线 | 是 |
| `backup/*` | 重大清理/重写历史前的安全备份 | 视情况保留 |
| `feature/*` | 高风险功能实验或多人协作隔离 | 可选 |

当前建议：

1. 单人快速开发默认直接在板端 `main`
2. 涉及大规模重构、历史清洗、目录迁移时，先打 `backup/*`
3. 如果要做明显高风险实验，再开 `feature/*`

## 9. 提交信息建议

| 前缀 | 用途 |
|---|---|
| `feat:` | 新功能 |
| `fix:` | 缺陷修复 |
| `refactor:` | 重构但不改功能 |
| `docs:` | 文档更新 |
| `chore:` | 维护性调整、仓库治理 |
| `test:` | 测试与验证脚本 |

示例：

```bash
git commit -m "docs: add git maintenance guide"
git commit -m "feat: add stronger knowledge import pipeline"
git commit -m "chore: rebuild canonical repo without bootstrap weights"
```

## 10. 提交前检查清单

| 检查项 | 要点 |
|---|---|
| `git status` | 只包含本次想提交的文件 |
| 板端验证 | 至少跑与改动直接相关的命令 |
| 大文件检查 | 没有误把 `.onnx` / `.bin` / 日志 / 缓存加进来 |
| 文档同步 | 路径、命令、目录说明与当前板端一致 |
| 真源确认 | 最终提交发生在 `/mnt/ssd/spacemit_project` |

## 11. 发生大文件误入 Git 时怎么办

这是高风险场景，处理原则是“先保底，再清历史”。

### 11.1 如果还没提交

```bash
git rm --cached <file>
```

然后：

1. 把对应规则补进 `.gitignore`
2. 确认文件仍保留在板端磁盘或模型目录中
3. 再重新检查 `git status`

### 11.2 如果已经提交但还没对外发布

推荐顺序：

| 步骤 | 动作 |
|---|---|
| 1 | 先打 `backup/*` 备份分支 |
| 2 | 把大文件从索引中摘掉，但保留磁盘文件 |
| 3 | 必要时重建一条干净历史 |
| 4 | 再推 GitHub |

本项目已经实际验证过的一种安全路线：

```bash
git branch backup/pre_cleanup_YYYYMMDD_HHMMSS
git rm --cached <large-files...>
git checkout --orphan github_ready_main
git add -A
git commit -m "chore: rebuild canonical repo without bootstrap weights"
git branch -M main
git push -u origin main --force-with-lease
```

注意：

1. 这种操作只适合在确认“要重建公开历史”时使用
2. 执行前必须先留 `backup/*`
3. 如果对当前历史是否需要保留有疑问，先不要直接 force push

## 12. 当前不推荐的做法

| 不推荐做法 | 原因 |
|---|---|
| 在 Windows 脏工作区直接定义正式版本 | 容易偏离板端真源 |
| 把 `.onnx` / `.bin` 权重提交到 GitHub | 会触发体积限制与仓库膨胀 |
| 不经板端验证就宣称“已跑通” | 不符合项目实际运行平台 |
| 在不备份的情况下重写历史 | 风险过高 |
| 在活跃工作区直接 `git reset --hard` | 容易误伤未整理改动 |

## 13. 后续可升级项

| 升级项 | 价值 |
|---|---|
| 给板端配置 GitHub SSH key | 让 canonical repo 直接推 GitHub |
| 为 Windows 辅助仓单独设置 `canonical` remote | 更方便对照板端 |
| 引入 GitHub Releases / 外部对象存储托管模型 | 进一步解耦代码与模型 |
| 加入简单发布检查脚本 | 自动拦截大文件、日志、缓存误提交 |

## 14. 当前维护结论

| 结论 | 说明 |
|---|---|
| 真源唯一 | `/mnt/ssd/spacemit_project` |
| GitHub 可用 | 当前已可稳定承载源码历史 |
| 模型边界已明确 | 运行模型在 `/mnt/ssd/models`，不再放入 GitHub |
| 后续开发默认路径 | 板端改动、板端验证、板端提交、再同步 GitHub |

后续只要继续遵守这四句话，仓库基本就不会再乱：

1. 先看板端，不先看 Windows
2. 先回收真源，再谈发布
3. 模型不进 Git，代码才进 Git
4. GitHub 是同步仓，不是板端运行时替身
