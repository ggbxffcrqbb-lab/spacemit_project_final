# Phase 3.5：更强知识库导入 + 板端状态页

| 项目 | 当前默认值 | 说明 |
|---|---|---|
| canonical repo | `/mnt/ssd/spacemit_project` | 板端正式工程唯一真源 |
| 知识库目录 | `/mnt/ssd/spacemit_project/data/knowledge` | 导入后的 `.md` 知识文档统一落点 |
| 索引缓存 | `/mnt/ssd/spacemit_project/data/index/knowledge_index.json` | 本地 RAG 索引缓存 |
| 导入目录分流 | `data/knowledge/imported/*` | 按 `--dest-subdir` 细分来源 |
| 状态页目录 | `/mnt/ssd/logs/spacemit_project/ui` | 输出 `html/json/txt` 三份状态文件 |
| 默认状态页 | `status_page.html` | 浏览器或 WebView 可直接打开 |
| 默认屏幕文本页 | `status_screen.txt` | 适合串口屏、命令行屏、轻量看板 |

## 板端命令

```bash
cd /mnt/ssd/spacemit_project

# 导入单文件
bash scripts/knowledge_import.sh /mnt/ssd/data/manuals/tank_check.txt --category 罐体巡检 --tag 现场

# 递归导入整个目录，并落到指定知识子目录
bash scripts/knowledge_import.sh /mnt/ssd/data/manuals --recursive \
  --category 历史资料 \
  --tag pdf \
  --tag 归档 \
  --title-prefix 导入- \
  --dest-subdir imported/manuals

# 如果只想导入，不立刻重建索引
bash scripts/knowledge_import.sh /mnt/ssd/data/manuals --recursive --no-rebuild

# 单独重建索引
bash scripts/voice.sh rag-rebuild

# 查看状态
bash scripts/voice.sh doctor
```

## 导入器增强点

| 能力 | 说明 |
|---|---|
| 多格式 | 支持 `.md` / `.txt` / `.pdf` / `.docx` |
| 编码兼容 | 文本文件优先尝试 `utf-8 / utf-8-sig / gb18030 / gbk` |
| 目录递归 | `--recursive` 可直接吃整批资料 |
| 元数据固化 | 自动写入来源文件、导入时间、分类、标签 |
| 去重 | 同批次按内容 hash 与目标文件名双重去重 |
| 导入台账 | 自动输出 `knowledge_import_catalog.json` |
| 自动重建 | 导入完成后默认自动重建本地 RAG 索引 |

## 状态页输出

| 文件 | 作用 |
|---|---|
| `/mnt/ssd/logs/spacemit_project/ui/status_page.html` | 图形状态页，适合浏览器/板载屏幕 |
| `/mnt/ssd/logs/spacemit_project/ui/status_state.json` | 给别的脚本或服务读取当前状态 |
| `/mnt/ssd/logs/spacemit_project/ui/status_screen.txt` | 纯文本状态页，适合终端/串口屏 |

状态页会持续显示：

1. 当前阶段与运行状态
2. 最近一次问题与回答
3. 最近引用与 RAG 命中
4. 最近一轮耗时指标
5. 最近问答历史
